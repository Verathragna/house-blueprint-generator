import os, sys, base64, threading, uuid, time, logging, queue, asyncio
from typing import Any, Dict, Optional, Tuple, List
from fastapi import (
    FastAPI,
    HTTPException,
    Request,
    WebSocket,
    WebSocketDisconnect,
    Depends,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
from fastapi.exceptions import RequestValidationError
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, ValidationError
from prometheus_client import (
    Counter,
    Histogram,
    CollectorRegistry,
    generate_latest,
    CONTENT_TYPE_LATEST,
)

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
sys.path.insert(0, REPO_ROOT)

from tokenizer.tokenizer import BlueprintTokenizer
from models.layout_transformer import LayoutTransformer
from models.decoding import decode
from dataset.render_svg import render_layout_svg
from evaluation.validators import enforce_min_separation
from Generate.params import Params

CHECKPOINT = os.path.join(REPO_ROOT, "checkpoints", "model_latest.pth")
DEVICE = os.environ.get("DEVICE", "cpu")


def _read_version(component: str) -> str:
    """Read the VERSION file for a component."""
    path = os.path.join(REPO_ROOT, component, "VERSION")
    if not os.path.exists(path):
        raise RuntimeError(f"Missing VERSION file for {component}")
    with open(path, "r", encoding="utf-8") as f:
        return f.read().strip()


MODEL_VERSION = _read_version("models")
TOKENIZER_VERSION = _read_version("tokenizer")
if MODEL_VERSION != TOKENIZER_VERSION:
    raise RuntimeError(
        f"Model version {MODEL_VERSION} is incompatible with tokenizer version {TOKENIZER_VERSION}"
    )


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("blueprint_api")

# Simple API key auth
API_KEYS = set(filter(None, os.environ.get("API_KEYS", "testkey").split(",")))


def _get_api_key(request: Request) -> str:
    api_key = request.headers.get("X-API-Key")
    if not api_key or api_key not in API_KEYS:
        raise HTTPException(
            status_code=401,
            detail={"code": "unauthorized", "message": "Invalid API key"},
        )
    return api_key


# Prometheus metrics
PROM_REGISTRY = CollectorRegistry()
REQUEST_COUNT = Counter(
    "request_total", "Total HTTP requests", ["method", "endpoint", "http_status"],
    registry=PROM_REGISTRY,
)
REQUEST_LATENCY = Histogram(
    "request_latency_seconds", "Latency of HTTP requests", ["endpoint"],
    registry=PROM_REGISTRY,
)
ERROR_COUNT = Counter(
    "request_errors_total", "Total HTTP errors",
    registry=PROM_REGISTRY,
)


class Metadata(BaseModel):
    processing_time: float


class GenerateRequest(BaseModel):
    params: Params
    strategy: str = Field(default="greedy", pattern="^(greedy|beam|sample)$")
    temperature: float = Field(default=1.0, ge=0.0)
    beam_size: int = Field(default=5, ge=1)
    min_separation: float = Field(default=1.0, ge=0.0)


class GenerateResponse(BaseModel):
    layout: Dict
    svg_data_url: str
    saved_svg_path: str
    saved_json_path: str
    svg_filename: str
    json_filename: str
    metadata: Metadata


class ErrorResponse(BaseModel):
    code: str
    message: str
    details: Optional[Any] = None
    metadata: Metadata


class JobResponse(BaseModel):
    job_id: str


class JobStatus(BaseModel):
    status: str
    logs: List[str]
    result: Optional[GenerateResponse] = None
    error: Optional[str] = None

app = FastAPI(title="Blueprint Generator API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def logging_middleware(request: Request, call_next):
    body = await request.body()
    if body:
        logger.info("Request %s %s body: %s", request.method, request.url.path, body.decode("utf-8", "ignore"))
    else:
        logger.info("Request %s %s", request.method, request.url.path)

    async def receive():
        return {"type": "http.request", "body": body}

    request._receive = receive  # type: ignore
    response = await call_next(request)

    resp_body = b""
    async for chunk in response.body_iterator:
        resp_body += chunk
    logger.info(
        "Response %s %s status %s body: %s",
        request.method,
        request.url.path,
        response.status_code,
        resp_body.decode("utf-8", "ignore"),
    )
    return Response(
        content=resp_body,
        status_code=response.status_code,
        headers=dict(response.headers),
        media_type=response.media_type,
    )


@app.middleware("http")
async def metrics_middleware(request: Request, call_next):
    start_time = time.perf_counter()
    request.state.start_time = start_time
    try:
        response = await call_next(request)
        status_code = response.status_code
    except Exception as exc:
        status_code = 500
        ERROR_COUNT.inc()
        logger.exception("Unhandled exception during request: %s", exc)
        raise
    finally:
        duration = time.perf_counter() - start_time
        endpoint = request.url.path
        REQUEST_COUNT.labels(request.method, endpoint, status_code).inc()
        REQUEST_LATENCY.labels(endpoint).observe(duration)
        logger.info(
            "%s %s -> %s in %.3fs",
            request.method,
            endpoint,
            status_code,
            duration,
        )
    return response


_tokenizer = BlueprintTokenizer()
_model = None
_lock = threading.Lock()

# simple in-memory rate limiter: requests per API key per minute
RATE_LIMIT = int(os.environ.get("RATE_LIMIT", "60"))
_WINDOW_SECONDS = 60
_request_counts: Dict[str, Tuple[float, int]] = {}

# Background job queue and status store
_task_queue: "queue.Queue[Tuple[str, Params, str, float, int, float]]" = queue.Queue()
_jobs: Dict[str, Dict[str, Any]] = {}


def _worker():
    while True:
        job_id, params, strategy, temperature, beam_size, min_sep = _task_queue.get()
        job = _jobs[job_id]
        try:
            job["status"] = "in_progress"
            job["start_time"] = time.perf_counter()
            job["logs"].append("Job started")
            job["event"].set()

            model = _get_model()
            job["logs"].append("Model loaded")
            job["event"].set()

            prefix = _tokenizer.encode_params(params.model_dump())
            job["logs"].append("Decoding layout")
            job["event"].set()
            layout_tokens = decode(
                model,
                prefix,
                max_len=160,
                strategy=strategy,
                temperature=temperature,
                beam_size=beam_size,
            )
            layout_json = _tokenizer.decode_layout_tokens(layout_tokens)
            if min_sep > 0:
                layout_json = enforce_min_separation(layout_json, min_sep)
            job["logs"].append("Rendering layout")
            job["event"].set()

            out_dir = os.path.join(REPO_ROOT, "generated")
            os.makedirs(out_dir, exist_ok=True)
            base = uuid.uuid4().hex
            json_filename = f"{base}.json"
            svg_filename = f"{base}.svg"
            json_path = os.path.join(out_dir, json_filename)
            svg_path = os.path.join(out_dir, svg_filename)

            with open(json_path, "w", encoding="utf-8") as f:
                import json as pyjson
                pyjson.dump(layout_json, f, indent=2)
            render_layout_svg(layout_json, svg_path)

            processing_time = time.perf_counter() - job["start_time"]
            result = GenerateResponse(
                layout=layout_json,
                svg_data_url=svg_to_data_url(svg_path),
                saved_svg_path=svg_path,
                saved_json_path=json_path,
                svg_filename=svg_filename,
                json_filename=json_filename,
                metadata=Metadata(processing_time=processing_time),
            )

            job["status"] = "completed"
            job["result"] = result
            job["logs"].append("Job completed")
            job["event"].set()
        except Exception as exc:
            logger.exception("Job %s failed: %s", job_id, exc)
            job["status"] = "failed"
            job["error"] = str(exc)
            job["logs"].append(f"Error: {exc}")
            job["event"].set()
        finally:
            _task_queue.task_done()


_worker_thread = threading.Thread(target=_worker, daemon=True)
_worker_thread.start()

def _get_model():
    global _model
    with _lock:
        if _model is None:
            import torch
            if not os.path.exists(CHECKPOINT):
                raise FileNotFoundError("Model checkpoint not found. Train first.")
            _model = LayoutTransformer(_tokenizer.get_vocab_size())
            _model.load_state_dict(torch.load(CHECKPOINT, map_location=DEVICE))
            _model.to(DEVICE)
            _model.eval()
        return _model


def svg_to_data_url(svg_path: str) -> str:
    with open(svg_path, "rb") as f:
        data = f.read()
    b64 = base64.b64encode(data).decode("utf-8")
    return f"data:image/svg+xml;base64,{b64}"

@app.get("/health")
def health():
    return {"ok": True}


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    logger.warning("HTTPException %s: %s", exc.status_code, exc.detail)
    detail = exc.detail
    if isinstance(detail, dict):
        content = detail
    else:
        content = {"code": "error", "message": str(detail)}
    processing_time = time.perf_counter() - getattr(
        request.state, "start_time", time.perf_counter()
    )
    content["metadata"] = {"processing_time": processing_time}
    return JSONResponse(status_code=exc.status_code, content=content)


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    logger.exception("Unhandled exception: %s", exc)
    processing_time = time.perf_counter() - getattr(
        request.state, "start_time", time.perf_counter()
    )
    return JSONResponse(
        status_code=500,
        content={
            "code": "internal_error",
            "message": "Internal server error",
            "metadata": {"processing_time": processing_time},
        },
    )


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    logger.warning("Validation error: %s", exc)
    processing_time = time.perf_counter() - getattr(
        request.state, "start_time", time.perf_counter()
    )
    return JSONResponse(
        status_code=422,
        content={
            "code": "validation_error",
            "message": "Invalid request",
            "details": exc.errors(),
            "metadata": {"processing_time": processing_time},
        },
    )


@app.get("/metrics")
def metrics():
    return Response(generate_latest(PROM_REGISTRY), media_type=CONTENT_TYPE_LATEST)


@app.post(
    "/generate",
    response_model=JobResponse,
    responses={
        429: {"model": ErrorResponse},
        500: {"model": ErrorResponse},
        422: {"model": ErrorResponse},
        401: {"model": ErrorResponse},
    },
)
def generate(
    request: Request,
    req: GenerateRequest,
    api_key: str = Depends(_get_api_key),
):
    try:
        params = Params.model_validate(req.params.model_dump())
    except ValidationError as e:
        raise HTTPException(
            status_code=422,
            detail={"code": "validation_error", "message": "Invalid parameters", "details": e.errors()},
        )

    strategy = req.strategy
    temperature = req.temperature
    beam_size = req.beam_size
    min_sep = req.min_separation

    # rate limiting per API key
    now = time.time()
    window_start, count = _request_counts.get(api_key, (now, 0))
    if now - window_start >= _WINDOW_SECONDS:
        window_start, count = now, 0
    if count >= RATE_LIMIT:
        raise HTTPException(
            status_code=429,
            detail={"code": "rate_limit_exceeded", "message": "Too many requests"},
        )
    _request_counts[api_key] = (window_start, count + 1)

    job_id = uuid.uuid4().hex
    _jobs[job_id] = {
        "status": "queued",
        "logs": ["Job queued"],
        "result": None,
        "error": None,
        "event": threading.Event(),
    }
    _task_queue.put((job_id, params, strategy, temperature, beam_size, min_sep))
    _jobs[job_id]["event"].set()
    return JobResponse(job_id=job_id)


@app.get(
    "/status/{job_id}",
    response_model=JobStatus,
    responses={404: {"model": ErrorResponse}, 401: {"model": ErrorResponse}},
)
def job_status(job_id: str, api_key: str = Depends(_get_api_key)):
    job = _jobs.get(job_id)
    if not job:
        raise HTTPException(
            status_code=404,
            detail={"code": "not_found", "message": "Job not found"},
        )
    data: Dict[str, Any] = {"status": job["status"], "logs": job["logs"]}
    if job["status"] == "completed":
        data["result"] = job["result"]
    if job["status"] == "failed":
        data["error"] = job["error"]
    return JobStatus(**data)


@app.websocket("/ws/{job_id}")
async def job_updates(websocket: WebSocket, job_id: str):
    api_key = websocket.headers.get("x-api-key")
    if not api_key or api_key not in API_KEYS:
        await websocket.close(code=1008)
        return
    job = _jobs.get(job_id)
    if job is None:
        await websocket.close(code=1008)
        return
    await websocket.accept()
    last = 0
    try:
        while True:
            while last < len(job["logs"]):
                await websocket.send_json({"log": job["logs"][last]})
                last += 1
            if job["status"] == "completed":
                await websocket.send_json(
                    {"status": "completed", "result": job["result"].model_dump()}
                )
                break
            if job["status"] == "failed":
                await websocket.send_json({"status": "failed", "error": job["error"]})
                break
            await asyncio.get_event_loop().run_in_executor(None, job["event"].wait)
            job["event"].clear()
    except WebSocketDisconnect:
        logger.info("WebSocket disconnected for job %s", job_id)
    finally:
        await websocket.close()

# Serve built front-end assets if present
frontend_dir = os.path.join(REPO_ROOT, "interface", "web", "dist")
if os.path.isdir(frontend_dir):
    app.mount("/app", StaticFiles(directory=frontend_dir, html=True), name="frontend")
