import os, sys, base64, threading, uuid, time, logging
from typing import Dict, Optional, Tuple
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel
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


class GenerateResponse(BaseModel):
    layout: Dict
    svg_data_url: str
    saved_svg_path: Optional[str] = None
    saved_json_path: Optional[str] = None
    svg_filename: Optional[str] = None
    json_filename: Optional[str] = None
    generation_time: float


class ErrorResponse(BaseModel):
    code: str
    message: str

app = FastAPI(title="Blueprint Generator API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def metrics_middleware(request: Request, call_next):
    start_time = time.perf_counter()
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

# simple in-memory rate limiter: requests per IP per minute
RATE_LIMIT = int(os.environ.get("RATE_LIMIT", "60"))
_WINDOW_SECONDS = 60
_request_counts: Dict[str, Tuple[float, int]] = {}

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
    return JSONResponse(status_code=exc.status_code, content=content)


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    logger.exception("Unhandled exception: %s", exc)
    return JSONResponse(
        status_code=500,
        content={"code": "internal_error", "message": "Internal server error"},
    )


@app.get("/metrics")
def metrics():
    return Response(generate_latest(PROM_REGISTRY), media_type=CONTENT_TYPE_LATEST)


@app.post(
    "/generate",
    response_model=GenerateResponse,
    responses={429: {"model": ErrorResponse}, 500: {"model": ErrorResponse}},
)
def generate(
    request: Request,
    params: Params,
    strategy: str = "greedy",
    temperature: float = 1.0,
    beam_size: int = 5,
):
    # rate limiting
    ip = request.client.host if request.client else "anonymous"
    now = time.time()
    window_start, count = _request_counts.get(ip, (now, 0))
    if now - window_start >= _WINDOW_SECONDS:
        window_start, count = now, 0
    if count >= RATE_LIMIT:
        raise HTTPException(
            status_code=429,
            detail={"code": "rate_limit_exceeded", "message": "Too many requests"},
        )
    _request_counts[ip] = (window_start, count + 1)

    model = _get_model()

    start_t = time.perf_counter()

    prefix = _tokenizer.encode_params(params.model_dump())
    layout_tokens = decode(
        model,
        prefix,
        max_len=160,
        strategy=strategy,
        temperature=temperature,
        beam_size=beam_size,
    )
    layout_json = _tokenizer.decode_layout_tokens(layout_tokens)
    layout_json = enforce_min_separation(layout_json)

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

    generation_time = time.perf_counter() - start_t

    logger.info(
        "Generated blueprint %s in %.3fs", base, generation_time
    )

    return GenerateResponse(
        layout=layout_json,
        svg_data_url=svg_to_data_url(svg_path),
        saved_svg_path=svg_path,
        saved_json_path=json_path,
        svg_filename=svg_filename,
        json_filename=json_filename,
        generation_time=generation_time,
    )
