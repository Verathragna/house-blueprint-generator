import os, sys, base64, threading, uuid
from typing import Dict, Optional
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
sys.path.insert(0, REPO_ROOT)

from tokenizer.tokenizer import BlueprintTokenizer
from models.layout_transformer import LayoutTransformer
from models.decoding import decode
from dataset.render_svg import render_layout_svg
from Generate.params import Params

CHECKPOINT = os.path.join(REPO_ROOT, "checkpoints", "model_latest.pth")
DEVICE = os.environ.get("DEVICE", "cpu")


class GenerateResponse(BaseModel):
    layout: Dict
    svg_data_url: str
    saved_svg_path: Optional[str] = None
    saved_json_path: Optional[str] = None
    svg_filename: Optional[str] = None
    json_filename: Optional[str] = None

app = FastAPI(title="Blueprint Generator API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_tokenizer = BlueprintTokenizer()
_model = None
_lock = threading.Lock()

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

@app.post("/generate", response_model=GenerateResponse)
def generate(
    params: Params,
    strategy: str = "greedy",
    temperature: float = 1.0,
    beam_size: int = 5,
):
    model = _get_model()

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

    return GenerateResponse(
        layout=layout_json,
        svg_data_url=svg_to_data_url(svg_path),
        saved_svg_path=svg_path,
        saved_json_path=json_path,
        svg_filename=svg_filename,
        json_filename=json_filename,
    )
