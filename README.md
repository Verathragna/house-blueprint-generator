# House Blueprint Generator

End-to-end system to generate residential blueprints from parameters:
- Synthetic data generator (JSON + SVG)
- Tokenizer with PAD/BOS/EOS/SEP + rich parameter support
- Transformer training (params + <SEP> → layout)
- Greedy decoding to produce layouts
- FastAPI service for web calls

## Quickstart

```bash
pip install -r requirements.txt

# 1) Build synthetic data + pairs
python dataset/generate_dataset.py
python scripts/build_jsonl.py

# 2) Train
python training/train.py --epochs 10 --batch 16

# 3) Generate locally (set device to "cpu" or "cuda")
python Generate/generate_blueprint.py --params_json sample_params.json --out_prefix my_blueprint --device cuda

# 4) Run API (optional DEVICE env var controls model device)
DEVICE=cuda uvicorn api.app:app --host 0.0.0.0 --port 8000 --reload
# POST to /generate with params JSON; response contains layout JSON and data-URL SVG
```
