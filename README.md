# House Blueprint Generator

End-to-end system to generate residential blueprints from parameters:
* Synthetic data generator (JSON + SVG)
* Tokenizer with PAD/BOS/EOS/SEP + rich parameter support
* Transformer training (params + `<SEP>` → layout)
* Greedy decoding to produce layouts
* FastAPI service for web calls

## Installation

```bash
pip install -r requirements.txt
```

## Dataset Generation

Generate a synthetic dataset and build shuffled train/validation pairs:

```bash
python dataset/generate_dataset.py
python scripts/build_jsonl.py --seed 42  # use --seed for reproducible shuffling
```

## Training

Train the transformer model on the generated dataset:

```bash
python training/train.py --epochs 10 --batch 16
```

## Inference (CPU/GPU)

### Command Line

Run local generation on CPU or GPU by setting the `--device` flag:

```bash
python Generate/generate_blueprint.py --params_json sample_params.json --out_prefix my_blueprint --device cuda
```

### API

Serve the model via FastAPI. The `DEVICE` environment variable controls whether the model runs on `cpu` or `cuda`:

```bash
DEVICE=cuda uvicorn api.app:app --host 0.0.0.0 --port 8000 --reload
# POST to /generate with params JSON; response contains layout JSON and data-URL SVG
```

### Simple CLI

For non-technical users, a basic command line interface is available:

```bash
python interface/cli.py --params sample_params.json
```

The `--seed` flag on `build_jsonl.py` ensures the shuffled train/val split is reproducible.
