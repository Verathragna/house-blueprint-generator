PY=python

data:
	$(PY) dataset/generate_dataset.py --strict
	$(PY) scripts/build_jsonl.py --strict

train:
	$(PY) training/train.py --epochs 20 --batch 16

infer:
	$(PY) Generate/generate_blueprint.py --params_json sample_params.json --out_prefix out/blueprint

api:
	uvicorn api.app:app --host 0.0.0.0 --port 8000 --reload
