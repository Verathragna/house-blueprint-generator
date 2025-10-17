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

Synthetic layouts are produced with parameter files in `dataset/params/` and
converted into paired JSONL records for training. Run:

```bash
python dataset/generate_dataset.py --seed 42 --strict       # create JSON + SVG pairs
python scripts/build_jsonl.py --seed 42 --strict            # shuffle into train/val splits
```

Use the same `--seed` value for both commands to ensure full reproducibility.

By default the generator rotates through several circulation templates (linear
chains, central corridors, and L-shaped wings) to encourage more realistic room
adjacencies. Override the sampler with `--patterns chain,corridor,l_shape` or
pick a single pattern when you want a focused dataset. You can also supply `--external_dir`
to ingest curated floor plans alongside the synthetic samples.

All layouts are scaled to fit within a 40×40 coordinate space. Rooms whose
positions or dimensions would exceed these bounds are scaled down before being
written. The preprocessing step in `scripts/build_jsonl.py` verifies that every
room supplies both `x` and `y` coordinates and, by default, enforces the `[0, 40]`
range, raising an error if a room is missing a coordinate or lies outside the
bounds. Pass `--skip-bounds-check` to disable the range validation.

## Evaluation

Use the helper script `evaluation/evaluate_sample.py` to validate generated
layouts. Passing `--strict` causes the command to exit with a non-zero status if
any geometry or parameter issues are detected, which is useful for automated
checks. Add `--json-report report.json` to emit a machine-readable summary:

```bash
python evaluation/evaluate_sample.py --params sample_params.json \
       --layout my_layout.json --svg_out check.svg --strict \
       --json-report report.json
```

The script renders an SVG for visual inspection and prints warnings or errors
for any detected problems.

## Training

Train the transformer using the prepared JSONL files. The example below trains
for 10 epochs with a batch size of 16 on the default device (CPU unless CUDA is
available). Each epoch updates `checkpoints/model_latest.pth`, so inference can
run immediately after training finishes. Pass `--save_weights` if you also want
per-epoch raw weight snapshots saved alongside the optimizer state checkpoints.

```bash
python training/train.py --epochs 10 --batch 16
```

After supervised training, you can run feedback-driven fine-tuning to penalise
layouts that still trigger validator issues:

```bash
python training/rl_finetune.py --episodes 2000 --batch-size 4 \
       --checkpoint-in checkpoints/model_latest.pth \
       --checkpoint-out checkpoints/model_rl_latest.pth
```

The script samples parameters, generates layouts, logs validator feedback to
`logs/rl_feedback.log`, and applies a reward that favours overlap-free, fully
specified floor plans.

## Inference (CPU/GPU)

### Command Line

Run local generation on CPU or GPU by setting the `--device` flag. The script
uses `checkpoints/model_latest.pth` by default, but it now falls back to the
newest `epoch_*.pt` checkpoint if the latest weights are missing. Provide
`--checkpoint` to point at a specific file when testing alternate runs:

```bash
python Generate/generate_blueprint.py --params_json sample_params.json \
       --out_prefix my_blueprint --device cuda
# or explicitly pick a checkpoint:
python Generate/generate_blueprint.py --params_json sample_params.json \
       --checkpoint checkpoints/epoch_9.pt
```

Layouts now emit adjacency edge tokens, so providing an `adjacency` block in
your input JSON will bias decoding toward the requested room neighbors. The
tokenizer also writes the realised adjacencies back into the generated layout
for downstream checks.

Use `--strategy guided` (optionally with `--guided_topk` and `--guided_beam`) to
enable constraint-guided search that runs geometry validation mid-decode and
prunes invalid branches before they complete.

Add `--refine_iters` and `--refine_temp` to run a post-decoding simulated-annealing
pass that nudges rooms apart, clamps bounds, and respects adjacency goals before
writing outputs.

Capture validator feedback for future RL fine-tuning with `--issues_log logs/run_issues.jsonl` while you experiment.

Decoding runs `validate_layout` after each attempt to ensure rooms stay within
the requested bounds. If any room exceeds the canvas, the script retries up to
`--max_attempts` times before clamping positions and dimensions. Increase this
value for higher accuracy at the cost of additional compute.

### API

Serve the model via FastAPI. The `DEVICE` environment variable controls whether
the model runs on `cpu` or `cuda`:

```bash
DEVICE=cuda uvicorn api.app:app --host 0.0.0.0 --port 8000 --reload
# POST to /generate with params JSON; response contains a job id
```

### Web Interface

A React front end is available for interactive blueprint generation. Start the API
server, then in a separate terminal run:

```bash
cd interface/web
npm install
VITE_API_BASE_URL=http://localhost:8000 npm run dev
```

Open `http://localhost:5173/design` in a browser to submit parameters and preview
the resulting blueprint.

### Simple CLI

For non-technical users, a basic command line interface is available:

```bash
python interface/cli.py --params sample_params.json
```

The CLI queues a job with the API, polls for completion, and saves both SVG and
JSON outputs to `generated_cli/`.

## Deployment

### Build the Container

The provided `Dockerfile` bundles the API server, model weights (if present in
`checkpoints/`), and the front-end assets. Build the image locally:

```bash
docker build -t blueprint-generator .
```

You can override the API URL baked into the front-end with the
`VITE_API_BASE_URL` build argument:

```bash
docker build -t blueprint-generator \
  --build-arg VITE_API_BASE_URL=https://example.com \
  .
```

### docker-compose

Use the included `docker-compose.yml` to run the web server, optional Redis
queue, and Prometheus monitoring:

```bash
docker-compose up --build
```

Services:

* **app** – FastAPI server serving the API and static front-end at
  `http://localhost:8000/app/`
* **queue** – Redis instance for background jobs (optional)
* **prometheus** – metrics collector available at `http://localhost:9090`

### Environment Variables

* `DEVICE` – set to `cpu` or `cuda` to control inference hardware (default:
  `cpu`).
* `RATE_LIMIT` – requests per API key per minute (default: `60`).

For production deployments, build the image with the desired API base URL and
run using docker-compose with the appropriate environment variables set (e.g.
`DEVICE=cuda`).
