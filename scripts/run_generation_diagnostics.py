
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from Generate.params import Params
from evaluation.validators import validate_layout


def build_generator_command(args: argparse.Namespace) -> list[str]:
    cmd = [
        sys.executable,
        "Generate/generate_blueprint.py",
        "--params_json",
        str(args.params),
        "--out_prefix",
        str(args.out_prefix),
        "--min_separation",
        str(args.min_separation),
        "--device",
        args.device,
        "--max_attempts",
        str(args.max_attempts),
        "--strategy",
        args.strategy,
    ]
    if args.strategy == "sample":
        cmd.extend(["--temperature", str(args.temperature)])
    if args.strategy == "beam":
        cmd.extend(["--beam_size", str(args.beam_size)])
    if args.debug_dump:
        cmd.extend(["--debug_dump", str(args.debug_dump)])
    return cmd


def load_params(params_path: Path) -> tuple[dict, Params]:
    raw = json.loads(params_path.read_text(encoding="utf-8"))
    return raw, Params.model_validate(raw)


def run_generator(cmd: list[str]) -> subprocess.CompletedProcess[str]:
    print("Running:", " ".join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.stdout:
        print(result.stdout)
    if result.stderr:
        print(result.stderr, file=sys.stderr)
    return result


def evaluate_layout_file(layout_path: Path, params_raw: dict, params: Params, min_separation: float) -> list[str]:
    if not layout_path.exists():
        raise FileNotFoundError(f"Layout JSON not found at {layout_path}")
    layout = json.loads(layout_path.read_text(encoding="utf-8"))
    dims = params_raw.get("dimensions") or {}
    max_w = float(dims.get("width", 40))
    max_h = float(dims.get("depth", dims.get("height", 40)))
    adjacency = params.adjacency.root if params.adjacency else None
    issues = validate_layout(
        layout,
        max_width=max_w,
        max_length=max_h,
        min_separation=min_separation,
        adjacency=adjacency,
    )
    return issues


def main() -> int:
    parser = argparse.ArgumentParser(description="Run blueprint generation and report validation results")
    parser.add_argument("--params", required=True, type=Path, help="Path to parameters JSON")
    parser.add_argument("--out_prefix", required=True, type=Path, help="Output prefix for generated files")
    parser.add_argument("--min_separation", type=float, default=0.0)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--max_attempts", type=int, default=5)
    parser.add_argument("--strategy", choices=["greedy", "sample", "beam"], default="greedy")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--beam_size", type=int, default=5)
    parser.add_argument("--debug_dump", type=Path, default=None)
    parser.add_argument("--skip_eval", action="store_true")

    args = parser.parse_args()

    params_raw, params = load_params(args.params)

    if args.debug_dump:
        os.makedirs(args.debug_dump, exist_ok=True)

    cmd = build_generator_command(args)
    result = run_generator(cmd)
    if result.returncode != 0:
        print("Generation failed; see output above.", file=sys.stderr)
        return result.returncode

    layout_path = Path(f"{args.out_prefix}.json")

    if args.skip_eval:
        print(f"Generation complete. Layout written to {layout_path}")
        return 0

    issues = evaluate_layout_file(layout_path, params_raw, params, args.min_separation)
    if issues:
        print("Validation issues detected:")
        for issue in issues:
            print(f" - {issue}")
        return 2

    print(f"Layout valid. JSON: {layout_path}, SVG: {args.out_prefix}.svg")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
