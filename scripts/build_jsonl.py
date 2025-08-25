import argparse
import json
import os
import random
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from tokenizer.tokenizer import BlueprintTokenizer
from dataset.augmentation import mirror_layout, rotate_layout

MAX_COORD = 40

try:
    import numpy as np
except Exception:  # NumPy not available
    np = None

if np is not None:
    try:
        import torch  # type: ignore
    except Exception:  # PyTorch not available
        torch = None
else:
    torch = None


def _validate_layout(
    layout: dict,
    enforce_bounds: bool = True,
    max_coord: int = MAX_COORD,
) -> None:
    """Ensure every room specifies an ``x`` and ``y`` position.

    Optionally verifies that coordinates fall within the 0–``max_coord`` range.
    Raises ``ValueError`` if validation fails.
    """

    for idx, room in enumerate(layout.get("layout", {}).get("rooms", [])):
        pos = room.get("position")
        if not pos or "x" not in pos or "y" not in pos:
            raise ValueError(f"Room {idx} missing x or y position")
        x, y = pos["x"], pos["y"]
        size = room.get("size", {})
        w = size.get("width", 0)
        l = size.get("length", 0)
        if enforce_bounds:
            if not (0 <= x <= max_coord and 0 <= y <= max_coord):
                raise ValueError(
                    f"Room {idx} position out of range: x={x}, y={y}, max={max_coord}"
                )
            if x + w > max_coord or y + l > max_coord:
                raise ValueError(
                    f"Room {idx} exceeds bounds: x={x}, width={w}, y={y}, length={l}, max={max_coord}"
                )

def main(seed: int = 42, augment: bool = False, check_bounds: bool = True) -> None:
    random.seed(seed)
    if np is not None:
        np.random.seed(seed)
    if torch is not None:
        torch.manual_seed(seed)

    in_dir = "./dataset/datasets/synthetic"
    out_dir = "./dataset"
    os.makedirs(out_dir, exist_ok=True)

    pairs = []
    tk = BlueprintTokenizer()
    files = sorted(
        [f for f in os.listdir(in_dir) if f.startswith("input_") and f.endswith(".json")]
    )
    for f in files:
        idx = f.split("_")[1].split(".")[0]
        inp = json.load(open(os.path.join(in_dir, f), "r", encoding="utf-8"))
        lp = os.path.join(in_dir, f"layout_{idx}.json")
        if os.path.exists(lp):
            layout = json.load(open(lp, "r", encoding="utf-8"))
            _validate_layout(layout, enforce_bounds=check_bounds)
            x_ids, y_ids = tk.build_training_pair(inp, layout)
            pairs.append({"params": inp, "layout": layout, "x": x_ids, "y": y_ids})

            if augment:
                for aug_layout in (mirror_layout(layout), rotate_layout(layout)):
                    _validate_layout(aug_layout, enforce_bounds=check_bounds)
                    ax, ay = tk.build_training_pair(inp, aug_layout)
                    pairs.append({"params": inp, "layout": aug_layout, "x": ax, "y": ay})

    random.shuffle(pairs)
    cut = int(0.9 * len(pairs)) if pairs else 0
    train, val = pairs[:cut], pairs[cut:]

    with open(os.path.join(out_dir, "train.jsonl"), "w", encoding="utf-8") as wf:
        for r in train:
            wf.write(json.dumps(r) + "\n")
    with open(os.path.join(out_dir, "val.jsonl"), "w", encoding="utf-8") as wf:
        for r in val:
            wf.write(json.dumps(r) + "\n")

    print(f"Wrote {len(train)} train and {len(val)} val rows to {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build train/val JSONL datasets.")
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help=(
            "Random seed for shuffling and any libraries in use. "
            "Should match the dataset generation seed for full reproducibility."
        ),
    )
    parser.add_argument(
        "--augment",
        action="store_true",
        help="Apply simple mirroring/rotation augmentations",
    )
    parser.add_argument(
        "--skip-bounds-check",
        action="store_true",
        help=f"Allow room positions outside the [0, {MAX_COORD}] range",
    )
    args = parser.parse_args()
    main(args.seed, augment=args.augment, check_bounds=not args.skip_bounds_check)
