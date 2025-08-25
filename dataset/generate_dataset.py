import os
import json
import sys
import random

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

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from dataset.render_svg import render_layout_svg
from dataset.augmentation import mirror_layout, rotate_layout
from evaluation.validators import clamp_bounds, check_bounds, check_overlaps

# Maximum width/height for any layout. Rooms will be scaled to fit within this
# square coordinate space.
MAX_COORD = 40

OUT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), 'datasets/synthetic'))

STYLES = ["Craftsman", "Modern", "Colonial", "Ranch", "Mediterranean"]
SIZES = ["small", "medium", "large"]
ROOM_TYPES = [
    "Bedroom",
    "Bathroom",
    "Kitchen",
    "Living Room",
    "Dining Room",
    "Office",
    "Garage",
]


def sample_parameters(i, rng=random):
    """Sample varied architectural parameters."""
    size = rng.choice(SIZES)
    square_feet = {"small": 1000, "medium": 2000, "large": 3000}[size]
    return {
        "houseStyle": rng.choice(STYLES),
        "squareFeet": square_feet,
        "bedrooms": rng.randint(2, 5),
        "bathrooms": {"full": rng.randint(1, 3)},
        "foundationType": rng.choice(["slab", "basement", "crawlspace"]),
        "bonusRoom": rng.choice([True, False]),
        "attic": rng.choice([True, False]),
        "fireplace": rng.choice([True, False]),
        "ownerSuiteLocation": rng.choice(["main floor", "second floor"]),
        "ceilingHeight": rng.choice([8, 9, 10]),
        "garage": {"attached": rng.choice([True, False]), "carCount": rng.randint(1, 3)},
    }


def _place_room(rooms, room_type, rng, max_coord=MAX_COORD, max_attempts=100):
    """Place a room so it shares a wall with an existing room.

    The first room is anchored at ``(0, 0)``.  Subsequent rooms are attached
    edge-to-edge to a randomly selected existing room.  This encourages the
    layout to form a contiguous chain of rooms rather than scattering them at
    random coordinates.
    """

    for _ in range(max_attempts):
        width = rng.randint(6, 12)
        length = rng.randint(6, 12)

        if not rooms:
            x, y = 0, 0
        else:
            base = rng.choice(rooms)
            bx, by = base["position"]["x"], base["position"]["y"]
            bw, bl = base["size"]["width"], base["size"]["length"]
            side = rng.choice(["right", "left", "top", "bottom"])
            if side == "right":
                x = bx + bw
                y = by
                length = bl
            elif side == "left":
                x = bx - width
                y = by
                length = bl
            elif side == "top":
                x = bx
                y = by + bl
                width = bw
            else:  # bottom
                x = bx
                y = by - length
                width = bw

        rect = (x, y, x + width, y + length)
        if x < 0 or y < 0 or rect[2] > max_coord or rect[3] > max_coord:
            continue

        overlap = False
        for r in rooms:
            rx, ry = r["position"]["x"], r["position"]["y"]
            rw, rl = r["size"]["width"], r["size"]["length"]
            if x < rx + rw and x + width > rx and y < ry + rl and y + length > ry:
                overlap = True
                break
        if not overlap:
            rooms.append(
                {
                    "type": room_type,
                    "position": {"x": x, "y": y},
                    "size": {"width": width, "length": length},
                }
            )
            return True
    return False


def random_layout(i, params, rng=random):
    """Generate a random layout matching requested room counts."""
    rooms: list[dict] = []

    bed_count = int(params.get("bedrooms", 0))
    baths = params.get("bathrooms") or {}
    bath_count = int(baths.get("full", 0)) + int(baths.get("half", 0))
    garage_needed = 1 if params.get("garage") else 0

    for _ in range(bed_count):
        if not _place_room(rooms, "Bedroom", rng):
            raise ValueError("Could not place all bedrooms")
    for _ in range(bath_count):
        if not _place_room(rooms, "Bathroom", rng):
            raise ValueError("Could not place all bathrooms")
    for _ in range(garage_needed):
        if not _place_room(rooms, "Garage", rng):
            raise ValueError("Could not place garage")

    extra_types = [r for r in ROOM_TYPES if r not in {"Bedroom", "Bathroom", "Garage"}]
    for _ in range(rng.randint(0, 3)):
        _place_room(rooms, rng.choice(extra_types), rng)

    return {"layout": {"rooms": rooms}}


def _scale_layout(layout, max_width=MAX_COORD, max_height=MAX_COORD):
    """Scale layout so all rooms fit within ``max_width`` × ``max_height``."""
    rooms = layout.get("layout", {}).get("rooms", [])
    if not rooms:
        return layout

    max_x = max(r["position"]["x"] + r["size"]["width"] for r in rooms)
    max_y = max(r["position"]["y"] + r["size"]["length"] for r in rooms)
    scale = min(max_width / max_x, max_height / max_y, 1.0)

    if scale < 1.0:
        for r in rooms:
            r["position"]["x"] = int(r["position"]["x"] * scale)
            r["position"]["y"] = int(r["position"]["y"] * scale)
            r["size"]["width"] = int(r["size"]["width"] * scale)
            r["size"]["length"] = int(r["size"]["length"] * scale)
    return layout


def validate_layout(layout, max_width=MAX_COORD, max_height=MAX_COORD):
    """Ensure rooms have coordinates, do not overlap, and respect the bounds."""
    rooms = layout.get("layout", {}).get("rooms", [])
    for idx, room in enumerate(rooms):
        pos = room.get("position")
        size = room.get("size", {})
        if not isinstance(pos, dict) or "x" not in pos or "y" not in pos:
            raise ValueError(f"Room {idx} missing coordinate fields")
        width = size.get("width", 0)
        length = size.get("length", 0)
        x, y = pos["x"], pos["y"]
        if not (0 <= x <= max_width and 0 <= y <= max_height):
            raise ValueError(f"Room {idx} position out of bounds")
        if x + width > max_width or y + length > max_height:
            raise ValueError(f"Room {idx} exceeds layout bounds")
    overlaps = check_overlaps(rooms)
    if overlaps:
        raise ValueError(overlaps[0])


def ingest_external_dataset(
    external_dir,
    out_dir=OUT_DIR,
    start_index=0,
    augment=False,
    strict=False,
):
    """Ingest external floor-plan JSON files and normalize to internal schema."""
    idx = start_index
    for fname in sorted(os.listdir(external_dir)):
        if not fname.endswith('.json'):
            continue
        with open(os.path.join(external_dir, fname), 'r', encoding='utf-8') as f:
            data = json.load(f)

        rooms = []
        for room in data.get("rooms", []):
            if "position" in room:
                pos = room["position"]
                x, y = pos.get("x"), pos.get("y")
            else:
                x, y = room.get("x"), room.get("y")
            if x is None or y is None:
                raise ValueError(f"Room missing coordinates in {fname}")
            size = {"width": room.get("width", 10), "length": room.get("length", 10)}
            rooms.append(
                {
                    "type": room.get("type", "Room"),
                    "position": {"x": x, "y": y},
                    "size": size,
                }
            )

        layout = {"layout": {"rooms": rooms}}
        params = data.get(
            "params", {"houseStyle": "External", "squareFeet": data.get("squareFeet", 0)}
        )

        try:
            _write_sample(params, layout, out_dir, idx)
        except ValueError:
            if strict:
                raise
            continue
        idx += 1
        if augment:
            for aug_layout in (mirror_layout(layout), rotate_layout(layout)):
                aug_layout = clamp_bounds(aug_layout, max_width=MAX_COORD, max_length=MAX_COORD)
                rooms = (aug_layout.get("layout") or {}).get("rooms", [])
                issues = check_bounds(rooms) + check_overlaps(rooms)
                if issues:
                    if strict:
                        raise ValueError("; ".join(issues))
                    continue
                try:
                    _write_sample(params, aug_layout, out_dir, idx)
                except ValueError:
                    if strict:
                        raise
                    continue
                idx += 1
    return idx - start_index


def _write_sample(params, layout, out_dir, idx):
    layout = _scale_layout(layout)
    layout = clamp_bounds(layout, max_width=MAX_COORD, max_length=MAX_COORD)
    validate_layout(layout)
    ip = os.path.join(out_dir, f"input_{idx:05d}.json")
    lp = os.path.join(out_dir, f"layout_{idx:05d}.json")
    sp = os.path.join(out_dir, f"layout_{idx:05d}.svg")
    with open(ip, "w", encoding="utf-8") as f:
        json.dump(params, f, indent=2)
    with open(lp, "w", encoding="utf-8") as f:
        json.dump(layout, f, indent=2)
    render_layout_svg(layout, sp)


def main(
    n=50,
    out_dir=OUT_DIR,
    external_dir=None,
    seed=None,
    augment=False,
    strict=False,
):
    random.seed(seed)
    if np is not None:
        np.random.seed(seed)
    if torch is not None:
        torch.manual_seed(seed)
    os.makedirs(out_dir, exist_ok=True)
    rng = random.Random(seed)
    idx = 0
    for _ in range(n):
        params = sample_parameters(idx, rng)
        try:
            layout = random_layout(idx, params, rng)
        except ValueError:
            if strict:
                raise
            continue
        try:
            _write_sample(params, layout, out_dir, idx)
        except ValueError:
            if strict:
                raise
            continue
        idx += 1
        if augment:
            for aug_layout in (mirror_layout(layout), rotate_layout(layout)):
                aug_layout = clamp_bounds(aug_layout, max_width=MAX_COORD, max_length=MAX_COORD)
                rooms = (aug_layout.get("layout") or {}).get("rooms", [])
                issues = check_bounds(rooms) + check_overlaps(rooms)
                if issues:
                    if strict:
                        raise ValueError("; ".join(issues))
                    continue
                try:
                    _write_sample(params, aug_layout, out_dir, idx)
                except ValueError:
                    if strict:
                        raise
                    continue
                idx += 1

    if external_dir:
        idx += ingest_external_dataset(
            external_dir, out_dir, start_index=idx, augment=augment, strict=strict
        )

    print(f"Wrote {idx} pairs to {out_dir}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate synthetic floor-plan dataset")
    parser.add_argument("--n", type=int, default=100, help="number of synthetic samples to generate")
    parser.add_argument("--external_dir", type=str, default=None, help="path to external JSON floor-plan dataset")
    parser.add_argument("--out_dir", type=str, default=OUT_DIR, help="output directory")
    parser.add_argument("--seed", type=int, default=None, help="random seed for reproducibility")
    parser.add_argument(
        "--augment", action="store_true", help="apply simple mirroring/rotation augmentations"
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail immediately on any boundary or overlap issues instead of skipping",
    )
    args = parser.parse_args()

    main(
        n=args.n,
        out_dir=args.out_dir,
        external_dir=args.external_dir,
        seed=args.seed,
        augment=args.augment,
        strict=args.strict,
    )
