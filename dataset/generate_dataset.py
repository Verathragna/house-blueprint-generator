import os
import json
import sys
import random
from typing import Iterable, List, Sequence, Tuple

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
from evaluation.validators import clamp_bounds, check_bounds, check_overlaps, resolve_overlaps

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
    "Laundry Room",
    "Garage",
    "Closet",
]

DEFAULT_PATTERNS = ("chain", "corridor", "l_shape")

ROOM_DIM_RANGES = {
    "Bedroom": ((10, 14), (10, 14)),
    "Bathroom": ((6, 10), (6, 10)),
    "Kitchen": ((12, 16), (10, 14)),
    "Living Room": ((16, 20), (12, 18)),
    "Dining Room": ((10, 14), (10, 14)),
    "Office": ((10, 14), (10, 14)),
    "Laundry Room": ((8, 12), (8, 10)),
    "Garage": ((18, 24), (20, 26)),
    "Closet": ((6, 8), (6, 8)),
}


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


def _random_room_dimensions(room_type: str, rng) -> Tuple[int, int]:
    ranges = ROOM_DIM_RANGES.get(room_type, ((8, 14), (8, 14)))
    w = rng.randint(*ranges[0])
    l = rng.randint(*ranges[1])
    return w, l


def _make_room(room_type: str, x: float, y: float, width: float, length: float) -> dict:
    return {
        "type": room_type,
        "position": {"x": int(round(x)), "y": int(round(y))},
        "size": {"width": int(round(width)), "length": int(round(length))},
    }


def _room_order_from_params(params: dict, rng) -> List[str]:
    rooms: List[str] = []
    bed_count = int(params.get("bedrooms", 0))
    baths = params.get("bathrooms") or {}
    bath_count = int(baths.get("full", 0)) + int(baths.get("half", 0))
    garage_needed = 1 if params.get("garage") else 0

    rooms.extend(["Bedroom"] * bed_count)
    rooms.extend(["Bathroom"] * bath_count)

    rooms.extend(["Kitchen"] * max(1, int(params.get("kitchen", 1))))
    rooms.extend(["Living Room"] * max(1, int(params.get("livingRooms", 1))))
    rooms.extend(["Dining Room"] * max(1, int(params.get("diningRooms", 1))))
    rooms.extend(["Laundry Room"] * max(1, int(params.get("laundryRooms", 1))))

    if garage_needed:
        rooms.append("Garage")

    extra_choices = [r for r in ROOM_TYPES if r not in {"Bedroom", "Bathroom"}]
    for _ in range(rng.randint(0, 3)):
        rooms.append(rng.choice(extra_choices))

    rng.shuffle(rooms)
    return rooms


def _generate_chain_layout(room_types: Sequence[str], rng) -> List[dict]:
    rooms: List[dict] = []
    for rtype in room_types:
        if not _place_room(rooms, rtype, rng):
            raise ValueError(f"Could not place all {rtype.lower()} rooms")
    return rooms


def _generate_corridor_layout(room_types: Sequence[str], rng) -> List[dict]:
    rooms: List[dict] = []
    corridor_center = MAX_COORD / 2
    corridor_half_width = 2
    spacing = 2
    offsets = [0.0, 0.0]

    for idx, rtype in enumerate(room_types):
        width, length = _random_room_dimensions(rtype, rng)
        if idx % 2 == 0:
            x = max(0.0, corridor_center - corridor_half_width - width)
            y = offsets[0]
            offsets[0] += length + spacing
        else:
            x = min(MAX_COORD - width, corridor_center + corridor_half_width)
            y = offsets[1]
            offsets[1] += length + spacing
        rooms.append(_make_room(rtype, x, y, width, length))
    return rooms


def _generate_l_layout(room_types: Sequence[str], rng) -> List[dict]:
    if not room_types:
        return []
    split = max(1, len(room_types) // 2)
    base_types = room_types[:split]
    vertical_types = room_types[split:]

    rooms: List[dict] = []
    spacing = 2
    x_cursor = 0.0
    base_max_height = 0.0

    for rtype in base_types:
        width, length = _random_room_dimensions(rtype, rng)
        rooms.append(_make_room(rtype, x_cursor, 0.0, width, length))
        x_cursor += width + spacing
        base_max_height = max(base_max_height, length)

    corner_x = max((room["position"]["x"] + room["size"]["width"] for room in rooms), default=0)
    y_cursor = base_max_height

    for idx, rtype in enumerate(vertical_types):
        width, length = _random_room_dimensions(rtype, rng)
        x = max(0.0, corner_x - width)
        if idx == 0:
            y = y_cursor
        else:
            y = y_cursor + spacing
        rooms.append(_make_room(rtype, x, y, width, length))
        y_cursor = y + length

    return rooms


PATTERN_GENERATORS = {
    "chain": _generate_chain_layout,
    "corridor": _generate_corridor_layout,
    "central_corridor": _generate_corridor_layout,
    "l_shape": _generate_l_layout,
    "l-shape": _generate_l_layout,
}


def random_layout(i, params, rng=random, pattern: str | None = None):
    """Generate a layout using one of the available circulation patterns."""
    room_sequence = _room_order_from_params(params, rng)
    pattern_key = (pattern or rng.choice(DEFAULT_PATTERNS)).lower()
    generator = PATTERN_GENERATORS.get(pattern_key)
    if generator is None:
        raise ValueError(f"Unknown pattern '{pattern}'")

    rooms = generator(room_sequence, rng)
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
    rng=None,
):
    rng = rng or random
    """Ingest external floor-plan JSON files and normalize to internal schema."""
    idx = start_index
    for fname in sorted(os.listdir(external_dir)):
        if not fname.endswith('.json'):
            continue
        path = os.path.join(external_dir, fname)
        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except (OSError, json.JSONDecodeError) as e:
            print(f"Skipping {fname}: {e}")
            continue

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
            _write_sample(params, layout, out_dir, idx, rng=rng)
        except (OSError, ValueError) as e:
            if strict:
                raise
            print(f"Skipping {fname}: {e}")
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
                    _write_sample(params, aug_layout, out_dir, idx, rng=rng)
                except (OSError, ValueError) as e:
                    if strict:
                        raise
                    print(f"Skipping augmented sample from {fname}: {e}")
                    continue
                idx += 1
    return idx - start_index


def _randomly_offset_layout(layout, max_width=MAX_COORD, max_height=MAX_COORD, rng=None):
    rng = rng or random
    rooms = (layout.get("layout") or {}).get("rooms", [])
    if not rooms:
        return layout
    min_x = min(r.get("position", {}).get("x", 0) for r in rooms)
    min_y = min(r.get("position", {}).get("y", 0) for r in rooms)
    max_x = max(r.get("position", {}).get("x", 0) + r.get("size", {}).get("width", 0) for r in rooms)
    max_y = max(r.get("position", {}).get("y", 0) + r.get("size", {}).get("length", 0) for r in rooms)
    width_extent = max_x - min_x
    height_extent = max_y - min_y
    if width_extent <= 0 or height_extent <= 0:
        return layout
    slack_x = max(0.0, max_width - width_extent)
    slack_y = max(0.0, max_height - height_extent)
    offset_x = rng.uniform(0.0, slack_x) - min_x if slack_x > 0 else -min_x
    offset_y = rng.uniform(0.0, slack_y) - min_y if slack_y > 0 else -min_y
    for room in rooms:
        pos = room.setdefault("position", {})
        pos["x"] = int(round(pos.get("x", 0) + offset_x))
        pos["y"] = int(round(pos.get("y", 0) + offset_y))
    return layout


def _write_sample(params, layout, out_dir, idx, rng=None):
    layout = _scale_layout(layout)
    layout = _randomly_offset_layout(layout, rng=rng)
    layout = clamp_bounds(layout, max_width=MAX_COORD, max_length=MAX_COORD)
    layout = resolve_overlaps(layout)
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
    patterns: Sequence[str] = DEFAULT_PATTERNS,
):
    random.seed(seed)
    if np is not None and seed is not None:
        np.random.seed(seed)
    if torch is not None and seed is not None:
        torch.manual_seed(seed)
    os.makedirs(out_dir, exist_ok=True)
    rng = random.Random(seed)
    idx = 0
    pattern_choices = [p for p in patterns if p in PATTERN_GENERATORS]
    if not pattern_choices:
        pattern_choices = list(DEFAULT_PATTERNS)

    for _ in range(n):
        params = sample_parameters(idx, rng)
        layout = None
        success = False
        for attempt in range(5):
            pattern = rng.choice(pattern_choices)
            try:
                candidate = random_layout(idx, params, rng, pattern=pattern)
            except ValueError as exc:
                if strict and attempt == 4:
                    raise
                if attempt == 4:
                    print(f"Failed to generate layout for sample {idx}: {exc}")
                continue
            try:
                _write_sample(params, candidate, out_dir, idx, rng=rng)
                layout = candidate
                success = True
                break
            except (OSError, ValueError) as e:
                if strict and attempt == 4:
                    raise
                if attempt == 4:
                    print(f"Skipping sample {idx}: {e}")
        if not success or layout is None:
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
                    _write_sample(params, aug_layout, out_dir, idx, rng=rng)
                except (OSError, ValueError) as e:
                    if strict:
                        raise
                    print(f"Skipping augmented sample {idx}: {e}")
                    continue
                idx += 1

    if external_dir:
        idx += ingest_external_dataset(
            external_dir, out_dir, start_index=idx, augment=augment, strict=strict, rng=rng
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
    parser.add_argument(
        "--patterns",
        type=str,
        default=",".join(DEFAULT_PATTERNS),
        help="Comma-separated list of layout patterns to sample (chain,corridor,l_shape)",
    )
    args = parser.parse_args()

    pattern_list = [p.strip() for p in args.patterns.split(",") if p.strip()]

    main(
        n=args.n,
        out_dir=args.out_dir,
        external_dir=args.external_dir,
        seed=args.seed,
        augment=args.augment,
        strict=args.strict,
        patterns=pattern_list,
    )
