import os
import json
import sys
import random

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from dataset.render_svg import render_layout_svg

OUT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), 'datasets/synthetic'))

STYLES = ["Craftsman", "Modern", "Colonial", "Ranch", "Mediterranean"]
SIZES = ["small", "medium", "large"]
ROOM_TYPES = ["Bedroom", "Bathroom", "Kitchen", "Living Room", "Dining Room", "Office"]


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


def random_layout(i, rng=random):
    """Generate a random layout with explicit x/y coordinates."""
    base_x = rng.randint(0, 20)
    base_y = rng.randint(0, 20)
    rooms = []
    for _ in range(rng.randint(3, 6)):
        room_type = rng.choice(ROOM_TYPES)
        x = base_x + rng.randint(0, 20)
        y = base_y + rng.randint(0, 20)
        width = rng.randint(8, 16)
        length = rng.randint(8, 16)
        rooms.append({
            "type": room_type,
            "position": {"x": x, "y": y},
            "size": {"width": width, "length": length},
        })
    return {"layout": {"rooms": rooms}}


def ingest_external_dataset(external_dir, out_dir=OUT_DIR, start_index=0):
    """Ingest external floor-plan JSON files and normalize to internal schema."""
    idx = start_index
    for fname in sorted(os.listdir(external_dir)):
        if not fname.endswith('.json'):
            continue
        with open(os.path.join(external_dir, fname), 'r', encoding='utf-8') as f:
            data = json.load(f)

        rooms = []
        for room in data.get("rooms", []):
            if "x" not in room or "y" not in room:
                raise ValueError(f"Room missing coordinates in {fname}")
            size = {"width": room.get("width", 10), "length": room.get("length", 10)}
            rooms.append({
                "type": room.get("type", "Room"),
                "position": {"x": room["x"], "y": room["y"]},
                "size": size,
            })

        layout = {"layout": {"rooms": rooms}}
        params = data.get("params", {"houseStyle": "External", "squareFeet": data.get("squareFeet", 0)})

        ip = os.path.join(out_dir, f"input_{idx:05d}.json")
        lp = os.path.join(out_dir, f"layout_{idx:05d}.json")
        sp = os.path.join(out_dir, f"layout_{idx:05d}.svg")
        with open(ip, "w", encoding="utf-8") as f:
            json.dump(params, f, indent=2)
        with open(lp, "w", encoding="utf-8") as f:
            json.dump(layout, f, indent=2)
        render_layout_svg(layout, sp)
        idx += 1
    return idx - start_index


def main(n=50, out_dir=OUT_DIR, external_dir=None, seed=None):
    os.makedirs(out_dir, exist_ok=True)
    rng = random.Random(seed)
    idx = 0
    for _ in range(n):
        params = sample_parameters(idx, rng)
        layout = random_layout(idx, rng)
        ip = os.path.join(out_dir, f"input_{idx:05d}.json")
        lp = os.path.join(out_dir, f"layout_{idx:05d}.json")
        sp = os.path.join(out_dir, f"layout_{idx:05d}.svg")
        with open(ip, "w", encoding="utf-8") as f:
            json.dump(params, f, indent=2)
        with open(lp, "w", encoding="utf-8") as f:
            json.dump(layout, f, indent=2)
        render_layout_svg(layout, sp)
        idx += 1

    if external_dir:
        idx += ingest_external_dataset(external_dir, out_dir, start_index=idx)

    print(f"✅ Wrote {idx} pairs to {out_dir}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate synthetic floor-plan dataset")
    parser.add_argument("--n", type=int, default=100, help="number of synthetic samples to generate")
    parser.add_argument("--external_dir", type=str, default=None, help="path to external JSON floor-plan dataset")
    parser.add_argument("--out_dir", type=str, default=OUT_DIR, help="output directory")
    parser.add_argument("--seed", type=int, default=None, help="random seed for reproducibility")
    args = parser.parse_args()

    main(n=args.n, out_dir=args.out_dir, external_dir=args.external_dir, seed=args.seed)
