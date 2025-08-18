import os, json, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from dataset.render_svg import render_layout_svg

OUT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), 'datasets/synthetic'))
os.makedirs(OUT_DIR, exist_ok=True)

def sample_parameters(i):
    # simple random-ish params; replace with your sampler
    return {
        "houseStyle": "Craftsman" if i % 2 == 0 else "Modern",
        "bedrooms": 3 + (i % 3),
        "bathrooms": {"full": 2 + (i % 2)},
        "foundationType": "basement" if i % 2 else "slab",
        "bonusRoom": bool(i % 2),
        "attic": bool((i+1) % 2),
        "fireplace": True,
        "ownerSuiteLocation": "main floor",
        "masterBathOption": "both",
        "ceilingHeight": 9,
        "garage": {"attached": True, "carCount": 2},
    }

def dummy_layout(i):
    # very simple rectangular rooms for demo; rules model can replace this
    return {
        "layout": {
            "rooms": [
                {"type": "Bedroom", "position": {"x": 0, "y": 0}, "size": {"width": 12, "length": 12}},
                {"type": "Bathroom", "position": {"x": 12, "y": 0}, "size": {"width": 8, "length": 8}},
                {"type": "Kitchen", "position": {"x": 0, "y": 12}, "size": {"width": 14, "length": 12}},
                {"type": "Living Room", "position": {"x": 14, "y": 12}, "size": {"width": 16, "length": 14}},
            ]
        }
    }

def main(n=50):
    for i in range(n):
        params = sample_parameters(i)
        layout = dummy_layout(i)
        ip = os.path.join(OUT_DIR, f"input_{i:05d}.json")
        lp = os.path.join(OUT_DIR, f"layout_{i:05d}.json")
        sp = os.path.join(OUT_DIR, f"layout_{i:05d}.svg")
        with open(ip, "w", encoding="utf-8") as f: json.dump(params, f, indent=2)
        with open(lp, "w", encoding="utf-8") as f: json.dump(layout, f, indent=2)
        render_layout_svg(layout, sp)
    print(f"✅ Wrote {n} pairs to {OUT_DIR}")

if __name__ == "__main__":
    main(100)
