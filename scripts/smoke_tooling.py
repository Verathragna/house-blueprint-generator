import json
from pathlib import Path
import sys

# Ensure repo root on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from Generate.generate_blueprint import main as _main  # reuse CLI for generation
from geometry.exporters import export_svg_clean
from evaluation.geometry_ops import snap_layout
from evaluation.ada import validate_accessibility

# Minimal smoke pipeline for tooling
# 1) Generate
# 2) Snap to 0.5ft grid
# 3) Run ADA-lite validation
# 4) Export cleaned SVG

params = {
    "dimensions": {"width": 40, "depth": 40},
    "bedrooms": 2,
    "bathrooms": {"full": 2, "half": 0},
    "kitchen": 1,
    "livingRooms": 1,
    "diningRooms": 1,
    "laundryRooms": 1,
}

work = Path("_smoke_tooling")
work.mkdir(exist_ok=True)
params_path = work / "params.json"
params_path.write_text(json.dumps(params))

# Use generate_blueprint CLI
import sys
sys.argv = [
    "generate_blueprint.py",
    "--params_json", str(params_path),
    "--out_prefix", str(work / "raw"),
]
try:
    _main()
except SystemExit:
    pass

layout_path = work / "raw.json"
if not layout_path.exists():
    # In generator, file is named with provided prefix
    layout_path = (work / "raw").with_suffix(".json")

layout = json.loads(layout_path.read_text())

# Snap and clamp
dims = params["dimensions"]
layout_snapped = snap_layout(layout, grid=0.5, max_width=dims["width"], max_length=dims["depth"])

# Validate accessibility
issues = validate_accessibility(layout_snapped)
print("Accessibility issues:", issues[:5] if issues else "none")

# Export clean SVG
export_svg_clean(layout_snapped, str(work / "clean.svg"), lot_dims=(dims["width"], dims["depth"]))
print("Wrote", (work / "clean.svg").as_posix())
