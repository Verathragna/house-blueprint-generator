import sys, json, types
from pathlib import Path

# Ensure repo root is on sys.path for top-level imports like 'Generate'
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

# Prepare dummy tokenizer/model and stubs like tests/test_generate_blueprint.py
class DummyTokenizer:
    def __init__(self, layout):
        self._layout = layout
        self.token_to_id = {
            "BEDROOM": 1,
            "BATHROOM": 2,
            "KITCHEN": 3,
            "LIVING": 4,
            "DINING": 5,
            "LAUNDRY": 6,
        }
        self.default_room_dims = {
            "BEDROOM": ("W12", "L12"),
            "BATHROOM": ("W8", "L8"),
            "KITCHEN": ("W12", "L12"),
            "LIVING": ("W16", "L14"),
            "DINING": ("W12", "L12"),
            "LAUNDRY": ("W10", "L10"),
        }

    def get_vocab_size(self):
        return 16

    def encode_params(self, params):
        return [0]

    def decode_layout_tokens(self, tokens):
        return json.loads(json.dumps(self._layout))

    def adjacency_requirements_from_params(self, adjacency):
        return dict(adjacency or {})

class DummyModel:
    def __init__(self, *_args):
        pass
    def load_state_dict(self, _state, strict=False):
        return None
    def to(self, _device):
        return self

# Layout template with reasonable positions
layout_template = {
    "layout": {
        "rooms": [
            {"type": "Master bedroom", "position": {"x": 0, "y": 0}, "size": {"width": 12, "length": 12}},
            {"type": "Master bathroom", "position": {"x": 12, "y": 0}, "size": {"width": 8, "length": 12}},
            {"type": "Kitchen", "position": {"x": 0, "y": 12}, "size": {"width": 12, "length": 12}},
            {"type": "Living room", "position": {"x": 12, "y": 12}, "size": {"width": 14, "length": 12}},
            {"type": "Dining room", "position": {"x": 26, "y": 12}, "size": {"width": 12, "length": 12}},
            {"type": "Laundry room", "position": {"x": 14, "y": 24}, "size": {"width": 10, "length": 10}},
        ]
    }
}

dummy_tokenizer = DummyTokenizer(layout_template)

# Stub pydantic
pydantic_module = types.ModuleType("pydantic")
class _ValidationError(Exception):
    ...
pydantic_module.ValidationError = _ValidationError
sys.modules["pydantic"] = pydantic_module

# Stub Generate.params
params_module = types.ModuleType("Generate.params")
class _Params:
    def __init__(self, data):
        self._raw = data
        baths = (data.get("bathrooms") or {})
        self.bathrooms = types.SimpleNamespace(full=int(baths.get("full", 0)), half=int(baths.get("half", 0)))
        self.bedrooms = int(data.get("bedrooms", 0))
        self.kitchen = int(data.get("kitchen", 1))
        self.livingRooms = int(data.get("livingRooms", 1))
        self.diningRooms = int(data.get("diningRooms", 1))
        self.laundryRooms = int(data.get("laundryRooms", 1))
        adjacency = data.get("adjacency")
        self.adjacency = types.SimpleNamespace(root=adjacency) if adjacency is not None else None
    @classmethod
    def model_validate(cls, data):
        return cls(data)
    def model_dump(self):
        return dict(self._raw)
params_module.Params = _Params
sys.modules["Generate.params"] = params_module

# Stub torch
torch_stub = types.SimpleNamespace(load=lambda *_a, **_k: {})
sys.modules["torch"] = torch_stub

# Stub model and tokenizer modules used by generate_blueprint
layout_module = types.ModuleType("models.layout_transformer")
layout_module.LayoutTransformer = DummyModel
sys.modules["models.layout_transformer"] = layout_module

decoding_module = types.ModuleType("models.decoding")
decoding_module.decode = lambda *_a, **_k: ["<DUMMY>"]
sys.modules["models.decoding"] = decoding_module

# Stub SVG renderer
from pathlib import Path as _Path

def fake_render(_layout, svg_path, lot_dims=None):
    _Path(svg_path).write_text("<svg />")

render_module = types.ModuleType("dataset.render_svg")
render_module.render_layout_svg = fake_render
sys.modules["dataset.render_svg"] = render_module

# Import and run main
import os as _os
import Generate.generate_blueprint as gb

# Force tokenizer and checkpoint behavior
orig_exists = gb.os.path.exists
setattr(gb, "BlueprintTokenizer", lambda: dummy_tokenizer)
setattr(gb.torch, "load", lambda *_a, **_k: {})
setattr(gb.os.path, "exists", lambda path: True if path == gb.CKPT else orig_exists(path))

# Build params and run
params = {
    "dimensions": {"width": 40, "depth": 40},
    "bedrooms": 0,
    "bathrooms": {"full": 0, "half": 0},
    "kitchen": 1,
    "livingRooms": 1,
    "diningRooms": 1,
    "laundryRooms": 1,
    "adjacency": {
        "Master bedroom": ["Master bathroom"],
        "Master bathroom": ["Master bedroom"],
        "Kitchen": ["Living room"],
        "Living room": ["Kitchen", "Dining room", "Laundry room"],
        "Dining room": ["Living room"],
        "Laundry room": ["Living room"],
    },
}

work = Path.cwd() / "_smoke"
work.mkdir(exist_ok=True)
params_path = work / "params.json"
params_path.write_text(json.dumps(params))

out_prefix = work / "generated"

gb.sys.argv = [
    "generate_blueprint.py",
    "--params_json", str(params_path),
    "--out_prefix", str(out_prefix),
    "--refine_iters", "30",
    "--refine_temp", "4.0",
]

gb.main()

json_path = out_prefix.with_suffix(".json")
svg_path = out_prefix.with_suffix(".svg")

if not json_path.exists() or not svg_path.exists():
    print("FAILED: outputs not created")
    sys.exit(2)

print("OK: generated", json_path.name, "and", svg_path.name)
