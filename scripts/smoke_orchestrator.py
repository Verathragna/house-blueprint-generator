import sys, types, json
from pathlib import Path

# Ensure repo root is on sys.path so we can import top-level packages like 'Generate'
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

# --- Test parameters ---
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

# --- Stub Generate.params to avoid pydantic dependency ---
params_module = types.ModuleType("Generate.params")

class _Params:
    def __init__(self, data):
        self._raw = dict(data)
        baths = (data.get("bathrooms") or {})
        self.bathrooms = types.SimpleNamespace(
            full=int(baths.get("full", 0)),
            half=int(baths.get("half", 0)),
        )
        self.bedrooms = int(data.get("bedrooms", 0))
        self.kitchen = int(data.get("kitchen", 1))
        self.livingRooms = int(data.get("livingRooms", 1))
        self.diningRooms = int(data.get("diningRooms", 1))
        self.laundryRooms = int(data.get("laundryRooms", 1))
        adjacency = data.get("adjacency")
        self.adjacency = (
            types.SimpleNamespace(root=adjacency) if adjacency is not None else None
        )

    @classmethod
    def model_validate(cls, data):
        return cls(data)

    def model_dump(self):
        return dict(self._raw)

params_module.Params = _Params
sys.modules["Generate.params"] = params_module

# --- Stub models.decoding.decode ---
decoding_module = types.ModuleType("models.decoding")
decoding_module.decode = lambda *_a, **_k: ["<DUMMY>"]
sys.modules["models.decoding"] = decoding_module

# --- Dummy tokenizer/model ---
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

    def encode_params(self, _params):
        return [0]

    def adjacency_requirements_from_params(self, adjacency):
        # Pass-through dict or empty mapping
        return dict(adjacency or {})

    def decode_layout_tokens(self, _tokens):
        # Return template layout directly
        return json.loads(json.dumps(self._layout))

class DummyModel:
    def load_state_dict(self, _state, strict=False):
        return None
    def to(self, _device):
        return self

layout_template = {
    "layout": {
        "rooms": [
            {
                "type": "Master bedroom",
                "position": {"x": 0, "y": 0},
                "size": {"width": 12, "length": 12},
            },
            {
                "type": "Master bathroom",
                "position": {"x": 12, "y": 0},
                "size": {"width": 8, "length": 12},
            },
            {
                "type": "Kitchen",
                "position": {"x": 0, "y": 12},
                "size": {"width": 12, "length": 12},
            },
            {
                "type": "Living room",
                "position": {"x": 12, "y": 12},
                "size": {"width": 14, "length": 12},
            },
            {
                "type": "Dining room",
                "position": {"x": 26, "y": 12},
                "size": {"width": 12, "length": 12},
            },
            {
                "type": "Laundry room",
                "position": {"x": 14, "y": 24},
                "size": {"width": 10, "length": 10},
            },
        ]
    }
}

# --- Run orchestrator with stubs ---
from Generate.orchestrator import generate_layout
from evaluation.validators import validate_layout

min_sep = 0.0

layout_json, meta = generate_layout(
    params=_Params(params),
    raw_params=params,
    backend="model",  # Force model path to avoid needing OR-Tools
    strategy="guided",
    temperature=1.0,
    beam_size=4,
    min_separation=min_sep,
    tokenizer=DummyTokenizer(layout_template),
    model=DummyModel(),
)

issues = validate_layout(
    layout_json,
    max_width=params["dimensions"]["width"],
    max_length=params["dimensions"]["depth"],
    min_separation=min_sep,
    adjacency=params["adjacency"],
)

if issues:
    print("Issues:", "; ".join(issues))
    sys.exit(2)

print("OK: layout valid with 0 issues; backend_used=", meta.get("backend_used"))
