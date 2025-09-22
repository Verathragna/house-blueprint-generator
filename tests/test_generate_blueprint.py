
import json
from copy import deepcopy
from pathlib import Path

import pytest
import subprocess

from evaluation.validators import validate_layout
import importlib
import sys
import types


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
        return deepcopy(self._layout)


class DummyModel:
    def __init__(self, *_args):
        pass

    def load_state_dict(self, _state):
        return None

    def to(self, _device):
        return self



@pytest.mark.parametrize("min_sep", [0.0, 2.0])
def test_generate_blueprint_produces_valid_layout(tmp_path, monkeypatch, min_sep):
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

    dummy_tokenizer = DummyTokenizer(layout_template)

    pydantic_module = types.ModuleType("pydantic")
    class _ValidationError(Exception):
        ...
    pydantic_module.ValidationError = _ValidationError
    monkeypatch.setitem(sys.modules, "pydantic", pydantic_module)

    params_module = types.ModuleType("Generate.params")

    class _Params:
        def __init__(self, data):
            self._raw = data
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
    monkeypatch.setitem(sys.modules, "Generate.params", params_module)

    torch_stub = types.SimpleNamespace(load=lambda *_a, **_k: {})
    monkeypatch.setitem(sys.modules, "torch", torch_stub)

    layout_module = types.ModuleType("models.layout_transformer")
    layout_module.LayoutTransformer = DummyModel
    monkeypatch.setitem(sys.modules, "models.layout_transformer", layout_module)

    decoding_module = types.ModuleType("models.decoding")
    decoding_module.decode = lambda *_a, **_k: ["<DUMMY>"]
    monkeypatch.setitem(sys.modules, "models.decoding", decoding_module)

    def fake_render(layout, svg_path, lot_dims=None):
        Path(svg_path).write_text("<svg />")

    render_module = types.ModuleType("dataset.render_svg")
    render_module.render_layout_svg = fake_render
    monkeypatch.setitem(sys.modules, "dataset.render_svg", render_module)

    sys.modules.pop("Generate.generate_blueprint", None)
    gb = importlib.import_module("Generate.generate_blueprint")

    original_exists = gb.os.path.exists
    monkeypatch.setattr(gb, "BlueprintTokenizer", lambda: dummy_tokenizer)
    monkeypatch.setattr(gb.torch, "load", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(gb.os.path, "exists", lambda path: True if path == gb.CKPT else original_exists(path))

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

    params_path = tmp_path / "params.json"
    params_path.write_text(json.dumps(params))

    out_prefix = tmp_path / "generated"
    argv = [
        "generate_blueprint.py",
        "--params_json",
        str(params_path),
        "--out_prefix",
        str(out_prefix),
    ]
    if min_sep:
        argv.extend(["--min_separation", str(min_sep)])

    monkeypatch.setattr(gb.sys, "argv", argv)

    gb.main()

    json_path = out_prefix.with_suffix(".json")
    assert json_path.exists()

    data = json.loads(json_path.read_text())
    adjacency = params["adjacency"]
    issues = validate_layout(
        data,
        max_width=params["dimensions"]["width"],
        max_length=params["dimensions"]["depth"],
        min_separation=min_sep,
        adjacency=adjacency,
    )
    assert issues == []



@pytest.mark.slow
def test_generate_blueprint_real_model(tmp_path):
    ckpt_path = Path("checkpoints/model_latest.pth")
    if not ckpt_path.exists():
        pytest.skip("real model checkpoint not available")

    params = {
        "dimensions": {"width": 40, "depth": 40},
        "bedrooms": 2,
        "bathrooms": {"full": 1, "half": 0},
        "kitchen": 1,
        "livingRooms": 1,
        "diningRooms": 1,
        "laundryRooms": 1,
    }
    params_path = tmp_path / "smoke_params.json"
    params_path.write_text(json.dumps(params))

    out_prefix = tmp_path / "smoke"
    cmd = [
        sys.executable,
        "Generate/generate_blueprint.py",
        "--params_json",
        str(params_path),
        "--out_prefix",
        str(out_prefix),
        "--min_separation",
        "0",
        "--device",
        "cpu",
        "--max_attempts",
        "5",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        msg = f"Generator failed: {result.stderr}\n{result.stdout}"
        pytest.fail(msg)

    layout_path = out_prefix.with_suffix(".json")
    assert layout_path.exists(), "Generator did not produce a JSON layout"

    layout = json.loads(layout_path.read_text())
    rooms = (layout.get("layout") or {}).get("rooms", [])
    assert rooms, "Generated layout contains no rooms"
