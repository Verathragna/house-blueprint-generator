import os
import sys
import uuid
from unittest.mock import patch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from fastapi.testclient import TestClient
from api.app import app, _tokenizer


def dummy_render(layout_json, svg_path):
    with open(svg_path, "w", encoding="utf-8") as f:
        f.write("<svg></svg>")


def test_generate_returns_paths_and_metadata(tmp_path):
    client = TestClient(app)
    payload = {"params": {}}
    with patch("api.app._get_model", lambda: object()), \
         patch("api.app.decode", lambda *args, **kwargs: []), \
         patch.object(_tokenizer, "encode_params", return_value=[]), \
         patch.object(_tokenizer, "decode_layout_tokens", return_value={}), \
         patch("api.app.enforce_min_separation", lambda x: x), \
         patch("api.app.render_layout_svg", dummy_render), \
         patch("api.app.REPO_ROOT", str(tmp_path)), \
         patch("api.app.uuid.uuid4") as mock_uuid:
        mock_uuid.side_effect = [uuid.UUID(int=1), uuid.UUID(int=2)]
        resp1 = client.post("/generate", json=payload)
        resp2 = client.post("/generate", json=payload)
    assert resp1.status_code == 200
    assert resp2.status_code == 200
    data1 = resp1.json()
    data2 = resp2.json()
    assert data1["svg_filename"] != data2["svg_filename"]
    assert "saved_svg_path" in data1 and "saved_json_path" in data1
    assert data1["metadata"]["processing_time"] >= 0


def test_validation_error_has_metadata():
    client = TestClient(app)
    payload = {"params": {}, "beam_size": 0}
    resp = client.post("/generate", json=payload)
    assert resp.status_code == 422
    data = resp.json()
    assert data["code"] == "validation_error"
    assert "processing_time" in data["metadata"]
