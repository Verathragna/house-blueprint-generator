import os
import sys
import uuid
import time
from unittest.mock import patch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from fastapi.testclient import TestClient
from api.app import app, _tokenizer


def dummy_render(layout_json, svg_path):
    with open(svg_path, "w", encoding="utf-8") as f:
        f.write("<svg></svg>")


def test_generate_returns_job_and_result(tmp_path):
    client = TestClient(app)
    headers = {"X-API-Key": "testkey"}
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
        resp = client.post("/generate", json=payload, headers=headers)
        assert resp.status_code == 200
        job_id = resp.json()["job_id"]
        for _ in range(10):
            status_resp = client.get(f"/status/{job_id}", headers=headers)
            assert status_resp.status_code == 200
            data = status_resp.json()
            if data["status"] == "completed":
                break
            time.sleep(0.1)
    assert data["status"] == "completed"
    result = data["result"]
    assert "saved_svg_path" in result and "saved_json_path" in result
    assert result["svg_filename"] and result["json_filename"]
    assert result["metadata"]["processing_time"] >= 0


def test_validation_error_has_metadata():
    client = TestClient(app)
    headers = {"X-API-Key": "testkey"}
    payload = {"params": {}, "beam_size": 0}
    resp = client.post("/generate", json=payload, headers=headers)
    assert resp.status_code == 422
    data = resp.json()
    assert data["code"] == "validation_error"
    assert "processing_time" in data["metadata"]
