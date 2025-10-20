import json
from pathlib import Path

from Generate.params import Params


def test_params_schema_contains_core_fields(tmp_path):
    schema = Params.model_json_schema()
    assert isinstance(schema, dict)
    props = schema.get("properties") or {}
    # Core fields
    assert "dimensions" in props
    assert "bedrooms" in props
    assert "bathrooms" in props
    # JSON-serializable
    out = tmp_path / "schema.json"
    out.write_text(json.dumps(schema))
    loaded = json.loads(out.read_text())
    assert "properties" in loaded
