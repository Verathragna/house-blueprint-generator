import json
from pathlib import Path

from dataset.render_svg import render_layout_svg


def test_svg_snapshot(tmp_path):
    layout = {
        "layout": {
            "rooms": [
                {"type": "Living Room", "position": {"x": 0, "y": 0}, "size": {"width": 12, "length": 10}},
                {"type": "Kitchen", "position": {"x": 12, "y": 0}, "size": {"width": 10, "length": 10}},
            ]
        }
    }
    svg_path = tmp_path / "plan.svg"
    render_layout_svg(layout, str(svg_path), lot_dims=(30, 30))
    data = svg_path.read_text(encoding="utf-8")
    assert data.strip().startswith("<svg"), "SVG output should start with <svg>"
    assert "Living Room" in data
    assert "Kitchen" in data
