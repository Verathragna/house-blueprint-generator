import xml.etree.ElementTree as ET
from dataset.render_svg import render_layout_svg
from evaluation.validators import clamp_bounds

def test_render_includes_lot_boundary_and_clamps(tmp_path):
    layout = {
        "layout": {
            "rooms": [
                {
                    "type": "Bedroom",
                    "position": {"x": 8, "y": 8},
                    "size": {"width": 5, "length": 5},
                }
            ]
        }
    }
    lot_dims = (10, 10)
    layout = clamp_bounds(layout, lot_dims[0], lot_dims[1])
    svg_path = tmp_path / "out.svg"
    render_layout_svg(layout, svg_path, lot_dims=lot_dims)
    tree = ET.parse(svg_path)
    root = tree.getroot()
    ns = {"svg": "http://www.w3.org/2000/svg"}
    rects = root.findall("svg:rect", ns)
    assert len(rects) == 2
    lot_rect = next(r for r in rects if r.get("fill") in (None, "none"))
    assert float(lot_rect.get("width")) == lot_dims[0] * 10
    assert float(lot_rect.get("height")) == lot_dims[1] * 10
    room_rect = next(r for r in rects if r is not lot_rect)
    x = float(room_rect.get("x"))
    y = float(room_rect.get("y"))
    w = float(room_rect.get("width"))
    h = float(room_rect.get("height"))
    assert x >= 0 and y >= 0
    assert x + w <= lot_dims[0] * 10
    assert y + h <= lot_dims[1] * 10
