import json
from evaluation.validators import check_connectivity, validate_layout


def test_detects_disconnected_rooms():
    rooms = [
        {"type": "Bedroom", "position": {"x": 0, "y": 0}, "size": {"width": 5, "length": 5}},
        {"type": "Kitchen", "position": {"x": 10, "y": 0}, "size": {"width": 5, "length": 5}},
    ]
    issues = check_connectivity(rooms)
    assert issues
    layout = {"layout": {"rooms": rooms}}
    issues2 = validate_layout(layout)
    assert any("not connected" in msg for msg in issues2)
