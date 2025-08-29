import json
from evaluation.validators import (
    check_connectivity,
    enforce_min_separation,
    validate_layout,
)


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


def test_required_adjacency_passes():
    rooms = [
        {
            "type": "Master bedroom",
            "position": {"x": 0, "y": 0},
            "size": {"width": 10, "length": 10},
        },
        {
            "type": "Master bathroom",
            "position": {"x": 10, "y": 0},
            "size": {"width": 5, "length": 10},
        },
    ]
    layout = {"layout": {"rooms": rooms}}
    adjacency = {"Master bedroom": ["Master bathroom"]}
    issues = validate_layout(layout, adjacency=adjacency)
    assert issues == []


def test_required_adjacency_detects_violation():
    rooms = [
        {
            "type": "Master bedroom",
            "position": {"x": 0, "y": 0},
            "size": {"width": 10, "length": 10},
        },
        {
            "type": "Master bathroom",
            "position": {"x": 12, "y": 0},
            "size": {"width": 5, "length": 10},
        },
    ]
    layout = {"layout": {"rooms": rooms}}
    adjacency = {"Master bedroom": ["Master bathroom"]}
    issues = validate_layout(layout, adjacency=adjacency)
    assert any("must be adjacent" in msg for msg in issues)


def test_enforce_min_separation_preserves_adjacency():
    rooms = [
        {
            "type": "Master bedroom",
            "position": {"x": 0, "y": 0},
            "size": {"width": 10, "length": 10},
        },
        {
            "type": "Master bathroom",
            "position": {"x": 10, "y": 0},
            "size": {"width": 5, "length": 10},
        },
        {
            "type": "Kitchen",
            "position": {"x": 5, "y": 0},
            "size": {"width": 5, "length": 10},
        },
    ]
    layout = {"layout": {"rooms": rooms}}
    adjacency = {"Master bedroom": ["Master bathroom"]}
    enforce_min_separation(layout, 2.0, adjacency=adjacency)
    issues = validate_layout(
        layout, min_separation=2.0, adjacency=adjacency, require_connectivity=False
    )
    assert issues == []
