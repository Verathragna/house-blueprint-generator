import copy
from typing import Dict

MAX_COORD = 40


def _clamp_rooms(rooms, max_coord: int = MAX_COORD) -> None:
    for r in rooms:
        w = r["size"]["width"]
        l = r["size"]["length"]
        r["position"]["x"] = max(0, min(r["position"]["x"], max_coord - w))
        r["position"]["y"] = max(0, min(r["position"]["y"], max_coord - l))


def mirror_layout(layout: Dict) -> Dict:
    """Mirror layout horizontally across its bounding box."""
    out = copy.deepcopy(layout)
    rooms = out.get("layout", {}).get("rooms", [])
    if not rooms:
        return out
    xs = [r["position"]["x"] for r in rooms]
    min_x, max_x = min(xs), max(xs)
    for r in rooms:
        r["position"]["x"] = max_x - (r["position"]["x"] - min_x)
    _clamp_rooms(rooms)
    return out


def rotate_layout(layout: Dict) -> Dict:
    """Rotate layout 90 degrees around its bounding box."""
    out = copy.deepcopy(layout)
    rooms = out.get("layout", {}).get("rooms", [])
    if not rooms:
        return out
    xs = [r["position"]["x"] for r in rooms]
    ys = [r["position"]["y"] for r in rooms]
    min_x, max_x = min(xs), max(xs)
    min_y = min(ys)
    width = max_x - min_x
    for r in rooms:
        x, y = r["position"]["x"], r["position"]["y"]
        r["position"]["x"] = y - min_y + min_x
        r["position"]["y"] = width - (x - min_x) + min_y
        w, l = r["size"]["width"], r["size"]["length"]
        r["size"]["width"], r["size"]["length"] = l, w
    _clamp_rooms(rooms)
    return out


def augment_layout(layout: Dict) -> Dict:
    """Return mirrored and rotated variants of the layout."""
    return [mirror_layout(layout), rotate_layout(layout)]
