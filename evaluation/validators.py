"""Geometry validators for generated house layouts.

This module provides simple geometric checks to ensure rooms do not
overlap and stay within a predefined bounding box.  The functions
return human readable strings describing any issues that are found.
"""

from __future__ import annotations

from typing import Dict, List, Tuple


def _room_bounds(room: Dict) -> Tuple[float, float, float, float]:
    """Return room bounds as (x1, y1, x2, y2)."""
    x = float((room.get("position") or {}).get("x", 0))
    y = float((room.get("position") or {}).get("y", 0))
    w = float((room.get("size") or {}).get("width", 0))
    l = float((room.get("size") or {}).get("length", 0))
    return x, y, x + w, y + l


def check_bounds(rooms: List[Dict], max_width: float = 40, max_length: float = 40) -> List[str]:
    """Check that rooms lie within the given bounds.

    Args:
        rooms: List of room dictionaries.
        max_width: Maximum allowed x extent.
        max_length: Maximum allowed y extent.

    Returns:
        List of issues describing boundary violations.
    """
    issues: List[str] = []
    for room in rooms:
        x1, y1, x2, y2 = _room_bounds(room)
        if x1 < 0 or y1 < 0 or x2 > max_width or y2 > max_length:
            issues.append(
                f"Room {room.get('type', 'Unknown')} at ({x1}, {y1}) "
                f"exceeds bounds {max_width}x{max_length}"
            )
    return issues


def check_overlaps(rooms: List[Dict]) -> List[str]:
    """Check for overlapping rooms.

    Args:
        rooms: List of room dictionaries.

    Returns:
        List of issues describing overlaps.
    """
    issues: List[str] = []
    for i, r1 in enumerate(rooms):
        x1, y1, x2, y2 = _room_bounds(r1)
        for r2 in rooms[i + 1 :]:
            xa, ya, xb, yb = _room_bounds(r2)
            if x1 < xb and x2 > xa and y1 < yb and y2 > ya:
                issues.append(
                    f"Room {r1.get('type', 'Unknown')} overlaps with {r2.get('type', 'Unknown')}"
                )
    return issues


def _too_close(r1: Dict, r2: Dict, min_sep: float) -> bool:
    """Return True if two rooms are closer than ``min_sep``."""
    ax1, ay1, ax2, ay2 = _room_bounds(r1)
    bx1, by1, bx2, by2 = _room_bounds(r2)
    return not (
        ax2 + min_sep <= bx1
        or bx2 + min_sep <= ax1
        or ay2 + min_sep <= by1
        or by2 + min_sep <= ay1
    )


def enforce_min_separation(layout: Dict, min_sep: float = 1.0) -> Dict:
    """Shift rooms to ensure a minimum separation.

    Rooms are moved to the right until they are at least ``min_sep`` units
    away from previously placed rooms. This is a simple post-processing step
    and does not guarantee a globally optimal arrangement, but it prevents
    obvious overlaps in the generated layouts.
    """
    rooms = (layout.get("layout") or {}).get("rooms", [])
    for i, room in enumerate(rooms):
        pos = room.setdefault("position", {})
        pos.setdefault("x", 0.0)
        pos.setdefault("y", 0.0)
        while any(_too_close(room, prev, min_sep) for prev in rooms[:i]):
            pos["x"] += min_sep
    return layout


def validate_layout(layout: Dict, max_width: float = 40, max_length: float = 40) -> List[str]:
    """Validate layout geometry.

    Args:
        layout: Layout dictionary containing rooms under ``layout['rooms']``.
        max_width: Maximum horizontal extent for bounds checking.
        max_length: Maximum vertical extent for bounds checking.

    Returns:
        A list of issues. Empty if layout passes validation.
    """
    rooms = (layout.get("layout") or {}).get("rooms", [])
    issues = []
    issues.extend(check_bounds(rooms, max_width=max_width, max_length=max_length))
    issues.extend(check_overlaps(rooms))
    return issues


__all__ = [
    "check_bounds",
    "check_overlaps",
    "validate_layout",
    "enforce_min_separation",
]
