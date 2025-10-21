from __future__ import annotations

from typing import Dict, List, Tuple

from evaluation.validators import _room_bounds as _rb


def _narrow_gap_between(a: Tuple[float,float,float,float], b: Tuple[float,float,float,float], min_gap: float) -> bool:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    # Parallel vertical strips
    if not (ay2 <= by1 or ay1 >= by2):
        if ax2 <= bx1:
            gap = bx1 - ax2
            if 0 < gap < min_gap and (min(ay2, by2) - max(ay1, by1)) >= min_gap:
                return True
        if bx2 <= ax1:
            gap = ax1 - bx2
            if 0 < gap < min_gap and (min(ay2, by2) - max(ay1, by1)) >= min_gap:
                return True
    # Parallel horizontal strips
    if not (ax2 <= bx1 or ax1 >= bx2):
        if ay2 <= by1:
            gap = by1 - ay2
            if 0 < gap < min_gap and (min(ax2, bx2) - max(ax1, bx1)) >= min_gap:
                return True
        if by2 <= ay1:
            gap = ay1 - by2
            if 0 < gap < min_gap and (min(ax2, bx2) - max(ax1, bx1)) >= min_gap:
                return True
    return False


def detect_clashes(layout: Dict, *, min_gap: float = 3.0) -> List[str]:
    """Detect potential clashes and narrow slivers between rooms (beyond pure overlaps)."""
    rooms = (layout.get("layout") or {}).get("rooms", [])
    bounds = [_rb(r) for r in rooms]
    issues: List[str] = []
    for i in range(len(rooms)):
        for j in range(i + 1, len(rooms)):
            if _narrow_gap_between(bounds[i], bounds[j], min_gap):
                a = rooms[i].get("type", "A")
                b = rooms[j].get("type", "B")
                issues.append(f"Narrow gap (< {min_gap}ft) between {a} and {b}")
    return issues
