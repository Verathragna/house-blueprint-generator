from __future__ import annotations

from typing import Dict, List, Optional

from evaluation.validators import _room_bounds as _rb, _shares_wall as _sw


def check_corridor_width(layout: Dict, *, min_width: float = 3.0) -> List[str]:
    issues: List[str] = []
    rooms = (layout.get("layout") or {}).get("rooms", [])
    for room in rooms:
        t = (room.get("type") or "").strip().lower()
        if "hallway" not in t:
            continue
        # Treat hallway width as the smaller side
        w = float((room.get("size") or {}).get("width", 0))
        l = float((room.get("size") or {}).get("length", 0))
        width = min(w, l)
        if width < min_width:
            issues.append(f"Hallway width {width:.1f}ft below minimum {min_width:.1f}ft")
    return issues


def check_min_door_openings(
    layout: Dict,
    *,
    adjacency: Optional[Dict[str, List[str]]] = None,
    min_opening: float = 3.0,
) -> List[str]:
    """Ensure required adjacencies have at least a shared wall segment >= min_opening (as door proxy)."""
    if not adjacency:
        return []
    rooms = (layout.get("layout") or {}).get("rooms", [])
    type_map: Dict[str, List[int]] = {}
    for idx, r in enumerate(rooms):
        t = (r.get("type") or "").strip().lower()
        if t:
            type_map.setdefault(t, []).append(idx)
    issues: List[str] = []
    def shared_wall_len(a, b) -> float:
        ax1, ay1, ax2, ay2 = _rb(a)
        bx1, by1, bx2, by2 = _rb(b)
        # vertical contact
        L = 0.0
        if abs(ax2 - bx1) < 1e-6 or abs(bx2 - ax1) < 1e-6:
            L = max(0.0, min(ay2, by2) - max(ay1, by1))
        # horizontal contact
        if abs(ay2 - by1) < 1e-6 or abs(by2 - ay1) < 1e-6:
            L = max(L, max(0.0, min(ax2, bx2) - max(ax1, bx1)))
        return L
    for a, targets in (adjacency or {}).items():
        akey = (a or "").strip().lower()
        for b in (targets or []):
            bkey = (b or "").strip().lower()
            a_idxs = type_map.get(akey) or []
            b_idxs = type_map.get(bkey) or []
            if not a_idxs or not b_idxs:
                continue
            ok = False
            best = 0.0
            for i in a_idxs:
                for j in b_idxs:
                    L = shared_wall_len(rooms[i], rooms[j])
                    best = max(best, L)
                    if L >= min_opening:
                        ok = True
                        break
                if ok:
                    break
            if not ok:
                issues.append(f"Insufficient door opening between {a} and {b} (max shared {best:.1f}ft, need {min_opening:.1f}ft)")
    return issues


def check_bathroom_turning(layout: Dict, *, min_diameter: float = 5.0) -> List[str]:
    """Simplified ADA: ensure bathrooms allow a 5' turning circle (min(width,length) >= 5ft)."""
    issues: List[str] = []
    rooms = (layout.get("layout") or {}).get("rooms", [])
    for room in rooms:
        t = (room.get("type") or "").strip().lower()
        if "bathroom" not in t:
            continue
        w = float((room.get("size") or {}).get("width", 0))
        l = float((room.get("size") or {}).get("length", 0))
        if min(w, l) < float(min_diameter):
            issues.append(f"Bathroom too narrow for {min_diameter:.1f}ft turning circle")
    return issues


def validate_accessibility(
    layout: Dict,
    *,
    adjacency: Optional[Dict[str, List[str]]] = None,
    min_corridor_width: float = 3.0,
    min_door_opening: float = 3.0,
    bathroom_turn_diameter: float = 5.0,
) -> List[str]:
    issues: List[str] = []
    issues += check_corridor_width(layout, min_width=min_corridor_width)
    issues += check_min_door_openings(layout, adjacency=adjacency, min_opening=min_door_opening)
    issues += check_bathroom_turning(layout, min_diameter=bathroom_turn_diameter)
    return issues
