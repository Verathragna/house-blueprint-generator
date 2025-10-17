"""Geometry validators for generated house layouts.

This module provides simple geometric checks to ensure rooms stay within
a predefined bounding box, do not overlap, and maintain a minimum
separation distance.  The functions return human readable strings
describing any issues that are found.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Set, Tuple


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


def clamp_bounds(layout: Dict, max_width: float = 40, max_length: float = 40) -> Dict:
    """Clamp room positions so they lie within the layout bounds.

    Each room's ``x`` and ``y`` coordinates are adjusted so that the
    resulting room rectangle remains within ``[0, max_width]`` and
    ``[0, max_length]``.
    """

    rooms = (layout.get("layout") or {}).get("rooms", [])
    for room in rooms:
        pos = room.setdefault("position", {})
        size = room.get("size") or {}
        w = float(size.get("width", 0))
        l = float(size.get("length", 0))
        x = float(pos.get("x", 0))
        y = float(pos.get("y", 0))
        x = max(0.0, min(x, max_width - w))
        y = max(0.0, min(y, max_length - l))
        pos["x"], pos["y"] = x, y
    return layout


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
    """Return ``True`` if two rooms are closer than ``min_sep``.

    The rooms are represented by axis-aligned bounding boxes.  Two rooms
    are considered "too close" if the distance between their rectangles
    is less than ``min_sep`` in either the x or y direction.  A
    ``min_sep`` of ``0`` therefore degenerates to a pure overlap check.
    """

    ax1, ay1, ax2, ay2 = _room_bounds(r1)
    bx1, by1, bx2, by2 = _room_bounds(r2)
    return not (
        ax2 + min_sep <= bx1
        or bx2 + min_sep <= ax1
        or ay2 + min_sep <= by1
        or by2 + min_sep <= ay1
    )


def _shares_wall(r1: Dict, r2: Dict, tol: float = 1e-6) -> bool:
    """Return ``True`` if two rooms share a wall segment."""

    ax1, ay1, ax2, ay2 = _room_bounds(r1)
    bx1, by1, bx2, by2 = _room_bounds(r2)

    vertical_touch = (abs(ax2 - bx1) < tol or abs(bx2 - ax1) < tol) and (
        min(ay2, by2) - max(ay1, by1) > 0
    )
    horizontal_touch = (abs(ay2 - by1) < tol or abs(by2 - ay1) < tol) and (
        min(ax2, bx2) - max(ax1, bx1) > 0
    )
    return vertical_touch or horizontal_touch


def check_connectivity(rooms: List[Dict]) -> List[str]:
    """Verify every room is adjacent to at least one other room."""

    issues: List[str] = []
    if len(rooms) <= 1:
        return issues

    for i, r1 in enumerate(rooms):
        if not any(_shares_wall(r1, r2) for j, r2 in enumerate(rooms) if i != j):
            issues.append(
                f"Room {r1.get('type', 'Unknown')} is not connected to any other room"
            )
    return issues


def check_adjacency(rooms: List[Dict], adjacency: Dict[str, List[str]]) -> List[str]:
    """Ensure each specified room type meets its adjacency requirements.

    Args:
        rooms: List of room dictionaries.
        adjacency: Mapping of room types to the list of room types that must
            share a wall with them.

    Returns:
        List of human-readable issue strings.
    """

    type_map: Dict[str, List[Tuple[int, Dict]]] = {}
    for room_idx, room in enumerate(rooms):
        type_map.setdefault(room.get("type", "").lower(), []).append((room_idx, room))

    issues: List[str] = []
    for room_type, required in (adjacency or {}).items():
        src_rooms = type_map.get(room_type.lower())
        if not src_rooms:
            issues.append(f"Room {room_type} required for adjacency check is missing")
            continue
        for target in required:
            tgt_rooms = type_map.get(target.lower())
            if not tgt_rooms:
                issues.append(
                    f"Room {target} required to be adjacent to {room_type} is missing"
                )
                continue
            for src_idx, src_room in src_rooms:
                if any(_shares_wall(src_room, tgt_room) for _, tgt_room in tgt_rooms):
                    continue
                pos = src_room.get("position") or {}
                x = pos.get("x", 0)
                y = pos.get("y", 0)
                label = src_room.get("type") or room_type
                issues.append(
                    f"{label} #{src_idx + 1} at ({x}, {y}) must share a wall with {target}"
                )
    return issues


def check_separation(
    rooms: List[Dict],
    min_sep: float,
    adjacency: Optional[Dict[str, List[str]]] = None,
) -> List[str]:
    """Ensure rooms are at least ``min_sep`` units apart.

    Pairs of rooms explicitly marked as adjacent are exempt from this check.
    """

    issues: List[str] = []
    if min_sep <= 0:
        return issues

    adj_pairs: Set[Tuple[str, str]] = set()
    for a, bs in (adjacency or {}).items():
        for b in bs:
            adj_pairs.add((a.lower(), b.lower()))
            adj_pairs.add((b.lower(), a.lower()))

    for i, r1 in enumerate(rooms):
        t1 = r1.get("type", "").lower()
        for r2 in rooms[i + 1 :]:
            t2 = r2.get("type", "").lower()
            if (t1, t2) in adj_pairs:
                continue
            if _too_close(r1, r2, min_sep):
                issues.append(
                    f"Room {r1.get('type', 'Unknown')} is within {min_sep} of {r2.get('type', 'Unknown')}"
                )
    return issues





def resolve_overlaps(
    layout: Dict,
    adjacency: Optional[Dict[str, List[str]]] = None,
    step: float = 0.5,
    max_iterations: int = 5,
    separation_iterations: int = 100,
) -> Dict:
    """Attempt to separate overlapping rooms while preserving adjacency."""

    rooms = (layout.get("layout") or {}).get("rooms", [])
    if not rooms:
        return layout

    current_step = max(step, 1e-3)
    logger = logging.getLogger(__name__)
    for iteration in range(max_iterations):
        if not check_overlaps(rooms):
            break
        layout = enforce_min_separation(
            layout,
            current_step,
            adjacency=adjacency,
            max_iterations=separation_iterations,
        )
        rooms = (layout.get("layout") or {}).get("rooms", [])
        current_step *= 1.5
    else:
        logger.warning(
            "resolve_overlaps reached iteration limit (%s) while overlaps remain",
            max_iterations,
        )
    return layout


def enforce_min_separation(
    layout: Dict,
    min_sep: float = 1.0,
    adjacency: Optional[Dict[str, List[str]]] = None,
    max_iterations: int = 100,
) -> Dict:
    """Shift rooms to ensure a minimum separation while preserving adjacency.

    Rooms that are required to be adjacent are moved as a group. The group is
    shifted either horizontally or vertically to satisfy the minimum separation
    from other rooms.
    """

    rooms = (layout.get("layout") or {}).get("rooms", [])
    if not rooms:
        return layout

    # Build adjacency graph of room indices
    type_map: Dict[str, List[int]] = {}
    for idx, room in enumerate(rooms):
        type_map.setdefault(room.get("type", "").lower(), []).append(idx)
        pos = room.setdefault("position", {})
        pos.setdefault("x", 0.0)
        pos.setdefault("y", 0.0)

    graph: Dict[int, Set[int]] = {i: set() for i in range(len(rooms))}
    for src, targets in (adjacency or {}).items():
        for tgt in targets:
            for i in type_map.get(src.lower(), []):
                for j in type_map.get(tgt.lower(), []):
                    graph[i].add(j)
                    graph[j].add(i)

    # Determine connected components (groups of rooms that must move together)
    groups: List[Set[int]] = []
    group_id: Dict[int, int] = {}
    visited: Set[int] = set()
    for i in range(len(rooms)):
        if i in visited:
            continue
        stack = [i]
        comp: Set[int] = set([i])
        visited.add(i)
        while stack:
            node = stack.pop()
            for nbr in graph[node]:
                if nbr not in visited:
                    visited.add(nbr)
                    stack.append(nbr)
                    comp.add(nbr)
        groups.append(comp)
        gid = len(groups) - 1
        for idx in comp:
            group_id[idx] = gid

    def group_bounds(indices: Set[int]) -> Tuple[float, float, float, float]:
        xs1: List[float] = []
        ys1: List[float] = []
        xs2: List[float] = []
        ys2: List[float] = []
        for idx in indices:
            x1, y1, x2, y2 = _room_bounds(rooms[idx])
            xs1.append(x1)
            ys1.append(y1)
            xs2.append(x2)
            ys2.append(y2)
        return min(xs1), min(ys1), max(xs2), max(ys2)

    def move_group(indices: Set[int], dx: float, dy: float) -> None:
        for idx in indices:
            pos = rooms[idx]["position"]
            pos["x"] += dx
            pos["y"] += dy

    changed = True
    iterations = 0
    logger = logging.getLogger(__name__)
    while changed and iterations < max_iterations:
        changed = False
        iterations += 1
        for i in range(len(rooms)):
            for j in range(i + 1, len(rooms)):
                gi, gj = group_id[i], group_id[j]
                if gi == gj:
                    continue
                if _too_close(rooms[i], rooms[j], min_sep):
                    ax1, ay1, ax2, ay2 = group_bounds(groups[gi])
                    bx1, by1, bx2, by2 = group_bounds(groups[gj])
                    axc = (ax1 + ax2) * 0.5
                    bxc = (bx1 + bx2) * 0.5
                    ayc = (ay1 + ay2) * 0.5
                    byc = (by1 + by2) * 0.5

                    candidates: List[Tuple[float, float]] = []

                    if bxc >= axc:
                        delta = ax2 + min_sep - bx1
                        if delta > 0:
                            candidates.append((delta, 0.0))
                    else:
                        delta = bx2 + min_sep - ax1
                        if delta > 0:
                            candidates.append((-delta, 0.0))

                    if byc >= ayc:
                        delta = ay2 + min_sep - by1
                        if delta > 0:
                            candidates.append((0.0, delta))
                    else:
                        delta = by2 + min_sep - ay1
                        if delta > 0:
                            candidates.append((0.0, -delta))

                    if not candidates:
                        continue

                    dx, dy = min(
                        candidates,
                        key=lambda shift: (
                            abs(shift[0]) + abs(shift[1]),
                            abs(shift[0]),
                            abs(shift[1]),
                        ),
                    )
                    move_group(groups[gj], dx, dy)
                    changed = True
                    break
            if changed:
                break
    if changed:
        logger.warning(
            "enforce_min_separation reached iteration limit (%s) with residual proximity",
            max_iterations,
        )
    return layout


def validate_layout(
    layout: Dict,
    max_width: float = 40,
    max_length: float = 40,
    min_separation: float = 0,
    require_connectivity: bool = True,
    adjacency: Optional[Dict[str, List[str]]] = None,
) -> List[str]:
    """Validate layout geometry.

    Args:
        layout: Layout dictionary containing rooms under ``layout['rooms']``.
        max_width: Maximum horizontal extent for bounds checking.
        max_length: Maximum vertical extent for bounds checking.
        min_separation: Required minimum spacing between rooms. ``0`` skips
            the check.

    Returns:
        A list of issues. Empty if layout passes validation.
    """

    rooms = (layout.get("layout") or {}).get("rooms", [])
    issues: List[str] = []
    issues.extend(check_bounds(rooms, max_width=max_width, max_length=max_length))
    issues.extend(check_overlaps(rooms))
    if require_connectivity:
        issues.extend(check_connectivity(rooms))
    if adjacency:
        issues.extend(check_adjacency(rooms, adjacency))
    if min_separation > 0:
        issues.extend(check_separation(rooms, min_separation, adjacency))
    return issues


__all__ = [
    "check_bounds",
    "check_overlaps",
    "check_separation",
    "check_connectivity",
    "check_adjacency",
    "validate_layout",
    "enforce_min_separation",
    "resolve_overlaps",
    "clamp_bounds",
]
