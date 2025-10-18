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


def check_overlaps(rooms: List[Dict], tol: float = 1e-6) -> List[str]:
    """Check for overlapping rooms.

    Args:
        rooms: List of room dictionaries.
        tol: Tolerance to ignore near-touching rectangles due to rounding.

    Returns:
        List of issues describing overlaps.
    """
    issues: List[str] = []
    for i, r1 in enumerate(rooms):
        x1, y1, x2, y2 = _room_bounds(r1)
        for r2 in rooms[i + 1 :]:
            xa, ya, xb, yb = _room_bounds(r2)
            if x1 < xb - tol and x2 > xa + tol and y1 < yb - tol and y2 > ya + tol:
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
    """Ensure rooms are at least ``min_sep`` units apart when they are not touching.

    Exemptions:
    - Pairs of rooms explicitly marked as adjacent (they may and should touch).
    - Any pair that already shares a wall segment (touching is allowed for connectivity).
    """

    issues: List[str] = []
    if min_sep <= 0:
        return issues

    adj_pairs: Set[Tuple[str, str]] = set()
    for a, bs in (adjacency or {}).items():
        for b in bs:
            a_key = (a or "").lower()
            b_key = (b or "").lower()
            if a_key and b_key:
                adj_pairs.add((a_key, b_key))
                adj_pairs.add((b_key, a_key))

    for i, r1 in enumerate(rooms):
        t1 = (r1.get("type") or "").lower()
        for r2 in rooms[i + 1 :]:
            t2 = (r2.get("type") or "").lower()
            # Skip explicitly-adjacent type pairs
            if (t1, t2) in adj_pairs:
                continue
            # Skip pairs that already share a wall (touching is OK)
            if _shares_wall(r1, r2):
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
    max_width: float = 40,
    max_length: float = 40,
) -> Dict:
    """Attempt to separate overlapping rooms while preserving adjacency.

    Strategy:
    1) Move adjacency-connected components as groups using enforce_min_separation.
    2) If overlaps remain (including intra-group), perform a lightweight pairwise
       repulsion pass that nudges individual rooms apart while respecting bounds
       and trying not to break explicit adjacency pairs.
    """

    rooms = (layout.get("layout") or {}).get("rooms", [])
    if not rooms:
        return layout

    # Build a quick lookup of explicitly-adjacent type pairs (both directions)
    adj_pairs: Set[Tuple[str, str]] = set()
    for a, bs in (adjacency or {}).items():
        for b in bs:
            adj_pairs.add((a.lower(), b.lower()))
            adj_pairs.add((b.lower(), a.lower()))

    def _room_type(room: Dict) -> str:
        return (room.get("type") or "").lower()

    def _nudge_apart(r1: Dict, r2: Dict) -> None:
        # Compute overlap rectangle and minimal separating shift
        ax1, ay1, ax2, ay2 = _room_bounds(r1)
        bx1, by1, bx2, by2 = _room_bounds(r2)
        overlap_x = min(ax2, bx2) - max(ax1, bx1)
        overlap_y = min(ay2, by2) - max(ay1, by1)
        if overlap_x <= 0 or overlap_y <= 0:
            return
        # Target gap: 0 for explicitly-adjacent types (touching is fine), else a small epsilon
        want_gap = 0.0 if (_room_type(r1), _room_type(r2)) in adj_pairs else 1e-3
        # Choose axis with smaller move
        # Positive dx means move r2 to the right, negative means left; same for dy downwards
        # Determine centers to set directions
        axc = (ax1 + ax2) * 0.5
        bxc = (bx1 + bx2) * 0.5
        ayc = (ay1 + ay2) * 0.5
        byc = (by1 + by2) * 0.5
        # Required separation along axes
        need_x = overlap_x + want_gap
        need_y = overlap_y + want_gap
        dx2 = need_x if bxc <= axc else -need_x
        dy2 = need_y if byc <= ayc else -need_y
        # Prefer the smaller magnitude shift
        if abs(dx2) <= abs(dy2):
            dx, dy = dx2, 0.0
        else:
            dx, dy = 0.0, dy2
        # Split displacement between both rooms to reduce boundary trapping
        dx1, dy1 = -0.5 * dx, -0.5 * dy
        dx2, dy2 = 0.5 * dx, 0.5 * dy

        def apply_shift(room: Dict, dx: float, dy: float) -> None:
            pos = room.setdefault("position", {})
            size = room.get("size") or {}
            w = float(size.get("width", 0))
            l = float(size.get("length", 0))
            new_x = max(0.0, min(float(pos.get("x", 0)) + dx, max_width - w))
            new_y = max(0.0, min(float(pos.get("y", 0)) + dy, max_length - l))
            pos["x"], pos["y"] = new_x, new_y

        apply_shift(r1, dx1, dy1)
        apply_shift(r2, dx2, dy2)

    current_step = max(step, 1e-3)
    logger = logging.getLogger(__name__)

    # Phase 1: group-based separation
    for _ in range(max_iterations):
        if not check_overlaps(rooms):
            break
        layout = enforce_min_separation(
            layout,
            current_step,
            adjacency=adjacency,
            max_iterations=separation_iterations,
            max_width=max_width,
            max_length=max_length,
        )
        layout = clamp_bounds(layout, max_width, max_length)
        rooms = (layout.get("layout") or {}).get("rooms", [])
        current_step *= 1.5
    else:
        logger.info(
            "resolve_overlaps reached iteration limit (%s) while overlaps remain",
            max_iterations,
        )

    # Phase 2: local pairwise repulsion to fix any residual overlaps (e.g., intra-group)
    # Limit to a small, deterministic number of passes to avoid long runtimes
    leftover = check_overlaps(rooms)
    if leftover:
        # Aggregated repulsion: push overlapping rooms apart collectively
        n = len(rooms)
        for _ in range(60):
            any_change = False
            disp = [(0.0, 0.0) for _ in range(n)]
            for i in range(n):
                for j in range(i + 1, n):
                    r1, r2 = rooms[i], rooms[j]
                    ax1, ay1, ax2, ay2 = _room_bounds(r1)
                    bx1, by1, bx2, by2 = _room_bounds(r2)
                    overlap_x = min(ax2, bx2) - max(ax1, bx1)
                    overlap_y = min(ay2, by2) - max(ay1, by1)
                    if overlap_x > 0 and overlap_y > 0:
                        # Determine preferred axis and direction
                        want_gap = 0.0 if (_room_type(r1), _room_type(r2)) in adj_pairs else 1e-3
                        axc = (ax1 + ax2) * 0.5
                        bxc = (bx1 + bx2) * 0.5
                        ayc = (ay1 + ay2) * 0.5
                        byc = (by1 + by2) * 0.5
                        need_x = overlap_x + want_gap
                        need_y = overlap_y + want_gap
                        if need_x <= need_y:
                            dx = need_x if bxc <= axc else -need_x
                            dy = 0.0
                        else:
                            dx = 0.0
                            dy = need_y if byc <= ayc else -need_y
                        # split equally
                        dxi, dyi = -0.5 * dx, -0.5 * dy
                        dxj, dyj = 0.5 * dx, 0.5 * dy
                        dix, diy = disp[i]
                        djx, djy = disp[j]
                        disp[i] = (dix + dxi, diy + dyi)
                        disp[j] = (djx + dxj, djy + dyj)
                        any_change = True
            # Apply a damped update and clamp
            alpha = 0.6
            if any_change:
                for idx, (dx, dy) in enumerate(disp):
                    if dx == 0.0 and dy == 0.0:
                        continue
                    pos = rooms[idx].setdefault("position", {})
                    size = rooms[idx].get("size") or {}
                    w = float(size.get("width", 0))
                    l = float(size.get("length", 0))
                    new_x = max(0.0, min(float(pos.get("x", 0)) + alpha * dx, max_width - w))
                    new_y = max(0.0, min(float(pos.get("y", 0)) + alpha * dy, max_length - l))
                    pos["x"], pos["y"] = new_x, new_y
                layout = clamp_bounds(layout, max_width, max_length)
            else:
                break
        # One more pass with individual (non-adjacent) separation to clean up residuals
        layout = enforce_min_separation(
            layout,
            min_sep=1e-3,
            adjacency=None,
            max_iterations=max(10, separation_iterations // 2),
            max_width=max_width,
            max_length=max_length,
        )
        # Final clamp after nudges
        layout = clamp_bounds(layout, max_width, max_length)

    return layout


def enforce_min_separation(
    layout: Dict,
    min_sep: float = 1.0,
    adjacency: Optional[Dict[str, List[str]]] = None,
    max_iterations: int = 100,
    max_width: float = 40.0,
    max_length: float = 40.0,
) -> Dict:
    """Shift rooms to ensure a minimum separation while preserving adjacency.

    Strategy:
    1) Move adjacency-connected components (rigid groups) apart to satisfy separation between groups.
    2) Locally nudge individual rooms apart for any residual too-close pairs, exempting explicitly-adjacent types.
    """

    rooms = (layout.get("layout") or {}).get("rooms", [])
    if not rooms or min_sep <= 0:
        return layout

    # Build adjacency graph of room indices (for required wall-sharing)
    type_map: Dict[str, List[int]] = {}
    for idx, room in enumerate(rooms):
        type_map.setdefault(room.get("type", "").lower(), []).append(idx)
        pos = room.setdefault("position", {})
        pos.setdefault("x", 0.0)
        pos.setdefault("y", 0.0)

    graph: Dict[int, Set[int]] = {i: set() for i in range(len(rooms))}
    adj_pairs: Set[Tuple[str, str]] = set()
    for src, targets in (adjacency or {}).items():
        for tgt in targets:
            for i in type_map.get(src.lower(), []):
                for j in type_map.get(tgt.lower(), []):
                    graph[i].add(j)
                    graph[j].add(i)
            # Track explicit adjacency exemptions by type (both directions)
            a = (src or "").strip().lower()
            b = (tgt or "").strip().lower()
            if a and b:
                adj_pairs.add((a, b))
                adj_pairs.add((b, a))

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
            size = rooms[idx].get("size", {})
            w = float(size.get("width", 0))
            l = float(size.get("length", 0))
            new_x = pos["x"] + dx
            new_y = pos["y"] + dy
            new_x = max(0.0, min(new_x, max_width - w))
            new_y = max(0.0, min(new_y, max_length - l))
            pos["x"] = new_x
            pos["y"] = new_y

    def would_overlap_after_move(indices: Set[int], dx: float, dy: float) -> bool:
        # Check if moving the given group by (dx, dy) would cause any overlap with rooms outside the group
        moved_bounds: List[Tuple[float, float, float, float]] = []
        for idx in indices:
            x1, y1, x2, y2 = _room_bounds(rooms[idx])
            moved_bounds.append((x1 + dx, y1 + dy, x2 + dx, y2 + dy))
        outside_indices = [k for k in range(len(rooms)) if k not in indices]
        for mb in moved_bounds:
            for k in outside_indices:
                ox1, oy1, ox2, oy2 = _room_bounds(rooms[k])
                if mb[0] < ox2 and mb[2] > ox1 and mb[1] < oy2 and mb[3] > oy1:
                    return True
        return False

    logger = logging.getLogger(__name__)

    # Phase 1: group-based separation (rigid components)
    changed = True
    iterations = 0
    while changed and iterations < max_iterations:
        changed = False
        iterations += 1
        for i in range(len(rooms)):
            for j in range(i + 1, len(rooms)):
                gi, gj = group_id[i], group_id[j]
                if gi == gj:
                    continue
                # Only act on pairs that are too close AND not already touching (touching is allowed)
                if _too_close(rooms[i], rooms[j], min_sep) and not _shares_wall(rooms[i], rooms[j]):
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

                    # Prefer candidates that do not introduce new overlaps
                    safe_candidates = [c for c in candidates if not would_overlap_after_move(groups[gj], c[0], c[1])]
                    if safe_candidates:
                        dx, dy = min(
                            safe_candidates,
                            key=lambda shift: (
                                abs(shift[0]) + abs(shift[1]),
                                abs(shift[0]),
                                abs(shift[1]),
                            ),
                        )
                        move_group(groups[gj], dx, dy)
                        changed = True
                        break
                    else:
                        # Fall back: split move across both groups in opposite directions (reduced magnitude)
                        dx, dy = min(
                            candidates,
                            key=lambda shift: (
                                abs(shift[0]) + abs(shift[1]),
                                abs(shift[0]),
                                abs(shift[1]),
                            ),
                        )
                        move_group(groups[gi], -0.5 * dx, -0.5 * dy)
                        move_group(groups[gj], 0.5 * dx, 0.5 * dy)
                        changed = True
                        break
            if changed:
                break

    # Phase 2: local pairwise separation for residual proximity (including intra-group)
    def _room_type(room: Dict) -> str:
        return (room.get("type") or "").strip().lower()

    def _pair_need(ax1, ay1, ax2, ay2, bx1, by1, bx2, by2) -> Tuple[float, float]:
        # Compute how much extra gap is needed along x and y to reach min_sep
        overlap_x = min(ax2, bx2) - max(ax1, bx1)
        overlap_y = min(ay2, by2) - max(ay1, by1)
        if overlap_x > 0:
            need_x = overlap_x + min_sep
        else:
            gap_x = -overlap_x  # >= 0
            need_x = max(0.0, min_sep - gap_x)
        if overlap_y > 0:
            need_y = overlap_y + min_sep
        else:
            gap_y = -overlap_y
            need_y = max(0.0, min_sep - gap_y)
        return need_x, need_y

    # Limit number of local passes for performance
    max_local_passes = max(10, min(2 * max_iterations, 200))
    for _ in range(max_local_passes):
        any_change = False
        n = len(rooms)
        disp = [(0.0, 0.0) for _ in range(n)]
        for i in range(n):
            ai1, aj1, ai2, aj2 = _room_bounds(rooms[i])
            atype = _room_type(rooms[i])
            axc = (ai1 + ai2) * 0.5
            ayc = (aj1 + aj2) * 0.5
            for j in range(i + 1, n):
                btype = _room_type(rooms[j])
                if (atype, btype) in adj_pairs:
                    continue  # explicitly-adjacent types may touch
                # Skip pairs already sharing a wall (touching is allowed)
                if _shares_wall(rooms[i], rooms[j]):
                    continue
                if not _too_close(rooms[i], rooms[j], min_sep):
                    continue
                bi1, bj1, bi2, bj2 = _room_bounds(rooms[j])
                bxc = (bi1 + bi2) * 0.5
                byc = (bj1 + bj2) * 0.5
                need_x, need_y = _pair_need(ai1, aj1, ai2, aj2, bi1, bj1, bi2, bj2)
                if need_x == 0.0 and need_y == 0.0:
                    continue
                if need_x <= need_y:
                    dx = need_x if bxc <= axc else -need_x
                    dy = 0.0
                else:
                    dx = 0.0
                    dy = need_y if byc <= ayc else -need_y
                # split equally
                dxi, dyi = -0.5 * dx, -0.5 * dy
                dxj, dyj = 0.5 * dx, 0.5 * dy
                dix, diy = disp[i]
                djx, djy = disp[j]
                disp[i] = (dix + dxi, diy + dyi)
                disp[j] = (djx + dxj, djy + dyj)
                any_change = True
        if not any_change:
            break
        # Apply damped updates with safeguards (avoid new overlaps, try to preserve connectivity)
        alpha = 0.6
        # Precompute which rooms currently have at least one wall contact
        had_neighbor = [False] * len(rooms)
        for i in range(len(rooms)):
            for j in range(len(rooms)):
                if i == j:
                    continue
                if _shares_wall(rooms[i], rooms[j]):
                    had_neighbor[i] = True
                    break
        for idx, (dx, dy) in enumerate(disp):
            if dx == 0.0 and dy == 0.0:
                continue
            pos = rooms[idx].setdefault("position", {})
            size = rooms[idx].get("size") or {}
            w = float(size.get("width", 0))
            l = float(size.get("length", 0))
            cur_x = float(pos.get("x", 0))
            cur_y = float(pos.get("y", 0))
            step = alpha
            applied = False
            while step > 1e-3:
                trial_x = max(0.0, min(cur_x + step * dx, max_width - w))
                trial_y = max(0.0, min(cur_y + step * dy, max_length - l))
                # Temporarily set and test
                old_x, old_y = pos.get("x", 0.0), pos.get("y", 0.0)
                pos["x"], pos["y"] = trial_x, trial_y
                # Check for overlaps
                creates_overlap = False
                for j in range(len(rooms)):
                    if j == idx:
                        continue
                    x1, y1, x2, y2 = _room_bounds(rooms[idx])
                    ox1, oy1, ox2, oy2 = _room_bounds(rooms[j])
                    if x1 < ox2 and x2 > ox1 and y1 < oy2 and y2 > oy1:
                        creates_overlap = True
                        break
                # Check connectivity preservation if previously had a neighbor
                breaks_connectivity = False
                if had_neighbor[idx]:
                    still_touches = False
                    for j in range(len(rooms)):
                        if j == idx:
                            continue
                        if _shares_wall(rooms[idx], rooms[j]):
                            still_touches = True
                            break
                    if not still_touches:
                        breaks_connectivity = True
                if not creates_overlap and not breaks_connectivity:
                    applied = True
                    break
                # revert and shrink step
                pos["x"], pos["y"] = old_x, old_y
                step *= 0.5
            if applied:
                # Already set pos to trial values inside loop
                pass
        layout = clamp_bounds(layout, max_width, max_length)

    # Final warning if residual proximity remains
    if any(
        _too_close(rooms[i], rooms[j], min_sep)
        and ((_room_type(rooms[i]), _room_type(rooms[j])) not in adj_pairs)
        and (not _shares_wall(rooms[i], rooms[j]))
        for i in range(len(rooms))
        for j in range(i + 1, len(rooms))
    ):
        logger.info(
            "enforce_min_separation reached iteration limit (%s) with residual proximity",
            max_iterations,
        )

    return layout


def _touches_perimeter(room: Dict, *, max_width: float, max_length: float, tol: float = 1e-6) -> Tuple[bool, float]:
    """Return (touches, max_contact_span) with the outer boundary.

    max_contact_span is the longest continuous contact length with any outer edge
    (in feet), considering the room rectangle.
    """
    x = float((room.get("position") or {}).get("x", 0))
    y = float((room.get("position") or {}).get("y", 0))
    w = float((room.get("size") or {}).get("width", 0))
    l = float((room.get("size") or {}).get("length", 0))
    spans = []
    if abs(x - 0.0) < tol:
        spans.append(l)
    if abs(y - 0.0) < tol:
        spans.append(w)
    if abs((x + w) - max_width) < tol:
        spans.append(l)
    if abs((y + l) - max_length) < tol:
        spans.append(w)
    return (len(spans) > 0, max(spans) if spans else 0.0)


def check_entrance(
    rooms: List[Dict],
    *,
    max_width: float,
    max_length: float,
    min_clear: float = 3.0,
    allowed_room_types: Optional[List[str]] = None,
) -> List[str]:
    """Ensure there is an exterior entrance opportunity.

    We require that at least one "public" room touches the perimeter with
    a straight contact span of at least ``min_clear`` feet to fit a door opening.

    By default allowed rooms are: Entry/Foyer, Living/Dining, Kitchen, Hallway,
    and Garage (as a fallback).
    """
    allowed = [
        "Entry",
        "Foyer",
        "Living Room",
        "Dining Room",
        "Kitchen",
        "Hallway",
        "Garage",
    ] if allowed_room_types is None else allowed_room_types
    allowed_l = {t.lower() for t in allowed}
    for r in rooms:
        rtype = (r.get("type") or "").lower()
        if rtype not in allowed_l:
            continue
        touches, span = _touches_perimeter(r, max_width=max_width, max_length=max_length)
        if touches and span >= max(0.1, float(min_clear)):
            return []
    return [
        f"No exterior entrance: no public room touches the boundary with >= {min_clear} ft straight run"
    ]


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
    # Entrance check (after basic geometry so bounds are known)
    issues.extend(
        check_entrance(
            rooms,
            max_width=max_width,
            max_length=max_length,
            min_clear=3.0,
        )
    )
    if adjacency:
        issues.extend(check_adjacency(rooms, adjacency))
    if min_separation > 0:
        issues.extend(check_separation(rooms, min_separation, adjacency))
    return issues


def pack_layout(
    layout: Dict,
    max_width: float = 40,
    max_length: float = 40,
    grid: float = 1.0,
    adjacency_hints: Optional[Dict[str, List[str]]] = None,
    zoning: bool = False,
    min_hall_width: float = 3.0,
    min_gap: float = 3.0,
    avoid_peninsulas: bool = True,
) -> Dict:
    """Re-pack rooms onto a grid to eliminate overlaps and ensure connectivity.

    Improvements over the basic packer:
    - Snap to ``grid`` (feet).
    - Prefer placements next to hinted neighbor types (e.g., Bedroom↔Bathroom, Kitchen↔Dining).
    - Optional zoning that places public/private/service clusters into coarse regions.
    - Simple hallway synthesis when there are multiple bedrooms.
    - Reject placements that create narrow leftover slivers (< ``min_gap``).
    - Avoid "peninsula" rooms that protrude from the envelope (touch boundary with poor internal contact).
    """
    rooms = (layout.get("layout") or {}).get("rooms", [])
    if not rooms:
        return layout

    def snap(v: float) -> float:
        return grid * round(v / max(grid, 1e-6))

    # Sort by area (largest first) for better packing
    indexed = []
    for idx, r in enumerate(rooms):
        x1, y1, x2, y2 = _room_bounds(r)
        area = (x2 - x1) * (y2 - y1)
        indexed.append((area, idx, r))
    indexed.sort(reverse=True)

    placed: List[Dict] = []
    placed_by_type: Dict[str, List[Dict]] = {}

    # Zone regions (coarse)
    zones: Dict[str, Tuple[float, float, float, float]] = {}
    if zoning:
        zones = {
            "public": (0.0, 0.0, max_width * 0.55, max_length * 0.65),
            "private": (max_width * 0.45, 0.0, max_width, max_length),
            "service": (0.0, max_length * 0.6, max_width, max_length),
        }

    def classify(room_type: str) -> str:
        t = (room_type or "").lower()
        if t in {"living room", "dining room", "kitchen"}:
            return "public"
        if t in {"laundry room", "garage", "office", "closet"}:
            return "service"
        return "private"  # bedrooms, bathrooms, etc.

    def _shared_wall_len(px: float, py: float, w: float, l: float) -> float:
        rx1, ry1, rx2, ry2 = px, py, px + w, py + l
        total = 0.0
        for q in placed:
            qx1, qy1, qx2, qy2 = _room_bounds(q)
            # vertical contact
            if abs(rx2 - qx1) < 1e-6 or abs(qx2 - rx1) < 1e-6:
                overlap = max(0.0, min(ry2, qy2) - max(ry1, qy1))
                total += overlap
            # horizontal contact
            if abs(ry2 - qy1) < 1e-6 or abs(qy2 - ry1) < 1e-6:
                overlap = max(0.0, min(rx2, qx2) - max(rx1, qx1))
                total += overlap
        return total

    def _creates_narrow_gap(px: float, py: float, w: float, l: float) -> bool:
        # Reject if the candidate leaves a corridor-like gap narrower than min_gap
        rx1, ry1, rx2, ry2 = px, py, px + w, py + l
        # Check gaps to boundary
        left_gap = rx1 - 0.0
        right_gap = max_width - rx2
        bottom_gap = ry1 - 0.0
        top_gap = max_length - ry2
        # If a long side is parallel and the free space on that side is narrow, reject
        if left_gap > 0 and left_gap < min_gap and (ry2 - ry1) >= min_gap:
            return True
        if right_gap > 0 and right_gap < min_gap and (ry2 - ry1) >= min_gap:
            return True
        if bottom_gap > 0 and bottom_gap < min_gap and (rx2 - rx1) >= min_gap:
            return True
        if top_gap > 0 and top_gap < min_gap and (rx2 - rx1) >= min_gap:
            return True
        # Check gaps against placed rooms (parallel strips)
        for q in placed:
            qx1, qy1, qx2, qy2 = _room_bounds(q)
            # vertical strip between rx1..rx2 and a q to its left/right
            if not (ry2 <= qy1 or ry1 >= qy2):  # y-overlap exists
                # gap to left neighbor
                if qx2 <= rx1:
                    gap = rx1 - qx2
                    if 0 < gap < min_gap and (min(ry2, qy2) - max(ry1, qy1)) >= min_gap:
                        return True
                # gap to right neighbor
                if qx1 >= rx2:
                    gap = qx1 - rx2
                    if 0 < gap < min_gap and (min(ry2, qy2) - max(ry1, qy1)) >= min_gap:
                        return True
            # horizontal strip between ry1..ry2 and a q below/above
            if not (rx2 <= qx1 or rx1 >= qx2):  # x-overlap exists
                # gap to bottom neighbor
                if qy2 <= ry1:
                    gap = ry1 - qy2
                    if 0 < gap < min_gap and (min(rx2, qx2) - max(rx1, qx1)) >= min_gap:
                        return True
                # gap to top neighbor
                if qy1 >= ry2:
                    gap = qy1 - ry2
                    if 0 < gap < min_gap and (min(rx2, qx2) - max(rx1, qx1)) >= min_gap:
                        return True
        return False

    def _would_be_peninsula(px: float, py: float, w: float, l: float) -> bool:
        if not avoid_peninsulas:
            return False
        rx1, ry1, rx2, ry2 = px, py, px + w, py + l
        touches_boundary = rx1 <= 1e-6 or ry1 <= 1e-6 or abs(rx2 - max_width) < 1e-6 or abs(ry2 - max_length) < 1e-6
        if not touches_boundary:
            return False
        shared = _shared_wall_len(px, py, w, l)
        # If little interior contact (< 0.5 of the shorter side), treat as peninsula
        min_side = max(1e-6, min(w, l))
        return shared < 0.5 * min_side

    def fits_here(r: Dict, px: float, py: float) -> bool:
        w = float((r.get("size") or {}).get("width", 0))
        l = float((r.get("size") or {}).get("length", 0))
        px, py = snap(px), snap(py)
        if px < 0 or py < 0 or px + w > max_width or py + l > max_length:
            return False
        rx1, ry1, rx2, ry2 = px, py, px + w, py + l
        for q in placed:
            qx1, qy1, qx2, qy2 = _room_bounds(q)
            if rx1 < qx2 and rx2 > qx1 and ry1 < qy2 and ry2 > qy1:
                return False
        if _creates_narrow_gap(px, py, w, l):
            return False
        if _would_be_peninsula(px, py, w, l):
            return False
        return True

    def touches(r: Dict, px: float, py: float, only_with: Optional[List[str]] = None) -> bool:
        # Check if placement at (px, py) would share a wall with any placed room, optionally filtered by type list.
        w = float((r.get("size") or {}).get("width", 0))
        l = float((r.get("size") or {}).get("length", 0))
        rx1, ry1, rx2, ry2 = px, py, px + w, py + l
        for q in placed:
            if only_with is not None and (q.get("type", "").lower() not in [s.lower() for s in only_with]):
                continue
            qx1, qy1, qx2, qy2 = _room_bounds(q)
            vertical_touch = (abs(rx2 - qx1) < 1e-6 or abs(qx2 - rx1) < 1e-6) and (
                min(ry2, qy2) - max(ry1, qy1) > 0
            )
            horizontal_touch = (abs(ry2 - qy1) < 1e-6 or abs(qy2 - ry1) < 1e-6) and (
                min(rx2, qx2) - max(rx1, qx1) > 0
            )
            if vertical_touch or horizontal_touch:
                return True
        return False

    def candidate_positions_near(target: Dict, w: float, l: float) -> List[Tuple[float, float]]:
        # Generate positions that would touch the target room on each side, snapped to grid
        px, py = float(target.get("position", {}).get("x", 0)), float(target.get("position", {}).get("y", 0))
        tw = float(target.get("size", {}).get("width", 0))
        tl = float(target.get("size", {}).get("length", 0))
        return [
            (snap(px - w), snap(py)),              # left
            (snap(px + tw), snap(py)),             # right
            (snap(px), snap(py - l)),              # bottom
            (snap(px), snap(py + tl)),             # top
        ]

    # Lay out rooms on grid with preferences
    for _, _, r in indexed:
        rtype = (r.get("type") or "").strip()
        size = r.get("size") or {}
        w = float(size.get("width", 0))
        l = float(size.get("length", 0))
        placed_ok = False

        # Determine search bounds by zone
        x_min, y_min, x_max, y_max = 0.0, 0.0, max_width - w, max_length - l
        if zoning:
            zone = zones.get(classify(rtype), (0.0, 0.0, max_width, max_length))
            x_min, y_min, x_max, y_max = zone[0], zone[1], max(zone[2] - w, 0.0), max(zone[3] - l, 0.0)

        # 1) Try to attach to preferred neighbor types if any already placed
        pref = (adjacency_hints or {}).get(rtype, [])
        pref_lower = [p.lower() for p in pref]
        near_targets = []
        for p in pref_lower:
            for q in placed_by_type.get(p, []) or []:
                near_targets.append(q)
        # Evaluate all candidates with a score to prefer interior, well-connected placements
        best: Optional[Tuple[float, float, float]] = None  # score, x, y
        def score_pos(x: float, y: float) -> float:
            if not fits_here(r, x, y):
                return float("-inf")
            shared = _shared_wall_len(x, y, w, l)
            # mild penalty for touching boundary
            boundary_pen = 0.0
            if x <= 0 or y <= 0 or x + w >= max_width or y + l >= max_length:
                boundary_pen = 0.5 * min(w, l)
            return shared - boundary_pen
        for tgt in near_targets:
            for x, y in candidate_positions_near(tgt, w, l):
                if x < x_min or y < y_min or x > x_max or y > y_max:
                    continue
                sc = score_pos(x, y)
                if best is None or sc > best[0]:
                    best = (sc, x, y)
        if best is not None and best[0] != float("-inf"):
            x, y = best[1], best[2]
            r.setdefault("position", {})["x"] = snap(x)
            r["position"]["y"] = snap(y)
            placed.append(r)
            placed_by_type.setdefault(rtype.lower(), []).append(r)
            placed_ok = True
        if placed_ok:
            continue

        # 2) General scan within zone, but try to ensure connectivity
        y_vals = [snap(y_min + i * grid) for i in range(int(((y_max - y_min) // grid)) + 1)]
        x_vals = [snap(x_min + i * grid) for i in range(int(((x_max - x_min) // grid)) + 1)]
        # Prefer positions closer to the zone center to avoid leaving a void in the middle
        cx, cy = (x_min + x_max) * 0.5, (y_min + y_max) * 0.5
        grid_points = [(x, y) for y in y_vals for x in x_vals]
        grid_points.sort(key=lambda p: (p[0] - cx) ** 2 + (p[1] - cy) ** 2)
        for x, y in grid_points:
            if fits_here(r, x, y) and (not placed or touches(r, x, y, None)):
                r.setdefault("position", {})["x"] = snap(x)
                r["position"]["y"] = snap(y)
                placed.append(r)
                placed_by_type.setdefault(rtype.lower(), []).append(r)
                placed_ok = True
                break
        if placed_ok:
            continue

        # 3) Last resort anywhere non-overlapping (still center-out ordering)
        for x, y in grid_points:
            if fits_here(r, x, y):
                r.setdefault("position", {})["x"] = snap(x)
                r["position"]["y"] = snap(y)
                placed.append(r)
                placed_by_type.setdefault(rtype.lower(), []).append(r)
                placed_ok = True
                break
        if not placed_ok:
            # Clamp to origin
            r.setdefault("position", {})["x"] = 0.0
            r["position"]["y"] = 0.0
            placed.append(r)
            placed_by_type.setdefault(rtype.lower(), []).append(r)

    # Simple hallway synthesis for multiple bedrooms
    bed_rooms = placed_by_type.get("bedroom", []) or []
    if len(bed_rooms) >= 2:
        # Vertical hall along the left of the private zone
        xs = [float(b.get("position", {}).get("x", 0)) for b in bed_rooms]
        ys1 = [float(b.get("position", {}).get("y", 0)) for b in bed_rooms]
        ys2 = [float(b.get("position", {}).get("y", 0)) + float(b.get("size", {}).get("length", 0)) for b in bed_rooms]
        y1 = max(0.0, min(ys1))
        y2 = min(max_length, max(ys2))
        h_w = float(max(min_hall_width, grid))
        x_left = max(0.0, min(xs) - h_w - grid)
        # Find a non-overlapping x by scanning
        def hall_fits(px: float) -> bool:
            rx1, ry1, rx2, ry2 = px, y1, px + h_w, y2
            if rx2 > max_width:
                return False
            for q in placed:
                qx1, qy1, qx2, qy2 = _room_bounds(q)
                if rx1 < qx2 and rx2 > qx1 and ry1 < qy2 and ry2 > qy1:
                    return False
            return True
        x_try = [x_left, max(0.0, min(xs) - h_w), max(0.0, min(xs) + grid)]
        hx = None
        for cand in x_try:
            cand = snap(cand)
            if hall_fits(cand):
                hx = cand
                break
        if hx is not None:
            hallway = {
                "type": "Hallway",
                "position": {"x": hx, "y": snap(y1)},
                "size": {"width": int(round(h_w)), "length": int(round(y2 - y1))},
            }
            placed.append(hallway)
            placed_by_type.setdefault("hallway", []).append(hallway)

    layout.setdefault("layout", {})["rooms"] = placed
    return clamp_bounds(layout, max_width, max_length)


__all__ = [
    "check_bounds",
    "check_overlaps",
    "check_separation",
    "check_connectivity",
    "check_adjacency",
    "check_entrance",
    "validate_layout",
    "enforce_min_separation",
    "resolve_overlaps",
    "clamp_bounds",
    "pack_layout",
]
