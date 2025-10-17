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


def pack_layout(
    layout: Dict,
    max_width: float = 40,
    max_length: float = 40,
    grid: float = 1.0,
) -> Dict:
    """Re-pack rooms onto a grid to eliminate overlaps and ensure connectivity.

    This discards existing positions and places rooms one-by-one so that they
    (a) remain within bounds, (b) do not overlap previously placed rooms,
    and (c) share a wall with at least one placed room whenever possible.
    """
    rooms = (layout.get("layout") or {}).get("rooms", [])
    if not rooms:
        return layout

    # Sort by area (largest first) for better packing
    indexed = []
    for idx, r in enumerate(rooms):
        x1, y1, x2, y2 = _room_bounds(r)
        area = (x2 - x1) * (y2 - y1)
        indexed.append((area, idx, r))
    indexed.sort(reverse=True)

    placed: List[Dict] = []

    def fits_here(r: Dict, px: float, py: float) -> bool:
        w = float((r.get("size") or {}).get("width", 0))
        l = float((r.get("size") or {}).get("length", 0))
        if px < 0 or py < 0 or px + w > max_width or py + l > max_length:
            return False
        rx1, ry1, rx2, ry2 = px, py, px + w, py + l
        for q in placed:
            qx1, qy1, qx2, qy2 = _room_bounds(q)
            if rx1 < qx2 and rx2 > qx1 and ry1 < qy2 and ry2 > qy1:
                return False
        return True

    def touches_any(r: Dict, px: float, py: float) -> bool:
        # Check if placement at (px, py) would share a wall with any placed room
        w = float((r.get("size") or {}).get("width", 0))
        l = float((r.get("size") or {}).get("length", 0))
        rx1, ry1, rx2, ry2 = px, py, px + w, py + l
        for q in placed:
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

    # Lay out rooms on grid
    for _, _, r in indexed:
        size = r.get("size") or {}
        w = float(size.get("width", 0))
        l = float(size.get("length", 0))
        placed_ok = False
        # First try positions that touch an existing room (connectivity)
        for y in [i * grid for i in range(int((max_length - l) // grid) + 1)]:
            if placed_ok:
                break
            for x in [i * grid for i in range(int((max_width - w) // grid) + 1)]:
                if fits_here(r, x, y) and (not placed or touches_any(r, x, y)):
                    r.setdefault("position", {})["x"] = x
                    r["position"]["y"] = y
                    placed.append(r)
                    placed_ok = True
                    break
        if placed_ok:
            continue
        # Fallback: place anywhere non-overlapping
        for y in [i * grid for i in range(int((max_length - l) // grid) + 1)]:
            if placed_ok:
                break
            for x in [i * grid for i in range(int((max_width - w) // grid) + 1)]:
                if fits_here(r, x, y):
                    r.setdefault("position", {})["x"] = x
                    r["position"]["y"] = y
                    placed.append(r)
                    placed_ok = True
                    break
        if not placed_ok:
            # As a last resort, clamp to origin
            r.setdefault("position", {})["x"] = 0.0
            r["position"]["y"] = 0.0
            placed.append(r)

    layout.setdefault("layout", {})["rooms"] = placed
    return clamp_bounds(layout, max_width, max_length)


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
    "pack_layout",
]
