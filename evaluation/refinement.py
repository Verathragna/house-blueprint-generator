import copy
import logging
import math
import random
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from evaluation.validators import clamp_bounds, resolve_overlaps


def _room_bounds(room: Dict) -> Tuple[float, float, float, float]:
    pos = room.get("position") or {}
    size = room.get("size") or {}
    x = float(pos.get("x", 0))
    y = float(pos.get("y", 0))
    w = float(size.get("width", 0))
    l = float(size.get("length", 0))
    return x, y, x + w, y + l


def _rect_overlap(bounds_a: Tuple[float, float, float, float], bounds_b: Tuple[float, float, float, float]) -> float:
    x1 = max(bounds_a[0], bounds_b[0])
    y1 = max(bounds_a[1], bounds_b[1])
    x2 = min(bounds_a[2], bounds_b[2])
    y2 = min(bounds_a[3], bounds_b[3])
    if x2 <= x1 or y2 <= y1:
        return 0.0
    return (x2 - x1) * (y2 - y1)


def _rect_distance(bounds_a: Tuple[float, float, float, float], bounds_b: Tuple[float, float, float, float]) -> float:
    dx = max(0.0, max(bounds_a[0], bounds_b[0]) - min(bounds_a[2], bounds_b[2]))
    dy = max(0.0, max(bounds_a[1], bounds_b[1]) - min(bounds_a[3], bounds_b[3]))
    if dx == 0.0 and dy == 0.0:
        return 0.0
    return math.hypot(dx, dy)


def _share_wall(bounds_a: Tuple[float, float, float, float], bounds_b: Tuple[float, float, float, float]) -> bool:
    overlaps_x = max(0.0, min(bounds_a[2], bounds_b[2]) - max(bounds_a[0], bounds_b[0]))
    overlaps_y = max(0.0, min(bounds_a[3], bounds_b[3]) - max(bounds_a[1], bounds_b[1]))
    touch_x = math.isclose(bounds_a[2], bounds_b[0]) or math.isclose(bounds_b[2], bounds_a[0])
    touch_y = math.isclose(bounds_a[3], bounds_b[1]) or math.isclose(bounds_b[3], bounds_a[1])
    return (touch_x and overlaps_y > 0) or (touch_y and overlaps_x > 0)


def _normalise_adjacency(adjacency: Optional[Dict[str, Sequence[str]]]) -> List[Tuple[str, str]]:
    if not adjacency:
        return []
    pairs = set()
    for room, targets in adjacency.items():
        room_key = room.strip().lower()
        if not room_key:
            continue
        for target in targets:
            target_key = (target or "").strip().lower()
            if not target_key or target_key == room_key:
                continue
            pairs.add(tuple(sorted((room_key, target_key))))
    return sorted(pairs)


def _build_type_index(rooms: Sequence[Dict]) -> Dict[str, List[Dict]]:
    index: Dict[str, List[Dict]] = {}
    for room in rooms:
        key = (room.get("type") or "").strip().lower()
        if not key:
            continue
        index.setdefault(key, []).append(room)
    return index


def _layout_cost(
    rooms: Sequence[Dict],
    max_width: float,
    max_length: float,
    adjacency_pairs: Sequence[Tuple[str, str]],
    min_separation: float,
) -> float:
    overlap_penalty = 1000.0
    bounds_penalty = 200.0
    separation_penalty = 200.0
    adjacency_penalty = 50.0

    total_cost = 0.0
    bounds = [_room_bounds(room) for room in rooms]

    # Overlap and separation penalties
    for i in range(len(rooms)):
        rect_a = bounds[i]
        # bounds checks
        over_left = max(0.0, -rect_a[0])
        over_bottom = max(0.0, -rect_a[1])
        over_right = max(0.0, rect_a[2] - max_width)
        over_top = max(0.0, rect_a[3] - max_length)
        total_cost += bounds_penalty * (over_left + over_bottom + over_right + over_top)

        for j in range(i + 1, len(rooms)):
            rect_b = bounds[j]
            overlap_area = _rect_overlap(rect_a, rect_b)
            if overlap_area > 0:
                total_cost += overlap_penalty * overlap_area
            else:
                dist = _rect_distance(rect_a, rect_b)
                if min_separation > 0 and dist < min_separation:
                    total_cost += separation_penalty * (min_separation - dist)

    # Adjacency penalties
    if adjacency_pairs:
        index = _build_type_index(rooms)
        for room_a, room_b in adjacency_pairs:
            rooms_a = index.get(room_a)
            rooms_b = index.get(room_b)
            if not rooms_a or not rooms_b:
                total_cost += adjacency_penalty * 20.0
                continue
            best = float("inf")
            for ra in rooms_a:
                ba = _room_bounds(ra)
                for rb in rooms_b:
                    bb = _room_bounds(rb)
                    if _share_wall(ba, bb):
                        best = 0.0
                        break
                    dist = _rect_distance(ba, bb)
                    if dist < best:
                        best = dist
                if best == 0.0:
                    break
            total_cost += adjacency_penalty * best

    return total_cost


def refine_layout(
    layout: Dict,
    max_width: float,
    max_length: float,
    min_separation: float = 0.0,
    adjacency: Optional[Dict[str, Sequence[str]]] = None,
    iterations: int = 200,
    temperature: float = 5.0,
    seed: Optional[int] = None,
) -> Dict:
    """Refine a layout by perturbing room positions using simulated annealing."""
    logger = logging.getLogger(__name__)
    if iterations <= 0:
        return layout

    rng = random.Random(seed)
    working = copy.deepcopy(layout)
    rooms = (working.get("layout") or {}).get("rooms", [])
    if not rooms:
        return layout

    working = clamp_bounds(working, max_width=max_width, max_length=max_length)
    working = resolve_overlaps(
        working,
        adjacency=adjacency,
        max_width=max_width,
        max_length=max_length,
    )

    rooms = (working.get("layout") or {}).get("rooms", [])
    if not rooms:
        return layout

    adjacency_pairs = _normalise_adjacency(adjacency)
    current_cost = _layout_cost(rooms, max_width, max_length, adjacency_pairs, min_separation)
    best_layout = copy.deepcopy(working)
    best_cost = current_cost

    total_extent = max(max_width, max_length)

    for step in range(iterations):
        temp_scale = temperature * max(0.05, 1.0 - step / max(1, iterations))
        move_radius = max(1.0, total_extent * 0.05 * temp_scale)
        if iterations >= 20 and step % max(1, iterations // 5) == 0 and step > 0:
            logger.info(
                "Refinement progress: step %s/%s (cost %.3f, best %.3f)",
                step,
                iterations,
                current_cost,
                best_cost,
            )

        idx = rng.randrange(len(rooms))
        room = rooms[idx]
        bounds_before = _room_bounds(room)
        dx = rng.uniform(-move_radius, move_radius)
        dy = rng.uniform(-move_radius, move_radius)

        pos = room.setdefault("position", {})
        size = room.get("size") or {}
        width = float(size.get("width", 0))
        length = float(size.get("length", 0))
        new_x = min(max(0.0, bounds_before[0] + dx), max_width - width)
        new_y = min(max(0.0, bounds_before[1] + dy), max_length - length)
        pos["x"] = round(new_x)
        pos["y"] = round(new_y)

        new_cost = _layout_cost(rooms, max_width, max_length, adjacency_pairs, min_separation)
        accept = False
        if new_cost <= current_cost:
            accept = True
        else:
            temp = max(0.01, temp_scale)
            prob = math.exp((current_cost - new_cost) / temp)
            if rng.random() < prob:
                accept = True

        if accept:
            current_cost = new_cost
            if current_cost < best_cost:
                best_cost = current_cost
                best_layout = copy.deepcopy(working)
        else:
            # revert
            pos["x"] = round(bounds_before[0])
            pos["y"] = round(bounds_before[1])

    best_layout = clamp_bounds(best_layout, max_width=max_width, max_length=max_length)
    best_layout = resolve_overlaps(
        best_layout,
        adjacency=adjacency,
        max_width=max_width,
        max_length=max_length,
    )
    return best_layout


__all__ = ["refine_layout"]
