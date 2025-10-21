from __future__ import annotations

import math
from typing import Dict, List, Tuple, Optional

# Utilities

def _room_bounds(room: Dict) -> Tuple[float, float, float, float]:
    pos = room.get("position") or {}
    size = room.get("size") or {}
    x = float(pos.get("x", 0))
    y = float(pos.get("y", 0))
    w = float(size.get("width", 0))
    l = float(size.get("length", 0))
    return x, y, x + w, y + l


def _center(bounds: Tuple[float, float, float, float]) -> Tuple[float, float]:
    x1, y1, x2, y2 = bounds
    return (x1 + x2) * 0.5, (y1 + y2) * 0.5


def _shares_wall(a: Tuple[float, float, float, float], b: Tuple[float, float, float, float], tol: float = 1e-6) -> bool:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    vertical_touch = (abs(ax2 - bx1) < tol or abs(bx2 - ax1) < tol) and (min(ay2, by2) - max(ay1, by1) > 0)
    horizontal_touch = (abs(ay2 - by1) < tol or abs(by2 - ay1) < tol) and (min(ax2, bx2) - max(ax1, bx1) > 0)
    return vertical_touch or horizontal_touch


def _shared_wall_length(a: Tuple[float, float, float, float], b: Tuple[float, float, float, float], tol: float = 1e-6) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    vertical = 0.0
    horizontal = 0.0
    if abs(ax2 - bx1) < tol or abs(bx2 - ax1) < tol:
        vertical = max(0.0, min(ay2, by2) - max(ay1, by1))
    if abs(ay2 - by1) < tol or abs(by2 - ay1) < tol:
        horizontal = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    return max(vertical, horizontal)


def _boundary_contact_span(bounds: Tuple[float, float, float, float], *, max_width: float, max_length: float, tol: float = 1e-6) -> float:
    x1, y1, x2, y2 = bounds
    spans: List[float] = []
    if abs(x1 - 0.0) < tol:
        spans.append(y2 - y1)
    if abs(y1 - 0.0) < tol:
        spans.append(x2 - x1)
    if abs(x2 - max_width) < tol:
        spans.append(y2 - y1)
    if abs(y2 - max_length) < tol:
        spans.append(x2 - x1)
    return max(spans) if spans else 0.0


def _build_graph(rooms: List[Dict]) -> Tuple[List[Tuple[float,float,float,float]], List[List[int]]]:
    bounds = [_room_bounds(r) for r in rooms]
    n = len(bounds)
    adj: List[List[int]] = [[] for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            if _shares_wall(bounds[i], bounds[j]):
                adj[i].append(j)
                adj[j].append(i)
    return bounds, adj


def _apsp_unweighted(adj: List[List[int]]) -> List[List[float]]:
    n = len(adj)
    dist = [[math.inf] * n for _ in range(n)]
    for s in range(n):
        queue = [s]
        dist[s][s] = 0.0
        head = 0
        while head < len(queue):
            v = queue[head]
            head += 1
            for w in adj[v]:
                if dist[s][w] == math.inf:
                    dist[s][w] = dist[s][v] + 1.0
                    queue.append(w)
    return dist


def _closeness_centrality(adj: List[List[int]]) -> List[float]:
    n = len(adj)
    if n == 0:
        return []
    dist = _apsp_unweighted(adj)
    centrality: List[float] = []
    for i in range(n):
        s = sum(d for d in dist[i] if d > 0 and d < math.inf)
        if s == 0:
            centrality.append(0.0)
        else:
            centrality.append((n - 1) / s)
    return centrality


def _betweenness_centrality(adj: List[List[int]]) -> List[float]:
    # Brandes algorithm (unweighted)
    n = len(adj)
    Cb = [0.0] * n
    for s in range(n):
        stack: List[int] = []
        pred: List[List[int]] = [[] for _ in range(n)]
        sigma = [0.0] * n
        dist = [-1] * n
        sigma[s] = 1.0
        dist[s] = 0
        queue: List[int] = [s]
        head = 0
        while head < len(queue):
            v = queue[head]
            head += 1
            stack.append(v)
            for w in adj[v]:
                if dist[w] < 0:
                    dist[w] = dist[v] + 1
                    queue.append(w)
                if dist[w] == dist[v] + 1:
                    sigma[w] += sigma[v]
                    pred[w].append(v)
        delta = [0.0] * n
        while stack:
            w = stack.pop()
            for v in pred[w]:
                if sigma[w] > 0:
                    delta_v = (sigma[v] / sigma[w]) * (1.0 + delta[w])
                    delta[v] += delta_v
            if w != s:
                Cb[w] += delta[w]
    # normalize (undirected)
    if n > 2:
        scale = 1.0 / ((n - 1) * (n - 2) / 2.0)
        Cb = [c * scale for c in Cb]
    return Cb


def compute_metrics(
    layout: Dict,
    *,
    max_width: float,
    max_length: float,
    adjacency: Optional[Dict[str, List[str]]] = None,
    target_fill: float = 0.6,
) -> Dict[str, float]:
    rooms = (layout.get("layout") or {}).get("rooms", [])
    bounds, adj = _build_graph(rooms)
    n = len(bounds)

    # Areas and perimeters
    lot_area = max(1e-6, float(max_width) * float(max_length))
    areas = [(b[2] - b[0]) * (b[3] - b[1]) for b in bounds]
    perims = [2.0 * ((b[2] - b[0]) + (b[3] - b[1])) for b in bounds]
    total_area = sum(areas)
    area_util = total_area / lot_area

    # Daylight proxy: boundary contact for public rooms
    public_types = {"living room", "dining room", "kitchen", "hallway", "entry", "foyer"}
    daylight_span = 0.0
    for i, r in enumerate(rooms):
        t = (r.get("type") or "").strip().lower()
        if t in public_types:
            daylight_span += _boundary_contact_span(bounds[i], max_width=max_width, max_length=max_length)

    # Wet-stack alignment: sum of distances among K, Bath, Laundry centers
    kbl_idx: List[int] = []
    for i, r in enumerate(rooms):
        t = (r.get("type") or "").lower()
        if any(key in t for key in ("kitchen", "bathroom", "laundry")):
            kbl_idx.append(i)
    wet_sum = 0.0
    for i in range(len(kbl_idx)):
        ci = _center(bounds[kbl_idx[i]])
        for j in range(i + 1, len(kbl_idx)):
            cj = _center(bounds[kbl_idx[j]])
            wet_sum += math.hypot(ci[0] - cj[0], ci[1] - cj[1])

    # Wall cost: sum perimeters minus shared wall lengths (encourages shared walls)
    shared = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            shared += _shared_wall_length(bounds[i], bounds[j])
    wall_cost = max(0.0, sum(perims) - 2.0 * shared)  # shared counted twice across two rooms

    # Space syntax metrics
    closeness = _closeness_centrality(adj)
    betweenness = _betweenness_centrality(adj)
    integration = sum(closeness) / max(1, len(closeness)) if closeness else 0.0
    choice = sum(betweenness) / max(1, len(betweenness)) if betweenness else 0.0

    # Circulation: average shortest path length over connected pairs (penalize disconnected as +5)
    dist = _apsp_unweighted(adj)
    total_d = 0.0
    pairs = 0
    for i in range(n):
        for j in range(i + 1, n):
            if dist[i][j] == math.inf:
                total_d += 5.0
            else:
                total_d += dist[i][j]
            pairs += 1
    circulation = (total_d / pairs) if pairs > 0 else 0.0

    return {
        "area_util": area_util,
        "area_fit": abs(target_fill - area_util),
        "daylight_span": daylight_span,
        "wet_stack": wet_sum,
        "wall_cost": wall_cost,
        "integration": integration,
        "choice": choice,
        "circulation": circulation,
    }


def aggregate_cost(metrics: Dict[str, float], *, weights: Optional[Dict[str, float]] = None) -> float:
    """Combine metrics into a scalar cost (lower is better). Defaults encourage spacious, connected, daylighted plans.

    Positive weights act as penalties; negative weights are rewards.
    """
    w = {
        # Fit target area utilization
        "area_fit": 80.0,
        # Reward daylight span on public perimeter
        "daylight_span": -2.0,
        # Encourage compact wet stacks
        "wet_stack": 0.6,
        # Reduce total wall exposure cost
        "wall_cost": 0.15,
        # Reward legibility/centrality
        "integration": -40.0,
        "choice": -20.0,
        # Penalize long/topologically distant circulation
        "circulation": 20.0,
    }
    if weights:
        w.update(weights)
    cost = 0.0
    for k, v in metrics.items():
        if k in w:
            cost += w[k] * v
    return cost
