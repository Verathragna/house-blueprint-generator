from __future__ import annotations

import copy
import math
import random
from typing import Dict, List, Optional, Tuple

from evaluation.scoring import compute_metrics, aggregate_cost
from evaluation.validators import check_overlaps, check_bounds


class _Node:
    def __init__(self, layout: Dict, parent: Optional["_Node"] = None):
        self.layout = layout
        self.parent = parent
        self.children: List[_Node] = []
        self.visits = 0
        self.value = 0.0  # we minimize cost; store negative cost for UCB maximizing

    def ucb(self, c: float = 1.4) -> float:
        if self.visits == 0:
            return float("inf")
        assert self.parent is not None
        return (self.value / self.visits) + c * math.sqrt(math.log(self.parent.visits + 1) / self.visits)


def _evaluate(layout: Dict, *, max_width: float, max_length: float) -> float:
    # Hard failures get large penalty
    rooms = (layout.get("layout") or {}).get("rooms", [])
    hard = 0.0
    hard += 10000.0 * len(check_overlaps(rooms))
    hard += 1000.0 * len(check_bounds(rooms, max_width=max_width, max_length=max_length))
    metrics = compute_metrics(layout, max_width=max_width, max_length=max_length)
    cost = aggregate_cost(metrics)
    return hard + cost


def _neighbors(layout: Dict, *, max_width: float, max_length: float, k: int = 6, radius: float = 2.0) -> List[Dict]:
    rng = random.Random()
    rooms = (layout.get("layout") or {}).get("rooms", [])
    outs: List[Dict] = []
    for _ in range(k):
        cand = copy.deepcopy(layout)
        rlist = (cand.get("layout") or {}).get("rooms", [])
        if not rlist:
            outs.append(cand)
            continue
        idx = rng.randrange(len(rlist))
        pos = rlist[idx].setdefault("position", {})
        size = rlist[idx].get("size") or {}
        w = float(size.get("width", 0))
        l = float(size.get("length", 0))
        dx = rng.uniform(-radius, radius)
        dy = rng.uniform(-radius, radius)
        x = max(0.0, min(float(pos.get("x", 0)) + dx, max_width - w))
        y = max(0.0, min(float(pos.get("y", 0)) + dy, max_length - l))
        pos["x"], pos["y"] = x, y
        outs.append(cand)
    return outs


def optimize_layout_mcts(
    layout: Dict,
    *,
    max_width: float,
    max_length: float,
    iterations: int = 200,
    exploration: float = 1.4,
    seed: Optional[int] = None,
) -> Dict:
    """Lightweight MCTS over local moves to minimize multi-objective cost.

    Returns the best layout encountered. This is a generic optimizer and not wired by default.
    """
    rng = random.Random(seed)
    root = _Node(copy.deepcopy(layout))
    best_layout = copy.deepcopy(layout)
    best_cost = _evaluate(best_layout, max_width=max_width, max_length=max_length)

    for _ in range(max(1, iterations)):
        # Selection
        node = root
        path = [node]
        while node.children:
            node = max(node.children, key=lambda ch: ch.ucb(c=exploration))
            path.append(node)

        # Expansion
        if node.visits > 0:
            for cand in _neighbors(node.layout, max_width=max_width, max_length=max_length):
                node.children.append(_Node(cand, parent=node))
            if node.children:
                node = rng.choice(node.children)
                path.append(node)

        # Simulation
        layout_eval = node.layout
        cost = _evaluate(layout_eval, max_width=max_width, max_length=max_length)
        if cost < best_cost:
            best_cost = cost
            best_layout = copy.deepcopy(layout_eval)
        reward = -cost  # maximize reward in UCB

        # Backprop
        for n in path:
            n.visits += 1
            n.value += reward

    return best_layout
