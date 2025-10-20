from __future__ import annotations

from typing import Dict, List, Tuple, Optional

# Import guarded so the repo still works without OR-Tools installed
try:
    from ortools.sat.python import cp_model
except Exception:  # pragma: no cover - handled by caller
    cp_model = None  # type: ignore


def _default_min_sizes() -> Dict[str, Tuple[int, int]]:
    return {
        "living room": (14, 14),
        "kitchen": (10, 12),
        "bedroom": (10, 10),
        "bathroom": (6, 8),
        "dining room": (10, 10),
        "garage": (20, 20),
        "laundry room": (6, 8),
        "office": (10, 10),
        "closet": (4, 6),
        "hallway": (3, 8),
    }


def build_intent_from_params(raw_params: Dict) -> List[Dict]:
    """Build a room intent list from the raw params JSON.

    Each item: {"type": str, "min_w": int, "min_l": int, "max_w": int, "max_l": int}
    """
    mins = _default_min_sizes()

    def add_many(lst: List[Dict], room_type: str, count: int) -> None:
        if count <= 0:
            return
        wmin, lmin = mins.get(room_type.lower(), (8, 8))
        # Cap max to a reasonable scale (rough heuristic)
        wmax = max(wmin, min(2 * wmin, 32))
        lmax = max(lmin, min(2 * lmin, 32))
        for _ in range(int(count)):
            lst.append({
                "type": room_type,
                "min_w": int(wmin),
                "min_l": int(lmin),
                "max_w": int(wmax),
                "max_l": int(lmax),
            })

    rooms: List[Dict] = []
    # Required/core rooms
    add_many(rooms, "Living Room", int(raw_params.get("livingRooms", 1)))
    add_many(rooms, "Kitchen", int(raw_params.get("kitchen", 1)))
    add_many(rooms, "Dining Room", int(raw_params.get("diningRooms", 1)))
    add_many(rooms, "Laundry Room", int(raw_params.get("laundryRooms", 0)))

    # Bedrooms and bathrooms
    add_many(rooms, "Bedroom", int(raw_params.get("bedrooms", 0)))
    baths = raw_params.get("bathrooms", {}) or {}
    add_many(rooms, "Bathroom", int(baths.get("full", 0)) + int(baths.get("half", 0)))

    # Optional rooms
    if raw_params.get("garage"):
        add_many(rooms, "Garage", 1)

    # If nothing specified, fall back to a small starter set
    if not rooms:
        add_many(rooms, "Living Room", 1)
        add_many(rooms, "Kitchen", 1)
        add_many(rooms, "Bedroom", 1)
        add_many(rooms, "Bathroom", 1)

    return rooms


def solve_layout_cpsat(
    intent: List[Dict],
    *,
    max_width: int,
    max_length: int,
    min_separation: int = 0,
    time_limit_s: float = 5.0,
    prefer_boundary: Optional[Dict[str, float]] = None,
) -> Optional[Dict]:
    """Solve a non-overlapping placement using CP-SAT.

    Returns a layout dict compatible with validators, or None if infeasible.
    """
    if cp_model is None:
        return None

    model = cp_model.CpModel()

    n = len(intent)
    # Decision vars
    xs: List = []
    ys: List = []
    ws: List = []
    ls: List = []

    # Bounds and domains (integer feet)
    for i, r in enumerate(intent):
        min_w = int(max(1, r.get("min_w", 6)))
        min_l = int(max(1, r.get("min_l", 6)))
        max_wi = int(max(min_w, min(r.get("max_w", min_w), max_width)))
        max_li = int(max(min_l, min(r.get("max_l", min_l), max_length)))
        w = model.NewIntVar(min_w, max_wi, f"w_{i}")
        l = model.NewIntVar(min_l, max_li, f"l_{i}")
        x = model.NewIntVar(0, max(0, max_width - min_w), f"x_{i}")
        y = model.NewIntVar(0, max(0, max_length - min_l), f"y_{i}")
        # Within bounds
        model.Add(x + w <= max_width)
        model.Add(y + l <= max_length)
        xs.append(x)
        ys.append(y)
        ws.append(w)
        ls.append(l)

    sep = int(max(0, round(min_separation)))

    # Non-overlap with separation
    for i in range(n):
        for j in range(i + 1, n):
            left = model.NewBoolVar(f"left_{i}_{j}")
            right = model.NewBoolVar(f"right_{i}_{j}")
            below = model.NewBoolVar(f"below_{i}_{j}")
            above = model.NewBoolVar(f"above_{i}_{j}")
            model.Add(xs[i] + ws[i] + sep <= xs[j]).OnlyEnforceIf(left)
            model.Add(xs[j] + ws[j] + sep <= xs[i]).OnlyEnforceIf(right)
            model.Add(ys[i] + ls[i] + sep <= ys[j]).OnlyEnforceIf(below)
            model.Add(ys[j] + ls[j] + sep <= ys[i]).OnlyEnforceIf(above)
            model.AddBoolOr([left, right, below, above])

    # Objective: maximize total room area + mild preference for boundary touch for some types
    # OR-Tools does not support multiplying two decision vars directly in a linear expression.
    # Introduce auxiliary area vars with multiplication equality constraints.
    area_terms = []
    max_area = int(max_width) * int(max_length)
    for i in range(n):
        area_i = model.NewIntVar(0, max_area, f"area_{i}")
        model.AddMultiplicationEquality(area_i, [ws[i], ls[i]])
        area_terms.append(area_i)

    # Encourage certain rooms to touch a boundary by minimizing distance to boundary
    # We linearize by adding negative weights for min distance to any side via helper vars
    boundary_weight: Dict[str, float] = prefer_boundary or {"living room": 2.0, "garage": 3.0}
    dist_terms = []
    for i, r in enumerate(intent):
        rtype = (r.get("type") or "").lower()
        wt = float(boundary_weight.get(rtype, 0.0))
        if wt <= 0:
            continue
        # Distances to each side: x, y, max_width - (x + w), max_length - (y + l)
        dx_left = model.NewIntVar(0, max_width, f"dxL_{i}")
        dy_bot = model.NewIntVar(0, max_length, f"dyB_{i}")
        dx_right = model.NewIntVar(0, max_width, f"dxR_{i}")
        dy_top = model.NewIntVar(0, max_length, f"dyT_{i}")
        model.Add(dx_left == xs[i])
        model.Add(dy_bot == ys[i])
        model.Add(dx_right == max_width - (xs[i] + ws[i]))
        model.Add(dy_top == max_length - (ys[i] + ls[i]))
        # min distance via auxiliary var m = min(dx_left, dy_bot, dx_right, dy_top)
        m = model.NewIntVar(0, max(max_width, max_length), f"mindist_{i}")
        # m <= each distance
        model.Add(m <= dx_left)
        model.Add(m <= dy_bot)
        model.Add(m <= dx_right)
        model.Add(m <= dy_top)
        # We cannot enforce m == min(...), but pushing m downwards in objective will tend to 0
        # Add with negative weight to prefer boundary contact
        # Scale by weight (convert to int by multiplying by 10)
        dist_terms.append((int(round(wt * 10)), m))

    # Build linear objective: maximize sum(area_terms) - sum(weight * mindist)
    # CP-SAT minimizes by default when using Add, but with Maximize we can directly maximize linear expression.
    lin = sum(area_terms)
    for w10, m in dist_terms:
        lin -= w10 * m
    model.Maximize(lin)

    if time_limit_s and time_limit_s > 0:
        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = float(time_limit_s)
        solver.parameters.num_search_workers = 8
    else:
        solver = cp_model.CpSolver()

    status = solver.Solve(model)
    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return None

    rooms: List[Dict] = []
    for i, r in enumerate(intent):
        rooms.append({
            "type": r.get("type", "Room"),
            "position": {
                "x": float(solver.Value(xs[i])),
                "y": float(solver.Value(ys[i])),
            },
            "size": {
                "width": int(solver.Value(ws[i])),
                "length": int(solver.Value(ls[i])),
            },
        })

    return {"layout": {"rooms": rooms, "dimensions": {"width": max_width, "depth": max_length}}}