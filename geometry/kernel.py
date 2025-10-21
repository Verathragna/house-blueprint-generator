from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple, Dict, List, Optional


@dataclass
class Rect:
    x: float
    y: float
    w: float
    h: float

    @property
    def bounds(self) -> Tuple[float, float, float, float]:
        return (self.x, self.y, self.x + self.w, self.y + self.h)

    def move_to_fit(self, max_w: float, max_h: float) -> None:
        self.x = max(0.0, min(self.x, max_w - self.w))
        self.y = max(0.0, min(self.y, max_h - self.h))

    def snap(self, grid: float) -> None:
        g = max(1e-6, float(grid))
        self.x = g * round(self.x / g)
        self.y = g * round(self.y / g)
        self.w = g * round(self.w / g)
        self.h = g * round(self.h / g)

    def area(self) -> float:
        return max(0.0, self.w) * max(0.0, self.h)


@dataclass
class DoorArc:
    # Minimal parametric door (for future use)
    # Located on a wall aligned to axis; swing as radius
    cx: float
    cy: float
    radius: float
    start_angle_deg: float
    end_angle_deg: float


def layout_to_rects(layout: Dict) -> List[Rect]:
    rooms = (layout.get("layout") or {}).get("rooms", [])
    out: List[Rect] = []
    for r in rooms:
        pos = r.get("position") or {}
        size = r.get("size") or {}
        out.append(Rect(float(pos.get("x", 0)), float(pos.get("y", 0)), float(size.get("width", 0)), float(size.get("length", 0))))
    return out


def rects_to_layout(rects: List[Rect], layout: Dict) -> Dict:
    rooms = (layout.get("layout") or {}).get("rooms", [])
    for rr, r in zip(rects, rooms):
        r.setdefault("position", {})["x"] = rr.x
        r["position"]["y"] = rr.y
        r.setdefault("size", {})["width"] = rr.w
        r["size"]["length"] = rr.h
    return layout
