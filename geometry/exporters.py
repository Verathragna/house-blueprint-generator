from __future__ import annotations

from typing import Dict, Tuple


def export_svg_clean(layout: Dict, svg_path: str, *, lot_dims: Tuple[float, float] | None = None, grid: float = 0.5) -> None:
    """Export a simple, orthogonally-snapped SVG from a layout.

    This does not require external deps and is meant for post-process cleanup.
    """
    max_w, max_h = (40.0, 40.0)
    if lot_dims is not None:
        max_w, max_h = lot_dims
    rooms = (layout.get("layout") or {}).get("rooms", [])

    def snap(v: float) -> float:
        g = max(1e-6, float(grid))
        return g * round(float(v) / g)

    # SVG header
    lines = [
        f"<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 {max_w} {max_h}' width='{max_w*10}' height='{max_h*10}'>",
        "  <g fill='none' stroke='black' stroke-width='0.1'>",
        f"    <rect x='0' y='0' width='{max_w}' height='{max_h}' stroke='gray' stroke-dasharray='1,1'/>",
    ]
    # Rooms
    for r in rooms:
        pos = r.get("position") or {}
        size = r.get("size") or {}
        x = snap(pos.get("x", 0))
        y = snap(pos.get("y", 0))
        w = snap(size.get("width", 0))
        h = snap(size.get("length", 0))
        lines.append(f"    <rect x='{x}' y='{y}' width='{w}' height='{h}'/>")
        # Label
        t = (r.get("type") or "").replace("&", "&amp;")
        cx = x + w / 2
        cy = y + h / 2
        lines.append(f"    <text x='{cx}' y='{cy}' font-size='1.5' text-anchor='middle' dominant-baseline='middle' fill='black'>{t}</text>")
    lines.append("  </g>")
    lines.append("</svg>")

    with open(svg_path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))
