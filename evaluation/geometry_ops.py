from __future__ import annotations

from typing import Dict
from geometry.kernel import layout_to_rects, rects_to_layout


def snap_layout(layout: Dict, *, grid: float = 0.5, max_width: float | None = None, max_length: float | None = None) -> Dict:
    """Snap room positions and sizes to the specified grid and clamp to bounds if provided."""
    rects = layout_to_rects(layout)
    for r in rects:
        r.snap(grid)
        if max_width is not None and max_length is not None:
            r.move_to_fit(max_width, max_length)
    return rects_to_layout(rects, layout)
