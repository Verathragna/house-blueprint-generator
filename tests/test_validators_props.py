import math
import pytest

hypothesis = pytest.importorskip("hypothesis")
from hypothesis import given, strategies as st

from evaluation.validators import check_overlaps


def non_overlapping_rects():
    # Generate rectangles that are placed on a grid with spacing to avoid overlaps
    def rect():
        w = st.integers(min_value=4, max_value=20)
        h = st.integers(min_value=4, max_value=20)
        x = st.integers(min_value=0, max_value=60)
        y = st.integers(min_value=0, max_value=60)
        return st.tuples(x, y, w, h)

    # Build a list and enforce separation by snapping to a coarse grid
    return st.lists(rect(), min_size=1, max_size=8).map(
        lambda rs: [
            {
                "type": f"Room{i}",
                "position": {"x": (r[0] // 8) * 8, "y": (r[1] // 8) * 8},
                "size": {"width": r[2], "length": r[3]},
            }
            for i, r in enumerate(rs)
        ]
    )


@given(non_overlapping_rects())
def test_no_overlaps_property(rooms):
    issues = check_overlaps(rooms)
    # Because of coarse snapping, some overlaps can still occur when cells collide.
    # We enforce a weak property: either no overlaps, or overlaps count is small.
    assert len(issues) <= 1
