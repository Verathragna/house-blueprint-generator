"""Evaluate a generated layout against input parameters.

This script renders an SVG for visual inspection, validates the geometry
of the layout, and compares simple statistics against the requested
parameters.
"""

import argparse
import json
import logging
import os
import sys
from collections import Counter
from pathlib import Path


# Ensure repository root is on ``sys.path`` when running as a script
current_dir = os.path.dirname(os.path.abspath(__file__))
repo_root = os.path.abspath(os.path.join(current_dir, ".."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from dataset.render_svg import render_layout_svg
from evaluation.validators import validate_layout, check_bounds


class BoundaryViolationError(RuntimeError):
    """Raised when a room falls outside the allowed layout bounds."""


log = logging.getLogger(__name__)


def assert_room_counts(layout: dict, params: dict) -> list[dict]:
    """Return a structured list describing missing or mismatched room counts.

    Each item in the returned list is a dictionary with keys ``room_type``,
    ``expected`` and ``found``. An empty list indicates that all requested room
    counts were satisfied.
    """
    rooms = (layout.get("layout") or {}).get("rooms", [])
    counts = Counter(r.get("type", "").lower() for r in rooms)

    missing: list[dict] = []

    bed_expected = int(params.get("bedrooms", 0))
    bed_found = counts.get("bedroom", 0)
    if bed_found < bed_expected:
        missing.append({"room_type": "bedroom", "expected": bed_expected, "found": bed_found})

    baths = params.get("bathrooms") or {}
    bath_expected = int(baths.get("full", 0)) + int(baths.get("half", 0))
    bath_found = counts.get("bathroom", 0)
    if bath_found < bath_expected:
        missing.append({"room_type": "bathroom", "expected": bath_expected, "found": bath_found})

    if params.get("garage"):
        gar_found = counts.get("garage", 0)
        if gar_found < 1:
            missing.append({"room_type": "garage", "expected": 1, "found": gar_found})

    return missing


def compare_with_params(layout: dict, params: dict) -> list[str]:
    """Compare layout statistics with desired parameters.

    Converts the structured output of :func:`assert_room_counts` into
    human-readable messages and checks optional room size expectations.
    """
    rooms = (layout.get("layout") or {}).get("rooms", [])
    issues: list[str] = []

    for item in assert_room_counts(layout, params):
        room = item["room_type"]
        issues.append(f"Expected {item['expected']} {room}s, found {item['found']}")

    size_expect = params.get("roomSizes") or {}
    for room in rooms:
        exp = size_expect.get(room.get("type"))
        if not exp:
            continue
        w = room.get("size", {}).get("width")
        l = room.get("size", {}).get("length")
        if ("width" in exp and w != exp["width"]) or ("length" in exp and l != exp["length"]):
            issues.append(
                f"{room.get('type')} size {w}x{l} differs from expected {exp.get('width')}x{exp.get('length')}"
            )

    return issues

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--params", required=True, help="Path to parameters JSON")
    ap.add_argument("--layout", required=True, help="Path to generated layout JSON")
    ap.add_argument("--svg_out", default="evaluation.svg", help="Path to write SVG rendering")
    ap.add_argument(
        "--min_sep",
        type=float,
        default=0.0,
        help="Minimum separation required between rooms during validation",
    )
    ap.add_argument(
        "--strict",
        action="store_true",
        help="Exit with a non-zero status if any validation issues are found",
    )
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    params = json.load(open(args.params, "r", encoding="utf-8"))
    layout = json.load(open(args.layout, "r", encoding="utf-8"))

    dims = params.get("dimensions") or {}
    max_w = float(dims.get("width", 40))
    max_h = float(dims.get("depth", dims.get("height", 40)))

    render_layout_svg(layout, args.svg_out)
    log.info("Rendered layout SVG to %s", Path(args.svg_out).resolve())

    bounds_issues = check_bounds(
        (layout.get("layout") or {}).get("rooms", []),
        max_width=max_w,
        max_length=max_h,
    )

    issues = [
        msg
        for msg in validate_layout(
            layout,
            max_width=max_w,
            max_length=max_h,
            min_separation=args.min_sep,
        )
        if msg not in bounds_issues
    ]
    issues.extend(compare_with_params(layout, params))

    all_issues = bounds_issues + issues

    if all_issues:
        log_func = log.error if args.strict else log.warning
        for msg in all_issues:
            log_func(msg)
        if args.strict:
            if bounds_issues:
                raise BoundaryViolationError("; ".join(all_issues))
            raise RuntimeError("; ".join(all_issues))
    else:
        log.info("No issues detected")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # pragma: no cover - CLI entry point
        log.error("%s", exc)
        sys.exit(1)
