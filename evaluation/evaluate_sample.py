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
from evaluation.validators import validate_layout


log = logging.getLogger(__name__)


def compare_with_params(layout: dict, params: dict) -> list[str]:
    """Compare layout statistics with desired parameters.

    Currently checks room counts for bedrooms, bathrooms and garage.
    Returns a list of mismatch descriptions.
    """
    rooms = (layout.get("layout") or {}).get("rooms", [])
    counts = Counter(r.get("type", "").lower() for r in rooms)
    issues: list[str] = []

    bed_expected = int(params.get("bedrooms", 0))
    bed_found = counts.get("bedroom", 0)
    if bed_expected != bed_found:
        issues.append(f"Expected {bed_expected} bedrooms, found {bed_found}")

    baths = params.get("bathrooms") or {}
    bath_expected = int(baths.get("full", 0)) + int(baths.get("half", 0))
    bath_found = counts.get("bathroom", 0)
    if bath_expected != bath_found:
        issues.append(f"Expected {bath_expected} bathrooms, found {bath_found}")

    if params.get("garage") and counts.get("garage", 0) == 0:
        issues.append("Expected a garage, found none")

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
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    params = json.load(open(args.params, "r", encoding="utf-8"))
    layout = json.load(open(args.layout, "r", encoding="utf-8"))

    render_layout_svg(layout, args.svg_out)
    log.info("Rendered layout SVG to %s", Path(args.svg_out).resolve())

    issues = validate_layout(layout, min_separation=args.min_sep)
    issues.extend(compare_with_params(layout, params))

    if issues:
        for msg in issues:
            log.warning(msg)
    else:
        log.info("No issues detected")


if __name__ == "__main__":
    main()
