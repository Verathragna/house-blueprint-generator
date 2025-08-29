import json
import subprocess
import sys
import pytest


def run_eval(params, layout, svg_path, strict=True, report=None):
    """Run the evaluate_sample script with provided params and layout."""
    params_file = svg_path.parent / "params.json"
    layout_file = svg_path.parent / "layout.json"
    params_file.write_text(json.dumps(params))
    layout_file.write_text(json.dumps(layout))

    cmd = [
        sys.executable,
        "evaluation/evaluate_sample.py",
        "--params",
        str(params_file),
        "--layout",
        str(layout_file),
        "--svg_out",
        str(svg_path),
    ]
    if strict:
        cmd.append("--strict")
    if report is not None:
        cmd.extend(["--json-report", str(report)])

    return subprocess.run(cmd, capture_output=True)


def test_strict_exits_on_issue(tmp_path):
    params = {"dimensions": {"width": 10, "depth": 10}, "bedrooms": 1}
    layout = {"layout": {"rooms": []}}

    result = run_eval(params, layout, tmp_path / "out.svg", strict=True)
    assert result.returncode != 0


def test_strict_allows_clean_layout(tmp_path):
    params = {"dimensions": {"width": 20, "depth": 20}, "bedrooms": 1}
    layout = {
        "layout": {
            "rooms": [
                {
                    "type": "Bedroom",
                    "position": {"x": 0, "y": 0},
                    "size": {"width": 5, "length": 5},
                },
                {
                    "type": "Kitchen",
                    "position": {"x": 5, "y": 0},
                    "size": {"width": 5, "length": 5},
                },
                {
                    "type": "Living Room",
                    "position": {"x": 10, "y": 0},
                    "size": {"width": 5, "length": 5},
                },
                {
                    "type": "Dining Room",
                    "position": {"x": 0, "y": 5},
                    "size": {"width": 5, "length": 5},
                },
                {
                    "type": "Laundry Room",
                    "position": {"x": 5, "y": 5},
                    "size": {"width": 5, "length": 5},
                },
            ]
        }
    }

    result = run_eval(params, layout, tmp_path / "out.svg", strict=True)
    assert result.returncode == 0


def test_json_report(tmp_path):
    params = {"dimensions": {"width": 10, "depth": 10}, "bedrooms": 1}
    layout = {"layout": {"rooms": []}}
    report = tmp_path / "report.json"
    result = run_eval(params, layout, tmp_path / "out.svg", strict=False, report=report)
    assert result.returncode == 0
    data = json.loads(report.read_text())
    assert data["bounds_issues"] or data["other_issues"]


@pytest.mark.parametrize("missing", [
    "kitchen",
    "living room",
    "dining room",
    "laundry room",
])
def test_missing_core_room_fails(tmp_path, missing):
    params = {"dimensions": {"width": 20, "depth": 20}}
    base_rooms = {
        "kitchen": {"type": "Kitchen", "position": {"x": 0, "y": 0}, "size": {"width": 5, "length": 5}},
        "living room": {"type": "Living Room", "position": {"x": 5, "y": 0}, "size": {"width": 5, "length": 5}},
        "dining room": {"type": "Dining Room", "position": {"x": 0, "y": 5}, "size": {"width": 5, "length": 5}},
        "laundry room": {"type": "Laundry Room", "position": {"x": 5, "y": 5}, "size": {"width": 5, "length": 5}},
    }
    rooms = [room for name, room in base_rooms.items() if name != missing]
    layout = {"layout": {"rooms": rooms}}

    result = run_eval(params, layout, tmp_path / f"out_{missing.replace(' ', '_')}.svg", strict=True)
    assert result.returncode != 0
