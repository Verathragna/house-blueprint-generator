import json
import subprocess
import sys


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
    params = {"dimensions": {"width": 10, "depth": 10}, "bedrooms": 1}
    layout = {
        "layout": {
            "rooms": [
                {
                    "type": "Bedroom",
                    "position": {"x": 0, "y": 0},
                    "size": {"width": 5, "length": 5},
                }
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
