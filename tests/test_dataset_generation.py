import json
from pathlib import Path


def dummy_render(layout, path, lot_dims=None):
    Path(path).write_text("<svg></svg>")


def test_generated_dataset_has_coordinates(tmp_path, monkeypatch):
    from dataset import generate_dataset as gd

    monkeypatch.setattr(gd, "render_layout_svg", dummy_render)
    out_dir = tmp_path / "synthetic"

    gd.main(n=5, out_dir=str(out_dir), seed=0, strict=True)

    coords = set()
    for layout_file in out_dir.glob("layout_*.json"):
        data = json.loads(layout_file.read_text())
        for room in data["layout"]["rooms"]:
            assert "x" in room["position"] and "y" in room["position"]
            coords.add((room["position"]["x"], room["position"]["y"]))

    # ensure coordinates vary across samples
    assert len(coords) > 1


def test_external_ingestion(tmp_path, monkeypatch):
    from dataset import generate_dataset as gd

    monkeypatch.setattr(gd, "render_layout_svg", dummy_render)

    external_dir = tmp_path / "external"
    external_dir.mkdir()
    sample = {
        "rooms": [{"type": "Bedroom", "x": 1, "y": 2, "width": 10, "length": 11}],
        "params": {"houseStyle": "External", "squareFeet": 1200},
    }
    (external_dir / "plan.json").write_text(json.dumps(sample))

    out_dir = tmp_path / "synthetic"
    gd.main(n=0, external_dir=str(external_dir), out_dir=str(out_dir), seed=0, strict=True)

    layout_file = next(out_dir.glob("layout_*.json"))
    data = json.loads(layout_file.read_text())
    room = data["layout"]["rooms"][0]
    assert room["position"] == {"x": 1, "y": 2}


def test_external_ingestion_with_position(tmp_path, monkeypatch):
    from dataset import generate_dataset as gd

    monkeypatch.setattr(gd, "render_layout_svg", dummy_render)

    external_dir = tmp_path / "external"
    external_dir.mkdir()
    sample = {
        "rooms": [
            {"type": "Kitchen", "position": {"x": 5, "y": 6}, "width": 12, "length": 14}
        ],
        "params": {"houseStyle": "External", "squareFeet": 900},
    }
    (external_dir / "plan.json").write_text(json.dumps(sample))

    out_dir = tmp_path / "synthetic"
    gd.main(n=0, external_dir=str(external_dir), out_dir=str(out_dir), seed=0, strict=True)

    layout_file = next(out_dir.glob("layout_*.json"))
    data = json.loads(layout_file.read_text())
    room = data["layout"]["rooms"][0]
    assert room["position"] == {"x": 5, "y": 6}


def test_skip_bad_external_file(tmp_path, monkeypatch, capsys):
    from dataset import generate_dataset as gd

    monkeypatch.setattr(gd, "render_layout_svg", dummy_render)

    external_dir = tmp_path / "external"
    external_dir.mkdir()
    (external_dir / "bad.json").write_text("{invalid")

    out_dir = tmp_path / "synthetic"
    gd.main(n=0, external_dir=str(external_dir), out_dir=str(out_dir), seed=0)

    captured = capsys.readouterr()
    assert "bad.json" in captured.out
    assert not list(out_dir.glob("layout_*.json"))


def test_main_skips_write_errors(tmp_path, monkeypatch, capsys):
    from dataset import generate_dataset as gd

    def failing_write(*args, **kwargs):
        raise OSError("boom")

    monkeypatch.setattr(gd, "render_layout_svg", dummy_render)
    monkeypatch.setattr(gd, "_write_sample", failing_write)

    out_dir = tmp_path / "synthetic"
    gd.main(n=1, out_dir=str(out_dir), seed=0)

    captured = capsys.readouterr()
    assert "Skipping sample 0" in captured.out
    assert not list(out_dir.glob("layout_*.json"))
