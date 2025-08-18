import os
import sys
from pathlib import Path
import importlib
import torch
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from tokenizer.tokenizer import EOS_ID


class DummyModel(torch.nn.Module):
    """Minimal model that always predicts EOS."""
    def __init__(self, vocab_size: int = 10):
        super().__init__()
        # single parameter so model has a device
        self.param = torch.nn.Parameter(torch.zeros(1))
        self.vocab_size = vocab_size

    def forward(self, x):
        batch, seq_len = x.shape
        logits = torch.zeros(batch, seq_len, self.vocab_size, device=x.device)
        logits[:, :, EOS_ID] = 10.0  # force EOS
        return logits


def dummy_render(layout_json, svg_path):
    Path(svg_path).write_text("<svg></svg>")


@pytest.fixture()
def dummy_checkpoint(tmp_path):
    ckpt_dir = Path("checkpoints")
    ckpt_dir.mkdir(exist_ok=True)
    ckpt_path = ckpt_dir / "model_latest.pth"
    torch.save(DummyModel().state_dict(), ckpt_path)
    yield ckpt_path
    ckpt_path.unlink(missing_ok=True)
    ckpt_dir.rmdir()


def test_generate_blueprint_device_flag(dummy_checkpoint, monkeypatch, tmp_path):
    import Generate.generate_blueprint as gb
    # patch model and rendering
    monkeypatch.setattr(gb, "LayoutTransformer", DummyModel)
    monkeypatch.setattr(gb, "render_layout_svg", dummy_render)

    out_prefix = tmp_path / "output"
    argv = [
        "generate_blueprint.py",
        "--params_json",
        str(Path("sample_params.json")),
        "--out_prefix",
        str(out_prefix),
        "--device",
        "cpu",
    ]
    monkeypatch.setattr(sys, "argv", argv)
    gb.main()
    assert (Path(str(out_prefix) + ".json").exists())
    assert (Path(str(out_prefix) + ".svg").exists())


def test_api_respects_device_env(dummy_checkpoint, monkeypatch):
    os.environ["DEVICE"] = "cpu"
    app_module = importlib.import_module("api.app")
    importlib.reload(app_module)
    monkeypatch.setattr(app_module, "LayoutTransformer", DummyModel)
    monkeypatch.setattr(app_module, "render_layout_svg", dummy_render)
    model = app_module._get_model()
    assert next(model.parameters()).device.type == "cpu"
