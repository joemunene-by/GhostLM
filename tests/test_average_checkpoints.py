"""Tests for scripts/average_checkpoints.py."""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.average_checkpoints import average


def _save(path, value, step):
    torch.save({"step": step, "config": {"d": 1},
                "model_state_dict": {"w": torch.full((2,), value), "n": torch.tensor(step)}}, path)
    return str(path)


def test_uniform_mean(tmp_path):
    paths = [_save(tmp_path / f"c{i}.pt", v, i) for i, v in enumerate([0.0, 3.0, 6.0])]
    out = average(paths)
    assert torch.allclose(out["model_state_dict"]["w"], torch.full((2,), 3.0))
    assert out["step"] == 2 and out["config"] == {"d": 1}
    # Integer buffers are kept from the first checkpoint, not averaged.
    assert out["model_state_dict"]["n"].dtype == torch.int64


def test_ema_weights_later_checkpoints_more(tmp_path):
    paths = [_save(tmp_path / f"c{i}.pt", v, i) for i, v in enumerate([0.0, 10.0])]
    out = average(paths, ema=0.5)
    assert torch.allclose(out["model_state_dict"]["w"], torch.full((2,), 5.0))
    out = average(paths, ema=0.9)
    assert torch.allclose(out["model_state_dict"]["w"], torch.full((2,), 1.0))
