"""One MLX optimizer step must match one GhostTrainer step from the same weights and batch."""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
mx = pytest.importorskip("mlx.core")

from ghostlm.config import GhostLMConfig  # noqa: E402
from ghostlm.mlx_model import GhostLMMLX, from_torch_state  # noqa: E402
from ghostlm.model import GhostLM  # noqa: E402
from ghostlm.trainer import GhostTrainer  # noqa: E402

spec = importlib.util.spec_from_file_location("mlx_trainer", ROOT / "scripts/train_ghost_base_mlx.py")
mt = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mt)


def _cfg(tmp_path, **overrides):
    kwargs = dict(n_layers=2, d_model=64, n_heads=4, n_kv_heads=2, d_ff=192, vocab_size=128,
                  context_length=32, dropout=0.0, bias=True, use_rope=True, use_swiglu=True,
                  use_rmsnorm=True, use_flash_attention=True, use_qk_norm=True,
                  intra_doc_mask=True, eos_token_id=127, device="cpu", batch_size=8,
                  grad_accum_steps=4, learning_rate=1e-3, warmup_steps=1, max_steps=10,
                  weight_decay=0.1, lr_schedule="wsd", checkpoint_dir=str(tmp_path / "ck"),
                  log_dir=str(tmp_path / "lg"))
    kwargs.update(overrides)
    return GhostLMConfig(**kwargs)


def test_one_step_matches_ghosttrainer(tmp_path):
    torch.manual_seed(0)
    cfg = _cfg(tmp_path)
    tm = GhostLM(cfg)
    mm = GhostLMMLX(cfg)
    from_torch_state(mm, tm.state_dict())

    rng = np.random.default_rng(0)
    x = rng.integers(0, 127, size=(8, 32))
    x[:, 15] = 127
    y = np.roll(x, -1, axis=1)

    tt = GhostTrainer(tm, cfg, use_amp=False)
    t_loss = tt.train_step((torch.tensor(x), torch.tensor(y)))

    mtr = mt.Trainer(cfg, mm, mx.float32)
    m_loss = mtr.train_step(x, y)

    assert abs(t_loss - m_loss) < 1e-4
    t_state = {k: v.detach().numpy() for k, v in tm.state_dict().items()}
    for name, value in mt.tree_flatten(mtr.master):
        assert np.abs(np.array(value) - t_state[name]).max() < 1e-5, name


def test_lr_schedule_matches_ghosttrainer(tmp_path):
    for schedule in ("wsd", "cosine"):
        cfg = _cfg(tmp_path, lr_schedule=schedule, max_steps=100, warmup_steps=10)
        tt = GhostTrainer(GhostLM(cfg), cfg, use_amp=False)
        for step in (0, 5, 10, 50, 85, 100):
            tt.step = step
            assert abs(tt.get_lr() - mt.lr_at(step, cfg)) < 1e-12


def test_checkpoint_loads_into_torch_and_resumes(tmp_path):
    cfg = _cfg(tmp_path, best_weights_only=True)
    mm = GhostLMMLX(cfg)
    mtr = mt.Trainer(cfg, mm, mx.float32)
    rng = np.random.default_rng(1)
    x = rng.integers(0, 127, size=(8, 32))
    mtr.train_step(x, x)
    mtr.save(2.5)

    ck = torch.load(tmp_path / "ck" / "checkpoint_step_1.pt", weights_only=False)
    GhostLM(cfg).load_state_dict(ck["model_state_dict"])
    best = torch.load(tmp_path / "ck" / "best_model.pt", weights_only=False)
    assert "mlx_optimizer_state" not in best and "mlx_optimizer_state" in ck

    resumed = mt.Trainer(cfg, GhostLMMLX(cfg), mx.float32)
    resumed.load(str(tmp_path / "ck" / "checkpoint_step_1.pt"))
    assert resumed.step == 1 and resumed.best_val_loss == 2.5
    a, b = resumed.train_step(x, x), mtr.train_step(x, x)
    assert abs(a - b) < 1e-5


@pytest.mark.parametrize("opt,cautious", [("muon", False), ("normuon", True)])
def test_muon_steps_match_ghosttrainer(tmp_path, opt, cautious):
    torch.manual_seed(0)
    cfg = _cfg(tmp_path, optimizer=opt, cautious_wd=cautious, attn_gate=True, value_residual=True)
    tm = GhostLM(cfg)
    mm = GhostLMMLX(cfg)
    from_torch_state(mm, tm.state_dict())
    tt = GhostTrainer(tm, cfg, use_amp=False)
    mtr = mt.Trainer(cfg, mm, mx.float32)
    assert mtr.muon is not None and mtr.muon.muon_names

    rng = np.random.default_rng(3)
    for _ in range(2):
        x = rng.integers(0, 127, size=(8, 32))
        t_loss = tt.train_step((torch.tensor(x), torch.tensor(x)))
        m_loss = mtr.train_step(x, x)
        assert abs(t_loss - m_loss) < 1e-4
    t_state = {k: v.detach().numpy() for k, v in tm.state_dict().items()}
    for name, value in mt.tree_flatten(mtr.master):
        assert np.abs(np.array(value) - t_state[name]).max() < 1e-4, name


def test_muon_resume_restores_optimizer_state(tmp_path):
    cfg = _cfg(tmp_path, optimizer="normuon")
    mtr = mt.Trainer(cfg, GhostLMMLX(cfg), mx.float32)
    x = np.random.default_rng(4).integers(0, 127, size=(8, 32))
    mtr.train_step(x, x)
    mtr.save(3.0)
    resumed = mt.Trainer(cfg, GhostLMMLX(cfg), mx.float32)
    resumed.load(str(tmp_path / "ck" / "checkpoint_step_1.pt"))
    assert abs(resumed.train_step(x, x) - mtr.train_step(x, x)) < 1e-5
    for name, value in mt.tree_flatten(mtr.master):
        assert np.abs(np.array(value) - np.array(dict(mt.tree_flatten(resumed.master))[name])).max() < 1e-6
