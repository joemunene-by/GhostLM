"""Tests for stop-and-resume training: graceful stop, atomic saves, and
best_val_loss surviving a resume."""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ghostlm.config import GhostLMConfig
from ghostlm.model import GhostLM
from ghostlm.trainer import GhostTrainer


def _tiny_config(tmp_path, **overrides) -> GhostLMConfig:
    kwargs = dict(
        n_layers=2, d_model=64, n_heads=4, d_ff=128,
        vocab_size=200, context_length=32, dropout=0.0,
        device="cpu", grad_accum_steps=1, warmup_steps=10,
        checkpoint_dir=str(tmp_path / "ckpt"),
        log_dir=str(tmp_path / "logs"),
    )
    kwargs.update(overrides)
    return GhostLMConfig(**kwargs)


def _loader(n=4):
    torch.manual_seed(0)
    return [(torch.randint(0, 200, (2, 8)), torch.randint(0, 200, (2, 8))) for _ in range(n)]


def test_stop_requested_saves_and_exits_early(tmp_path):
    cfg = _tiny_config(tmp_path, max_steps=50, eval_interval=1000, save_interval=1000)
    trainer = GhostTrainer(GhostLM(cfg), cfg, use_amp=False)

    original_step = trainer.train_step

    def step_then_stop(batch):
        loss = original_step(batch)
        if trainer.step >= 3:
            trainer.stop_requested = True
        return loss

    trainer.train_step = step_then_stop
    trainer.train(_loader(), _loader(2))

    assert trainer.step == 3
    ckpt = tmp_path / "ckpt" / "checkpoint_step_3.pt"
    assert ckpt.exists()
    assert not list((tmp_path / "ckpt").glob("*.tmp"))
    # A stop checkpoint is only a resume point, never a new best.
    assert not (tmp_path / "ckpt" / "best_model.pt").exists()


def test_resume_restores_best_not_latest_val_loss(tmp_path):
    cfg = _tiny_config(tmp_path)
    trainer = GhostTrainer(GhostLM(cfg), cfg, use_amp=False)
    trainer.step = 10
    trainer.save_checkpoint(2.0)
    trainer.step = 20
    trainer.save_checkpoint(3.0)

    resumed = GhostTrainer(GhostLM(cfg), cfg, use_amp=False)
    resumed.load_checkpoint(str(tmp_path / "ckpt" / "checkpoint_step_20.pt"))

    assert resumed.step == 20
    assert resumed.best_val_loss == 2.0
    resumed.save_checkpoint(2.5)
    best = torch.load(tmp_path / "ckpt" / "best_model.pt", weights_only=False)
    assert best["step"] == 10


def test_resume_from_legacy_checkpoint_falls_back_to_val_loss(tmp_path):
    cfg = _tiny_config(tmp_path)
    trainer = GhostTrainer(GhostLM(cfg), cfg, use_amp=False)
    trainer.step = 5
    trainer.save_checkpoint(1.5)
    path = tmp_path / "ckpt" / "checkpoint_step_5.pt"
    ckpt = torch.load(path, weights_only=False)
    del ckpt["best_val_loss"]
    torch.save(ckpt, path)

    resumed = GhostTrainer(GhostLM(cfg), cfg, use_amp=False)
    resumed.load_checkpoint(str(path))
    assert resumed.best_val_loss == 1.5


def test_wsd_schedule_is_flat_then_decays(tmp_path):
    cfg = _tiny_config(tmp_path, max_steps=100, warmup_steps=10, learning_rate=1e-3,
                       lr_schedule="wsd", wsd_decay_frac=0.2)
    trainer = GhostTrainer(GhostLM(cfg), cfg, use_amp=False)

    def lr_at(step):
        trainer.step = step
        return trainer.get_lr()

    assert lr_at(5) < 1e-3
    assert lr_at(10) == lr_at(50) == lr_at(79) == 1e-3
    assert 1e-5 < lr_at(90) < 1e-3
    assert abs(lr_at(100) - 1e-5) < 1e-9


def test_wsd_saves_pre_decay_checkpoint(tmp_path):
    cfg = _tiny_config(tmp_path, max_steps=10, warmup_steps=2, lr_schedule="wsd",
                       wsd_decay_frac=0.5, eval_interval=1000, save_interval=1000)
    trainer = GhostTrainer(GhostLM(cfg), cfg, use_amp=False)
    trainer.train(_loader(), _loader(2))

    pre = torch.load(tmp_path / "ckpt" / "pre_decay.pt", weights_only=False)
    assert pre["step"] == 5


def test_best_weights_only_drops_optimizer_state(tmp_path):
    cfg = _tiny_config(tmp_path, best_weights_only=True)
    trainer = GhostTrainer(GhostLM(cfg), cfg, use_amp=False)
    trainer.save_checkpoint(1.0)
    best = torch.load(tmp_path / "ckpt" / "best_model.pt", weights_only=False)
    latest = torch.load(tmp_path / "ckpt" / "checkpoint_step_0.pt", weights_only=False)
    assert "optimizer_state_dict" not in best and "model_state_dict" in best
    assert "optimizer_state_dict" in latest
