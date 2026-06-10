"""Tests for the GPU-run-prep trainer features: wandb wiring,
torch.compile checkpoint unwrapping, and gradient checkpointing."""

import sys
import types
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ghostlm.config import GhostLMConfig
from ghostlm.model import GhostLM
from ghostlm.trainer import GhostTrainer


def _tiny_config(tmp_path, **overrides) -> GhostLMConfig:
    cfg = GhostLMConfig(
        n_layers=2, d_model=64, n_heads=4, d_ff=128,
        vocab_size=200, context_length=32, dropout=0.0,
        device="cpu", grad_accum_steps=1, warmup_steps=10,
        checkpoint_dir=str(tmp_path / "ckpt"),
        log_dir=str(tmp_path / "logs"),
        **overrides,
    )
    return cfg


def _batch():
    torch.manual_seed(0)
    return torch.randint(0, 200, (2, 8)), torch.randint(0, 200, (2, 8))


class _WandbStub(types.ModuleType):
    """Minimal wandb stand-in capturing init/log/finish calls."""

    def __init__(self):
        super().__init__("wandb")
        self.init_kwargs = None
        self.logged = []
        self.finished = False

    def init(self, **kwargs):
        self.init_kwargs = kwargs
        return self

    def log(self, metrics, step=None):
        self.logged.append((dict(metrics), step))

    def finish(self):
        self.finished = True


def test_wandb_wiring(tmp_path, monkeypatch):
    """use_wandb=True must init wandb, stream eval metrics, and finish."""
    stub = _WandbStub()
    monkeypatch.setitem(sys.modules, "wandb", stub)

    cfg = _tiny_config(tmp_path, use_wandb=True, max_steps=2,
                       eval_interval=1, save_interval=2)
    trainer = GhostTrainer(GhostLM(cfg), cfg, use_amp=False)

    assert trainer.wandb_run is stub
    assert stub.init_kwargs["project"] == "ghostlm"
    # the default "logs" dir gives no run name (wandb autogenerates one)
    assert stub.init_kwargs["name"] is None

    trainer._log({"step": 1, "train_loss": 2.0, "val_loss": 2.1, "lr": 1e-4})
    eval_metrics = [m for m, _ in stub.logged if "eval/val_loss" in m]
    assert eval_metrics and eval_metrics[0]["eval/val_loss"] == 2.1
    # the literal "step" field must not be re-logged as a metric
    assert "eval/step" not in eval_metrics[0]


def test_wandb_missing_module_degrades_gracefully(tmp_path, monkeypatch):
    """A broken/missing wandb must warn, not crash the trainer."""
    monkeypatch.setitem(sys.modules, "wandb", None)  # import returns None -> AttributeError
    cfg = _tiny_config(tmp_path, use_wandb=True)
    trainer = GhostTrainer(GhostLM(cfg), cfg, use_amp=False)
    assert trainer.wandb_run is None
    # training still works
    loss = trainer.train_step(_batch())
    assert loss > 0


def test_compile_wrap_and_clean_checkpoint(tmp_path):
    """use_compile must wrap the model, and checkpoints must contain
    plain GhostLM keys (no _orig_mod. prefix)."""
    if not hasattr(torch, "compile"):
        pytest.skip("torch.compile unavailable")
    cfg = _tiny_config(tmp_path, use_compile=True)
    trainer = GhostTrainer(GhostLM(cfg), cfg, use_amp=False)

    assert hasattr(trainer.model, "_orig_mod")
    assert isinstance(trainer._unwrap_model(), GhostLM)

    trainer.save_checkpoint(val_loss=1.0)
    ckpt = torch.load(tmp_path / "ckpt" / "checkpoint_step_0.pt",
                      map_location="cpu", weights_only=False)
    assert all(not k.startswith("_orig_mod.")
               for k in ckpt["model_state_dict"])
    assert "token_embedding.weight" in ckpt["model_state_dict"]


def test_gradient_checkpointing_matches_plain_backward(tmp_path):
    """Same weights + same batch: loss and gradients must be identical
    with and without gradient checkpointing."""
    x, y = _batch()

    losses, grads = [], []
    for ckpt_on in (False, True):
        torch.manual_seed(7)
        cfg = _tiny_config(tmp_path, gradient_checkpointing=ckpt_on,
                           use_rope=True, use_swiglu=True, use_rmsnorm=True)
        model = GhostLM(cfg)
        model.train()
        _, loss = model(x, targets=y)
        loss.backward()
        losses.append(loss.item())
        grads.append(model.blocks[0].attn.c_qkv.weight.grad.clone())

    assert losses[0] == pytest.approx(losses[1], rel=1e-6)
    assert torch.allclose(grads[0], grads[1], atol=1e-6)


def test_gradient_checkpointing_skipped_in_eval_and_cache():
    """Checkpointing must not interfere with eval-mode KV-cached decoding."""
    torch.manual_seed(0)
    cfg = GhostLMConfig(
        n_layers=2, d_model=64, n_heads=4, d_ff=128,
        vocab_size=200, context_length=32, dropout=0.0,
        gradient_checkpointing=True,
    )
    model = GhostLM(cfg)
    model.eval()
    x = torch.randint(0, 200, (1, 8))
    full, _ = model(x)
    logits, _, kv = model(x[:, :4], use_cache=True)
    pieces = [logits]
    for t in range(4, 8):
        logits, _, kv = model(x[:, t:t + 1], past_kv=kv, use_cache=True)
        pieces.append(logits)
    assert torch.allclose(full, torch.cat(pieces, dim=1), atol=1e-4)
