"""Tests for the Muon optimizer and its wiring into configure_optimizers."""

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ghostlm.config import GhostLMConfig
from ghostlm.model import GhostLM
from ghostlm.muon import MuonWithAuxAdam, zeropower_via_newtonschulz5
from ghostlm.trainer import GhostTrainer


def _tiny_config(tmp_path, **overrides) -> GhostLMConfig:
    return GhostLMConfig(
        n_layers=2, d_model=64, n_heads=4, d_ff=128,
        vocab_size=200, context_length=32, dropout=0.0,
        device="cpu", grad_accum_steps=1, warmup_steps=5,
        checkpoint_dir=str(tmp_path / "ckpt"),
        log_dir=str(tmp_path / "logs"),
        **overrides,
    )


@pytest.mark.parametrize("shape", [(64, 64), (32, 96), (96, 32)])
def test_newton_schulz_flattens_singular_values(shape):
    torch.manual_seed(0)
    g = torch.randn(*shape)
    s = torch.linalg.svdvals(zeropower_via_newtonschulz5(g))
    # The quintic iteration trades exactness for speed: singular values land
    # in a band around 1 instead of exactly 1, and a random square matrix's
    # near-zero tail is only partly lifted in 5 steps, so check the bulk.
    assert torch.quantile(s, 0.05) > 0.5 and s.max() < 1.3
    assert zeropower_via_newtonschulz5(g).shape == g.shape


def test_muon_groups_cover_every_param_once(tmp_path):
    cfg = _tiny_config(tmp_path, optimizer="muon", use_rope=True, use_swiglu=True, use_rmsnorm=True)
    model = GhostLM(cfg)
    opt = model.configure_optimizers(cfg)
    assert isinstance(opt, MuonWithAuxAdam)

    seen = [id(p) for g in opt.param_groups for p in g["params"]]
    assert len(seen) == len(set(seen))
    assert set(seen) == {id(p) for p in model.parameters()}

    muon = {id(p) for g in opt.param_groups if g["use_muon"] for p in g["params"]}
    assert id(model.token_embedding.weight) not in muon
    assert all(p.ndim == 2 for g in opt.param_groups if g["use_muon"] for p in g["params"])
    assert muon


def test_muon_rejects_non_matrix_params():
    with pytest.raises(ValueError):
        MuonWithAuxAdam([{"params": [torch.nn.Parameter(torch.zeros(4))], "use_muon": True}])


def test_muon_reduces_loss_and_resumes(tmp_path):
    torch.manual_seed(0)
    cfg = _tiny_config(tmp_path, optimizer="muon", learning_rate=3e-3)
    trainer = GhostTrainer(GhostLM(cfg), cfg, use_amp=False)
    x = torch.randint(0, 200, (4, 16))
    batch = (x, x)

    first = trainer.train_step(batch)
    for _ in range(30):
        last = trainer.train_step(batch)
    assert last < first * 0.8

    trainer.save_checkpoint(last)
    resumed = GhostTrainer(GhostLM(cfg), cfg, use_amp=False)
    resumed.load_checkpoint(str(tmp_path / "ckpt" / f"checkpoint_step_{trainer.step}.pt"))
    assert isinstance(resumed.optimizer, MuonWithAuxAdam)
    assert any("momentum_buffer" in s for s in resumed.optimizer.state.values())


def test_normuon_and_cautious_wd_train_and_balance_rows(tmp_path):
    torch.manual_seed(0)
    cfg = _tiny_config(tmp_path, optimizer="normuon", cautious_wd=True, learning_rate=3e-3)
    trainer = GhostTrainer(GhostLM(cfg), cfg, use_amp=False)
    assert trainer.optimizer.param_groups[0]["normuon"]
    x = torch.randint(0, 200, (4, 16))
    first = trainer.train_step((x, x))
    for _ in range(30):
        last = trainer.train_step((x, x))
    assert last < first * 0.8
    assert any("row_sq" in s for s in trainer.optimizer.state.values())


def test_cautious_wd_skips_disagreeing_coordinates():
    p = torch.tensor([1.0, 1.0, -1.0])
    update = torch.tensor([1.0, -1.0, -1.0])
    MuonWithAuxAdam._decay(p, update, 0.1, cautious=True)
    assert torch.allclose(p, torch.tensor([0.9, 1.0, -0.9]))
