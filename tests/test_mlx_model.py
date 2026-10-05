"""The MLX port must match the PyTorch model given identical weights."""

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

mx = pytest.importorskip("mlx.core")

from ghostlm.config import GhostLMConfig  # noqa: E402
from ghostlm.mlx_model import GhostLMMLX, from_torch_state, to_torch_state  # noqa: E402
from ghostlm.model import GhostLM  # noqa: E402

EOS = 199


def _cfg(**overrides):
    kwargs = dict(n_layers=3, d_model=96, n_heads=6, n_kv_heads=2, d_ff=256, vocab_size=200,
                  context_length=64, dropout=0.0, bias=True, use_rope=True, use_swiglu=True,
                  use_rmsnorm=True, use_flash_attention=True, use_qk_norm=True,
                  intra_doc_mask=True, eos_token_id=EOS, device="cpu")
    kwargs.update(overrides)
    return GhostLMConfig(**kwargs)


def _tokens():
    rng = np.random.default_rng(0)
    idx = rng.integers(0, EOS, size=(2, 48))
    idx[0, 10] = idx[0, 30] = idx[1, 20] = EOS
    return idx


@pytest.mark.parametrize("flags", [{}, {"attn_gate": True, "value_residual": True},
                                   {"intra_doc_mask": False}])
def test_logits_and_loss_match_torch(flags):
    torch.manual_seed(0)
    cfg = _cfg(**flags)
    tm = GhostLM(cfg).eval()
    if flags.get("value_residual"):
        for b in tm.blocks:
            b.attn.v_mix.data.fill_(0.3)
    mm = GhostLMMLX(cfg)
    from_torch_state(mm, tm.state_dict())

    idx = _tokens()
    with torch.no_grad():
        t_logits, t_loss = tm(torch.tensor(idx), torch.tensor(idx))
    m_logits = np.array(mm(mx.array(idx)))
    m_loss = mm.loss(mx.array(idx), mx.array(idx)).item()

    assert np.abs(m_logits - t_logits.numpy()).max() < 2e-3
    assert abs(m_loss - t_loss.item()) < 1e-4


def test_round_trip_back_into_torch():
    torch.manual_seed(1)
    cfg = _cfg(attn_gate=True, value_residual=True)
    mm = GhostLMMLX(cfg)
    tm = GhostLM(cfg).eval()
    state = {k: torch.from_numpy(v) for k, v in to_torch_state(mm).items()}
    tm.load_state_dict(state)
    idx = _tokens()
    with torch.no_grad():
        t_logits, _ = tm(torch.tensor(idx))
    assert np.abs(np.array(mm(mx.array(idx))) - t_logits.numpy()).max() < 2e-3


def test_gradient_checkpointing_matches():
    cfg = _cfg()
    plain, ckpt = GhostLMMLX(cfg), GhostLMMLX(cfg, gradient_checkpointing=True)
    ckpt.update(plain.parameters())
    import mlx.nn as nn
    idx = mx.array(_tokens())
    g1 = nn.value_and_grad(plain, lambda m: m.loss(idx, idx))(plain)[1]
    g2 = nn.value_and_grad(ckpt, lambda m: m.loss(idx, idx))(ckpt)[1]
    a = np.array(g1["blocks"][1]["attn"]["c_qkv"]["weight"])
    b = np.array(g2["blocks"][1]["attn"]["c_qkv"]["weight"])
    assert np.abs(a - b).max() < 1e-5


def test_rejects_dropout():
    with pytest.raises(ValueError):
        GhostLMMLX(_cfg(dropout=0.1))
