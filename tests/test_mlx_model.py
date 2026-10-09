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


def test_moe_matches_torch_sparse_moe():
    import mlx.nn as nn
    torch.manual_seed(0)
    cfg = _cfg(use_moe=True, n_experts=4, n_experts_active=2, moe_aux_loss_coef=0.01)
    tm = GhostLM(cfg).eval()
    mm = GhostLMMLX(cfg)
    from_torch_state(mm, tm.state_dict())

    idx = _tokens()
    with torch.no_grad():
        t_logits, t_loss = tm(torch.tensor(idx), torch.tensor(idx))
    m_logits = np.array(mm(mx.array(idx)))
    assert np.abs(m_logits - t_logits.numpy()).max() < 2e-3
    assert abs(mm.loss(mx.array(idx), mx.array(idx)).item() - t_loss.item()) < 1e-4

    # Router gradients must flow through the top-k weights and the balancing loss.
    tm.train()
    _, loss = tm(torch.tensor(idx), torch.tensor(idx))
    loss.backward()
    g = nn.value_and_grad(mm, lambda m: m.loss(mx.array(idx), mx.array(idx)))(mm)[1]
    t_gate = tm.blocks[1].ffn.gate.weight.grad.numpy()
    assert np.abs(np.array(g["blocks"][1]["ffn"]["gate"]["weight"]) - t_gate).max() < 1e-4


def test_moe_state_round_trips_per_expert_names():
    cfg = _cfg(use_moe=True, n_experts=3, n_experts_active=2)
    mm = GhostLMMLX(cfg)
    state = to_torch_state(mm)
    assert "blocks.0.ffn.experts.2.fc3.weight" in state and "blocks.0.ffn.fc3" not in state
    tm = GhostLM(cfg)
    tm.load_state_dict({k: torch.from_numpy(v) for k, v in state.items()})
    back = GhostLMMLX(cfg)
    from_torch_state(back, tm.state_dict())
    idx = mx.array(_tokens())
    assert np.abs(np.array(back(idx)) - np.array(mm(idx))).max() < 1e-5
