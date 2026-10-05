"""Tests for the opt-in gated-attention and value-residual architecture switches."""

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ghostlm.config import GhostLMConfig
from ghostlm.model import GhostLM


def _cfg(**overrides):
    kwargs = dict(n_layers=3, d_model=64, n_heads=4, n_kv_heads=2, d_ff=128, vocab_size=200,
                  context_length=32, dropout=0.0, use_rope=True, use_swiglu=True,
                  use_rmsnorm=True, use_flash_attention=True, device="cpu")
    kwargs.update(overrides)
    return GhostLMConfig(**kwargs)


@pytest.mark.parametrize("flags", [{"attn_gate": True}, {"value_residual": True},
                                   {"attn_gate": True, "value_residual": True}])
def test_options_train_and_get_gradients(flags):
    torch.manual_seed(0)
    model = GhostLM(_cfg(**flags))
    x = torch.randint(0, 200, (2, 16))
    _, loss = model(x, x)
    loss.backward()
    if flags.get("value_residual"):
        mixes = [b.attn.v_mix for b in model.blocks]
        # Layer 0 supplies v_first and never mixes; later layers learn the mix.
        assert mixes[0].grad is None
        assert all(m.grad is not None for m in mixes[1:])
    if flags.get("attn_gate"):
        assert all(b.attn.gate.weight.grad is not None for b in model.blocks)


@pytest.mark.parametrize("flags", [{"attn_gate": True}, {"value_residual": True}])
def test_kv_cache_matches_full_forward(flags):
    torch.manual_seed(0)
    model = GhostLM(_cfg(**flags)).eval()
    x = torch.randint(0, 200, (1, 12))
    with torch.no_grad():
        full, _ = model(x)
        logits, _, cache = model(x[:, :8], use_cache=True)
        step, _, _ = model(x[:, 8:], past_kv=cache, use_cache=True)
    assert torch.allclose(full[:, 8:], step, atol=1e-5)


def test_gradient_checkpointing_matches_with_value_residual():
    torch.manual_seed(0)
    base = GhostLM(_cfg(value_residual=True))
    ckpt = GhostLM(_cfg(value_residual=True, gradient_checkpointing=True))
    ckpt.load_state_dict(base.state_dict())
    x = torch.randint(0, 200, (2, 16))
    for m in (base, ckpt):
        m.train()
        _, loss = m(x, x)
        loss.backward()
    assert torch.allclose(base.blocks[1].attn.v_mix.grad, ckpt.blocks[1].attn.v_mix.grad, atol=1e-6)


def test_options_off_keep_checkpoint_shape():
    keys = set(GhostLM(_cfg()).state_dict())
    assert not any("gate" in k or "v_mix" in k for k in keys)
