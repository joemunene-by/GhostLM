"""Tests for scripts/btx_merge.py: dense domain branches merged into one MoE."""

import sys
from dataclasses import asdict
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ghostlm.config import GhostLMConfig  # noqa: E402
from ghostlm.model import GhostLM  # noqa: E402
from scripts.btx_merge import merge  # noqa: E402


def _cfg(**overrides):
    kwargs = dict(n_layers=2, d_model=48, n_heads=4, n_kv_heads=2, d_ff=128, vocab_size=120,
                  context_length=32, dropout=0.0, use_rope=True, use_swiglu=True, use_rmsnorm=True,
                  use_flash_attention=True, device="cpu")
    kwargs.update(overrides)
    return GhostLMConfig(**kwargs)


def _branches(n=3):
    """Branches that share every non-FFN weight but have their own FFNs, like BTX branches would."""
    torch.manual_seed(0)
    seed = GhostLM(_cfg())
    out = []
    for i in range(n):
        m = GhostLM(_cfg())
        m.load_state_dict(seed.state_dict())
        with torch.no_grad():
            for b in m.blocks:
                for fc in (b.ffn.fc1, b.ffn.fc2, b.ffn.fc3):
                    fc.weight.add_(torch.randn_like(fc.weight) * 0.05 * (i + 1))
        out.append((f"d{i}", {"model_state_dict": m.state_dict(), "config": asdict(_cfg()), "step": 100 + i}, m))
    return out


def test_merged_model_loads_and_reports_layout():
    branches = _branches()
    ckpt = merge([(n, c) for n, c, _ in branches], top_k=2)
    cfg = GhostLMConfig(**ckpt["config"])
    assert cfg.use_moe and cfg.n_experts == 3 and cfg.n_experts_active == 2
    moe = GhostLM(cfg)
    moe.load_state_dict(ckpt["model_state_dict"])
    assert ckpt["btx"] == {"experts": ["d0", "d1", "d2"], "branch_steps": [100, 101, 102]}
    for e, (_, _, dense) in enumerate(branches):
        assert torch.equal(moe.blocks[1].ffn.experts[e].fc2.weight, dense.blocks[1].ffn.fc2.weight)


def test_shared_weights_are_averaged():
    branches = _branches(2)
    with torch.no_grad():
        branches[1][1]["model_state_dict"]["ln_f.weight"] = torch.full_like(
            branches[1][1]["model_state_dict"]["ln_f.weight"], 3.0)
        branches[0][1]["model_state_dict"]["ln_f.weight"] = torch.full_like(
            branches[0][1]["model_state_dict"]["ln_f.weight"], 1.0)
    ckpt = merge([(n, c) for n, c, _ in branches], top_k=1)
    assert torch.allclose(ckpt["model_state_dict"]["ln_f.weight"], torch.full_like(
        ckpt["model_state_dict"]["ln_f.weight"], 2.0))


class _FixedRouter(torch.nn.Module):
    """Router stand-in that always prefers one expert."""

    def __init__(self, expert: int, n: int):
        super().__init__()
        self.expert, self.n = expert, n

    def forward(self, x):
        logits = torch.full((*x.shape[:-1], self.n), -10.0)
        logits[..., self.expert] = 10.0
        return logits


def test_routing_everything_to_one_expert_reproduces_that_branch():
    branches = _branches()
    ckpt = merge([(n, c) for n, c, _ in branches], top_k=1)
    moe = GhostLM(GhostLMConfig(**ckpt["config"])).eval()
    moe.load_state_dict(ckpt["model_state_dict"])
    idx = torch.randint(0, 120, (2, 20))
    for e, (_, _, dense) in enumerate(branches):
        for b in moe.blocks:
            b.ffn.gate = _FixedRouter(e, 3)
        with torch.no_grad():
            moe_logits, _ = moe(idx)
            dense_logits, _ = dense.eval()(idx)
        assert torch.allclose(moe_logits, dense_logits, atol=1e-4)
