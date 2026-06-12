"""Tests for KV-cached generation, attention masking, and the memmap dataset.

The KV cache must be a pure optimization: cached incremental decoding
has to produce exactly the same logits as a full forward pass, for
every architecture variant (learned-pos/GELU/LayerNorm, RoPE/SwiGLU/
RMSNorm, flash attention, MoE). These tests pin that invariant down so
future attention changes can't silently break it.
"""

import json
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ghostlm.config import GhostLMConfig
from ghostlm.model import GhostLM


VARIANTS = {
    "legacy": dict(),
    "modern": dict(use_rope=True, use_swiglu=True, use_rmsnorm=True),
    "flash": dict(use_rope=True, use_swiglu=True, use_rmsnorm=True,
                  use_flash_attention=True),
    "moe": dict(use_rope=True, use_moe=True, n_experts=4,
                n_experts_active=2, use_flash_attention=True),
    "gqa": dict(use_rope=True, use_swiglu=True, use_rmsnorm=True,
                n_kv_heads=2),
    "gqa_flash": dict(use_rope=True, use_swiglu=True, use_rmsnorm=True,
                      use_flash_attention=True, n_kv_heads=1),
    "qknorm": dict(use_rope=True, use_swiglu=True, use_rmsnorm=True,
                   use_qk_norm=True),
}


def _tiny_config(**overrides) -> GhostLMConfig:
    return GhostLMConfig(
        n_layers=2, d_model=64, n_heads=4, d_ff=128,
        vocab_size=200, context_length=32, dropout=0.0, **overrides,
    )


def _tiny_model(**overrides) -> GhostLM:
    torch.manual_seed(0)
    model = GhostLM(_tiny_config(**overrides))
    model.eval()
    return model


@pytest.mark.parametrize("variant", sorted(VARIANTS))
def test_cached_decoding_matches_full_forward(variant):
    """Prefill + per-token cached steps must equal one full forward."""
    model = _tiny_model(**VARIANTS[variant])
    x = torch.randint(0, 200, (2, 10))

    full_logits, _ = model(x)

    logits, _, kv = model(x[:, :6], use_cache=True)
    pieces = [logits]
    for t in range(6, 10):
        logits, _, kv = model(x[:, t:t + 1], past_kv=kv, use_cache=True)
        pieces.append(logits)
    incremental = torch.cat(pieces, dim=1)

    assert torch.allclose(full_logits, incremental, atol=1e-4), (
        f"{variant}: max diff {(full_logits - incremental).abs().max():.2e}"
    )


@pytest.mark.parametrize("variant", sorted(VARIANTS))
def test_generate_runs_past_context_window(variant):
    """generate() must handle sequences that outgrow the context window."""
    model = _tiny_model(**VARIANTS[variant])
    prompt = torch.randint(0, 200, (2, 10))
    out = model.generate(prompt, max_new_tokens=30, top_k=5)
    assert out.shape == (2, 40)
    assert (out[:, :10] == prompt).all()


@pytest.mark.parametrize("variant", sorted(VARIANTS))
def test_right_padding_does_not_change_real_logits(variant):
    """attn_mask must make padded positions invisible to real tokens."""
    model = _tiny_model(**VARIANTS[variant])
    x = torch.randint(0, 200, (2, 10))

    base_logits, _ = model(x)

    padded = torch.cat([x, torch.zeros(2, 4, dtype=torch.long)], dim=1)
    mask = torch.cat([torch.ones(2, 10), torch.zeros(2, 4)], dim=1)
    masked_logits, _ = model(padded, attn_mask=mask)

    assert torch.allclose(masked_logits[:, :10], base_logits, atol=1e-4)
    assert not torch.isnan(masked_logits).any()


def test_left_padding_does_not_nan():
    """Fully-padded leading rows (left padding) must not produce NaNs."""
    model = _tiny_model(use_rope=True, use_swiglu=True, use_rmsnorm=True)
    x = torch.randint(0, 200, (2, 8))
    padded = torch.cat([torch.zeros(2, 4, dtype=torch.long), x], dim=1)
    mask = torch.cat([torch.zeros(2, 4), torch.ones(2, 8)], dim=1)
    logits, _ = model(padded, attn_mask=mask)
    assert not torch.isnan(logits).any()


def test_cache_overflow_assertion():
    """Feeding more tokens than the context can hold must raise."""
    model = _tiny_model()
    x = torch.randint(0, 200, (1, 32))
    _, _, kv = model(x, use_cache=True)
    with pytest.raises(AssertionError):
        model(x[:, :1], past_kv=kv, use_cache=True)


def test_swiglu_residual_projection_gets_scaled_init():
    """SwiGLU's fc3 (residual-path output) must use the depth-scaled std;
    fc2 (the gate) must keep the base 0.02 std."""
    import math
    torch.manual_seed(0)
    cfg = GhostLMConfig(
        n_layers=8, d_model=256, n_heads=4, d_ff=1024,
        vocab_size=200, context_length=32, dropout=0.0, use_swiglu=True,
    )
    model = GhostLM(cfg)
    resid_std = 0.02 / math.sqrt(2 * cfg.n_layers)
    ffn = model.blocks[0].ffn
    assert abs(ffn.fc3.weight.std().item() - resid_std) < resid_std * 0.2
    assert abs(ffn.fc2.weight.std().item() - 0.02) < 0.02 * 0.2


def test_bin_dataset_matches_jsonl_dataset(tmp_path):
    """GhostBinDataset over pretokenized output must yield the same
    chunks as GhostDataset over the source JSONL."""
    from ghostlm.dataset import GhostBinDataset, GhostDataset
    from ghostlm.tokenizer import GhostTokenizer

    records = [
        {"text": "SQL injection lets attackers modify queries."},
        {"text": "EternalBlue exploits SMBv1 (CVE-2017-0144)."},
        {"text": "Cross-site scripting injects script into pages." * 20},
    ]
    jsonl_path = tmp_path / "corpus.jsonl"
    with open(jsonl_path, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

    tokenizer = GhostTokenizer()
    config = GhostLMConfig(context_length=16)

    # Pretokenize via the script's helper.
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
    from pretokenize import pretokenize_file
    bin_path = tmp_path / "corpus.bin"
    total = pretokenize_file(jsonl_path, bin_path, tokenizer)

    jsonl_ds = GhostDataset(str(jsonl_path), tokenizer, config)
    bin_ds = GhostBinDataset(str(bin_path), config)

    assert total == len(jsonl_ds.tokens)
    assert len(bin_ds) == len(jsonl_ds)
    for i in range(len(bin_ds)):
        xj, yj = jsonl_ds[i]
        xb, yb = bin_ds[i]
        assert torch.equal(xj, xb)
        assert torch.equal(yj, yb)

    meta = json.loads((tmp_path / "corpus.meta.json").read_text())
    assert meta["dtype"] == "uint16"
    assert meta["num_tokens"] == total


def test_dataset_inserts_eos_between_records(tmp_path):
    """Every JSONL record must be EOS-terminated in the token stream."""
    from ghostlm.dataset import GhostDataset
    from ghostlm.tokenizer import GhostTokenizer

    jsonl_path = tmp_path / "two.jsonl"
    with open(jsonl_path, "w") as f:
        f.write(json.dumps({"text": "alpha"}) + "\n")
        f.write(json.dumps({"text": "beta"}) + "\n")

    tokenizer = GhostTokenizer()
    ds = GhostDataset(str(jsonl_path), tokenizer, GhostLMConfig(context_length=4))
    eos = tokenizer._special_tokens[tokenizer.EOS]
    assert ds.tokens.count(eos) == 2
    assert ds.tokens[-1] == eos


def test_warmup_lr_applied_on_first_optimizer_step(tmp_path):
    """The first update must run at the warmup LR, not the base LR."""
    from ghostlm.trainer import GhostTrainer

    cfg = _tiny_config()
    cfg.device = "cpu"
    cfg.warmup_steps = 100
    cfg.learning_rate = 3e-4
    cfg.checkpoint_dir = str(tmp_path / "ckpt")
    cfg.log_dir = str(tmp_path / "logs")
    cfg.grad_accum_steps = 1

    model = GhostLM(cfg)
    trainer = GhostTrainer(model, cfg, use_amp=False)

    x = torch.randint(0, 200, (2, 8))
    y = torch.randint(0, 200, (2, 8))
    trainer.train_step((x, y))

    # After the first step the optimizer must have used (and still hold)
    # the step-0 warmup LR: base_lr * 1/warmup.
    expected = 3e-4 * 1 / 100
    for group in trainer.optimizer.param_groups:
        assert group["lr"] == pytest.approx(expected)
