"""Tests for intra-document attention masking (packed-sequence isolation).

The defining property: with the mask on, a token can only attend within its
own document, so changing an earlier packed document must not change the
logits of a later one. With the mask off, it does. These tests prove that
isolation holds and that the feature is inert by default.
"""

import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from ghostlm.config import GhostLMConfig
from ghostlm.model import GhostLM, build_intra_doc_bias

EOS = 7


def _packed_idx():
    # Three documents separated by EOS (=7), padded to length 12.
    # doc A: 1 2 3 EOS | doc B: 4 5 EOS | doc C: 6 8 9 EOS | filler 2
    return torch.tensor([[1, 2, 3, EOS, 4, 5, EOS, 6, 8, 9, EOS, 2]])


def _cfg(intra: bool, flash: bool = False):
    return GhostLMConfig(
        vocab_size=16, n_layers=2, d_model=32, n_heads=4, d_ff=64,
        context_length=12, dropout=0.0, intra_doc_mask=intra,
        eos_token_id=EOS, use_flash_attention=flash,
    )


# ---------- bias construction ----------

def test_bias_is_causal_and_intra_document():
    idx = _packed_idx()
    bias = build_intra_doc_bias(idx, EOS, torch.float32)[0, 0]  # (T, T)
    allowed = torch.isfinite(bias)
    # Position 0 (doc A) may attend only to itself.
    assert allowed[0].tolist() == [True] + [False] * 11
    # Position 4 is doc B's first token (after EOS at 3): may attend to 4 only,
    # NOT back into doc A (0-3).
    assert allowed[4, :4].tolist() == [False, False, False, False]
    assert allowed[4, 4].item() is True
    # Position 9 (doc C) may attend within doc C (7,8,9) but not B or A.
    assert allowed[9, 7:10].tolist() == [True, True, True]
    assert not allowed[9, :7].any()
    # No row is fully masked (diagonal always open) -> no softmax NaN.
    assert allowed.diagonal().all()
    # Strictly causal: never attend to the future.
    assert not torch.triu(allowed, diagonal=1).any()


# ---------- functional isolation ----------

def _logits(cfg, idx):
    torch.manual_seed(0)
    model = GhostLM(cfg).eval()
    with torch.no_grad():
        logits, _ = model(idx)
    return model, logits


def test_earlier_doc_change_does_not_affect_later_doc_when_masked():
    idx = _packed_idx()
    cfg = _cfg(intra=True)
    model, base = _logits(cfg, idx)

    # Perturb only doc A's content tokens (positions 0-2); keep B and C.
    perturbed = idx.clone()
    perturbed[0, 0], perturbed[0, 1], perturbed[0, 2] = 5, 6, 8
    with torch.no_grad():
        new, _ = model(perturbed)

    # doc C positions (7,8,9) must be identical: C cannot attend to A.
    assert torch.allclose(base[0, 7:10], new[0, 7:10], atol=1e-5)
    # doc A positions DID change (sanity: the perturbation actually mattered).
    assert not torch.allclose(base[0, 0:3], new[0, 0:3], atol=1e-4)


def test_without_mask_earlier_doc_change_leaks_into_later_doc():
    idx = _packed_idx()
    cfg = _cfg(intra=False)
    model, base = _logits(cfg, idx)

    perturbed = idx.clone()
    perturbed[0, 0], perturbed[0, 1], perturbed[0, 2] = 5, 6, 8
    with torch.no_grad():
        new, _ = model(perturbed)

    # Without masking, doc C attends back to doc A, so its logits move.
    assert not torch.allclose(base[0, 7:10], new[0, 7:10], atol=1e-4)


def test_flash_path_matches_manual_path_under_mask():
    idx = _packed_idx()
    _, manual = _logits(_cfg(intra=True, flash=False), idx)
    _, flash = _logits(_cfg(intra=True, flash=True), idx)
    assert torch.allclose(manual, flash, atol=1e-4)


def test_default_off_is_inert():
    # intra_doc_mask defaults False: forward must run and not apply isolation.
    cfg = GhostLMConfig(vocab_size=16, n_layers=2, d_model=32, n_heads=4,
                        d_ff=64, context_length=12, dropout=0.0)
    assert cfg.intra_doc_mask is False
    model, base = _logits(cfg, _packed_idx())
    assert base.shape == (1, 12, 16)
