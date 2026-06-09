"""Tests for the six differentiation-bet artifacts.

Covers:
  - bet 3: v1 BPE backend (GhostTokenizerV1) + load_tokenizer factory routing
  - bet 5: MoE aux loss wired into GhostLM.forward + ghost-1b / ghost-3b
           presets construct cleanly and report MoE-aware sizes
  - bet 6: format validators (parse_stix, parse_yara, parse_sigma,
           parse_misp) accept canonical examples and reject obvious
           bad ones

The expensive bits (running an LLM teacher, instantiating ghost-3b at
its real 6B parameter count) are out of scope; those need
GPU/budget. These tests are M4-cheap and run in CI with the rest of
the suite.
"""

from pathlib import Path

import pytest
import torch

from ghostlm.config import GhostLMConfig
from ghostlm.model import GhostLM, SparseMoE
from ghostlm.tokenizer import (
    GhostTokenizer, GhostTokenizerV1, load_tokenizer,
)

REPO_ROOT = Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------------------
# Bet 3: v1 BPE backend
# ---------------------------------------------------------------------------


V1_PATH = REPO_ROOT / "data" / "tokenizer" / "v1" / "tokenizer.json"


@pytest.mark.skipif(not V1_PATH.exists(), reason="v1 BPE artifact not present")
def test_v1_tokenizer_loads_with_full_special_set():
    """V1 backend exposes all 11 special tokens (4 base + 3 chat + 4 tool)."""
    tok = GhostTokenizerV1(str(V1_PATH))
    assert tok.vocab_size == 32000
    assert len(tok._special_tokens) == 11
    expected = {
        tok.BOS, tok.EOS, tok.PAD, tok.UNK,
        tok.USER, tok.ASSISTANT, tok.END,
        tok.TOOL_CALL, tok.TOOL_CALL_END,
        tok.TOOL_RESPONSE, tok.TOOL_RESPONSE_END,
    }
    assert expected == set(tok._special_tokens.keys())


@pytest.mark.skipif(not V1_PATH.exists(), reason="v1 BPE artifact not present")
def test_v1_tool_tags_are_atomic():
    """Tool-use tags must tokenize to a single id each so SFT loss
    masks (bet 1) align cleanly across tokenizer choices."""
    tok = GhostTokenizerV1(str(V1_PATH))
    for name in (
        tok.TOOL_CALL, tok.TOOL_CALL_END,
        tok.TOOL_RESPONSE, tok.TOOL_RESPONSE_END,
    ):
        ids = tok._tok.encode(name).ids
        assert len(ids) == 1, f"{name} did not tokenize to one id: {ids}"


@pytest.mark.skipif(not V1_PATH.exists(), reason="v1 BPE artifact not present")
def test_v1_round_trip_preserves_text():
    """encode then decode must round-trip on cybersec-typical text."""
    tok = GhostTokenizerV1(str(V1_PATH))
    text = "CVE-2017-0144 is the EternalBlue SMB exploit"
    ids = tok.encode(text)
    decoded = tok.decode(ids)
    assert text in decoded


@pytest.mark.skipif(not V1_PATH.exists(), reason="v1 BPE artifact not present")
def test_load_tokenizer_factory_routes_to_v1():
    """Factory must pick V1 when sibling special_tokens_map.json
    contains the four tool-use tags."""
    tok = load_tokenizer(str(V1_PATH))
    assert isinstance(tok, GhostTokenizerV1)


def test_load_tokenizer_factory_falls_back_to_legacy():
    """No path: falls back to GPT-2 legacy backend."""
    tok = load_tokenizer()
    assert isinstance(tok, GhostTokenizer)


# ---------------------------------------------------------------------------
# Bet 5: MoE aux loss wiring + ghost-1b/3b presets
# ---------------------------------------------------------------------------


def _tiny_moe_config() -> GhostLMConfig:
    """Smallest MoE config that exercises the wiring without burning RAM."""
    return GhostLMConfig(
        vocab_size=128, context_length=16, d_model=32, n_heads=4,
        n_layers=2, d_ff=64,
        use_rope=True, use_swiglu=True, use_rmsnorm=True,
        use_moe=True, n_experts=4, n_experts_active=2,
        moe_aux_loss_coef=0.01,
    )


def test_moe_layers_constructed_in_blocks():
    """With use_moe=True every TransformerBlock's FFN is a SparseMoE."""
    cfg = _tiny_moe_config()
    model = GhostLM(cfg)
    moe_layers = [m for m in model.modules() if isinstance(m, SparseMoE)]
    assert len(moe_layers) == cfg.n_layers


def test_moe_aux_loss_added_to_total_loss():
    """When targets are supplied, total loss must equal CE + coef*sum(aux)."""
    cfg = _tiny_moe_config()
    model = GhostLM(cfg)
    x = torch.randint(0, cfg.vocab_size, (2, cfg.context_length))
    y = torch.randint(0, cfg.vocab_size, (2, cfg.context_length))

    logits, total_loss = model(x, targets=y)
    # Recompute CE without aux for the comparison.
    import torch.nn.functional as F
    ce = F.cross_entropy(
        logits.view(-1, cfg.vocab_size), y.view(-1), ignore_index=-1,
    )
    aux = sum(
        m.last_aux_loss for m in model.modules() if isinstance(m, SparseMoE)
    )
    expected = ce + cfg.moe_aux_loss_coef * aux

    assert torch.allclose(total_loss, expected, atol=1e-5)


def test_moe_no_aux_when_targets_omitted():
    """Without targets the model returns logits-only and skips aux."""
    cfg = _tiny_moe_config()
    model = GhostLM(cfg)
    x = torch.randint(0, cfg.vocab_size, (1, cfg.context_length))
    logits, loss = model(x)
    assert loss is None


def test_moe_gate_weights_receive_gradient():
    """Backward over the total loss must populate gradients on every
    SparseMoE.gate weight, otherwise the router never learns."""
    cfg = _tiny_moe_config()
    model = GhostLM(cfg)
    x = torch.randint(0, cfg.vocab_size, (2, cfg.context_length))
    y = torch.randint(0, cfg.vocab_size, (2, cfg.context_length))
    _, loss = model(x, targets=y)
    loss.backward()
    for m in model.modules():
        if isinstance(m, SparseMoE):
            assert m.gate.weight.grad is not None
            assert m.gate.weight.grad.norm().item() > 0


def test_moe_disabled_path_unchanged():
    """With use_moe=False the model contains no SparseMoE layers and
    forward(targets=...) returns plain CE."""
    cfg = GhostLMConfig.from_preset("ghost-tiny")
    cfg.vocab_size = 128
    cfg.context_length = 16
    model = GhostLM(cfg)
    moe_layers = [m for m in model.modules() if isinstance(m, SparseMoE)]
    assert moe_layers == []


def test_ghost_1b_preset_shape():
    """ghost-1b preset must report the expected MoE shape and ~2.1B size."""
    cfg = GhostLMConfig.from_preset("ghost-1b")
    assert cfg.use_moe is True
    assert cfg.n_experts == 4
    assert cfg.n_experts_active == 2
    assert cfg.d_model == 1536
    assert cfg.n_layers == 24
    # head_dim must divide cleanly
    assert cfg.d_model % cfg.n_heads == 0
    size = cfg.model_size()
    assert size.endswith("B")
    # lossy parse: pull the leading float
    val = float(size.rstrip("B"))
    assert 1.5 <= val <= 2.5  # 2.1B target with some slack


def test_ghost_3b_preset_shape():
    """ghost-3b preset must report the expected MoE shape and ~6B size."""
    cfg = GhostLMConfig.from_preset("ghost-3b")
    assert cfg.use_moe is True
    assert cfg.n_experts == 4
    assert cfg.d_model == 2048
    assert cfg.n_layers == 32
    assert cfg.d_model % cfg.n_heads == 0
    size = cfg.model_size()
    assert size.endswith("B")
    val = float(size.rstrip("B"))
    assert 5.0 <= val <= 7.0  # 6B target with slack


def test_legacy_presets_unchanged():
    """Adding ghost-1b/3b must not perturb the existing presets."""
    for name in ("ghost-tiny", "ghost-small", "ghost-medium", "ghost-small-v0.5"):
        cfg = GhostLMConfig.from_preset(name)
        assert cfg.use_moe is False


# ---------------------------------------------------------------------------
# Bet 6: format validators
# ---------------------------------------------------------------------------


def _import_format_validators():
    """Lazy import so the test file can still load when distill_format_aware
    deps are missing in some environment."""
    import sys
    sys.path.insert(0, str(REPO_ROOT))
    from scripts.distill_format_aware import (
        parse_stix, parse_yara, parse_sigma, parse_misp,
    )
    return parse_stix, parse_yara, parse_sigma, parse_misp


def test_parse_stix_accepts_valid_indicator():
    parse_stix, _, _, _ = _import_format_validators()
    valid = (
        '{"type": "indicator", "spec_version": "2.1", '
        '"id": "indicator--abc12345-1234-1234-1234-123456789abc", '
        '"created": "2026-05-08T00:00:00Z", '
        '"modified": "2026-05-08T00:00:00Z", '
        '"pattern_type": "stix", '
        '"pattern": "[file:hashes.SHA-256 = \'abc\']", '
        '"valid_from": "2026-05-08T00:00:00Z", '
        '"labels": ["malicious-activity"], "name": "x"}'
    )
    assert parse_stix(valid) is not None


def test_parse_stix_rejects_wrong_spec_version():
    parse_stix, _, _, _ = _import_format_validators()
    bad = (
        '{"type": "indicator", "spec_version": "2.0", '
        '"id": "x", "created": "x", "modified": "x"}'
    )
    assert parse_stix(bad) is None


def test_parse_stix_rejects_garbage():
    parse_stix, _, _, _ = _import_format_validators()
    assert parse_stix("not even json") is None
    assert parse_stix("") is None


def test_parse_yara_accepts_valid_rule():
    _, parse_yara, _, _ = _import_format_validators()
    valid = (
        "rule x {\n"
        "  meta:\n    author = \"a\"\n"
        "  strings:\n    $s1 = \"foo\"\n"
        "  condition:\n    $s1\n"
        "}\n"
    )
    assert parse_yara(valid) is not None


def test_parse_yara_rejects_missing_condition():
    _, parse_yara, _, _ = _import_format_validators()
    bad = "rule x { strings: $s1 = \"x\" foo: $s1 }"
    assert parse_yara(bad) is None


def test_parse_sigma_regex_fallback():
    _, _, parse_sigma, _ = _import_format_validators()
    # The required-fields regex fallback path; runs even without PyYAML.
    valid = (
        "title: Detect X\n"
        "logsource:\n  category: process_creation\n"
        "detection:\n  selection:\n    Image: foo\n  condition: selection\n"
    )
    assert parse_sigma(valid) is not None


def test_parse_sigma_rejects_no_detection():
    _, _, parse_sigma, _ = _import_format_validators()
    assert parse_sigma("title: foo\nlogsource: bar\n") is None


def test_parse_misp_accepts_valid_event():
    _, _, _, parse_misp = _import_format_validators()
    valid = (
        '{"Event": {"info": "x", "Attribute": ['
        '{"type": "ip-dst", "value": "1.2.3.4", "category": "Network"}]}}'
    )
    assert parse_misp(valid) is not None


def test_parse_misp_rejects_empty_attribute_array():
    _, _, _, parse_misp = _import_format_validators()
    assert parse_misp('{"Event": {"info": "x", "Attribute": []}}') is None


def test_parse_misp_rejects_missing_event_key():
    _, _, _, parse_misp = _import_format_validators()
    assert parse_misp('{"info": "x"}') is None
