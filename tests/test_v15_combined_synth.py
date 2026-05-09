"""Tests for build_v15_combined_synth.py categorisation + file-list integrity.

After v0.9.32 the combined-synth wiring grew to include code_explain and
code_write (the two general-code templated bets that surpassed cybersec
SFT scale). Lock the CATEGORY_RULES + SYNTH_FILES in so the next bet
addition shows up here too.
"""

import importlib.util
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "build_v15", str(REPO_ROOT / "scripts" / "build_v15_combined_synth.py"),
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_synth_files_complete():
    m = _load_module()
    expected = {
        "synth_format_aware.jsonl",
        "synth_tool_use.jsonl",
        "synth_tool_use_provenance.jsonl",
        "synth_code_security.jsonl",
        "synth_binary_literacy.jsonl",
        "synth_log_analysis.jsonl",
        "synth_iac_security.jsonl",
        "synth_protocol_fields.jsonl",
        "synth_code_explain.jsonl",
        "synth_code_write.jsonl",
    }
    assert set(m.SYNTH_FILES) == expected


def test_category_rules_cover_every_synth_file():
    m = _load_module()
    rule_sources = {k[0] for k in m.CATEGORY_RULES.keys()}
    file_sources = {f.removeprefix("synth_").removesuffix(".jsonl")
                    for f in m.SYNTH_FILES}
    file_sources = {f"synth_{s}" for s in file_sources}
    missing = file_sources - rule_sources
    assert not missing, f"sources without categorisation rules: {missing}"


def test_code_explain_variants():
    m = _load_module()
    expected = {"pretrain_prose", "identify_lang", "explain_purpose",
                "walkthrough", "concepts"}
    actual = {k[1] for k in m.CATEGORY_RULES if k[0] == "synth_code_explain"}
    assert actual == expected


def test_code_write_variants():
    m = _load_module()
    expected = {"pretrain_prose", "write_function", "write_idiomatic",
                "compare"}
    actual = {k[1] for k in m.CATEGORY_RULES if k[0] == "synth_code_write"}
    assert actual == expected


def test_pretrain_prose_always_pretrain():
    """Every templated bet's `pretrain_prose` variant must be tagged pretrain."""
    m = _load_module()
    for (source, variant), tag in m.CATEGORY_RULES.items():
        if variant == "pretrain_prose":
            assert tag == "pretrain", f"{source}:{variant} = {tag}"


def test_known_sft_qa_variants_are_sft():
    """Spot-check that the new code-explain/code-write Q&A variants are SFT."""
    m = _load_module()
    sft_must = [
        ("synth_code_explain", "identify_lang"),
        ("synth_code_explain", "walkthrough"),
        ("synth_code_write", "write_function"),
        ("synth_code_write", "compare"),
    ]
    for key in sft_must:
        assert m.CATEGORY_RULES[key] == "sft", key
