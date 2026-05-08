"""Tests for the v0.9.17 bet 7 expansion (multi-language code-security bank).

Covers:
  - The patterns bank loads without JSON errors.
  - The bank includes the original 12 + 20 new patterns covering 7
    languages (Python / JavaScript / Java / Go / C / Ruby / PHP).
  - Each new pattern has the required fields and non-empty
    vulnerable / patched code.
  - The held-out eval covers the new languages.
"""

import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))


PATTERNS_PATH = REPO_ROOT / "data" / "raw" / "code_security_patterns.jsonl"
EVAL_PATH = REPO_ROOT / "data" / "raw" / "code_security_eval.jsonl"


def _load_jsonl(path):
    out = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


class TestBank:
    def test_bank_loads(self):
        recs = _load_jsonl(PATTERNS_PATH)
        assert len(recs) >= 32, f"expected >= 32 patterns, got {len(recs)}"

    def test_languages_covered(self):
        recs = _load_jsonl(PATTERNS_PATH)
        langs = {r.get("language") for r in recs}
        # Original bank had python/javascript/c. Expansion adds java,
        # go, ruby, php.
        for lang in ("python", "javascript", "java", "go", "c",
                     "ruby", "php"):
            assert lang in langs, f"language missing: {lang}"

    def test_required_fields_present(self):
        recs = _load_jsonl(PATTERNS_PATH)
        for r in recs:
            for field in ("id", "cwe", "name", "language",
                          "vulnerable", "patched", "explanation"):
                assert r.get(field), f"{r.get('id')} missing {field}"
            assert r["vulnerable"] != r["patched"], (
                f"{r['id']} has identical vulnerable + patched code")
            assert len(r["explanation"]) >= 80, (
                f"{r['id']} explanation too short: "
                f"{len(r['explanation'])} chars")

    def test_unique_ids(self):
        recs = _load_jsonl(PATTERNS_PATH)
        ids = [r["id"] for r in recs]
        assert len(ids) == len(set(ids)), (
            f"duplicate IDs in bank: "
            f"{[i for i in ids if ids.count(i) > 1]}")

    def test_new_cwe_classes_added(self):
        """Expansion adds CWE classes the original 12 didn't cover."""
        recs = _load_jsonl(PATTERNS_PATH)
        cwes = {r["cwe"] for r in recs}
        # Sample of CWEs introduced by the expansion.
        for cwe in ("CWE-1321", "CWE-1333", "CWE-134", "CWE-190",
                     "CWE-285", "CWE-326", "CWE-915", "CWE-98"):
            assert cwe in cwes, f"expected CWE {cwe} in expanded bank"


class TestEval:
    def test_eval_loads(self):
        recs = _load_jsonl(EVAL_PATH)
        assert len(recs) >= 32

    def test_eval_record_shape(self):
        recs = _load_jsonl(EVAL_PATH)
        for r in recs:
            assert "prompt" in r
            assert "required_substrings" in r
            assert isinstance(r["required_substrings"], list)
            assert len(r["required_substrings"]) >= 1

    def test_eval_covers_new_languages(self):
        """Eval prompts mention the new languages so the model gets
        scored on cross-language coverage, not just Python."""
        recs = _load_jsonl(EVAL_PATH)
        all_text = " ".join(r["prompt"] for r in recs).lower()
        for term in ("javascript", "java ", "go ", "ruby", "php"):
            assert term in all_text, (
                f"eval set missing prompt mentioning {term!r}")


class TestSynthIntegration:
    def test_synth_runs_on_expanded_bank(self, tmp_path):
        """The existing synth_code_security.py should accept the
        expanded bank and produce a record for every pattern."""
        import subprocess
        out = tmp_path / "synth.jsonl"
        result = subprocess.run(
            [sys.executable, "scripts/synth_code_security.py",
             "--bank", str(PATTERNS_PATH),
             "--out", str(out)],
            cwd=REPO_ROOT, capture_output=True, text=True, timeout=60,
        )
        assert result.returncode == 0, result.stderr
        recs = _load_jsonl(out)
        # 32 patterns * 4 variants = 128, minus a few that the
        # cwe_mapping quality filter rejects. Assert at least 100.
        assert len(recs) >= 100, f"got only {len(recs)} records"
        # Variants should be distributed.
        sources = {r.get("seed_source") for r in recs}
        assert "pretrain_prose" in sources
        assert "identify_and_fix" in sources
