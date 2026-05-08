"""Tests for the v0.9.22 bet 8 binary-literacy expansion.

Bank size, category coverage, eval coverage, synth integration.
"""

import json
import sys
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

PATTERNS_PATH = REPO_ROOT / "data" / "raw" / "binary_literacy_patterns.jsonl"
EVAL_PATH = REPO_ROOT / "data" / "raw" / "binary_literacy_eval.jsonl"


def _load_jsonl(path):
    out = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


class TestBank:
    def test_bank_size(self):
        recs = _load_jsonl(PATTERNS_PATH)
        assert len(recs) >= 38, f"expected >= 38 patterns, got {len(recs)}"

    def test_categories_covered(self):
        """v0.9.5 had file_magic / packer / shellcode / pe_field /
        disassembly. v0.9.22 adds elf_field / encoding / hash."""
        recs = _load_jsonl(PATTERNS_PATH)
        cats = {r.get("category") for r in recs}
        for c in ("file_magic", "shellcode", "disassembly",
                   "pe_field", "elf_field", "encoding", "hash"):
            assert c in cats, f"missing category: {c}"

    def test_required_fields(self):
        recs = _load_jsonl(PATTERNS_PATH)
        for r in recs:
            for f in ("id", "category", "name", "explanation"):
                assert r.get(f), f"{r.get('id')} missing {f}"
            assert len(r["explanation"]) >= 80

    def test_unique_ids(self):
        recs = _load_jsonl(PATTERNS_PATH)
        ids = [r["id"] for r in recs]
        assert len(ids) == len(set(ids)), "duplicate IDs"

    def test_new_file_magics_present(self):
        """v0.9.22 adds JPEG / GIF / MP4 / Java class / WASM /
        GZIP / SQLite / DEX file magics."""
        recs = _load_jsonl(PATTERNS_PATH)
        names_lower = " ".join(r["name"].lower() for r in recs)
        for term in ("jpeg", "gif", "mp4", "java class",
                     "webassembly", "gzip", "sqlite", "dex"):
            assert term in names_lower, f"missing pattern for: {term}"

    def test_new_disassembly_patterns_present(self):
        recs = _load_jsonl(PATTERNS_PATH)
        names_lower = " ".join(r["name"].lower() for r in recs)
        for term in ("syscall", "indirect call",
                     "rop gadget", "function epilogue",
                     "arm64"):
            assert term in names_lower, f"missing pattern for: {term}"

    def test_hash_recognition_patterns_present(self):
        """Hash format recognition: MD5 / SHA-256 / bcrypt at minimum."""
        recs = _load_jsonl(PATTERNS_PATH)
        hash_recs = [r for r in recs if r.get("category") == "hash"]
        names = " ".join(r["name"].lower() for r in hash_recs)
        for term in ("md5", "sha-256", "bcrypt"):
            assert term in names, f"missing hash pattern: {term}"


class TestEval:
    def test_eval_size(self):
        recs = _load_jsonl(EVAL_PATH)
        assert len(recs) >= 33

    def test_eval_record_shape(self):
        recs = _load_jsonl(EVAL_PATH)
        for r in recs:
            assert "prompt" in r
            assert isinstance(r.get("required_substrings"), list)

    def test_eval_covers_new_categories(self):
        """Eval prompts + their required-substrings (the answer
        terms) should collectively cover the new categories."""
        recs = _load_jsonl(EVAL_PATH)
        all_text = " ".join(
            r["prompt"] + " " + " ".join(r.get("required_substrings", []))
            for r in recs
        ).lower()
        for term in ("webassembly", "sqlite", "base64", "bcrypt",
                     "syscall", "arm64", "wasm", "sha-256"):
            assert term in all_text, (
                f"eval set missing prompt or required substring "
                f"matching {term!r}")


class TestSynthIntegration:
    def test_synth_runs_on_expanded_bank(self, tmp_path):
        out = tmp_path / "synth.jsonl"
        result = subprocess.run(
            [sys.executable, "scripts/synth_binary_literacy.py",
             "--bank", str(PATTERNS_PATH),
             "--out", str(out)],
            cwd=REPO_ROOT, capture_output=True, text=True, timeout=60,
        )
        assert result.returncode == 0, result.stderr
        recs = _load_jsonl(out)
        # 40 patterns producing 2-3 variants each = ~100 records.
        assert len(recs) >= 95, f"got only {len(recs)} records"
