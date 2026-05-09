"""Tests for the v0.9.30 open-source code corpus collector.

Network and git operations are mocked. The fast test path validates:
config schema, license-allowlist filtering, file walker, dedup,
manifest aggregation. A live integration test that actually clones a
tiny real repo is gated behind ``RUN_LIVE_CODE_CORPUS_TEST=1`` so the
default ``pytest`` run stays offline.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

CONFIG = REPO_ROOT / "data" / "code_corpus_repos.json"
SCRIPT = REPO_ROOT / "scripts" / "collect_code_corpus.py"


def _load_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


class TestRepoConfig:
    def test_loads(self):
        repos = _load_json(CONFIG)
        assert isinstance(repos, list)
        assert len(repos) >= 100, f"only {len(repos)} repos configured"

    def test_required_fields(self):
        repos = _load_json(CONFIG)
        for r in repos:
            for k in ("url", "license", "name"):
                assert r.get(k), f"{r.get('name', '?')} missing {k}"
            assert r["url"].startswith("https://github.com/")

    def test_unique_names(self):
        repos = _load_json(CONFIG)
        names = [r["name"] for r in repos]
        assert len(names) == len(set(names)), "duplicate repo names"

    def test_language_diversity(self):
        repos = _load_json(CONFIG)
        langs = {r.get("language") for r in repos if r.get("language")}
        assert len(langs) >= 12, f"only {len(langs)} languages: {langs}"
        for must in ("python", "go", "rust", "javascript", "c", "cpp",
                     "java", "ruby"):
            assert must in langs, f"missing {must}"

    def test_permissive_majority(self):
        """Permissively-licensed repos should dominate the config."""
        repos = _load_json(CONFIG)
        permissive = {
            "MIT", "MIT-0", "MIT-CMU", "Apache-2.0",
            "BSD-2-Clause", "BSD-3-Clause", "ISC", "MPL-2.0",
            "PSF-2.0", "Unlicense", "CC0-1.0", "Zlib",
            "blessing", "PostgreSQL",
        }
        n_perm = sum(1 for r in repos if r["license"] in permissive)
        assert n_perm / len(repos) >= 0.95, (
            f"only {n_perm}/{len(repos)} repos under a permissive license"
        )


class TestDryRun:
    def test_dry_run_lists_repos(self):
        result = subprocess.run(
            [sys.executable, str(SCRIPT), "--dry-run"],
            cwd=REPO_ROOT, capture_output=True, text=True, timeout=30,
        )
        assert result.returncode == 0, result.stderr
        # Should list each repo (sample a few).
        assert "cpython" in result.stdout
        assert "tokio" in result.stdout
        assert "kubernetes" in result.stdout
        # Should print the language breakdown summary.
        assert "Language breakdown" in result.stdout

    def test_dry_run_license_filter_skips_gpl(self):
        result = subprocess.run(
            [sys.executable, str(SCRIPT), "--dry-run"],
            cwd=REPO_ROOT, capture_output=True, text=True, timeout=30,
        )
        assert result.returncode == 0, result.stderr
        assert "License filter" in result.stdout
        # Default allowlist excludes GPL/LGPL — should not appear.
        assert "git/git" not in result.stdout
        assert "sidekiq" not in result.stdout

    def test_dry_run_all_includes_gpl(self):
        result = subprocess.run(
            [sys.executable, str(SCRIPT),
             "--dry-run", "--license-allowlist", "all"],
            cwd=REPO_ROOT, capture_output=True, text=True, timeout=30,
        )
        assert result.returncode == 0, result.stderr
        assert "git" in result.stdout
        assert "sidekiq" in result.stdout

    def test_only_language_filter(self):
        result = subprocess.run(
            [sys.executable, str(SCRIPT),
             "--dry-run", "--only-language", "rust"],
            cwd=REPO_ROOT, capture_output=True, text=True, timeout=30,
        )
        assert result.returncode == 0, result.stderr
        assert "tokio" in result.stdout
        assert "cpython" not in result.stdout


class TestModuleAPIs:
    """Direct unit tests against the module's helper functions."""

    def test_imports(self):
        sys.path.insert(0, str(REPO_ROOT / "scripts"))
        try:
            mod = __import__("collect_code_corpus")
        finally:
            sys.path.pop(0)
        assert hasattr(mod, "EXT_TO_LANG")
        assert hasattr(mod, "SKIP_DIRS")
        assert hasattr(mod, "DEFAULT_LICENSE_ALLOWLIST")
        assert hasattr(mod, "walk_source_files")
        assert hasattr(mod, "hash_text")
        assert hasattr(mod, "compute_totals")

    def test_walk_excludes_skip_dirs(self, tmp_path):
        sys.path.insert(0, str(REPO_ROOT / "scripts"))
        try:
            mod = __import__("collect_code_corpus")
        finally:
            sys.path.pop(0)
        (tmp_path / "src").mkdir()
        (tmp_path / "src" / "main.py").write_text("print('x')\n")
        (tmp_path / "node_modules").mkdir()
        (tmp_path / "node_modules" / "junk.py").write_text("noise\n")
        (tmp_path / "build").mkdir()
        (tmp_path / "build" / "out.js").write_text("bundle\n")
        files = mod.walk_source_files(tmp_path, {".py", ".js"})
        names = [f.name for f in files]
        assert "main.py" in names
        assert "junk.py" not in names
        assert "out.js" not in names

    def test_walk_skips_lockfiles_and_minified(self, tmp_path):
        sys.path.insert(0, str(REPO_ROOT / "scripts"))
        try:
            mod = __import__("collect_code_corpus")
        finally:
            sys.path.pop(0)
        (tmp_path / "real.js").write_text("var x;")
        (tmp_path / "lib.min.js").write_text("var x;")
        (tmp_path / "package-lock.json").write_text("{}")
        (tmp_path / "main_test.go").write_text("package main")
        (tmp_path / "main.go").write_text("package main")
        files = mod.walk_source_files(tmp_path, {".js", ".json", ".go"})
        names = [f.name for f in files]
        assert "real.js" in names
        assert "main.go" in names
        assert "lib.min.js" not in names
        assert "package-lock.json" not in names
        assert "main_test.go" not in names

    def test_hash_text_stable(self):
        sys.path.insert(0, str(REPO_ROOT / "scripts"))
        try:
            mod = __import__("collect_code_corpus")
        finally:
            sys.path.pop(0)
        a = mod.hash_text("hello world\n")
        b = mod.hash_text("hello world\n")
        c = mod.hash_text("hello world!\n")
        assert a == b
        assert a != c

    def test_compute_totals_aggregates(self):
        sys.path.insert(0, str(REPO_ROOT / "scripts"))
        try:
            mod = __import__("collect_code_corpus")
        finally:
            sys.path.pop(0)
        sources = [
            {"name": "a", "license": "MIT", "language": "python",
             "files_written": 10, "chars_written": 1000,
             "duplicates_skipped": 1, "status": "ok"},
            {"name": "b", "license": "MIT", "language": "python",
             "files_written": 5, "chars_written": 500,
             "duplicates_skipped": 0, "status": "ok"},
            {"name": "c", "license": "Apache-2.0", "language": "go",
             "files_written": 8, "chars_written": 800,
             "duplicates_skipped": 2, "status": "ok"},
            {"name": "d", "license": "MIT", "language": "python",
             "files_written": 0, "chars_written": 0,
             "duplicates_skipped": 0, "status": "clone_failed"},
        ]
        totals = mod.compute_totals(sources)
        assert totals["repos"] == 3
        assert totals["files"] == 23
        assert totals["chars"] == 2300
        assert totals["duplicates"] == 3
        assert totals["by_language"]["python"] == 15
        assert totals["by_language"]["go"] == 8
        assert totals["by_license"]["MIT"] == 15
        assert totals["by_license"]["Apache-2.0"] == 8

    def test_parse_license_allowlist_all(self):
        sys.path.insert(0, str(REPO_ROOT / "scripts"))
        try:
            mod = __import__("collect_code_corpus")
        finally:
            sys.path.pop(0)
        assert mod.parse_license_allowlist("all") is None
        assert mod.parse_license_allowlist("ALL") is None

    def test_parse_license_allowlist_specific(self):
        sys.path.insert(0, str(REPO_ROOT / "scripts"))
        try:
            mod = __import__("collect_code_corpus")
        finally:
            sys.path.pop(0)
        allow = mod.parse_license_allowlist("MIT, Apache-2.0 ,BSD-3-Clause")
        assert allow == {"MIT", "Apache-2.0", "BSD-3-Clause"}


@pytest.mark.skipif(
    os.environ.get("RUN_LIVE_CODE_CORPUS_TEST") != "1",
    reason="Live network test — set RUN_LIVE_CODE_CORPUS_TEST=1 to enable",
)
class TestLiveSmoke:
    def test_collect_one_tiny_repo(self, tmp_path):
        """Clone a tiny permissively-licensed repo and validate the pipeline."""
        cfg = tmp_path / "tiny.json"
        cfg.write_text(json.dumps([
            {
                "url": "https://github.com/pallets/click",
                "license": "BSD-3-Clause",
                "name": "click-tiny",
                "subdir": "src/click",
                "language": "python",
            },
        ]))
        out = tmp_path / "out.jsonl"
        manifest = tmp_path / "manifest.json"
        result = subprocess.run(
            [sys.executable, str(SCRIPT),
             "--config", str(cfg),
             "--output", str(out),
             "--manifest", str(manifest),
             "--max-files-per-repo", "20"],
            cwd=REPO_ROOT, capture_output=True, text=True, timeout=600,
        )
        assert result.returncode == 0, result.stderr
        assert out.exists()
        recs = []
        with out.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    recs.append(json.loads(line))
        assert len(recs) >= 5, f"only {len(recs)} records"
        for r in recs:
            assert r["source"] == "code_corpus"
            assert r["language"] == "python"
            assert r["license"] == "BSD-3-Clause"
            assert r["text"]
        m = _load_json(manifest)
        assert m["totals"]["repos"] == 1
        assert m["totals"]["files"] >= 5
