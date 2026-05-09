#!/usr/bin/env python3
"""Collect a broad open-source code corpus for GhostLM pretrain.

The v0.9 pretrain corpus is heavily cybersec-skewed: of the 363M tokens,
code is around 2.4% (mostly the security_code collector's pull from
~30 cybersec-tool repos). For ghost-base to be a competent code-aware
LM the pretrain mix needs a much broader code share. This collector
expands beyond cybersec tooling to general open-source code across
the major language ecosystems.

Sources are configured in ``data/code_corpus_repos.json`` as an array
of ``{url, license, name, language, [subdir], [branch], [category]}``
entries. The default config covers ~110 permissively-licensed repos:
Python (cpython stdlib + numpy/scipy/pandas + Flask/Django/FastAPI +
ML stack), Go (stdlib + gin/cobra/k8s/terraform/docker), Rust (std +
tokio/serde/clap/cargo/ripgrep/ruff/uv), JS/TS (express/node/react/
vue/typescript/vite/nestjs), C/C++ (redis/sqlite/curl/postgres/grpc/
abseil), Java/Kotlin/Scala (spring/guava/kafka/kotlin), Ruby (rails/
sinatra), Elixir/Erlang/Zig/Swift.

Licensing is enforced by an allowlist (``--license-allowlist``); the
default permits MIT / Apache-2.0 / BSD-2/3-Clause / ISC / MPL-2.0 /
PSF-2.0 / Unlicense / CC0 / blessing (sqlite) / MIT-0 / MIT-CMU /
PostgreSQL. GPL/LGPL/AGPL repos in the config are skipped by default
(set ``--license-allowlist all`` to include them anyway).

Filters baked in:
- Skip vendored dirs (node_modules/, vendor/, third_party/, target/,
  __pycache__/, .venv/, build/, dist/, .git/).
- Skip lockfiles, minified bundles, generated bindings, large test
  fixtures (``*_test.go``, ``*.test.ts``, ``*.spec.js``, etc).
- Drop files under ``--min-chars`` (boilerplate __init__ / mod.rs).
- Truncate files over ``--max-chars`` (predictable training cost).
- Per-language file cap (``--per-language-cap``) so any one language
  can't dominate.
- Per-repo file cap (``--max-files-per-repo``) so any one mega-repo
  can't dominate.
- Content-hash deduplication within the run (sha256 of stripped text).

Output: ``data/raw/code_corpus.jsonl`` with the standard
``{"id", "source", "text"}`` schema plus ``language``, ``license``,
``repo``, ``path``, ``category``, ``truncated``. ``source`` is
``code_corpus``. A sidecar manifest at
``data/raw/code_corpus_manifest.json`` records per-source totals
(license, file count, char count, repo URL) for downstream audit and
the CORPUS.md update.

Designed for the Mac (long-running task per the project conventions).
A full pull touches dozens of GB of git history; expect ~60-120 min
on M-series with a fast pipe. ``--append`` lets you resume after an
interrupt or re-run with a config diff.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
import time
from collections import defaultdict
from pathlib import Path


EXT_TO_LANG = {
    ".py": "python", ".pyx": "python", ".pyi": "python",
    ".c": "c", ".h": "c",
    ".cpp": "cpp", ".cc": "cpp", ".cxx": "cpp",
    ".hpp": "cpp", ".hh": "cpp", ".hxx": "cpp",
    ".js": "javascript", ".mjs": "javascript", ".cjs": "javascript",
    ".jsx": "javascript",
    ".ts": "typescript", ".tsx": "typescript",
    ".go": "go",
    ".rs": "rust",
    ".java": "java",
    ".kt": "kotlin", ".kts": "kotlin",
    ".scala": "scala",
    ".rb": "ruby",
    ".php": "php",
    ".cs": "csharp",
    ".swift": "swift",
    ".m": "objc", ".mm": "objcpp",
    ".sh": "shell", ".bash": "shell", ".zsh": "shell",
    ".pl": "perl",
    ".lua": "lua",
    ".ex": "elixir", ".exs": "elixir",
    ".erl": "erlang", ".hrl": "erlang",
    ".zig": "zig",
    ".hs": "haskell",
    ".clj": "clojure", ".cljs": "clojure",
    ".sql": "sql",
    ".dart": "dart",
    ".jl": "julia",
}

SKIP_DIRS = {
    "node_modules", "vendor", "third_party", "third-party",
    "__pycache__", ".venv", "venv", "env", ".env", ".git",
    "build", "dist", "target", ".tox", ".pytest_cache",
    ".mypy_cache", "site-packages", ".eggs", "egg-info",
    ".gradle", ".idea", ".vscode", ".cache", "out", "_build",
    "bazel-bin", "bazel-out", "bazel-testlogs", "bazel-genfiles",
    "Pods", "DerivedData",
}

SKIP_FILE_PATTERNS = (
    ".min.js", ".min.css", ".min.mjs",
    ".bundle.js", ".bundle.css",
    ".test.ts", ".test.js", ".test.tsx", ".test.jsx",
    ".spec.ts", ".spec.js",
    "_test.go", "_mock.go",
    ".pb.go", ".pb.cc", ".pb.h",
    "_pb2.py", "_pb2_grpc.py",
    ".generated.ts", ".generated.js",
    ".d.ts.map", ".js.map", ".css.map",
    "package-lock.json", "yarn.lock", "pnpm-lock.yaml",
    "Cargo.lock", "Pipfile.lock", "poetry.lock", "go.sum",
    "Gemfile.lock", "composer.lock",
)

DEFAULT_LICENSE_ALLOWLIST = {
    "MIT", "MIT-0", "MIT-CMU",
    "Apache-2.0",
    "BSD-2-Clause", "BSD-3-Clause", "BSD-3-Clause-Clear",
    "ISC",
    "MPL-2.0",
    "PSF-2.0",
    "Unlicense",
    "CC0-1.0",
    "Zlib",
    "blessing",
    "PostgreSQL",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Collect open-source code into a JSONL pretrain corpus",
    )
    p.add_argument(
        "--config",
        default="data/code_corpus_repos.json",
        help="JSON array of {url, license, name, language, [subdir], [branch], [category]}",
    )
    p.add_argument(
        "--output",
        default="data/raw/code_corpus.jsonl",
        help="JSONL output path",
    )
    p.add_argument(
        "--manifest",
        default="data/raw/code_corpus_manifest.json",
        help="Sidecar manifest with per-source totals",
    )
    p.add_argument("--min-chars", type=int, default=200)
    p.add_argument("--max-chars", type=int, default=15_000)
    p.add_argument(
        "--max-files-per-repo", type=int, default=600,
        help="Cap files per repo to keep distribution balanced",
    )
    p.add_argument(
        "--per-language-cap", type=int, default=0,
        help="Cap total files per language across all repos (0 = no cap)",
    )
    p.add_argument(
        "--license-allowlist",
        default=",".join(sorted(DEFAULT_LICENSE_ALLOWLIST)),
        help="Comma-separated SPDX list, or 'all' to skip license filtering",
    )
    p.add_argument(
        "--exts", nargs="+", default=list(EXT_TO_LANG.keys()),
        help="File extensions to include (with leading dot)",
    )
    p.add_argument(
        "--append", action="store_true",
        help="Append to existing output and skip repos already in the manifest",
    )
    p.add_argument(
        "--only-language", default=None,
        help="Restrict to one language label (matches config 'language' field)",
    )
    p.add_argument(
        "--only-repo", default=None,
        help="Restrict to one repo name (matches config 'name' field). Useful for resume.",
    )
    p.add_argument(
        "--dry-run", action="store_true",
        help="Print plan without cloning anything",
    )
    return p.parse_args()


def load_config(path: Path) -> list[dict]:
    if not path.exists():
        sys.exit(f"config not found: {path}")
    repos = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(repos, list):
        sys.exit("config must be a JSON array of {url, license, ...} entries")
    for r in repos:
        for k in ("url", "license", "name"):
            if k not in r:
                sys.exit(f"entry missing {k!r}: {r!r}")
    return repos


def parse_license_allowlist(spec: str) -> set[str] | None:
    spec = spec.strip()
    if spec.lower() == "all":
        return None
    allow = {s.strip() for s in spec.split(",") if s.strip()}
    return allow


def shallow_clone(url: str, dest: Path, branch: str | None = None) -> bool:
    cmd = ["git", "clone", "--depth", "1", "--filter=blob:none"]
    if branch:
        cmd += ["--branch", branch]
    cmd += [url, str(dest)]
    try:
        subprocess.run(cmd, check=True, capture_output=True, timeout=900)
        return True
    except subprocess.CalledProcessError as e:
        err = e.stderr.decode()[:300] if e.stderr else "unknown"
        print(f"  clone failed: {err}")
        return False
    except subprocess.TimeoutExpired:
        print(f"  clone timeout (>15min): {url}")
        return False


def walk_source_files(root: Path, ext_set: set[str]) -> list[Path]:
    out: list[Path] = []
    for p in root.rglob("*"):
        if not p.is_file():
            continue
        if any(part in SKIP_DIRS for part in p.parts):
            continue
        if any(p.name.endswith(pat) for pat in SKIP_FILE_PATTERNS):
            continue
        if p.suffix.lower() not in ext_set:
            continue
        out.append(p)
    return out


def hash_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="ignore")).hexdigest()


def process_repo(
    entry: dict,
    out_fh,
    args: argparse.Namespace,
    repo_idx: int,
    total: int,
    seen_hashes: set[str],
    lang_counts: dict[str, int],
) -> dict:
    """Clone one repo, walk sources, write records. Return per-source manifest."""
    url = entry["url"]
    license_spdx = entry["license"]
    name = entry["name"]
    branch = entry.get("branch")
    subdir = entry.get("subdir")
    category = entry.get("category", "lib")
    cfg_lang = entry.get("language")
    print(f"\n[{repo_idx}/{total}] {name} ({license_spdx}) -> {url}")

    started = time.monotonic()
    summary = {
        "name": name, "url": url, "license": license_spdx,
        "language": cfg_lang, "category": category,
        "files_written": 0, "files_skipped": 0,
        "chars_written": 0, "duplicates_skipped": 0,
        "elapsed_sec": 0.0, "status": "ok",
    }

    with tempfile.TemporaryDirectory(prefix=f"code_corpus_{name}_") as tmp:
        tmp_path = Path(tmp)
        clone_root = tmp_path / name
        if not shallow_clone(url, clone_root, branch=branch):
            summary["status"] = "clone_failed"
            return summary
        walk_root = clone_root / subdir if subdir else clone_root
        if subdir and not walk_root.is_dir():
            print(f"  subdir not found: {subdir}")
            summary["status"] = "subdir_missing"
            return summary

        ext_set = {e.lower() for e in args.exts}
        files = walk_source_files(walk_root, ext_set)
        if len(files) > args.max_files_per_repo:
            print(f"  capping {len(files)} files to {args.max_files_per_repo}")
            files = sorted(files)[: args.max_files_per_repo]

        for fp in files:
            try:
                text = fp.read_text(encoding="utf-8", errors="ignore")
            except Exception:
                summary["files_skipped"] += 1
                continue
            if len(text) < args.min_chars:
                summary["files_skipped"] += 1
                continue

            truncated = False
            if len(text) > args.max_chars:
                text = text[: args.max_chars]
                truncated = True

            h = hash_text(text)
            if h in seen_hashes:
                summary["duplicates_skipped"] += 1
                continue
            seen_hashes.add(h)

            rel = fp.relative_to(walk_root)
            file_lang = EXT_TO_LANG.get(fp.suffix.lower(), fp.suffix.lstrip("."))

            if args.per_language_cap and lang_counts.get(file_lang, 0) >= args.per_language_cap:
                summary["files_skipped"] += 1
                continue
            lang_counts[file_lang] = lang_counts.get(file_lang, 0) + 1

            rec = {
                "id": f"{name}/{rel}",
                "source": "code_corpus",
                "text": text,
                "language": file_lang,
                "license": license_spdx,
                "repo": url,
                "path": str(rel),
                "category": category,
                "truncated": truncated,
            }
            out_fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            summary["files_written"] += 1
            summary["chars_written"] += len(text)

    summary["elapsed_sec"] = round(time.monotonic() - started, 1)
    print(
        f"  wrote {summary['files_written']} files, "
        f"{summary['chars_written']:,} chars, "
        f"{summary['duplicates_skipped']} dups, "
        f"{summary['elapsed_sec']}s",
    )
    return summary


def load_existing_manifest(path: Path) -> list[dict]:
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data.get("sources", [])
    except Exception:
        return []


def write_manifest(path: Path, sources: list[dict], totals: dict) -> None:
    payload = {"totals": totals, "sources": sources}
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()
    repos = load_config(Path(args.config))
    allowlist = parse_license_allowlist(args.license_allowlist)

    if args.only_language:
        repos = [r for r in repos if r.get("language") == args.only_language]
    if args.only_repo:
        repos = [r for r in repos if r.get("name") == args.only_repo]

    if allowlist is not None:
        before = len(repos)
        repos = [r for r in repos if r["license"] in allowlist]
        skipped = before - len(repos)
        if skipped:
            print(f"License filter: skipping {skipped} repo(s) outside allowlist")

    out_path = Path(args.output)
    manifest_path = Path(args.manifest)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    existing_sources: list[dict] = []
    seen_hashes: set[str] = set()
    if args.append and out_path.exists():
        print(f"Append mode: scanning existing {out_path} for hashes...")
        with out_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                seen_hashes.add(hash_text(rec.get("text", "")))
        print(f"  loaded {len(seen_hashes)} existing hashes")
        existing_sources = load_existing_manifest(manifest_path)
        done_names = {s["name"] for s in existing_sources if s.get("status") == "ok"}
        repos = [r for r in repos if r["name"] not in done_names]
        print(f"  resuming with {len(repos)} repos to process")

    if args.dry_run:
        print(f"\nDRY RUN — would process {len(repos)} repo(s):")
        by_lang: dict[str, int] = defaultdict(int)
        for r in repos:
            by_lang[r.get("language", "?")] += 1
            print(f"  {r['name']:<30} {r['license']:<14} {r.get('language', '?'):<12} {r['url']}")
        print("\nLanguage breakdown:")
        for lang, n in sorted(by_lang.items(), key=lambda kv: -kv[1]):
            print(f"  {lang:<12} {n}")
        return

    mode = "a" if args.append else "w"
    out_fh = out_path.open(mode, encoding="utf-8")
    sources = list(existing_sources)
    lang_counts: dict[str, int] = defaultdict(int)
    if args.per_language_cap and args.append:
        with out_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                lang_counts[rec.get("language", "?")] += 1
    try:
        for i, entry in enumerate(repos, 1):
            summary = process_repo(
                entry, out_fh, args, i, len(repos), seen_hashes, lang_counts,
            )
            sources.append(summary)
            totals = compute_totals(sources)
            write_manifest(manifest_path, sources, totals)
    finally:
        out_fh.close()

    totals = compute_totals(sources)
    write_manifest(manifest_path, sources, totals)
    print("\n=== done ===")
    print(f"  total repos: {totals['repos']}")
    print(f"  total files: {totals['files']}")
    print(f"  total chars: {totals['chars']:,}")
    print(f"  duplicates skipped: {totals['duplicates']}")
    print(f"  output: {out_path}")
    print(f"  manifest: {manifest_path}")


def compute_totals(sources: list[dict]) -> dict:
    by_lang: dict[str, int] = defaultdict(int)
    by_license: dict[str, int] = defaultdict(int)
    for s in sources:
        if s.get("language"):
            by_lang[s["language"]] += s.get("files_written", 0)
        by_license[s["license"]] += s.get("files_written", 0)
    return {
        "repos": sum(1 for s in sources if s.get("status") == "ok"),
        "files": sum(s.get("files_written", 0) for s in sources),
        "chars": sum(s.get("chars_written", 0) for s in sources),
        "duplicates": sum(s.get("duplicates_skipped", 0) for s in sources),
        "by_language": dict(by_lang),
        "by_license": dict(by_license),
    }


if __name__ == "__main__":
    main()
