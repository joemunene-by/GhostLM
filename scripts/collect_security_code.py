#!/usr/bin/env python3
"""Collect source code from curated cybersecurity-tool GitHub repos.

The cybersec corpus through v0.9 is text-only (writeups, advisories,
RFCs, MITRE entries). For ghost-base we want the model to also know the
*code shape* of the field: pwntools idioms, scapy packet construction,
volatility plugin patterns, sqlmap injection chains, impacket SMB
flows, etc. Pulling source from the actual tools the field uses gives
the LM exposure to those patterns.

This collector clones a JSON-config'd list of repos (with explicit
SPDX license per entry) and walks source files matching an extension
whitelist (.py / .c / .h / .cpp / .js / .ts / .go / .rs / .sh by
default). Each file becomes one JSONL record with full attribution
(repo URL, file path, language, license).

Filters baked in:
- Skips vendored deps (``node_modules/``, ``vendor/``, ``third_party/``,
  ``__pycache__/``, ``.venv/``).
- Skips test fixtures and binary blobs (``*.test.*``, ``*.min.*``).
- Drops files smaller than ``--min-chars`` (boilerplate, empty
  ``__init__.py``).
- Truncates files longer than ``--max-chars`` (some repos commit huge
  generated files; truncation keeps training-time compute predictable).
- Per-record SPDX license string drives downstream audit.

Output: ``data/raw/security_code.jsonl`` with the standard
``{"id", "source", "text"}`` schema plus extra metadata. ``source`` is
``security_code``.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


# Default extension → language label
EXT_TO_LANG = {
    ".py": "python",
    ".c": "c",
    ".h": "c",
    ".cpp": "cpp",
    ".cc": "cpp",
    ".hpp": "cpp",
    ".js": "javascript",
    ".ts": "typescript",
    ".jsx": "javascript",
    ".tsx": "typescript",
    ".go": "go",
    ".rs": "rust",
    ".sh": "shell",
    ".bash": "shell",
    ".rb": "ruby",
    ".pl": "perl",
    ".java": "java",
}

SKIP_DIRS = {
    "node_modules", "vendor", "third_party", "__pycache__",
    ".venv", "venv", "env", ".env", ".git", "build", "dist",
    "target", ".tox", ".pytest_cache", ".mypy_cache",
    "site-packages", ".eggs", "egg-info",
}

SKIP_FILE_PATTERNS = (
    ".min.js", ".min.css", ".bundle.js", ".bundle.css",
    ".test.ts", ".test.js", ".test.tsx", ".test.jsx",
    "_test.go",
)


def parse_args() -> argparse.Namespace:
    """CLI args."""
    p = argparse.ArgumentParser(description="Collect source code from cybersec tool repos")
    p.add_argument("--config", required=True,
                   help="JSON config: [{url, license, [name], [branch], [subdir]}]")
    p.add_argument("--output", default="data/raw/security_code.jsonl")
    p.add_argument("--min-chars", type=int, default=200)
    p.add_argument("--max-chars", type=int, default=15000)
    p.add_argument("--exts", nargs="+", default=list(EXT_TO_LANG.keys()),
                   help="File extensions to include (with leading dot)")
    p.add_argument("--max-files-per-repo", type=int, default=2000,
                   help="Cap files per repo so one mega-repo can't dominate")
    return p.parse_args()


def shallow_clone(url: str, dest: Path, branch: str | None = None) -> bool:
    """Shallow clone the repo. Return True on success."""
    cmd = ["git", "clone", "--depth", "1"]
    if branch:
        cmd += ["--branch", branch]
    cmd += [url, str(dest)]
    try:
        subprocess.run(cmd, check=True, capture_output=True, timeout=300)
        return True
    except subprocess.CalledProcessError as e:
        print(f"  clone failed: {e.stderr.decode()[:200] if e.stderr else 'unknown'}")
        return False
    except subprocess.TimeoutExpired:
        print(f"  clone timeout (>5min): {url}")
        return False


def walk_source_files(root: Path, exts: list[str]) -> list[Path]:
    """List source files under root matching the extension whitelist."""
    out: list[Path] = []
    ext_set = set(e.lower() for e in exts)
    for p in root.rglob("*"):
        if not p.is_file():
            continue
        # Skip files inside excluded dirs
        if any(part in SKIP_DIRS for part in p.parts):
            continue
        # Skip patterns
        if any(p.name.endswith(pat) for pat in SKIP_FILE_PATTERNS):
            continue
        if p.suffix.lower() not in ext_set:
            continue
        out.append(p)
    return out


def process_repo(entry: dict, out_fh, args, repo_idx: int) -> tuple[int, int]:
    """Clone one repo, walk sources, write records. Return (written, skipped)."""
    url = entry["url"]
    license_spdx = entry["license"]
    name = entry.get("name") or url.rstrip("/").split("/")[-1].replace(".git", "")
    branch = entry.get("branch")
    subdir = entry.get("subdir")
    print(f"\n[{repo_idx}] {name} ({license_spdx}) -> {url}")

    with tempfile.TemporaryDirectory(prefix=f"sec_code_{name}_") as tmp:
        tmp_path = Path(tmp)
        if not shallow_clone(url, tmp_path / name, branch=branch):
            return 0, 0
        root = tmp_path / name
        if subdir:
            root = root / subdir
            if not root.is_dir():
                print(f"  subdir not found: {subdir}")
                return 0, 0

        files = walk_source_files(root, args.exts)
        if len(files) > args.max_files_per_repo:
            print(f"  capping {len(files)} files to {args.max_files_per_repo}")
            files = sorted(files)[: args.max_files_per_repo]

        written = 0
        skipped = 0
        for fp in files:
            try:
                text = fp.read_text(encoding="utf-8", errors="ignore")
            except Exception:
                skipped += 1
                continue
            if len(text) < args.min_chars:
                skipped += 1
                continue
            truncated = False
            if len(text) > args.max_chars:
                text = text[: args.max_chars]
                truncated = True
            rel = fp.relative_to(root if not subdir else (tmp_path / name))
            lang = EXT_TO_LANG.get(fp.suffix.lower(), fp.suffix.lstrip("."))
            rec = {
                "id": f"{name}/{rel}",
                "source": "security_code",
                "text": text,
                "language": lang,
                "license": license_spdx,
                "repo": url,
                "path": str(rel),
                "truncated": truncated,
            }
            out_fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            written += 1
        print(f"  wrote {written} files (skipped {skipped})")
        return written, skipped


def main() -> None:
    """Walk every configured repo, emit JSONL."""
    args = parse_args()
    cfg_path = Path(args.config)
    if not cfg_path.exists():
        sys.exit(f"config not found: {cfg_path}")
    repos = json.loads(cfg_path.read_text(encoding="utf-8"))
    if not isinstance(repos, list):
        sys.exit("config must be a JSON array of {url, license, ...} entries")
    for r in repos:
        if "url" not in r or "license" not in r:
            sys.exit(f"entry missing url or license: {r!r}")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    total_written = 0
    total_skipped = 0
    out_fh = out_path.open("w", encoding="utf-8")
    try:
        for i, entry in enumerate(repos, 1):
            w, s = process_repo(entry, out_fh, args, i)
            total_written += w
            total_skipped += s
    finally:
        out_fh.close()

    print(f"\n=== done ===")
    print(f"  wrote {total_written} files, skipped {total_skipped}")
    print(f"  output: {out_path}")


if __name__ == "__main__":
    main()
