#!/usr/bin/env python3
"""Templated synthesis of code-explanation training records (v0.9.23).

Mirrors the bet 7 / bet 8 pattern: a hand-curated bank of code
snippets + metadata, multiplied through 5 templated variants per
pattern.

Variants per snippet:
  1. pretrain_prose   markdown article describing what the code
                       does and why it's useful.
  2. identify_lang    USER shows the snippet, asks 'what language?';
                       ASSISTANT names the language and gives one
                       reason it's recognisable from the snippet.
  3. explain_purpose  USER shows the snippet, asks 'what does this
                       do?'; ASSISTANT gives the one-sentence purpose
                       plus a longer explanation.
  4. walkthrough      USER shows the snippet, asks for a step-by-
                       step walk-through; ASSISTANT walks through
                       the logic line by line.
  5. concepts         USER shows the snippet, asks 'what programming
                       concepts does this demonstrate?'; ASSISTANT
                       names the key concepts.

40 patterns × 5 variants = 200 records, all parser-clean. The
output is shaped identically to synth_code_security.jsonl so it
drops into the same combined-corpus build pipeline.

Run:

    PYTHONPATH=. python3 scripts/synth_code_explain.py \\
        --bank data/raw/code_explain_patterns.jsonl \\
        --out data/processed/synth_code_explain.jsonl
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Dict, Iterator, List

REPO_ROOT = Path(__file__).resolve().parent.parent


def build_record(seed_id: str, variant: str, text: str) -> Dict[str, str]:
    h = hashlib.sha1(
        f"{seed_id}\n{variant}\n{text}".encode("utf-8")
    ).hexdigest()[:10]
    return {
        "id": f"synth_code_explain#{seed_id}#{variant}#{h}",
        "source": "synth_code_explain",
        "teacher": "templated",
        "seed_source": variant,
        "seed_id": seed_id,
        "text": text,
    }


def fence(code: str, lang: str) -> str:
    return f"```{lang}\n{code}\n```"


def concepts_phrase(concepts: List[str]) -> str:
    if not concepts:
        return ""
    if len(concepts) == 1:
        return f"Demonstrates: {concepts[0]}."
    return f"Demonstrates: {', '.join(concepts[:-1])} and {concepts[-1]}."


# ---------------------------------------------------------------------------
# Variant templates
# ---------------------------------------------------------------------------


def pretrain_prose(p: Dict) -> str:
    """Markdown article with snippet + purpose + explanation."""
    parts = [
        f"# {p['language'].title()}: {p['purpose']}",
        "",
        fence(p["snippet"], p["language"]),
        "",
        p["explanation"],
    ]
    if p.get("key_concepts"):
        parts.append("")
        parts.append(concepts_phrase(p["key_concepts"]))
    return "\n".join(parts)


def identify_lang_qa(p: Dict) -> str:
    """USER shows snippet, asks 'what language?'."""
    user = (
        "What programming language is this snippet written in, "
        "and what's the most distinctive clue?\n\n"
        + fence(p["snippet"], p["language"])
    )
    asst = (
        f"This is {p['language'].title()}. "
        + p.get("language_clue", _default_lang_clue(p["language"]))
    )
    return _qa(user, asst)


def explain_purpose_qa(p: Dict) -> str:
    """USER shows snippet, asks 'what does this do?'."""
    user = (
        "What does this code do?\n\n"
        + fence(p["snippet"], p["language"])
    )
    asst = (
        f"{p['purpose'].rstrip('.')}.\n\n"
        + p["explanation"]
    )
    return _qa(user, asst)


def walkthrough_qa(p: Dict) -> str:
    """USER shows snippet, asks for step-by-step walkthrough."""
    user = (
        "Walk me through this code step by step.\n\n"
        + fence(p["snippet"], p["language"])
    )
    asst = p.get("walkthrough") or p["explanation"]
    return _qa(user, asst)


def concepts_qa(p: Dict) -> str:
    """USER asks 'what programming concepts does this demonstrate?'."""
    if not p.get("key_concepts"):
        return ""
    user = (
        "What programming concepts does this code demonstrate?\n\n"
        + fence(p["snippet"], p["language"])
    )
    bullets = "\n".join(f"- {c}" for c in p["key_concepts"])
    asst = (
        "The snippet illustrates several concepts:\n\n"
        + bullets
        + "\n\n"
        + p["explanation"]
    )
    return _qa(user, asst)


def _qa(user: str, asst: str) -> str:
    return f"USER: {user}\n\nASSISTANT: {asst}\n"


_LANG_CLUES = {
    "python": "The `def` keyword, `:` block syntax, and indentation-"
              "as-syntax are unambiguous Python.",
    "javascript": "`const`/`let`, arrow functions, and the absence of "
                   "type annotations point to JavaScript.",
    "typescript": "`const`/`let` plus inline type annotations like "
                   "`: string` are TypeScript.",
    "go": "`func` keyword, `:=` short variable declaration, and "
          "explicit error returns are Go.",
    "rust": "`fn` keyword, `let` binding, and either `&` borrow / "
            "lifetime syntax or `Result<T, E>` error type point to "
            "Rust.",
    "c": "`#include`, explicit pointer types, and `void` parameters "
         "are classic C.",
    "java": "`public class` or `public static void main` plus type "
            "annotations on every variable point to Java.",
    "ruby": "`def end` blocks, `do...end`, and method names ending in "
            "`?`/`!` are Ruby.",
    "kotlin": "`fun` keyword + JVM-style type annotations + null-"
              "safety operators (`?.`, `!!`) are Kotlin.",
    "swift": "`func` keyword + Swift-style optionals + `let`/`var` "
             "with type inference are Swift.",
    "php": "`<?php` tag and `$variable` sigils.",
    "shell": "shebang `#!/bin/bash`, `$VAR` syntax, command "
             "substitution `$(...)`.",
}


def _default_lang_clue(lang: str) -> str:
    return _LANG_CLUES.get(
        lang.lower(),
        f"The syntax of this snippet is characteristic of {lang}.",
    )


# ---------------------------------------------------------------------------
# Bank loading + main
# ---------------------------------------------------------------------------


def stream_bank(path: Path) -> Iterator[Dict]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def quality_ok(text: str, min_words: int = 30,
                max_words: int = 1500) -> bool:
    n = len(text.split())
    return min_words <= n <= max_words


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="scripts/synth_code_explain.py")
    p.add_argument("--bank",
                    default="data/raw/code_explain_patterns.jsonl")
    p.add_argument("--out",
                    default="data/processed/synth_code_explain.jsonl")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    bank = Path(args.bank)
    if not bank.exists():
        print(f"[error] bank not found: {bank}", file=sys.stderr)
        return 1
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    counts: Dict[str, int] = {}
    rejects: Dict[str, int] = {}
    n_total = 0
    with out.open("w", encoding="utf-8") as fout:
        for pattern in stream_bank(bank):
            pid = pattern["id"]
            variants = [
                ("pretrain_prose", pretrain_prose),
                ("identify_lang", identify_lang_qa),
                ("explain_purpose", explain_purpose_qa),
                ("walkthrough", walkthrough_qa),
                ("concepts", concepts_qa),
            ]
            for vname, fn in variants:
                text = fn(pattern)
                if not text or not quality_ok(text):
                    rejects[vname] = rejects.get(vname, 0) + 1
                    continue
                rec = build_record(pid, vname, text)
                fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                counts[vname] = counts.get(vname, 0) + 1
                n_total += 1

    print(f"Wrote {n_total} records to {out}")
    print(f"  by variant: {counts}")
    if rejects:
        print(f"  rejects:    {rejects}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
