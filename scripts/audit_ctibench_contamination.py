#!/usr/bin/env python3
"""Audit CTIBench questions for contamination in the GhostLM pretrain corpus.

The v0.9 chat regression on CTIBench while leading on the in-repo CTF eval
needs an explanation. One candidate: PRIMUS-FineWeb's TinyBERT-filtered
crawl text contains CTIBench source pages or near-paraphrases, which
would let earlier checkpoints (no FineWeb in pretrain) do better on
CTIBench by NOT being confused, while v0.9 (saw FineWeb) gets confused
by near-duplicate text it half-remembers.

This script does a cheap 8-gram shingle overlap check between every
CTIBench MCQ question and the v0.9 corpus (PRIMUS-Seed + PRIMUS-FineWeb).
A single 8-word phrase appearing verbatim in both is unlikely by chance
in natural English; multiple hits per question is high-confidence
contamination.

Output: ``logs/ctibench_contamination.json`` with per-question overlap
counts and a top-N list of the most-contaminated questions for manual
review.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Iterable


SHINGLE_N = 8  # Word shingle length. 8 is conservative for natural English.


def parse_args() -> argparse.Namespace:
    """CLI args."""
    p = argparse.ArgumentParser(description="CTIBench contamination audit")
    p.add_argument("--corpus", nargs="+",
                   default=["data/raw/primus_seed.jsonl",
                            "data/raw/primus_fineweb.jsonl"],
                   help="JSONL files to scan for matches")
    p.add_argument("--out", default="logs/ctibench_contamination.json")
    p.add_argument("--top-n", type=int, default=30,
                   help="Show top N most-contaminated questions")
    p.add_argument("--include-options", action="store_true",
                   help="Also shingle option A/B/C/D text, not just question")
    return p.parse_args()


def normalize(s: str) -> str:
    """Lowercase, strip punctuation, collapse whitespace."""
    s = s.lower()
    s = re.sub(r"[^\w\s]", " ", s)
    s = re.sub(r"\s+", " ", s)
    return s.strip()


def shingles(text: str, n: int = SHINGLE_N) -> Iterable[str]:
    """Generate n-word shingles from text. Empty if text < n words."""
    words = normalize(text).split()
    if len(words) < n:
        return
    for i in range(len(words) - n + 1):
        yield " ".join(words[i:i + n])


def shingle_hash(s: str) -> int:
    """Deterministic 64-bit hash for set membership (cross-process stable)."""
    return int(hashlib.md5(s.encode()).hexdigest()[:16], 16)


def load_ctibench() -> list[dict]:
    """Load the CTIBench cti-mcq test split via the datasets library."""
    from datasets import load_dataset
    ds = load_dataset("AI4Sec/cti-bench", "cti-mcq", split="test")
    return [dict(r) for r in ds]


def build_corpus_index(paths: list[str]) -> set[int]:
    """Build set of shingle hashes from all corpus files."""
    seen: set[int] = set()
    for p in paths:
        path = Path(p)
        if not path.exists():
            print(f"  skip missing {p}")
            continue
        n = 0
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                text = rec.get("text") or rec.get("content") or ""
                if not text:
                    continue
                for sh in shingles(text):
                    seen.add(shingle_hash(sh))
                n += 1
                if n % 50_000 == 0:
                    print(f"  {p}: {n} records, {len(seen)} unique shingles")
        print(f"  {p}: {n} records done, total {len(seen)} shingles")
    return seen


def audit_question(rec: dict, corpus: set[int],
                   include_options: bool) -> tuple[int, int]:
    """Return (overlap_shingles, total_shingles) for one CTIBench record."""
    parts = [rec.get("Question", "")]
    if include_options:
        for k in ("Option A", "Option B", "Option C", "Option D"):
            v = rec.get(k)
            if v:
                parts.append(v)
    blob = " ".join(parts)
    qs = list(shingles(blob))
    if not qs:
        return 0, 0
    hits = sum(1 for s in qs if shingle_hash(s) in corpus)
    return hits, len(qs)


def main() -> None:
    """Run the audit and emit JSON + summary."""
    args = parse_args()
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print("Loading CTIBench cti-mcq...")
    ctibench = load_ctibench()
    print(f"  {len(ctibench)} questions")

    print(f"\nBuilding shingle index over {len(args.corpus)} corpus files "
          f"(n={SHINGLE_N})...")
    corpus = build_corpus_index(args.corpus)

    print(f"\nAuditing {len(ctibench)} CTIBench questions...")
    results = []
    contaminated_count = 0
    total_overlap_pct = 0.0
    for i, rec in enumerate(ctibench):
        hits, total = audit_question(rec, corpus,
                                     include_options=args.include_options)
        pct = hits / total if total else 0.0
        results.append({
            "idx": i,
            "question_preview": rec.get("Question", "")[:150],
            "url": rec.get("URL", ""),
            "overlap_shingles": hits,
            "total_shingles": total,
            "overlap_pct": pct,
        })
        if hits > 0:
            contaminated_count += 1
        total_overlap_pct += pct

    avg_overlap = total_overlap_pct / len(ctibench) if ctibench else 0.0

    # Sort by overlap_shingles desc for the worst offenders
    top = sorted(results, key=lambda r: -r["overlap_shingles"])[: args.top_n]

    summary = {
        "corpus_files": args.corpus,
        "shingle_n": SHINGLE_N,
        "n_questions": len(ctibench),
        "n_corpus_shingles": len(corpus),
        "n_contaminated_questions": contaminated_count,
        "contaminated_pct": contaminated_count / len(ctibench) if ctibench else 0.0,
        "avg_overlap_pct_per_question": avg_overlap,
        "include_options": args.include_options,
        "top_contaminated": top,
        "per_question": [
            {"idx": r["idx"], "overlap_shingles": r["overlap_shingles"],
             "total_shingles": r["total_shingles"]}
            for r in results
        ],
    }
    out_path.write_text(json.dumps(summary, indent=2))

    print()
    print(f"=== Contamination audit ===")
    print(f"  questions checked: {len(ctibench)}")
    print(f"  with at least one shingle overlap: {contaminated_count} "
          f"({contaminated_count / len(ctibench) * 100:.1f}%)")
    print(f"  avg overlap pct per question: {avg_overlap * 100:.2f}%")
    print(f"  saved: {out_path}")
    if top and top[0]["overlap_shingles"] > 0:
        print(f"\n  worst-offender preview:")
        for r in top[:5]:
            if r["overlap_shingles"] == 0:
                break
            print(f"    [{r['overlap_shingles']:3d} hits / {r['total_shingles']:3d} sh] "
                  f"{r['question_preview'][:100]}")


if __name__ == "__main__":
    main()
