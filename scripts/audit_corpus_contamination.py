#!/usr/bin/env python3
"""Contamination audit: do CTIBench MCQ questions (or near-paraphrases of
them) appear inside the GhostLM training corpus?

Why this matters: PRIMUS-FineWeb is CommonCrawl-derived, so any
CTIBench question that originated on a public web page may have been
ingested during corpus build. If even a few percent of CTIBench is
contaminated, the v0.9 chat numbers are inflated by memorization rather
than by actual capability, and the ghost-base GPU spend would be
chasing a poisoned target.

Audit strategy (two tiers):

    Tier 1: exact question substring.
        Lowercase + collapse-whitespace each CTIBench question, then
        scan every corpus record for that exact substring. Matches here
        are smoking-gun contamination: the question text itself is in
        the training data verbatim.

    Tier 2: long-phrase shingle (12-word windows).
        For each question, take every contiguous 12-word window of the
        question text. If three or more windows from the same question
        appear inside a single corpus record, that's strong evidence of
        a near-paraphrase or copy-paste of the question's stem.

Outputs:

    docs/contamination_audit.md   summary + per-source breakdown
    logs/contamination_hits.jsonl one record per match, keyed by
                                  (ctibench_idx, source_file)

Run on the Mac alongside the v0.9 corpus:

    cd ~/Desktop/GhostLM && PYTHONPATH=. python3 scripts/audit_corpus_contamination.py
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Set, Tuple

WS_RE = re.compile(r"\s+")
NON_WORD_RE = re.compile(r"[^\w\s]")


def normalize(text: str) -> str:
    """Lowercase, collapse whitespace, drop punctuation. Used for both
    CTIBench questions and corpus text. The audit is intentionally
    indifferent to capitalization and minor formatting tweaks; we want
    to catch reformatted copies of the same sentence."""
    text = text.lower()
    text = NON_WORD_RE.sub(" ", text)
    text = WS_RE.sub(" ", text).strip()
    return text


def load_ctibench_questions() -> List[Dict]:
    """Load the 2500-record CTIBench MCQ test split via huggingface
    datasets. Returns a list of {idx, question, normalized}."""
    from datasets import load_dataset
    ds = load_dataset("AI4Sec/cti-bench", "cti-mcq", split="test")
    out: List[Dict] = []
    for i, r in enumerate(ds):
        q = r.get("Question") or r.get("question") or r.get("prompt") or ""
        if not q:
            continue
        out.append({
            "idx": i,
            "question": q.strip(),
            "normalized": normalize(q),
        })
    return out


def shingles(words: List[str], n: int = 12) -> Set[str]:
    """Return the set of contiguous n-word shingles from a token list.
    Shingles shorter than n words are skipped (yields empty set)."""
    if len(words) < n:
        return set()
    return {" ".join(words[i:i + n]) for i in range(len(words) - n + 1)}


def iter_records(jsonl_path: Path) -> Iterable[Tuple[int, str]]:
    """Yield (line_idx, text) from a jsonl. Pulls 'text' or 'content'
    field; falls back to the raw record JSON serialized if neither is
    present (so we still scan exotic shards)."""
    with jsonl_path.open() as f:
        for line_idx, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(rec, dict):
                text = rec.get("text") or rec.get("content") or rec.get("body") or json.dumps(rec)
            else:
                text = str(rec)
            yield line_idx, text


def discover_corpus_files(data_dir: Path) -> List[Path]:
    """Find every .jsonl shard under data/. We scan both raw/ (per-source
    shards) and processed/ (final blended training set). Any individual
    file under 1 KB is skipped as it's not a real corpus."""
    files: List[Path] = []
    for sub in ("raw", "processed"):
        d = data_dir / sub
        if not d.is_dir():
            continue
        for p in sorted(d.glob("*.jsonl")):
            if p.stat().st_size < 1024:
                continue
            files.append(p)
    return files


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir", default="data",
                   help="Path containing data/raw/*.jsonl and data/processed/*.jsonl")
    p.add_argument("--out-dir", default="docs",
                   help="Where to write contamination_audit.md")
    p.add_argument("--logs-dir", default="logs",
                   help="Where to write contamination_hits.jsonl")
    p.add_argument("--limit-questions", type=int, default=0,
                   help="Audit only the first N CTIBench questions (for smoke testing). 0 = all 2500.")
    p.add_argument("--shingle-n", type=int, default=12,
                   help="Word-window size for tier-2 shingle matching")
    p.add_argument("--min-shingles", type=int, default=3,
                   help="Minimum shingle hits in a single record to count as suspicious")
    args = p.parse_args()

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    logs_dir = Path(args.logs_dir); logs_dir.mkdir(parents=True, exist_ok=True)

    print("Loading CTIBench MCQ test split...", flush=True)
    questions = load_ctibench_questions()
    if args.limit_questions:
        questions = questions[: args.limit_questions]
    print(f"  {len(questions)} questions loaded", flush=True)

    # Build tier-1 lookup: normalized question -> ctibench idx.
    # Skip questions shorter than 30 chars after normalization (those are
    # too generic, would false-positive on common phrasing).
    tier1: Dict[str, int] = {}
    skipped_short = 0
    for q in questions:
        n = q["normalized"]
        if len(n) < 30:
            skipped_short += 1
            continue
        tier1[n] = q["idx"]
    print(f"  tier-1 keys (>= 30 chars): {len(tier1)}; skipped {skipped_short} too-short", flush=True)

    # Build tier-2 shingle lookup: shingle -> set of ctibench idx that contain it.
    tier2: Dict[str, Set[int]] = {}
    for q in questions:
        words = q["normalized"].split()
        for sh in shingles(words, n=args.shingle_n):
            tier2.setdefault(sh, set()).add(q["idx"])
    print(f"  tier-2 shingles ({args.shingle_n}-word windows): {len(tier2)}", flush=True)

    corpus_files = discover_corpus_files(data_dir)
    print(f"\nScanning {len(corpus_files)} corpus files:", flush=True)

    hits: List[Dict] = []
    per_source: Dict[str, Dict[str, int]] = {}

    for cf in corpus_files:
        scanned = 0
        tier1_hits = 0
        tier2_hits = 0
        for line_idx, text in iter_records(cf):
            scanned += 1
            normalized = normalize(text)
            if not normalized:
                continue

            # Tier 1: substring of any CTIBench question.
            for key, qidx in tier1.items():
                if key in normalized:
                    tier1_hits += 1
                    hit = {
                        "tier": 1,
                        "ctibench_idx": qidx,
                        "source": str(cf.relative_to(data_dir)),
                        "line_idx": line_idx,
                        "evidence": key[:240],
                    }
                    hits.append(hit)

            # Tier 2: count shingle matches grouped by question.
            words = normalized.split()
            if len(words) >= args.shingle_n:
                hits_per_q: Dict[int, int] = {}
                for i in range(len(words) - args.shingle_n + 1):
                    sh = " ".join(words[i:i + args.shingle_n])
                    if sh in tier2:
                        for qidx in tier2[sh]:
                            hits_per_q[qidx] = hits_per_q.get(qidx, 0) + 1
                for qidx, count in hits_per_q.items():
                    if count >= args.min_shingles:
                        tier2_hits += 1
                        hit = {
                            "tier": 2,
                            "ctibench_idx": qidx,
                            "source": str(cf.relative_to(data_dir)),
                            "line_idx": line_idx,
                            "shingle_count": count,
                        }
                        hits.append(hit)

        per_source[str(cf.relative_to(data_dir))] = {
            "records_scanned": scanned,
            "tier1_hits": tier1_hits,
            "tier2_hits": tier2_hits,
        }
        print(f"  {cf.name:35s}  scanned={scanned:>7d}  T1={tier1_hits:>4d}  T2={tier2_hits:>4d}", flush=True)

    # Persist per-hit log.
    log_path = logs_dir / "contamination_hits.jsonl"
    with log_path.open("w") as f:
        for h in hits:
            f.write(json.dumps(h) + "\n")
    print(f"\nWrote {len(hits)} hit records to {log_path}", flush=True)

    # Summary report.
    total_t1 = sum(s["tier1_hits"] for s in per_source.values())
    total_t2 = sum(s["tier2_hits"] for s in per_source.values())
    leaked_questions = {h["ctibench_idx"] for h in hits if h["tier"] == 1}
    near_para_questions = {h["ctibench_idx"] for h in hits if h["tier"] == 2}

    md = []
    md.append("# Corpus contamination audit (CTIBench vs GhostLM v1.0 corpus)\n")
    md.append("This is an automated audit produced by `scripts/audit_corpus_contamination.py`.\n")
    md.append("Run it again whenever the corpus changes; the answer is only true for\n")
    md.append(f"the corpus that was on disk at audit time.\n\n")
    md.append("## Summary\n\n")
    md.append(f"- CTIBench MCQ test split: **{len(questions)} questions**\n")
    md.append(f"- Corpus files scanned: **{len(corpus_files)}**\n")
    md.append(f"- Tier-1 (exact-substring) hits: **{total_t1}** total, "
              f"{len(leaked_questions)} distinct CTIBench questions\n")
    md.append(f"- Tier-2 (>= {args.min_shingles}× {args.shingle_n}-word shingle) hits: "
              f"**{total_t2}** total, {len(near_para_questions)} distinct CTIBench questions\n")
    md.append(f"- Tier-1 contamination rate: "
              f"**{100.0 * len(leaked_questions) / max(1, len(questions)):.2f}%** of CTIBench\n")
    md.append(f"- Combined (T1 ∪ T2) contamination rate: "
              f"**{100.0 * len(leaked_questions | near_para_questions) / max(1, len(questions)):.2f}%**\n\n")
    md.append("## Per-source breakdown\n\n")
    md.append("| Source | Records | Tier-1 hits | Tier-2 hits |\n")
    md.append("|---|---:|---:|---:|\n")
    for src, s in sorted(per_source.items()):
        md.append(f"| `{src}` | {s['records_scanned']:,} | "
                  f"{s['tier1_hits']} | {s['tier2_hits']} |\n")
    md.append("\n## Verdict\n\n")
    if len(leaked_questions) == 0 and len(near_para_questions) == 0:
        md.append("**Clean.** No CTIBench questions found in the corpus by either tier.\n"
                  "v1.0 ghost-base GPU spend can proceed without contamination concern.\n"
                  "(Caveat: this audit catches verbatim and 12-word-shingle matches.\n"
                  "Heavy paraphrase or translation-back-translation would still slip\n"
                  "through; for that we'd need embedding-similarity, which is a\n"
                  "follow-up if the bench numbers come in suspiciously high.)\n")
    elif len(leaked_questions) > 0:
        md.append(f"**{len(leaked_questions)} CTIBench questions are present verbatim "
                  "in the training corpus.** This is direct contamination. The v0.9 chat "
                  "bench numbers on those specific questions are memorization, not "
                  "capability. Decide whether to: (a) excise the contaminated records "
                  "from the corpus and rebuild, or (b) report the contaminated subset "
                  "separately in the bench tables. Either way the v1.0 ghost-base GPU "
                  "spend should not run on the corpus as-is until this is resolved.\n")
    else:
        md.append(f"**{len(near_para_questions)} CTIBench questions are near-paraphrased "
                  "in the corpus** (3+ shared 12-word shingles in a single corpus record). "
                  "Probably real overlap, possibly coincidence on common cybersec phrasing. "
                  "Spot-check the top hits in `logs/contamination_hits.jsonl` before deciding "
                  "whether to scrub. Tier-1 is clean, so the bulk of CTIBench is uncompromised.\n")
    md.append(f"\nRaw per-hit log at `logs/contamination_hits.jsonl` ({len(hits)} records).\n")

    out_path = out_dir / "contamination_audit.md"
    out_path.write_text("".join(md))
    print(f"Wrote summary to {out_path}", flush=True)

    return 0


if __name__ == "__main__":
    sys.exit(main())
