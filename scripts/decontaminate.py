#!/usr/bin/env python3
"""Decontaminate the training corpus against every evaluation benchmark.

`audit_corpus_contamination.py` answers one question ("is CTIBench in the
corpus?") and only reports. This is its generalization: fingerprint *all*
benchmarks the project scores on (the cybersec MCQ sets and the new
general rulers, ARC / OpenBookQA), find training records that overlap a
benchmark question, and optionally **remove** them to write a clean
corpus. Every number GhostLM reports is only trustworthy if the model was
never trained on the questions it is graded with, and the generalist
pivot added general web / Wikipedia sources where ARC and OpenBookQA
questions are very plausibly present.

Detection (same methodology as the CTIBench audit, two tiers):

  Tier 1  exact normalized-question substring inside a corpus record.
  Tier 2  >= ``--min-shingles`` shared n-word shingles between a single
          corpus record and one benchmark question.

A corpus record that trips either tier against any benchmark question is
flagged as contaminated. ``--write-clean`` emits the corpus with flagged
records dropped; without it the tool only reports (audit mode).

Benchmarks are auto-discovered from ``data/raw`` (``*_eval.jsonl``,
``*_bench*.jsonl``, ``secqa.jsonl``, ``general_mcq_bench.jsonl``) and any
extra files passed with ``--bench``. Each is parsed flexibly for a
question/prompt field and, with ``--include-answers``, its answer text.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Set, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from scripts.audit_corpus_contamination import normalize, shingles, iter_records

# Benchmark files in data/raw that the corpus must never contain.
DEFAULT_BENCH_GLOBS = ("*_eval.jsonl", "*_bench.jsonl", "*_bench_v2.jsonl",
                       "secqa.jsonl", "general_mcq_bench.jsonl")


def _extract_bench_texts(rec: dict, include_answers: bool) -> List[str]:
    """Pull the question (and optionally answer) text out of one bench record.

    Handles the MCQ schema ({question, choices, answer}), the fact-recall
    schema ({prompt|question, answer|answers}), and free-text variants.
    Returns a list of strings to fingerprint.
    """
    texts: List[str] = []
    q = rec.get("question") or rec.get("prompt") or rec.get("Question") or ""
    if q:
        texts.append(str(q))
    if include_answers:
        choices = rec.get("choices")
        if isinstance(choices, dict):
            texts.extend(str(v) for v in choices.values() if v)
        elif isinstance(choices, list):
            texts.extend(str(v) for v in choices if v)
        for key in ("answer", "answers", "gold", "response"):
            v = rec.get(key)
            if isinstance(v, list):
                texts.extend(str(x) for x in v if x)
            elif v:
                texts.append(str(v))
    return texts


def load_benchmarks(bench_paths: List[Path], *, include_answers: bool, shingle_n: int,
                    min_exact_words: int = 6) -> Tuple[Dict[int, Set[str]], Set[str]]:
    """Build the benchmark fingerprint index, fully set-based for speed.

    Returns ``(exact_by_len, shingle_set)``:

    - ``shingle_set``: union of all benchmark ``shingle_n``-word shingles
      (tier-2 detector). Covers any question/answer >= ``shingle_n`` words.
    - ``exact_by_len``: maps a word-length L (``min_exact_words`` <= L <
      ``shingle_n``) to the set of normalized questions of exactly that
      length. Catches short questions that produce no full-length shingle,
      via an L-gram set intersection rather than a per-record substring scan.

    Both detectors are O(corpus words) at scan time. Questions shorter than
    ``min_exact_words`` are ignored as too generic to be smoking-gun.
    """
    exact_by_len: Dict[int, Set[str]] = {}
    shingle_set: Set[str] = set()
    for path in bench_paths:
        n_recs = 0
        with path.open(encoding="utf-8") as f:
            for raw in f:
                raw = raw.strip()
                if not raw:
                    continue
                try:
                    rec = json.loads(raw)
                except json.JSONDecodeError:
                    continue
                if not isinstance(rec, dict):
                    continue
                texts = _extract_bench_texts(rec, include_answers)
                if not texts:
                    continue
                n_recs += 1
                q_words = normalize(texts[0]).split()
                if min_exact_words <= len(q_words) < shingle_n:
                    exact_by_len.setdefault(len(q_words), set()).add(" ".join(q_words))
                for t in texts:
                    shingle_set |= shingles(normalize(t).split(), n=shingle_n)
        print(f"  bench {path.name}: {n_recs} records indexed")
    return exact_by_len, shingle_set


def scan_corpus(corpus_path: Path, exact_by_len: Dict[int, Set[str]], shingle_set: Set[str],
                *, shingle_n: int, min_shingles: int) -> Tuple[List[int], List[Dict]]:
    """Return (contaminated_line_indices, hit_details) for one corpus file.

    Both tiers are set intersections over the record's own n-grams, so the
    cost is linear in corpus size regardless of how many benchmark questions
    were indexed.
    """
    flagged: List[int] = []
    hits: List[Dict] = []
    lengths = sorted(exact_by_len)
    for line_idx, text in iter_records(corpus_path):
        words = normalize(text).split()
        reason = None
        # Tier 1: exact short-question match via per-length L-gram sets.
        for L in lengths:
            if len(words) < L:
                continue
            grams = {" ".join(words[i:i + L]) for i in range(len(words) - L + 1)}
            if grams & exact_by_len[L]:
                reason = ("exact", f"{L}-word question")
                break
        # Tier 2: shingle overlap for longer content.
        if reason is None and shingle_set:
            rec_sh = shingles(words, n=shingle_n)
            if rec_sh:
                overlap = rec_sh & shingle_set
                if len(overlap) >= min_shingles:
                    reason = ("shingle", f"{len(overlap)} shared")
        if reason is not None:
            flagged.append(line_idx)
            hits.append({"line": line_idx, "tier": reason[0], "detail": reason[1]})
    return flagged, hits


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--corpus", default="data/processed/train.jsonl",
                   help="Training corpus JSONL to scan/clean")
    p.add_argument("--raw-dir", default="data/raw",
                   help="Directory scanned for benchmark files")
    p.add_argument("--bench", action="append", default=None,
                   help="Extra benchmark JSONL path (repeatable)")
    p.add_argument("--include-answers", action="store_true",
                   help="Fingerprint answer/choice text too, not just questions")
    p.add_argument("--shingle-n", type=int, default=12)
    p.add_argument("--min-shingles", type=int, default=3)
    p.add_argument("--write-clean", default=None,
                   help="If set, write the decontaminated corpus here (drops flagged records)")
    p.add_argument("--report", default="docs/decontamination_report.md")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    raw_dir = Path(args.raw_dir)
    bench_paths: List[Path] = []
    seen = set()
    for g in DEFAULT_BENCH_GLOBS:
        for p in sorted(raw_dir.glob(g)):
            if p not in seen:
                bench_paths.append(p); seen.add(p)
    for extra in (args.bench or []):
        p = Path(extra)
        if p.exists() and p not in seen:
            bench_paths.append(p); seen.add(p)

    if not bench_paths:
        print("No benchmark files found — nothing to decontaminate against.")
        return 1

    corpus = Path(args.corpus)
    if not corpus.exists():
        print(f"Corpus not found: {corpus}")
        return 1

    print(f"Decontaminating {corpus} against {len(bench_paths)} benchmark files")
    exact_by_len, shingle_set = load_benchmarks(
        bench_paths, include_answers=args.include_answers, shingle_n=args.shingle_n)
    n_exact = sum(len(s) for s in exact_by_len.values())
    print(f"  indexed {n_exact} short exact questions, {len(shingle_set):,} shingles")

    flagged, hits = scan_corpus(
        corpus, exact_by_len, shingle_set,
        shingle_n=args.shingle_n, min_shingles=args.min_shingles)

    total = sum(1 for _ in iter_records(corpus))
    pct = 100 * len(flagged) / total if total else 0.0
    tier1 = sum(1 for h in hits if h["tier"] == "exact")
    print(f"\n  corpus records: {total:,}")
    print(f"  contaminated:   {len(flagged):,} ({pct:.3f}%)  "
          f"[exact={tier1}, shingle={len(flagged) - tier1}]")

    report = Path(args.report)
    report.parent.mkdir(parents=True, exist_ok=True)
    with report.open("w", encoding="utf-8") as f:
        f.write("# Corpus decontamination report\n\n")
        f.write(f"- Corpus: `{corpus}` ({total:,} records)\n")
        f.write(f"- Benchmarks: {', '.join(p.name for p in bench_paths)}\n")
        f.write(f"- Shingle window: {args.shingle_n} words, min overlap {args.min_shingles}\n")
        f.write(f"- Answers fingerprinted: {args.include_answers}\n\n")
        f.write(f"**Contaminated: {len(flagged):,} / {total:,} ({pct:.3f}%)** "
                f"(exact {tier1}, shingle {len(flagged) - tier1})\n\n")
        if hits[:50]:
            f.write("## First matches\n\n")
            for h in hits[:50]:
                f.write(f"- line {h['line']}: {h['tier']} ({h['detail']})\n")
    print(f"  report -> {report}")

    if args.write_clean:
        drop = set(flagged)
        out = Path(args.write_clean)
        out.parent.mkdir(parents=True, exist_ok=True)
        kept = 0
        with corpus.open(encoding="utf-8") as fin, out.open("w", encoding="utf-8") as fout:
            line_idx = -1
            for line in fin:
                if not line.strip():
                    continue
                line_idx += 1
                if line_idx in drop:
                    continue
                fout.write(line)
                kept += 1
        print(f"  clean corpus -> {out} ({kept:,} kept, {len(drop):,} dropped)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
