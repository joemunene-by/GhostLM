#!/usr/bin/env python3
"""Combine the v0.9.5 templated-synth outputs into a single training-ready
corpus, categorised by intended training-time use (pretrain vs SFT).

After v0.9.5 ships the five templated-synth pipelines produce:

  data/processed/synth_format_aware.jsonl       (560 records, bet 6)
  data/processed/synth_tool_use.jsonl           (424 records, bet 1)
  data/processed/synth_tool_use_provenance.jsonl (429 records, bet 9)
  data/processed/synth_code_security.jsonl      (48 records, bet 7)
  data/processed/synth_binary_literacy.jsonl    (44 records, bet 8)

Each pipeline emits records in its own shape (pretrain prose with
flat ``text``, or four-message tool-use traces with USER / ASSISTANT
/ TOOL turns). The ghost-base trainer needs to know which records
to mix into pretrain vs SFT, so this script:

  1. Reads all five synth files.
  2. Tags each record with ``format_type`` (pretrain | sft) based
     on the ``seed_source`` field (variant name from the synth
     pipeline).
  3. Writes one combined JSONL where every record carries the tag.
  4. Reports per-source and per-format-type counts so the operator
     can audit the mix before training.

Categorisation rules:

  pretrain shape (flat ``text`` blob):
    - synth_format_aware (all variants; the `Source/Format/Prompt/Artifact`
      block is pretrain-prose shape)
    - synth_code_security:pretrain_prose
    - synth_binary_literacy:pretrain_prose

  sft shape (chat-tagged USER / ASSISTANT / TOOL traces):
    - synth_tool_use (all four tool variants)
    - synth_tool_use_provenance (all four tool variants)
    - synth_code_security:identify_and_fix / explain_the_diff / cwe_mapping
    - synth_binary_literacy:identify_hex / show_magic

The combined output goes to ``data/processed/synth_v15_combined.jsonl``.
Train code paths consume it by filtering on ``format_type``.

Run:

    PYTHONPATH=. python3 scripts/build_v15_combined_synth.py \\
        --in-dir data/processed \\
        --out data/processed/synth_v15_combined.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterator, List

REPO_ROOT = Path(__file__).resolve().parent.parent


# Per-source rules: which seed_source values are pretrain shape vs SFT.
# Keys are (record_source, record_seed_source) tuples; value is
# "pretrain" or "sft".
CATEGORY_RULES = {
    # Bet 6: format-aware records are pretrain-prose shape regardless
    # of the format family.
    ("synth_format_aware", "stix_indicator"): "pretrain",
    ("synth_format_aware", "yara_rule"):       "pretrain",
    ("synth_format_aware", "sigma_rule"):      "pretrain",
    ("synth_format_aware", "misp_event"):      "pretrain",

    # Bet 1: every tool-use trace is chat shape.
    ("synth_tool_use", "search_cve_nvd"):         "sft",
    ("synth_tool_use", "lookup_mitre_technique"): "sft",
    ("synth_tool_use", "lookup_cwe"):             "sft",
    ("synth_tool_use", "rag_retrieve"):           "sft",

    # Bet 9: cite-augmented traces are also chat shape.
    ("synth_tool_use_provenance", "search_cve_nvd"):         "sft",
    ("synth_tool_use_provenance", "lookup_mitre_technique"): "sft",
    ("synth_tool_use_provenance", "lookup_cwe"):             "sft",
    ("synth_tool_use_provenance", "rag_retrieve"):           "sft",

    # Bet 7: pretrain_prose is pretrain; the three Q&A variants are SFT.
    ("synth_code_security", "pretrain_prose"):    "pretrain",
    ("synth_code_security", "identify_and_fix"):  "sft",
    ("synth_code_security", "explain_the_diff"):  "sft",
    ("synth_code_security", "cwe_mapping"):       "sft",

    # Bet 8: pretrain_prose is pretrain; the two Q&A variants are SFT.
    ("synth_binary_literacy", "pretrain_prose"): "pretrain",
    ("synth_binary_literacy", "identify_hex"):   "sft",
    ("synth_binary_literacy", "show_magic"):     "sft",

    # Bet 10: log analysis. Pretrain prose is pretrain; the three Q&A
    # variants are SFT.
    ("synth_log_analysis", "pretrain_prose"):     "pretrain",
    ("synth_log_analysis", "identify_technique"): "sft",
    ("synth_log_analysis", "explain_detection"):  "sft",
    ("synth_log_analysis", "field_citation"):     "sft",

    # Bet 11: cloud IaC security. Same shape as bet 7.
    ("synth_iac_security", "pretrain_prose"):    "pretrain",
    ("synth_iac_security", "identify_and_fix"):  "sft",
    ("synth_iac_security", "explain_the_diff"):  "sft",
    ("synth_iac_security", "severity_mapping"):  "sft",

    # Bet 12: protocol field reading. Pretrain prose + 2 Q&A variants.
    ("synth_protocol_fields", "pretrain_prose"):   "pretrain",
    ("synth_protocol_fields", "identify_protocol"): "sft",
    ("synth_protocol_fields", "read_field"):       "sft",

    # Bets 23/24: general code-explain + code-write (not security-only).
    # Five explain variants and four write variants per pattern.
    ("synth_code_explain", "pretrain_prose"):    "pretrain",
    ("synth_code_explain", "identify_lang"):     "sft",
    ("synth_code_explain", "explain_purpose"):   "sft",
    ("synth_code_explain", "walkthrough"):       "sft",
    ("synth_code_explain", "concepts"):          "sft",

    ("synth_code_write", "pretrain_prose"):  "pretrain",
    ("synth_code_write", "write_function"):  "sft",
    ("synth_code_write", "write_idiomatic"): "sft",
    ("synth_code_write", "compare"):         "sft",

    # Code-reasoning / debugging (synth_code_reasoning.py): prose is
    # pretrain; find / trace / fix are chat-shape SFT.
    ("synth_code_reasoning", "prose"):  "pretrain",
    ("synth_code_reasoning", "find"):   "sft",
    ("synth_code_reasoning", "trace"):  "sft",
    ("synth_code_reasoning", "fix"):    "sft",
}


SYNTH_FILES = [
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
    "synth_code_reasoning.jsonl",
]


def stream_jsonl(path: Path) -> Iterator[Dict]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--in-dir", default="data/processed",
                   help="Directory holding the five synth_*.jsonl outputs")
    p.add_argument("--out", default="data/processed/synth_v15_combined.jsonl",
                   help="Combined JSONL output path")
    p.add_argument("--strict", action="store_true",
                   help="Exit non-zero on any (source, seed_source) tuple "
                        "missing from CATEGORY_RULES")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    in_dir = REPO_ROOT / args.in_dir if not Path(args.in_dir).is_absolute() \
             else Path(args.in_dir)
    out_path = REPO_ROOT / args.out if not Path(args.out).is_absolute() \
               else Path(args.out)

    by_source: Counter = Counter()
    by_format_type: Counter = Counter()
    by_pair: Counter = Counter()
    unknown_pairs: set = set()
    n_total = 0
    n_dropped = 0

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as fout:
        for fname in SYNTH_FILES:
            fpath = in_dir / fname
            if not fpath.exists():
                print(f"  [warn] missing {fpath}; skipping")
                continue
            print(f"  reading {fpath}")
            for rec in stream_jsonl(fpath):
                src = rec.get("source", "")
                seed_src = rec.get("seed_source", "")
                key = (src, seed_src)
                fmt_type = CATEGORY_RULES.get(key)
                if fmt_type is None:
                    unknown_pairs.add(key)
                    n_dropped += 1
                    continue
                rec["format_type"] = fmt_type
                fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                n_total += 1
                by_source[src] += 1
                by_format_type[fmt_type] += 1
                by_pair[key] += 1

    print()
    print("=== per (source, seed_source) ===")
    for (s, ss), c in by_pair.most_common():
        print(f"  {s} / {ss}: {c}")
    print()
    print("=== by format_type ===")
    for ft, c in by_format_type.most_common():
        print(f"  {ft}: {c}")
    print()
    print(f"Wrote {n_total} records to {out_path} "
          f"({n_dropped} dropped to unknown source pairs)")

    if unknown_pairs:
        print(f"\nUnknown (source, seed_source) tuples seen "
              f"(add to CATEGORY_RULES if expected):")
        for pair in sorted(unknown_pairs):
            print(f"  {pair}")
        if args.strict:
            return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
