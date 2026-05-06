#!/usr/bin/env python3
"""Split a CTIBench eval result by contamination subset.

Cross-references a per-question text-scoring run (saved by
``eval_text_scoring.py --out-json``, which now records
``per_perm_per_question``) with the contamination audit JSON
(``logs/ctibench_contamination.json`` from
``audit_ctibench_contamination.py``).

Reports per-permutation accuracy on the contaminated subset
(questions with at least one shingle overlap to PRIMUS) vs. the
clean subset, and the delta. If a checkpoint is gaining on the
contaminated subset and losing on the clean subset, the FineWeb
exposure is helping where the model saw the source and hurting
where it didn't, which is one of the candidate explanations for
the v0.9 CTIBench regression.

Usage::

    python scripts/analyze_contamination_split.py \\
        --eval-json logs/text_scoring/chat-v09.json \\
        --contam-json logs/ctibench_contamination.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    """CLI args."""
    p = argparse.ArgumentParser(description="Contamination subset split for CTIBench")
    p.add_argument("--eval-json", required=True,
                   help="Output of eval_text_scoring.py with per_perm_per_question")
    p.add_argument("--contam-json", required=True,
                   help="Output of audit_ctibench_contamination.py")
    p.add_argument("--min-overlap", type=int, default=1,
                   help="Min shingles overlap for a question to count as contaminated")
    return p.parse_args()


def main() -> None:
    """Cross-reference and report."""
    args = parse_args()
    eval_data = json.loads(Path(args.eval_json).read_text())
    contam_data = json.loads(Path(args.contam_json).read_text())

    # Build a per-idx contamination flag from the audit's top_contaminated +
    # the n_contaminated_questions count. The audit JSON only stores top-N
    # explicitly, but we have per-question records in the broader scan; if
    # the audit only saved top-N, fall back to the count.
    # The audit stores an entry for EVERY question at any overlap level;
    # check by reconstructing from the top_contaminated list (which is
    # ALL the records sorted by overlap_shingles).
    # Actually `top_contaminated` is sliced to `--top-n`. We need every
    # question's overlap. Re-run the audit with --top-n big OR look at
    # the file.
    # Workaround: assume the audit kept all questions if --top-n was
    # >= n_questions. Look at length.
    n_q = contam_data.get("n_questions", 0)
    per_q_audit = contam_data.get("per_question") or contam_data.get("top_contaminated", [])
    if len(per_q_audit) < n_q:
        raise SystemExit(
            f"Contamination JSON has only {len(per_q_audit)} of {n_q} per-question entries. "
            f"Re-run audit_ctibench_contamination.py (this version saves every "
            f"question into a 'per_question' field automatically)."
        )

    contam_by_idx: dict[int, int] = {
        r["idx"]: r["overlap_shingles"] for r in per_q_audit
    }

    perms = eval_data["permutations"]
    per_q_results = eval_data["per_perm_per_question"]
    n_records = eval_data["n_records"]

    print(f"Eval: {eval_data['label']}")
    print(f"  n_records: {n_records}")
    print(f"  per-perm avg overall: {eval_data['per_perm_avg']:.4f}")
    print()
    print(f"Contamination cutoff: overlap_shingles >= {args.min_overlap}")
    print()

    n_contam = 0
    n_clean = 0
    for idx, ovr in contam_by_idx.items():
        if ovr >= args.min_overlap:
            n_contam += 1
        else:
            n_clean += 1
    print(f"  contaminated questions: {n_contam}")
    print(f"  clean questions:        {n_clean}")
    print()

    print(f"{'perm':<6} {'overall':>8} {'contam':>8} {'clean':>8} {'delta':>8}")
    overall_contam_correct = 0
    overall_clean_correct = 0
    overall_contam_total = 0
    overall_clean_total = 0
    for j, per_q in enumerate(per_q_results):
        contam_correct = contam_total = 0
        clean_correct = clean_total = 0
        for idx, c in enumerate(per_q):
            if c < 0:
                continue
            ovr = contam_by_idx.get(idx, 0)
            if ovr >= args.min_overlap:
                contam_total += 1
                contam_correct += c
            else:
                clean_total += 1
                clean_correct += c
        contam_acc = contam_correct / contam_total if contam_total else 0.0
        clean_acc = clean_correct / clean_total if clean_total else 0.0
        overall_acc = (contam_correct + clean_correct) / (contam_total + clean_total) if (contam_total + clean_total) else 0.0
        print(f"{''.join(perms[j]):<6} {overall_acc:>8.4f} {contam_acc:>8.4f} "
              f"{clean_acc:>8.4f} {(contam_acc - clean_acc):>+8.4f}")
        overall_contam_correct += contam_correct
        overall_clean_correct += clean_correct
        overall_contam_total += contam_total
        overall_clean_total += clean_total

    print()
    avg_contam = overall_contam_correct / overall_contam_total if overall_contam_total else 0.0
    avg_clean = overall_clean_correct / overall_clean_total if overall_clean_total else 0.0
    avg_overall = (overall_contam_correct + overall_clean_correct) / (overall_contam_total + overall_clean_total) if (overall_contam_total + overall_clean_total) else 0.0
    print(f"{'avg':<6} {avg_overall:>8.4f} {avg_contam:>8.4f} "
          f"{avg_clean:>8.4f} {(avg_contam - avg_clean):>+8.4f}")
    print()
    print("Reading: a positive contam-clean delta means the checkpoint does better")
    print("on questions whose source it saw during pretrain, which would be the")
    print("contamination-helps signature. A flat or negative delta means")
    print("contamination isn't the lever.")


if __name__ == "__main__":
    main()
