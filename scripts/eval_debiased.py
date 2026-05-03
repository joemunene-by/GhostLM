#!/usr/bin/env python3
"""Debiased CTIBench MCQ evaluation that controls for positional bias.

The CTIBench MCQ gold-letter distribution is heavily skewed: A=15%, B=32%,
C=37%, D=15%. A model that always picks C scores 37.1% on it, which is
roughly the same as our "canonical" chat-v3 at 36.9%. This means a single-
ordering eval cannot tell us whether the model has learned cybersec or
just learned to pick C.

This script fixes that by scoring each record under N different option
permutations and counting a record correct only when the model picks the
gold answer regardless of which letter the gold is mapped to. A pure
positional-bias model collapses to 25% (random); a model with real
capability stays close to its single-ordering score.

Outputs:
- single-order accuracy (matches run_bench.py for sanity check)
- debiased accuracy (consistency-correct across permutations)
- prediction-letter distribution (audit for C-bias)
- per-gold-letter accuracy (where is the model strong vs weak)

Usage:
    PYTHONPATH=. python3 scripts/eval_debiased.py \\
        --checkpoint checkpoints/phase5_chat_v3/best_model.pt \\
        --label "chat-v3 canonical" \\
        --device mps \\
        --out-json logs/phase5_chat_v3/bench_debiased.json
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from dataclasses import fields
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

from ghostlm.config import GhostLMConfig
from ghostlm.model import GhostLM
from ghostlm.tokenizer import GhostTokenizer, load_tokenizer

from scripts.run_bench import (
    CHOICES, format_mcq_prompt, load_ctibench_mcq,
)


def parse_args() -> argparse.Namespace:
    """CLI args."""
    p = argparse.ArgumentParser(description="Debiased CTIBench MCQ eval")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--tokenizer", default=None,
                   help="Optional v0.5 tokenizer.json")
    p.add_argument("--label", required=True)
    p.add_argument("--device", default="mps")
    p.add_argument("--limit", type=int, default=None,
                   help="Cap records (for smoke testing)")
    p.add_argument("--n-permutations", type=int, default=4,
                   help="Number of option-order permutations per record")
    p.add_argument("--out-json", default=None)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def score_one(
    model: GhostLM,
    tokenizer: GhostTokenizer,
    record: Dict,
    *,
    chat_format: bool,
    device: str,
) -> str:
    """Score a single record (with whatever option order it currently has)
    and return the predicted letter."""
    prompt_ids = format_mcq_prompt(record, tokenizer, chat_format=chat_format)
    x = torch.tensor(prompt_ids, dtype=torch.long, device=device).unsqueeze(0)
    ctx = model.config.context_length
    x = x[:, -ctx:]
    with torch.no_grad():
        logits, _ = model(x)
    next_logits = logits[0, -1, :]
    log_probs = F.log_softmax(next_logits, dim=-1)
    scores: Dict[str, float] = {}
    for ch in CHOICES:
        ids_space = tokenizer.encode(f" {ch}")
        ids_plain = tokenizer.encode(ch)
        candidates = [ids_space[0]] if ids_space else []
        if ids_plain:
            candidates.append(ids_plain[0])
        scores[ch] = max(log_probs[c].item() for c in candidates)
    return max(scores.items(), key=lambda kv: kv[1])[0]


def permute_record(record: Dict, perm: List[str]) -> Tuple[Dict, str]:
    """Return a copy of ``record`` whose options are reordered such that
    the original A-option is now at perm[0], original B at perm[1], etc.

    Also returns the new letter that the original gold answer maps to.
    """
    orig_choices = record["choices"]
    orig_gold = record["answer"]
    # Map original-letter -> new-letter
    orig_to_new = dict(zip(CHOICES, perm))
    new_to_orig = {v: k for k, v in orig_to_new.items()}
    new_choices = {
        new_letter: orig_choices.get(new_to_orig[new_letter])
        for new_letter in CHOICES
    }
    new_gold = orig_to_new.get(orig_gold)
    return (
        {**record, "choices": new_choices, "answer": new_gold},
        new_gold,
    )


def main() -> None:
    """Run debiased eval across N permutations per record."""
    args = parse_args()
    rng = random.Random(args.seed)

    # Load checkpoint and rebuild model
    print(f"Checkpoint: {args.label}  device={args.device}")
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    saved = ckpt["config"]
    cfg = GhostLMConfig(**{
        f.name: saved[f.name] for f in fields(GhostLMConfig) if f.name in saved
    })
    cfg.device = args.device
    tokenizer = load_tokenizer(args.tokenizer) if args.tokenizer else GhostTokenizer()
    print(f"Tokenizer vocab: {tokenizer.vocab_size}")

    model = GhostLM(cfg).to(args.device)
    state = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt["model"]
    model.load_state_dict(state, strict=False)
    model.eval()

    chat_format = "ghost_user" in str(getattr(tokenizer, "_special_tokens", {}))

    # Load CTIBench
    ds = load_ctibench_mcq()
    if args.limit:
        ds = ds[: args.limit]
    print(f"Loaded {len(ds)} CTIBench MCQ records")

    # Generate N permutations (deterministic per seed)
    perms: List[List[str]] = [list(CHOICES)]  # identity first (matches run_bench)
    seen = {tuple(perms[0])}
    while len(perms) < args.n_permutations:
        cand = list(CHOICES)
        rng.shuffle(cand)
        if tuple(cand) not in seen:
            perms.append(cand)
            seen.add(tuple(cand))
    print(f"Permutations: {perms}")

    # For each record, score under each permutation. Track:
    # - single_order_correct (perm 0 only) — should match run_bench.py
    # - all_correct (all permutations correct = consistent capability)
    # - any_correct (at least one permutation correct = upper bound)
    # - pred_letter_dist (across all perms)
    # - per_gold_acc (dict of gold letter -> accuracy at perm 0)
    single_correct = 0
    all_correct = 0
    any_correct = 0
    counted = 0
    pred_dist: Counter = Counter()
    per_gold_correct: Dict[str, int] = {ch: 0 for ch in CHOICES}
    per_gold_total: Dict[str, int] = {ch: 0 for ch in CHOICES}
    per_perm_correct: List[int] = [0] * len(perms)

    for i, rec in enumerate(ds):
        if not rec["answer"] or rec["answer"] not in CHOICES:
            continue
        counted += 1
        per_gold_total[rec["answer"]] += 1

        results = []
        for j, perm in enumerate(perms):
            permuted_rec, new_gold = permute_record(rec, perm)
            pred = score_one(
                model, tokenizer, permuted_rec,
                chat_format=chat_format, device=args.device,
            )
            ok = (pred == new_gold)
            results.append(ok)
            per_perm_correct[j] += int(ok)
            if j == 0:
                pred_dist[pred] += 1
                if ok:
                    per_gold_correct[rec["answer"]] += 1

        if results[0]:
            single_correct += 1
        if all(results):
            all_correct += 1
        if any(results):
            any_correct += 1

        if (i + 1) % 200 == 0:
            print(f"  [{i + 1}/{len(ds)}] single={single_correct}/{counted} "
                  f"({single_correct / counted:.3f})  "
                  f"all-perm={all_correct}/{counted} "
                  f"({all_correct / counted:.3f})")

    # Aggregate
    print()
    print(f"=== {args.label} ===")
    print(f"  single-order accuracy: {single_correct}/{counted} = {single_correct / counted:.3f}")
    print(f"  any-perm accuracy:     {any_correct}/{counted} = {any_correct / counted:.3f}  (upper bound)")
    print(f"  all-perm accuracy:     {all_correct}/{counted} = {all_correct / counted:.3f}  (debiased: must be right under every order)")
    avg_perm = sum(per_perm_correct) / (len(perms) * counted)
    print(f"  per-perm avg accuracy: {avg_perm:.3f}  (mean across {len(perms)} permutations — true expected accuracy of a fresh shuffle)")
    print()
    print(f"  prediction-letter distribution (perm 0):")
    for ch in CHOICES:
        n = pred_dist.get(ch, 0)
        print(f"    {ch}: {n:4d} ({n / counted * 100:.1f}%)")
    print()
    print(f"  per-gold-letter accuracy (perm 0):")
    for ch in CHOICES:
        n = per_gold_correct[ch]
        d = per_gold_total[ch]
        pct = n / d * 100 if d else 0.0
        print(f"    gold={ch}: {n:4d}/{d:4d} ({pct:.1f}%)")
    print()
    print(f"  per-permutation accuracy:")
    for j, (perm, n_correct) in enumerate(zip(perms, per_perm_correct)):
        print(f"    perm {j} {''.join(perm)}: {n_correct}/{counted} = {n_correct / counted:.3f}")

    if args.out_json:
        out_path = Path(args.out_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps({
            "label": args.label,
            "checkpoint": args.checkpoint,
            "n_records": counted,
            "single_order_acc": single_correct / counted,
            "any_perm_acc": any_correct / counted,
            "all_perm_acc": all_correct / counted,
            "per_perm_avg_acc": avg_perm,
            "per_perm_acc": [n / counted for n in per_perm_correct],
            "permutations": [list(p) for p in perms],
            "pred_dist_perm0": dict(pred_dist),
            "per_gold_correct_perm0": per_gold_correct,
            "per_gold_total": per_gold_total,
        }, indent=2))
        print(f"  saved: {out_path}")


if __name__ == "__main__":
    main()
