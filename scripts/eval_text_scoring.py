#!/usr/bin/env python3
"""CTIBench MCQ eval that scores option TEXT, not the letter token.

Hypothesis: our chat-tunes might know cybersec content but only express it
through letter emission. The eval_debiased.py results show every model is
a single-letter emitter under letter-token scoring. But what if we scored
each option's full text logprob given the prompt and picked the highest?
A model that learned content (but happens to also emit letters) should
score above chance under this scoring.

For each record, we compute log P(option_text | prompt) for each of the 4
options and pick the highest. To control for option length, we use
per-token average log-prob (the standard length-normalized score from
lm-eval-harness).

We also run this under N permutations of option order so positional bias
on the LETTER side doesn't pollute the result. Since we are scoring text,
the letter-bias should not matter, but we verify by checking that the
permutation accuracies are consistent.
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

from scripts.run_bench import CHOICES, load_ctibench_mcq
from scripts.eval_debiased import permute_record


def parse_args() -> argparse.Namespace:
    """CLI args."""
    p = argparse.ArgumentParser(description="MCQ eval scoring option TEXT (not letter)")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--tokenizer", default=None)
    p.add_argument("--label", required=True)
    p.add_argument("--device", default="mps")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--n-permutations", type=int, default=4)
    p.add_argument("--score-mode", choices=["per-token-avg", "total"], default="per-token-avg")
    p.add_argument("--out-json", default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--bench-jsonl", default=None,
                   help="Path to a JSONL file with {question, choices, answer} "
                        "records. If unset, uses CTIBench MCQ.")
    return p.parse_args()


def load_jsonl_mcq(path: str) -> List[Dict]:
    """Load a generic MCQ JSONL: {question, choices: {A,B,C,D}, answer}."""
    out: List[Dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if not r.get("question") or not r.get("choices") or not r.get("answer"):
                continue
            out.append(r)
    return out


def format_prompt(record: Dict, tokenizer: GhostTokenizer, *, chat_format: bool) -> List[int]:
    """Build the prompt token ids that end with 'Answer:' (no letter yet).
    Same as run_bench.py format_mcq_prompt without RAG."""
    question = record["question"]
    choices = record["choices"]
    body_lines = [f"{k}) {v}" for k, v in choices.items() if v]
    body = (
        f"Pick the best answer (A, B, C, or D) for this multiple-choice "
        f"cybersecurity question.\n\nQuestion: {question}\n\n"
        + "\n".join(body_lines)
        + "\n\nAnswer:"
    )
    if chat_format:
        return tokenizer.format_chat_prompt([{"role": "user", "content": body}])
    return tokenizer.encode(body)


def score_option_text(
    model: GhostLM,
    tokenizer: GhostTokenizer,
    prompt_ids: List[int],
    option_text: str,
    *,
    device: str,
    score_mode: str,
) -> float:
    """Compute log P(option_text | prompt) for one option, length-normalized.

    The prompt ends with "Answer:". We score the continuation
    " <letter>. <option_text>" and return per-token avg log-prob (or total
    log-prob, depending on score_mode). Higher = model thinks this option
    is more likely.
    """
    # We want to score the option text itself, not just one letter, so we
    # encode " " + option_text as the continuation. The leading space
    # matches natural completion after "Answer:".
    cont_ids = tokenizer.encode(" " + option_text)
    if not cont_ids:
        return float("-inf")

    # Build full sequence: prompt + continuation
    full_ids = list(prompt_ids) + list(cont_ids)
    ctx = model.config.context_length
    # Crop on the left if too long; we MUST keep all continuation tokens
    if len(full_ids) > ctx:
        excess = len(full_ids) - ctx
        full_ids = full_ids[excess:]
        # Adjust where the continuation starts
        cont_start = len(full_ids) - len(cont_ids)
    else:
        cont_start = len(prompt_ids)

    x = torch.tensor(full_ids, dtype=torch.long, device=device).unsqueeze(0)
    with torch.no_grad():
        logits, _ = model(x)
    # logits[0, t] predicts token at position t+1
    # We want log P(token[t]) for t in [cont_start, cont_start+len(cont_ids))
    log_probs = F.log_softmax(logits[0], dim=-1)
    target_log_probs = []
    for i, tok_id in enumerate(cont_ids):
        # The model at position (cont_start + i - 1) predicts token at (cont_start + i)
        pred_pos = cont_start + i - 1
        if pred_pos < 0:
            continue
        target_log_probs.append(log_probs[pred_pos, tok_id].item())

    if not target_log_probs:
        return float("-inf")

    total = sum(target_log_probs)
    if score_mode == "per-token-avg":
        return total / len(target_log_probs)
    return total


def evaluate_one_perm(
    model: GhostLM,
    tokenizer: GhostTokenizer,
    dataset: List[Dict],
    *,
    chat_format: bool,
    device: str,
    score_mode: str,
    perm: List[str],
    progress_label: str,
) -> Tuple[int, int, Counter, List[int]]:
    """Score every record under one permutation.

    Returns (correct, total, pred_dist, per_question_correct) where the
    last item is a list with 1 if the model got record i right, 0 if
    wrong, -1 if the record was skipped (no gold answer). Same length
    as the input dataset.
    """
    correct = 0
    total = 0
    pred_dist: Counter = Counter()
    per_q: List[int] = []
    for i, rec in enumerate(dataset):
        if not rec["answer"] or rec["answer"] not in CHOICES:
            per_q.append(-1)
            continue
        permuted_rec, new_gold = permute_record(rec, perm)
        prompt_ids = format_prompt(permuted_rec, tokenizer, chat_format=chat_format)

        scores: Dict[str, float] = {}
        for letter in CHOICES:
            option_text = permuted_rec["choices"].get(letter)
            if not option_text:
                scores[letter] = float("-inf")
                continue
            scores[letter] = score_option_text(
                model, tokenizer, prompt_ids, option_text,
                device=device, score_mode=score_mode,
            )
        pred = max(scores.items(), key=lambda kv: kv[1])[0]
        pred_dist[pred] += 1
        is_correct = 1 if pred == new_gold else 0
        per_q.append(is_correct)
        if is_correct:
            correct += 1
        total += 1

        if (i + 1) % 200 == 0:
            print(f"  [{progress_label}] {i + 1}/{len(dataset)} "
                  f"acc={correct / total:.3f}")
    return correct, total, pred_dist, per_q


def main() -> None:
    """Run option-text scoring across N permutations."""
    args = parse_args()
    rng = random.Random(args.seed)

    print(f"Checkpoint: {args.label}  device={args.device}  score_mode={args.score_mode}")
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    saved = ckpt["config"]
    cfg = GhostLMConfig(**{
        f.name: saved[f.name] for f in fields(GhostLMConfig) if f.name in saved
    })
    cfg.device = args.device
    tokenizer = load_tokenizer(args.tokenizer) if args.tokenizer else GhostTokenizer()

    model = GhostLM(cfg).to(args.device)
    state = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt["model"]
    model.load_state_dict(state, strict=False)
    model.eval()

    chat_format = "ghost_user" in str(getattr(tokenizer, "_special_tokens", {}))

    if args.bench_jsonl:
        ds = load_jsonl_mcq(args.bench_jsonl)
        bench_name = args.bench_jsonl
    else:
        ds = load_ctibench_mcq()
        bench_name = "CTIBench"
    if args.limit:
        ds = ds[: args.limit]
    print(f"Loaded {len(ds)} {bench_name} MCQ records")

    perms: List[List[str]] = [list(CHOICES)]
    seen = {tuple(perms[0])}
    while len(perms) < args.n_permutations:
        cand = list(CHOICES)
        rng.shuffle(cand)
        if tuple(cand) not in seen:
            perms.append(cand)
            seen.add(tuple(cand))

    per_perm_results: List[Tuple[int, int, Counter, List[int]]] = []
    for j, perm in enumerate(perms):
        print(f"=== perm {j} {''.join(perm)} ===")
        correct, total, pred_dist, per_q = evaluate_one_perm(
            model, tokenizer, ds,
            chat_format=chat_format, device=args.device,
            score_mode=args.score_mode, perm=perm,
            progress_label=f"{args.label} perm{j}",
        )
        per_perm_results.append((correct, total, pred_dist, per_q))

    print()
    print(f"=== {args.label} text-scoring results ===")
    counted = per_perm_results[0][1]
    for j, (correct, total, pred_dist, _) in enumerate(per_perm_results):
        print(f"  perm {j} {''.join(perms[j])}: {correct}/{total} = {correct / total:.3f}  "
              f"pred_dist={dict(pred_dist)}")
    avg = sum(c for c, _, _, _ in per_perm_results) / (len(perms) * counted)
    print(f"  per-perm avg: {avg:.3f}  (random baseline = 0.250)")

    if args.out_json:
        out_path = Path(args.out_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps({
            "label": args.label,
            "checkpoint": args.checkpoint,
            "score_mode": args.score_mode,
            "n_records": counted,
            "per_perm_acc": [c / t for c, t, _, _ in per_perm_results],
            "per_perm_avg": avg,
            "per_perm_pred_dist": [dict(pd) for _, _, pd, _ in per_perm_results],
            "per_perm_per_question": [pq for _, _, _, pq in per_perm_results],
            "permutations": [list(p) for p in perms],
        }, indent=2))
        print(f"  saved: {out_path}")


if __name__ == "__main__":
    main()
