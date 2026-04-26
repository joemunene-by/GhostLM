"""Per-source perplexity for a GhostLM checkpoint.

The standard val_loss is a single number averaged over the entire
validation set. After Phase 3.5 the val set is heterogeneous — NVD
descriptions, MITRE technique entries, CAPEC patterns, real CTFtime
writeups, arXiv abstracts — and a single average smears all of those
together. The rebalance-vs-baseline question is *not* "which model
has lower overall val_loss" but "which model models each source
better, and how do those per-source numbers shift when the training
mix shifts".

This script answers that question directly. Load a checkpoint, walk
the validation JSONL once, partition by record-level ``source`` field,
compute mean cross-entropy per token within each group, and report
perplexity per source plus the overall.

Run on a v0.3.3 checkpoint vs a v0.3.5 checkpoint with identical val
data and the deltas reveal whether the rebalance specialized the model
on diversity sources at the expected cost on NVD.
"""

import argparse
import json
import math
import sys
import time
from collections import defaultdict
from dataclasses import fields
from pathlib import Path
from typing import Dict, List

import torch

from ghostlm.config import GhostLMConfig
from ghostlm.model import GhostLM
from ghostlm.tokenizer import GhostTokenizer


def load_model(checkpoint_path: str, device: str):
    """Load a GhostLM model from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    saved_config = checkpoint["config"]
    config = GhostLMConfig(**{
        f.name: saved_config[f.name]
        for f in fields(GhostLMConfig)
        if f.name in saved_config
    })
    model = GhostLM(config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    model = model.to(device)
    return model, config


def score_record_chunk(
    model: GhostLM,
    token_ids: List[int],
    device: str,
    context_length: int,
) -> Dict[str, float]:
    """Cross-entropy summed and counted for a single token sequence.

    Returns ``{"loss_sum": float, "n_tokens": int}``. The caller aggregates
    across many records and divides at the end so per-record CE doesn't
    bias toward records with longer texts.

    Sequences longer than the model's context length are processed in
    non-overlapping chunks so every token is scored exactly once. (We
    don't bother with sliding-window scoring — the per-source ranking
    is robust to that detail at this scale.)
    """
    if len(token_ids) < 2:
        return {"loss_sum": 0.0, "n_tokens": 0}

    total_loss = 0.0
    total_n = 0

    chunk_size = context_length
    for start in range(0, len(token_ids), chunk_size):
        chunk = token_ids[start : start + chunk_size]
        if len(chunk) < 2:
            continue

        x = torch.tensor(chunk[:-1], dtype=torch.long, device=device).unsqueeze(0)
        y = torch.tensor(chunk[1:], dtype=torch.long, device=device).unsqueeze(0)

        with torch.no_grad():
            _, loss = model(x, targets=y)

        n = y.numel()
        total_loss += loss.item() * n
        total_n += n

    return {"loss_sum": total_loss, "n_tokens": total_n}


def parse_args():
    p = argparse.ArgumentParser(
        description="Compute per-source perplexity on the validation split."
    )
    p.add_argument(
        "--checkpoint",
        required=True,
        help="Path to GhostLM checkpoint (.pt).",
    )
    p.add_argument(
        "--val",
        default="data/processed/val.jsonl",
        help="Validation JSONL path. Records must carry a 'source' field.",
    )
    p.add_argument(
        "--device",
        default="auto",
        help="Device: auto, cpu, cuda, mps.",
    )
    p.add_argument(
        "--output",
        default="logs/eval_perplexity_by_source.json",
        help="Where to save per-source perplexity numbers as JSON.",
    )
    p.add_argument(
        "--max-per-source",
        type=int,
        default=None,
        help=(
            "Cap on records per source (random sample). Default: all. "
            "Use to bound runtime when the val split has a long tail "
            "in one source."
        ),
    )
    p.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for the per-source subsample (reproducible).",
    )
    return p.parse_args()


def main():
    args = parse_args()

    if args.device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    else:
        device = args.device

    print(f"Loading {args.checkpoint} on {device}...")
    model, config = load_model(args.checkpoint, device)
    tokenizer = GhostTokenizer()
    context_length = config.context_length

    print(f"Loading {args.val}...")
    by_source: Dict[str, List[dict]] = defaultdict(list)
    with open(args.val, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            src = rec.get("source", "unknown")
            by_source[src].append(rec)

    if args.max_per_source is not None:
        import random
        rng = random.Random(args.seed)
        for src in list(by_source.keys()):
            if len(by_source[src]) > args.max_per_source:
                by_source[src] = rng.sample(by_source[src], args.max_per_source)

    print(f"Sources: {', '.join(f'{s}({len(v)})' for s, v in sorted(by_source.items()))}")

    t0 = time.time()
    per_source_stats: Dict[str, Dict[str, float]] = {}

    for src, records in sorted(by_source.items()):
        loss_sum = 0.0
        n_tokens = 0
        for rec in records:
            ids = tokenizer.encode(rec.get("text", ""))
            if not ids:
                continue
            r = score_record_chunk(model, ids, device, context_length)
            loss_sum += r["loss_sum"]
            n_tokens += r["n_tokens"]

        if n_tokens == 0:
            ce = float("inf")
            ppl = float("inf")
        else:
            ce = loss_sum / n_tokens
            ppl = math.exp(min(ce, 50))  # clamp to avoid overflow on degenerate sources

        per_source_stats[src] = {
            "records": len(records),
            "n_tokens": n_tokens,
            "cross_entropy": round(ce, 4),
            "perplexity": round(ppl, 2),
        }
        print(f"  {src:<14}  n={len(records):>5}  tokens={n_tokens:>8}  CE={ce:.4f}  PPL={ppl:.2f}")

    # Token-weighted overall (matches what train_loop computes for val_loss)
    total_loss = sum(s["cross_entropy"] * s["n_tokens"] for s in per_source_stats.values())
    total_tokens = sum(s["n_tokens"] for s in per_source_stats.values())
    overall_ce = total_loss / total_tokens if total_tokens > 0 else float("inf")
    overall_ppl = math.exp(min(overall_ce, 50))

    elapsed = time.time() - t0

    print()
    print(f"Overall  CE={overall_ce:.4f}  PPL={overall_ppl:.2f}  ({total_tokens:,} tokens, {elapsed:.1f}s)")

    save_data = {
        "checkpoint": args.checkpoint,
        "device": device,
        "val": args.val,
        "max_per_source": args.max_per_source,
        "seed": args.seed,
        "elapsed_seconds": round(elapsed, 1),
        "by_source": per_source_stats,
        "overall": {
            "cross_entropy": round(overall_ce, 4),
            "perplexity": round(overall_ppl, 2),
            "n_tokens": total_tokens,
        },
    }

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(save_data, f, indent=2)
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
