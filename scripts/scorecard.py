#!/usr/bin/env python3
"""GhostLM generalist scorecard: one table, all benchmarks, peer context.

Scores a checkpoint across the cybersecurity benchmarks AND the new
general rulers (ARC-Easy, ARC-Challenge, OpenBookQA) with the project's
debiased multi-permutation text-scoring, then prints a single markdown
scorecard that places each number next to a random baseline and a
published peer small-model number. The point is to answer "is GhostLM
good *for its size*?" with evidence, not vibes.

Peer reference numbers are published zero-shot accuracies for models in
the 50-360M class; sources are listed under ``PEER_REFERENCE``. They are
context, not exact apples-to-apples (eval harnesses differ), so the
scorecard labels them as reference.

Usage:
    python scripts/scorecard.py --checkpoint <ckpt> --label ghost-small-gen \
        --device mps --out docs/scorecard.md
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

# Published zero-shot reference points for the small-model class. Sources:
#   Pythia-160M: EleutherAI Pythia paper / model card (ARC-E 43.5, ARC-C 18.8,
#     HellaSwag 30.3).
#   ~111M / ~256M rows: small-LM survey (arXiv 2409.15790) zero-shot
#     (OpenBookQA 27.8 / ARC-E 34.8 at 111M; 25.4 / 37.6 at 256M).
#   SmolLM2-360M: model card (ARC-C 36.6, MMLU 20.2).
# Numbers are percentages. None = not reported in the sources gathered.
PEER_REFERENCE: Dict[str, Dict[str, Optional[float]]] = {
    "arc_easy":      {"random": 25.0, "pythia_160m": 43.5, "small_111m": 34.8, "small_256m": 37.6},
    "arc_challenge": {"random": 25.0, "pythia_160m": 18.8, "smollm2_360m": 36.6},
    "openbookqa":    {"random": 25.0, "small_111m": 27.8, "small_256m": 25.4, "lamini_35m": 26.2},
    "secqa":         {"random": 25.0},
    "ctf_eval_bench": {"random": 25.0},
    "math":          {"random": 25.0},
}

# "Competitive for a 50-100M model" bands, from the survey discussion.
COMPETITIVE_BAND = {
    "arc_easy": "35-45%", "openbookqa": "25-35%",
    "arc_challenge": ">25%", "secqa": ">25%", "ctf_eval_bench": ">25%",
    "math": ">25%",
}


@dataclass
class BenchSpec:
    key: str
    path: str
    prompt_style: str
    label: str
    bench_filter: Optional[str] = None  # for general_mcq_bench sub-benches


# The benchmark set the scorecard runs. General rulers come from the single
# general_mcq_bench file, split by their 'bench' field.
def default_benches(raw_dir: Path) -> List[BenchSpec]:
    gmcq = str(raw_dir / "general_mcq_bench.jsonl")
    return [
        BenchSpec("arc_easy", gmcq, "general", "ARC-Easy", "arc_easy"),
        BenchSpec("arc_challenge", gmcq, "general", "ARC-Challenge", "arc_challenge"),
        BenchSpec("openbookqa", gmcq, "general", "OpenBookQA", "openbookqa"),
        BenchSpec("secqa", str(raw_dir / "secqa.jsonl"), "cybersec", "SecQA"),
        BenchSpec("ctf_eval_bench", str(raw_dir / "ctf_eval_bench.jsonl"),
                  "cybersec", "CTF eval"),
        # Math numeracy ruler (scripts/build_math_eval.py); single aggregate row.
        BenchSpec("math", str(raw_dir / "math_mcq_bench.jsonl"), "general",
                  "Math (numeracy)"),
    ]


def _load_records(spec: BenchSpec, limit: Optional[int] = None):
    from scripts.eval_text_scoring import load_jsonl_mcq
    recs = load_jsonl_mcq(spec.path)
    if spec.bench_filter:
        recs = [r for r in recs if r.get("bench") == spec.bench_filter]
    if limit:
        recs = recs[:limit]
    return recs


def bootstrap_ci(question_accs: List[float], n_resamples: int = 2000,
                 seed: int = 0, alpha: float = 0.05) -> tuple:
    """Percentile bootstrap CI on mean accuracy, resampling over questions.

    ``question_accs`` is per-question accuracy (mean correctness across
    permutations). Returns ``(lo, hi)`` as percentages at the (1-alpha)
    level. Resampling questions (not individual perm outcomes) is the
    honest unit, the questions are the sample, the permutations are just
    de-biasing. Pure and deterministic given the seed.
    """
    import random as _random
    if not question_accs:
        return (0.0, 0.0)
    rng = _random.Random(seed)
    n = len(question_accs)
    means = []
    for _ in range(n_resamples):
        s = 0.0
        for _ in range(n):
            s += question_accs[rng.randrange(n)]
        means.append(s / n)
    means.sort()
    lo = means[int(alpha / 2 * n_resamples)]
    hi = means[int((1 - alpha / 2) * n_resamples) - 1]
    return (100 * lo, 100 * hi)


def score_checkpoint(args) -> Dict[str, Dict]:
    """Score the checkpoint on every bench; return {key: {acc, n, perms}}."""
    import torch
    from dataclasses import fields as dc_fields
    from ghostlm.config import GhostLMConfig
    from ghostlm.model import GhostLM
    from ghostlm.tokenizer import GhostTokenizer, load_tokenizer
    from scripts.run_bench import CHOICES
    from scripts.eval_text_scoring import evaluate_one_perm
    from scripts.eval_debiased import permute_record  # noqa: F401 (used indirectly)
    import random

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    cfg_dict = ckpt.get("config", {})
    valid = {f.name for f in dc_fields(GhostLMConfig)}
    cfg = GhostLMConfig(**{k: v for k, v in cfg_dict.items() if k in valid})
    cfg.device = args.device
    tokenizer = load_tokenizer(args.tokenizer) if args.tokenizer else GhostTokenizer()
    model = GhostLM(cfg).to(args.device)
    state = ckpt.get("model_state_dict", ckpt.get("model", ckpt))
    model.load_state_dict(state)
    model.eval()
    chat_format = "ghost_user" in str(getattr(tokenizer, "_special_tokens", {}))

    rng = random.Random(args.seed)
    perms: List[List[str]] = [list(CHOICES)]
    seen = {tuple(perms[0])}
    while len(perms) < args.n_permutations:
        cand = list(CHOICES)
        rng.shuffle(cand)
        if tuple(cand) not in seen:
            perms.append(cand)
            seen.add(tuple(cand))

    results: Dict[str, Dict] = {}
    for spec in default_benches(Path(args.raw_dir)):
        if not Path(spec.path).exists():
            continue
        recs = _load_records(spec, limit=args.limit_per_bench)
        if not recs:
            continue
        accs = []
        per_q_sum = [0.0] * len(recs)   # summed correctness per question
        per_q_cnt = [0] * len(recs)     # scored permutations per question
        for j, perm in enumerate(perms):
            correct, total, _, per_q = evaluate_one_perm(
                model, tokenizer, recs, chat_format=chat_format, device=args.device,
                score_mode="per-token-avg", perm=perm,
                progress_label=f"{args.label}:{spec.key}", prompt_style=spec.prompt_style)
            if total:
                accs.append(correct / total)
            for i, c in enumerate(per_q):
                if c >= 0:
                    per_q_sum[i] += c
                    per_q_cnt[i] += 1
        if accs:
            mean = 100 * sum(accs) / len(accs)
            q_accs = [per_q_sum[i] / per_q_cnt[i] for i in range(len(recs)) if per_q_cnt[i] > 0]
            lo, hi = bootstrap_ci(q_accs, seed=args.seed)
            results[spec.key] = {"acc": mean, "ci": (lo, hi), "n": len(q_accs),
                                 "perms": len(accs), "label": spec.label}
            print(f"  {spec.label}: {mean:.1f}% (95% CI {lo:.1f}-{hi:.1f}, "
                  f"n={len(q_accs)}, {len(accs)} perms)")
    return results


def render_scorecard(label: str, results: Dict[str, Dict]) -> str:
    lines = [f"# GhostLM scorecard — {label}", "",
             "Debiased multi-permutation text-scoring; 95% CI is a percentile "
             "bootstrap over questions. A score whose CI lower bound is above "
             "the 25% random baseline is significantly better than chance "
             "(marked +). Peer numbers are published zero-shot references for "
             "the small-model class (different harnesses; context, not exact "
             "comparison).", "",
             "| Benchmark | n | GhostLM | 95% CI | vs random | Competitive band | Peer reference |",
             "|---|---:|---:|---:|:--:|---|---|"]
    for key, ref in PEER_REFERENCE.items():
        r = results.get(key)
        if r:
            gh = f"**{r['acc']:.1f}%**"
            lo, hi = r["ci"]
            ci = f"{lo:.1f}-{hi:.1f}"
            sig = "+" if lo > ref["random"] else ("~" if hi > ref["random"] else "-")
            n = r["n"]
        else:
            gh, ci, sig, n = "—", "—", "", "—"
        band = COMPETITIVE_BAND.get(key, "")
        peers = ", ".join(f"{k}={v:.1f}" for k, v in ref.items() if k != "random" and v is not None)
        lines.append(f"| {key} | {n} | {gh} | {ci} | {sig} | {band} | {peers or '—'} |")
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--label", required=True)
    p.add_argument("--tokenizer", default=None)
    p.add_argument("--device", default="mps")
    p.add_argument("--raw-dir", default="data/raw")
    p.add_argument("--n-permutations", type=int, default=4)
    p.add_argument("--limit-per-bench", type=int, default=None,
                   help="Cap questions per bench (cheap mid-run reads; omit for full).")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", default="docs/scorecard.md")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    results = score_checkpoint(args)
    md = render_scorecard(args.label, results)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(md, encoding="utf-8")
    print("\n" + md)
    print(f"scorecard -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
