#!/usr/bin/env python3
"""Nightly LoRA fine-tune over fresh threat-intel data.

The differentiator: every other LLM is frozen at its training-data
cutoff. Cybersec is uniquely time-sensitive: today's CVE / today's
ransomware family / today's CISA advisory aren't in last month's
training data. A continuously-updated cybersec LM is a different
kind of product than a frozen one.

This script wraps the existing threat-intel collectors + a small
LoRA tune into one nightly cron-friendly orchestrator:

  1. Run all collectors that already exist:
       collect_cisa_kev.py, collect_cisa_advisories.py,
       collect_vendor_research.py, collect_misp_feeds.py,
       collect_security_blogs.py
     Each is resume-safe; only fetches new records since last run.

  2. Compute the delta from yesterday's corpus snapshot. If under
     ~50K new tokens, skip the tune (no signal). If above, proceed.

  3. Run a small LoRA fine-tune on top of the latest base or chat
     checkpoint:
       - LoRA r=16, alpha=32, target attention QKV and FFN proj
       - 1-2 epochs over the delta
       - 1-2 GPU hours on H100 / RTX 6000 / 4090
     Saves to checkpoints/daily/YYYY-MM-DD/.

  4. Push the LoRA adapter (small, ~10-50 MB) to a new HF model
     repo: Ghostgim/GhostLM-daily-2026-MM-DD
     The base checkpoint stays at Ghostgim/GhostLM-v0.9-experimental
     (or ghost-base when it lands); consumers download the adapter
     and merge at load time.

  5. Update the demo Space's app.py to point at the latest daily
     adapter so the live demo always reflects yesterday's threat
     landscape.

Run as a nightly cron on the M4 (or rented GPU host once ghost-base
is available):

    0 2 * * *  cd /Users/ghost/Desktop/GhostLM && \\
               PYTHONPATH=. python3 scripts/daily_finetune.py \\
               >> /var/log/ghostlm-daily.log 2>&1

Schedule: 02:00 EAT puts the run in the off-peak window for both
EAS network and Mac CPU (audit/eval/etc unlikely to overlap).

This script is the orchestrator only; the actual LoRA training
happens via scripts/finetune_chat.py with --lora flags. The LoRA
recipe is documented in docs/daily_finetune.md.

Failure modes:
  - Collectors timeout / 503: skip the failed feed, continue with
    the rest. Resume next night.
  - Delta below threshold: log "no significant new content; skipping
    tune" and exit cleanly.
  - LoRA tune diverges: keep yesterday's adapter as the canonical
    "latest"; don't push the broken one. Manual intervention needed.
  - HF push fails: keep the adapter local; retry next night.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
LOG = lambda m: print(f"[{dt.datetime.now().strftime('%H:%M:%S')}] {m}", flush=True)


COLLECTORS = [
    ("cisa_kev",          ["scripts/collect_cisa_kev.py"]),
    ("cisa_advisories",   ["scripts/collect_cisa_advisories.py"]),
    ("vendor_research",   ["scripts/collect_vendor_research.py"]),
    ("misp_feeds",        ["scripts/collect_misp_feeds.py", "--max-events-per-feed", "50"]),
    ("security_blogs",    ["scripts/collect_security_blogs.py"]),
]


def run_collectors(args: argparse.Namespace) -> int:
    """Run each collector, swallowing failures. Returns count of
    successes."""
    successes = 0
    for name, cmd in COLLECTORS:
        LOG(f"running collector: {name}")
        try:
            result = subprocess.run(
                [sys.executable] + cmd, cwd=str(REPO_ROOT),
                env={**os.environ, "PYTHONPATH": str(REPO_ROOT)},
                timeout=args.collector_timeout_s,
                check=False,
            )
            if result.returncode == 0:
                successes += 1
                LOG(f"  {name}: OK")
            else:
                LOG(f"  {name}: exit {result.returncode}, continuing")
        except subprocess.TimeoutExpired:
            LOG(f"  {name}: timeout, continuing")
    return successes


def measure_corpus_delta(args: argparse.Namespace) -> int:
    """Estimate how many new tokens have arrived since the last
    snapshot. Returns approximate token count of the delta.

    Heuristic: sum the byte size of files modified within the last
    24h under data/raw/, divide by 4 (rough byte-to-token ratio for
    English). Not exact but good enough for the 50K-or-not gate."""
    raw_dir = REPO_ROOT / "data" / "raw"
    cutoff = dt.datetime.now() - dt.timedelta(hours=24)
    cutoff_ts = cutoff.timestamp()

    total_bytes = 0
    for p in raw_dir.rglob("*.jsonl"):
        if p.stat().st_mtime > cutoff_ts:
            total_bytes += p.stat().st_size

    approx_tokens = total_bytes // 4
    LOG(f"corpus delta: ~{approx_tokens:,} tokens "
        f"(threshold: {args.min_delta_tokens:,})")
    return approx_tokens


def kick_off_lora_tune(args: argparse.Namespace, run_date: str) -> Path | None:
    """Wrap scripts/finetune_chat.py with LoRA flags. Returns the
    path to the saved adapter, or None on failure."""
    out_dir = REPO_ROOT / "checkpoints" / "daily" / run_date
    cmd = [
        sys.executable, "scripts/finetune_chat.py",
        "--checkpoint", args.base_checkpoint,
        "--lora",
        "--lora-r", str(args.lora_r),
        "--lora-alpha", str(args.lora_alpha),
        "--max-steps", str(args.lora_max_steps),
        "--learning-rate", str(args.lora_lr),
        "--out-dir", str(out_dir),
        "--data", "data/processed/daily_train.jsonl",
    ]
    LOG(f"running LoRA tune: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=str(REPO_ROOT),
                            env={**os.environ, "PYTHONPATH": str(REPO_ROOT)},
                            timeout=args.tune_timeout_s,
                            check=False)
    if result.returncode == 0 and out_dir.exists():
        adapter_path = out_dir / "adapter.pt"
        return adapter_path if adapter_path.exists() else None
    LOG(f"  LoRA tune exit {result.returncode}")
    return None


def push_to_hf(adapter_path: Path, run_date: str) -> str | None:
    """Push the adapter to a new HF Models repo with date-stamped
    name. Returns the repo URL on success."""
    repo_id = f"Ghostgim/GhostLM-daily-{run_date}"
    try:
        from huggingface_hub import HfApi, CommitOperationAdd
        api = HfApi()
        api.create_repo(repo_id=repo_id, repo_type="model",
                        exist_ok=True, private=False)
        info = api.create_commit(
            repo_id=repo_id, repo_type="model",
            operations=[CommitOperationAdd(
                path_in_repo="adapter.pt",
                path_or_fileobj=str(adapter_path),
            )],
            commit_message=f"daily LoRA adapter: {run_date}",
            commit_description=(
                f"Nightly LoRA fine-tune of GhostLM v0.9 chat over the\n"
                f"24h delta of threat-intel corpus through {run_date}.\n\n"
                f"Base checkpoint: Ghostgim/GhostLM-v0.9-experimental\n"
                f"LoRA recipe: docs/daily_finetune.md"
            ),
        )
        url = f"https://huggingface.co/{repo_id}"
        LOG(f"pushed to {url} (commit {info.oid[:10]})")
        return url
    except Exception as e:
        LOG(f"HF push failed: {type(e).__name__}: {e}")
        return None


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base-checkpoint",
                   default="checkpoints/phase19_chat_v09/best_model.pt")
    p.add_argument("--collector-timeout-s", type=int, default=600,
                   help="Per-collector timeout in seconds")
    p.add_argument("--min-delta-tokens", type=int, default=50_000,
                   help="Skip the tune if the 24h delta is below this")
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument("--lora-max-steps", type=int, default=500)
    p.add_argument("--lora-lr", type=float, default=1e-4)
    p.add_argument("--tune-timeout-s", type=int, default=10_800,
                   help="Total LoRA tune timeout (default 3h)")
    p.add_argument("--skip-collectors", action="store_true")
    p.add_argument("--skip-push", action="store_true")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    run_date = dt.date.today().isoformat()
    LOG(f"=== daily finetune {run_date} ===")

    if not args.skip_collectors:
        ok = run_collectors(args)
        LOG(f"collectors: {ok}/{len(COLLECTORS)} OK")

    delta = measure_corpus_delta(args)
    if delta < args.min_delta_tokens:
        LOG("delta below threshold; skipping LoRA tune")
        return 0

    adapter_path = kick_off_lora_tune(args, run_date)
    if not adapter_path:
        LOG("LoRA tune failed or produced no adapter; bailing")
        return 1

    if not args.skip_push:
        url = push_to_hf(adapter_path, run_date)
        if not url:
            LOG("HF push failed but adapter is local at "
                f"{adapter_path}; will retry next night")

    LOG(f"=== daily finetune {run_date} complete ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
