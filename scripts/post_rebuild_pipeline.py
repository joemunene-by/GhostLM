#!/usr/bin/env python3
"""Wait for the v1.0 RAG index rebuild to finish, then run the full
post-rebuild pipeline:

  1. Wait for the rebuild PID to exit (data/rag_v1/index.npy lands).
  2. Run scripts/eval_rag_recall.py against the new index. Captures
     retrieval@4 over the v1.0 corpus and writes
     logs/rag_retrieval_at_k_v1.jsonl.
  3. Run scripts/subsample_rag_index.py to build data/rag_v1_lite/
     with --max-per-source 25000 --cast-fp16 (Space-shippable size).
  4. Push data/rag_v1_lite/ to Ghostgim/GhostLM-v0.9-experimental
     under rag/ via huggingface_hub.create_commit. Replaces the
     existing v0.4-era index with the v1.0 lite version.
  5. Call api.restart_space("Ghostgim/ghostlm") so the Space
     re-pulls the new index on launch.
  6. Poll the Space until stage = RUNNING with no errors, write a
     summary to /tmp/post_rebuild_summary.txt.

Designed to run on the Mac via:

    nohup python3 -u scripts/post_rebuild_pipeline.py \\
        --rebuild-pid 43793 \\
        > /tmp/post_rebuild_pipeline.log 2>&1 &

The orchestrator handles the multi-hour wait so the loop completes
unattended. Failure at any step writes the error and exits non-zero;
re-running with the same args picks up from where it stopped because
each step is idempotent.
"""

from __future__ import annotations

import argparse
import os
import shlex
import shutil
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
HUB_REPO = "Ghostgim/GhostLM-v0.9-experimental"
SPACE_REPO = "Ghostgim/ghostlm"


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def wait_for_pid(pid: int, poll_seconds: int = 60) -> None:
    """Block until ``pid`` is no longer running. Cheap polling on
    /proc-style; on macOS uses ``kill -0`` (signal 0 doesn't actually
    kill, just probes existence)."""
    log(f"Waiting for PID {pid} to exit (poll every {poll_seconds}s)...")
    while True:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            log(f"PID {pid} has exited.")
            return
        except PermissionError:
            log(f"PID {pid} exists (no perm to signal); treating as alive.")
        time.sleep(poll_seconds)


def run(cmd: list, env: dict | None = None) -> None:
    """Run a subprocess, raise on non-zero exit. Inherit stdout/stderr."""
    log(f"$ {shlex.join(cmd)}")
    full_env = os.environ.copy()
    if env:
        full_env.update(env)
    result = subprocess.run(cmd, env=full_env, cwd=str(REPO_ROOT))
    if result.returncode != 0:
        raise SystemExit(f"step failed (exit {result.returncode}): {' '.join(cmd)}")


def ensure_index_built(rag_dir: Path) -> None:
    """Sanity-check that the rebuild actually produced an index."""
    idx = rag_dir / "index.npy"
    chunks = rag_dir / "chunks.jsonl"
    meta = rag_dir / "meta.json"
    for p in (idx, chunks, meta):
        if not p.exists():
            raise SystemExit(f"Expected output missing: {p}")
    log(f"Verified rebuild output: {idx} ({idx.stat().st_size / 1e6:.1f} MB), "
        f"{chunks} ({chunks.stat().st_size / 1e6:.1f} MB)")


def push_lite_to_models_repo(lite_dir: Path) -> str:
    """Upload data/rag_v1_lite/ to the Models repo's rag/ subfolder.
    Returns the commit OID."""
    from huggingface_hub import HfApi, CommitOperationAdd
    api = HfApi()
    log(f"HF auth: {api.whoami()['name']}")
    ops = [
        CommitOperationAdd(path_in_repo="rag/index.npy",
                           path_or_fileobj=str(lite_dir / "index.npy")),
        CommitOperationAdd(path_in_repo="rag/chunks.jsonl",
                           path_or_fileobj=str(lite_dir / "chunks.jsonl")),
        CommitOperationAdd(path_in_repo="rag/meta.json",
                           path_or_fileobj=str(lite_dir / "meta.json")),
    ]
    log(f"Uploading {len(ops)} files to {HUB_REPO}...")
    info = api.create_commit(
        repo_id=HUB_REPO, repo_type="model", operations=ops,
        commit_message="feat: v1.0 corpus RAG index (lite, fp16)",
        commit_description=(
            "Replaces the v0.4-era 83K-chunk index with the v1.0 corpus rebuild,\n"
            "subsampled to <=25K chunks per source and cast to fp16. The full\n"
            "1.2M-chunk index lives offline (data/rag_v1/) for diagnostics; this\n"
            "lite version is what the Space pulls and loads at runtime.\n\n"
            "Built by scripts/build_rag_index.py over data/processed/train.jsonl\n"
            "(516,736 records / ~363M tokens / 26 sources / six domains) and\n"
            "subsampled by scripts/subsample_rag_index.py."
        ),
    )
    log(f"  commit oid: {info.oid}")
    return info.oid


def restart_space() -> None:
    from huggingface_hub import HfApi
    api = HfApi()
    log(f"Restarting Space {SPACE_REPO}...")
    api.restart_space(SPACE_REPO)
    log("  restart called.")


def wait_for_space_running(timeout_s: int = 600) -> None:
    """Poll until the Space leaves BUILDING and reaches RUNNING (or errors out)."""
    from huggingface_hub import HfApi
    api = HfApi()
    deadline = time.time() + timeout_s
    last_stage = None
    while time.time() < deadline:
        runtime = api.get_space_runtime(SPACE_REPO)
        if runtime.stage != last_stage:
            log(f"  Space stage: {runtime.stage}")
            last_stage = runtime.stage
        if runtime.stage == "RUNNING":
            err = runtime.raw.get("errorMessage")
            if err:
                log(f"  Space RUNNING but with error: {err[:300]}")
            else:
                log("  Space is RUNNING with no error. Done.")
            return
        if runtime.stage in {"RUNTIME_ERROR", "BUILD_ERROR", "APP_FAILED"}:
            err = runtime.raw.get("errorMessage", "(no message)")
            raise SystemExit(f"Space failed: stage={runtime.stage}, err={err[:500]}")
        time.sleep(15)
    raise SystemExit(f"Space did not reach RUNNING within {timeout_s}s")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--rebuild-pid", type=int, required=True,
                   help="PID of the build_rag_index.py process to wait on")
    p.add_argument("--rebuild-dir", default="data/rag_v1",
                   help="Where the rebuild writes its output")
    p.add_argument("--lite-dir", default="data/rag_v1_lite",
                   help="Where to write the subsampled index")
    p.add_argument("--max-per-source", type=int, default=25000)
    p.add_argument("--skip-eval", action="store_true",
                   help="Skip the retrieval@4 diagnostic step (faster cycle)")
    p.add_argument("--skip-push", action="store_true",
                   help="Skip the HF upload + Space restart (local-only test)")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    rebuild_dir = Path(args.rebuild_dir)
    lite_dir = Path(args.lite_dir)

    log(f"=== post-rebuild pipeline ===")
    log(f"  rebuild PID:    {args.rebuild_pid}")
    log(f"  rebuild dir:    {rebuild_dir}")
    log(f"  lite dir:       {lite_dir}")
    log(f"  max-per-source: {args.max_per_source}")
    log(f"  skip eval:      {args.skip_eval}")
    log(f"  skip push:      {args.skip_push}")

    # Step 1: wait for the rebuild to finish.
    wait_for_pid(args.rebuild_pid)
    ensure_index_built(rebuild_dir)

    # Step 2: retrieval@4 diagnostic on the new index.
    if not args.skip_eval:
        log("\n=== Step 2: retrieval@4 diagnostic on v1.0 index ===")
        run([
            sys.executable, "-u", "scripts/eval_rag_recall.py",
            "--rag-dir", str(rebuild_dir),
            "--bench", "data/raw/fact_recall_bench_v2.jsonl",
            "--top-k", "4",
            "--out", "logs/rag_retrieval_at_k_v1.jsonl",
        ])

    # Step 3: subsample to a Space-friendly index.
    log("\n=== Step 3: subsample to lite index ===")
    if lite_dir.exists():
        shutil.rmtree(lite_dir)
    run([
        sys.executable, "-u", "scripts/subsample_rag_index.py",
        "--src-dir", str(rebuild_dir),
        "--out-dir", str(lite_dir),
        "--max-per-source", str(args.max_per_source),
        "--cast-fp16",
    ])

    # Step 4 + 5 + 6: push to Models + restart Space + verify.
    if not args.skip_push:
        log("\n=== Step 4: push lite index to Models repo ===")
        push_lite_to_models_repo(lite_dir)

        log("\n=== Step 5: restart Space ===")
        restart_space()

        log("\n=== Step 6: wait for Space to come back RUNNING ===")
        wait_for_space_running()

    log("\n=== Pipeline complete ===")
    summary_path = Path("/tmp/post_rebuild_summary.txt")
    with summary_path.open("w") as f:
        f.write("post-rebuild pipeline complete\n")
        f.write(f"  rebuild dir:  {rebuild_dir}\n")
        f.write(f"  lite dir:     {lite_dir}\n")
        f.write(f"  pushed to:    https://huggingface.co/{HUB_REPO}/tree/main/rag\n")
        f.write(f"  space:        https://huggingface.co/spaces/{SPACE_REPO}\n")
        f.write(f"  retrieval log: logs/rag_retrieval_at_k_v1.jsonl\n")
    log(f"Summary at {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
