#!/usr/bin/env python3
"""Subsample a large RAG index down to a Space-shippable size.

Why: rebuilding the index over the v1.0 corpus (363M tokens, 26
sources) produces ~1.2M chunks. At 384-dim fp16 that is ~925 MB of
index plus ~280 MB of chunks.jsonl, well past the cpu-basic Space's
practical RAM budget for an LM + embedder + index simultaneously
loaded. The local copy is fine for offline diagnostics but the
Space needs a smaller version.

This script keeps the per-source distribution but caps each source's
chunk count, defaulting to a max of 25,000 chunks per source. With
26 sources that comes out to ~120-200K chunks total, which is a
~150-250 MB fp16 index, comfortably loadable on cpu-basic.

Sampling strategy is deterministic: for each source, take the first
N chunks. This biases toward the start of each source file but avoids
random draws producing different indices on different machines.
For corpus shards that are themselves randomly ordered (PRIMUS-FineWeb,
fineweb_edu) the first-N is essentially random anyway. For sequenced
shards (CVE list, MITRE techniques) the first-N favors the
alphanumerically-earlier IDs, which is a known bias but documented.

Output is the same shape as the input: index.npy + chunks.jsonl +
meta.json under --out-dir.

Usage:

    PYTHONPATH=. python3 scripts/subsample_rag_index.py \\
        --src-dir data/rag_v1 \\
        --out-dir data/rag_v1_lite \\
        --max-per-source 25000

Run after `scripts/build_rag_index.py` completes the v1.0 corpus
rebuild. The Space pulls from the lite version; local development
and offline diagnostics use the full index.
"""

from __future__ import annotations

import argparse
import json
from collections import OrderedDict
from pathlib import Path
from typing import Dict, List


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--src-dir", required=True,
                   help="Source directory with index.npy + chunks.jsonl + meta.json")
    p.add_argument("--out-dir", required=True,
                   help="Where to write the subsampled index + chunks + meta")
    p.add_argument("--max-per-source", type=int, default=25000,
                   help="Max chunks to keep per source")
    p.add_argument("--max-total", type=int, default=0,
                   help="Optional total cap; 0 = no cap")
    p.add_argument("--cast-fp16", action="store_true",
                   help="Save the subsampled index in fp16 to halve disk + memory")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    import numpy as np

    src = Path(args.src_dir)
    dst = Path(args.out_dir)
    dst.mkdir(parents=True, exist_ok=True)

    print(f"Loading {src / 'index.npy'}...")
    full_idx = np.load(src / "index.npy")
    print(f"  shape={full_idx.shape}, dtype={full_idx.dtype}")
    print(f"Loading {src / 'chunks.jsonl'}...")
    full_chunks: List[Dict] = []
    with (src / "chunks.jsonl").open("r", encoding="utf-8") as f:
        for line in f:
            full_chunks.append(json.loads(line))
    print(f"  {len(full_chunks)} chunks")

    if len(full_chunks) != full_idx.shape[0]:
        raise SystemExit(
            f"Mismatch: index has {full_idx.shape[0]} rows, chunks.jsonl has "
            f"{len(full_chunks)}. Refusing to subsample inconsistent input."
        )

    # Bucket chunks by source, preserving original order.
    by_source: "OrderedDict[str, List[int]]" = OrderedDict()
    for i, ch in enumerate(full_chunks):
        s = ch.get("source") or "unknown"
        by_source.setdefault(s, []).append(i)

    print(f"\nSources: {len(by_source)}")
    keep_indices: List[int] = []
    for s, idxs in by_source.items():
        kept = idxs[: args.max_per_source]
        keep_indices.extend(kept)
        print(f"  {s:30s}  {len(idxs):>7d} -> {len(kept):>6d}")
        if args.max_total and len(keep_indices) >= args.max_total:
            keep_indices = keep_indices[: args.max_total]
            print(f"  (hit --max-total cap of {args.max_total})")
            break

    # Stable order is important (caller expects index row N to match
    # chunk row N). Sort the kept indices to preserve corpus order;
    # this also makes diff between subsamples reproducible.
    keep_indices.sort()
    print(f"\nKept {len(keep_indices)} of {len(full_chunks)} chunks "
          f"({100 * len(keep_indices) / len(full_chunks):.1f}%)")

    sub_idx = full_idx[keep_indices]
    if args.cast_fp16 and sub_idx.dtype != np.float16:
        sub_idx = sub_idx.astype(np.float16)
        print(f"Cast index to fp16 for the subsampled output")
    np.save(dst / "index.npy", sub_idx)
    print(f"Wrote {dst / 'index.npy'} ({sub_idx.nbytes / 1e6:.1f} MB)")

    with (dst / "chunks.jsonl").open("w", encoding="utf-8") as f:
        for i in keep_indices:
            f.write(json.dumps(full_chunks[i], ensure_ascii=False) + "\n")
    print(f"Wrote {dst / 'chunks.jsonl'} ({(dst / 'chunks.jsonl').stat().st_size / 1e6:.1f} MB)")

    # Carry over meta.json with the new chunk count and a note.
    meta_path = src / "meta.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
    else:
        meta = {}
    meta["n_chunks"] = len(keep_indices)
    meta["subsample_max_per_source"] = args.max_per_source
    meta["subsample_source"] = str(src)
    if args.cast_fp16:
        meta["dtype"] = "float16"
    (dst / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"Wrote {dst / 'meta.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
