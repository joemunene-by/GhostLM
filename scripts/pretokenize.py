"""Pretokenize JSONL corpora into flat .bin token files for training.

Tokenizing the v1.0 corpus (~422M tokens) inside ``GhostDataset`` costs
>10 GB of RAM (Python list of ints) and several minutes of startup on
every training launch. This script does that work once, streaming each
record (terminated with the EOS document separator) into a flat
``uint16``/``uint32`` array on disk. Training then memory-maps the file
via ``GhostBinDataset`` — instant startup, near-zero resident memory.

Usage:

    python scripts/pretokenize.py \
        --train data/processed/train.jsonl \
        --val data/processed/val.jsonl \
        --out-dir data/processed

    # then train against the .bin outputs:
    python scripts/train.py --train-data data/processed/train.bin \
                            --val-data data/processed/val.bin ...

A sidecar ``<name>.meta.json`` records the dtype, vocab size, and token
count so ``GhostBinDataset`` can open the file without guessing.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ghostlm.tokenizer import load_tokenizer  # noqa: E402
from data.collect import domain_of  # noqa: E402


def pretokenize_file(jsonl_path: Path, out_path: Path, tokenizer) -> int:
    """Stream-tokenize one JSONL file into a flat .bin of token ids.

    Returns the total token count written.
    """
    vocab_size = len(tokenizer)
    dtype = np.uint16 if vocab_size <= np.iinfo(np.uint16).max + 1 else np.uint32

    total = 0
    records = 0
    t0 = time.time()
    with open(jsonl_path, "r", encoding="utf-8") as f_in, open(out_path, "wb") as f_out:
        for line in f_in:
            line = line.strip()
            if not line:
                continue
            text = json.loads(line).get("text", "")
            if not text:
                continue
            # EOS-terminate every record: explicit document boundaries.
            ids = tokenizer.encode(text, add_eos=True)
            np.asarray(ids, dtype=dtype).tofile(f_out)
            total += len(ids)
            records += 1
            if records % 50_000 == 0:
                print(f"  {records:,} records / {total:,} tokens "
                      f"({time.time() - t0:.0f}s)")

    meta = {
        "dtype": np.dtype(dtype).name,
        "vocab_size": vocab_size,
        "num_tokens": total,
        "num_records": records,
        "source": str(jsonl_path),
    }
    meta_path = out_path.with_suffix(".meta.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"  Wrote {total:,} tokens ({records:,} records) -> {out_path} "
          f"[{np.dtype(dtype).name}] + {meta_path.name}")
    return total


def pretokenize_by_domain(jsonl_path: Path, out_dir: Path, tokenizer, stem: str) -> dict:
    """Write one ``.bin`` per training domain for curriculum sampling.

    Each record's ``source`` is mapped to a domain (``data.collect.domain_of``)
    and its EOS-terminated tokens are appended to that domain's stream. Emits
    ``<stem>.<domain>.bin`` (+ per-bin meta) and a ``<stem>.domains.json``
    manifest mapping domain -> bin path, which the curriculum trainer loads.
    """
    vocab_size = len(tokenizer)
    dtype = np.uint16 if vocab_size <= np.iinfo(np.uint16).max + 1 else np.uint32

    handles: dict = {}
    totals: dict = {}
    t0 = time.time()
    records = 0
    with open(jsonl_path, "r", encoding="utf-8") as f_in:
        for line in f_in:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            text = rec.get("text", "")
            if not text:
                continue
            domain = domain_of(rec.get("source", ""))
            if domain not in handles:
                handles[domain] = open(out_dir / f"{stem}.{domain}.bin", "wb")
                totals[domain] = 0
            ids = tokenizer.encode(text, add_eos=True)
            np.asarray(ids, dtype=dtype).tofile(handles[domain])
            totals[domain] += len(ids)
            records += 1
            if records % 50_000 == 0:
                print(f"  {records:,} records ({time.time() - t0:.0f}s)")

    manifest = {}
    for domain, fh in handles.items():
        fh.close()
        bin_path = out_dir / f"{stem}.{domain}.bin"
        with open(bin_path.with_suffix(".meta.json"), "w") as f:
            json.dump({"dtype": np.dtype(dtype).name, "vocab_size": vocab_size,
                       "num_tokens": totals[domain], "domain": domain}, f, indent=2)
        manifest[domain] = str(bin_path)
        print(f"  domain {domain:12s} {totals[domain]:>12,} tokens -> {bin_path.name}")

    manifest_path = out_dir / f"{stem}.domains.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"  manifest -> {manifest_path}")
    return manifest


def main() -> int:
    p = argparse.ArgumentParser(
        description="Pretokenize JSONL corpora into memmap-ready .bin files.",
    )
    p.add_argument("--train", default="data/processed/train.jsonl",
                   help="Training JSONL (records with a 'text' field).")
    p.add_argument("--val", default="data/processed/val.jsonl",
                   help="Validation JSONL.")
    p.add_argument("--out-dir", default=None,
                   help="Output directory (default: alongside the inputs).")
    p.add_argument("--tokenizer", default=None,
                   help="Optional tokenizer.json path (v0.5/v1 BPE). "
                        "Omit for the default GPT-2 BPE tokenizer.")
    p.add_argument("--by-domain", action="store_true",
                   help="Also write one train.<domain>.bin per training domain "
                        "plus a train.domains.json manifest, for curriculum "
                        "(domain-weighted) training. The val split stays a single bin.")
    args = p.parse_args()

    tokenizer = load_tokenizer(args.tokenizer)
    print(f"Tokenizer: {tokenizer!r}")

    for src in (args.train, args.val):
        src = Path(src)
        if not src.exists():
            print(f"  Skipping {src} (not found)")
            continue
        out_dir = Path(args.out_dir) if args.out_dir else src.parent
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / (src.stem + ".bin")
        print(f"Pretokenizing {src} -> {out_path}")
        pretokenize_file(src, out_path, tokenizer)
        # Per-domain shards only make sense for the training split.
        if args.by_domain and "train" in src.stem:
            print(f"Pretokenizing {src} by domain")
            pretokenize_by_domain(src, out_dir, tokenizer, src.stem)

    return 0


if __name__ == "__main__":
    sys.exit(main())
