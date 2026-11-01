#!/usr/bin/env python3
"""Stream large public corpora into deduplicated, decontaminated, tokenised shards.

    python scripts/corpus/build_shards.py run sources.json --out gs-staging/ [--max-tokens-per-source 20e9]
    python scripts/corpus/build_shards.py concat gs-staging/ --out data/processed_tpu/

This is the data side of the TPU run (docs/tpu_training.md). It is written to run on a
cloud VM next to the bucket, not on the Mac: a 3B model wants 150 to 300B tokens and
that is hundreds of gigabytes of download. ``run`` streams each source (a Hugging Face
dataset or a local JSONL), applies a length and character filter, drops exact
duplicates by 64-bit content hash, drops anything that overlaps a benchmark question
using the fingerprints from ``scripts/decontaminate.py``, tokenises with GhostTokenizer
and writes fixed-size ``.bin`` shards per domain with the same sidecar ``meta.json`` the
trainers expect. ``concat`` joins a domain's shards into the single
``train.<domain>.bin`` the curriculum sampler reads and writes the manifest.

Exact-hash dedup is in memory (8 bytes per kept document, so 100M documents is under a
gigabyte). Near-duplicate (MinHash) dedup is left to a later pass; FineWeb-Edu and The
Stack are already near-deduplicated upstream.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
from pathlib import Path
from typing import Dict, Iterator

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from decontaminate import load_benchmarks, normalize, shingles  # noqa: E402

SHARD_TOKENS = 100_000_000
NON_ALNUM = re.compile(r"[^A-Za-z0-9\s]")


def iter_source(src: dict) -> Iterator[str]:
    """Yield raw texts from one source entry (``type`` is ``hf`` or ``jsonl``)."""
    field = src.get("text_field", "text")
    if src["type"] == "jsonl":
        with open(src["path"], encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    rec = json.loads(line)
                    if rec.get(field):
                        yield rec[field]
        return
    from datasets import load_dataset
    ds = load_dataset(src["dataset"], src.get("subset"), split=src.get("split", "train"), streaming=True,
                      token=src.get("token"))
    for rec in ds:
        text = rec.get(field)
        if text:
            yield text


def passes_filters(text: str, src: dict) -> bool:
    n = len(text)
    if n < src.get("min_chars", 200) or n > src.get("max_chars", 200_000):
        return False
    if src.get("max_symbol_frac") is not None:
        if len(NON_ALNUM.findall(text)) / n > src["max_symbol_frac"]:
            return False
    return True


def content_hash(text: str) -> int:
    return int.from_bytes(hashlib.blake2b(" ".join(text.split()).encode("utf-8"), digest_size=8).digest(), "big")


class Decontaminator:
    def __init__(self, bench_paths, shingle_n: int = 13, min_shingles: int = 2):
        self.shingle_n, self.min_shingles = shingle_n, min_shingles
        self.exact_by_len, self.shingle_set = load_benchmarks(bench_paths, include_answers=True, shingle_n=shingle_n)
        self.lengths = sorted(self.exact_by_len)

    def contaminated(self, text: str) -> bool:
        words = normalize(text).split()
        for L in self.lengths:
            if len(words) >= L:
                grams = {" ".join(words[i:i + L]) for i in range(len(words) - L + 1)}
                if grams & self.exact_by_len[L]:
                    return True
        if self.shingle_set:
            overlap = shingles(words, n=self.shingle_n) & self.shingle_set
            if len(overlap) >= self.min_shingles:
                return True
        return False


class ShardWriter:
    """Fixed-size token shards for one domain: ``<domain>/shard_00000.bin`` plus meta."""

    def __init__(self, out_dir: Path, domain: str, dtype: np.dtype, vocab_size: int, shard_tokens: int):
        self.dir = out_dir / domain
        self.dir.mkdir(parents=True, exist_ok=True)
        self.dtype, self.vocab_size, self.shard_tokens = dtype, vocab_size, shard_tokens
        existing = sorted(self.dir.glob("shard_*.bin"))
        self.index = len(existing)
        self.buffer: list = []
        self.buffered = 0
        self.total = sum(int(json.loads(p.with_suffix(".meta.json").read_text())["tokens"]) for p in existing
                         if p.with_suffix(".meta.json").exists())

    def add(self, ids) -> None:
        arr = np.asarray(ids, dtype=self.dtype)
        self.buffer.append(arr)
        self.buffered += len(arr)
        if self.buffered >= self.shard_tokens:
            self.flush()

    def flush(self) -> None:
        if not self.buffer:
            return
        path = self.dir / f"shard_{self.index:05d}.bin"
        tmp = path.with_suffix(".bin.tmp")
        np.concatenate(self.buffer).tofile(tmp)
        tmp.replace(path)
        path.with_suffix(".meta.json").write_text(json.dumps({
            "dtype": np.dtype(self.dtype).name, "vocab_size": self.vocab_size, "tokens": self.buffered}))
        self.total += self.buffered
        self.index += 1
        self.buffer, self.buffered = [], 0


def run(args: argparse.Namespace) -> None:
    from ghostlm.tokenizer import GhostTokenizer
    sources = json.loads(Path(args.sources).read_text())
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    tok = GhostTokenizer()
    dtype = np.uint16 if tok.vocab_size <= np.iinfo(np.uint16).max + 1 else np.uint32
    bench = [Path(p) for p in args.bench] or sorted((ROOT / "data/raw").glob("*_eval*.jsonl")) + \
        sorted((ROOT / "data/raw").glob("*bench*.jsonl")) + [p for p in [ROOT / "data/raw/secqa.jsonl"] if p.exists()]
    decon = Decontaminator(bench) if bench else None
    seen: set = set()
    state_path = out / "state.json"
    state = json.loads(state_path.read_text()) if state_path.exists() else {"done": [], "stats": {}}
    writers: Dict[str, ShardWriter] = {}
    for src in sources:
        name = src["name"]
        if name in state["done"]:
            print(f"skip {name}: already done", flush=True)
            continue
        domain = src["domain"]
        writer = writers.setdefault(domain, ShardWriter(out, domain, dtype, tok.vocab_size, args.shard_tokens))
        budget = int(float(src.get("max_tokens", args.max_tokens_per_source)))
        stats = {"docs": 0, "kept": 0, "filtered": 0, "dups": 0, "contaminated": 0, "tokens": 0}
        t0 = time.time()
        for text in iter_source(src):
            stats["docs"] += 1
            if not passes_filters(text, src):
                stats["filtered"] += 1
                continue
            h = content_hash(text)
            if h in seen:
                stats["dups"] += 1
                continue
            if decon is not None and decon.contaminated(text):
                stats["contaminated"] += 1
                continue
            seen.add(h)
            ids = tok.encode(text, add_eos=True)
            writer.add(ids)
            stats["kept"] += 1
            stats["tokens"] += len(ids)
            if stats["docs"] % 10_000 == 0:
                rate = stats["tokens"] / max(1, time.time() - t0)
                print(f"{name}: {stats['docs']:,} docs, {stats['tokens'] / 1e6:,.0f}M tokens, {rate / 1e3:,.0f}K tok/s",
                      flush=True)
            if stats["tokens"] >= budget:
                break
        writer.flush()
        state["done"].append(name)
        state["stats"][name] = stats
        state_path.write_text(json.dumps(state, indent=2))
        print(f"done {name}: {json.dumps(stats)}", flush=True)
    for w in writers.values():
        w.flush()
    totals = {d: w.total for d, w in writers.items()}
    print("domain totals:", json.dumps(totals), flush=True)


def concat(args: argparse.Namespace) -> None:
    src, out = Path(args.staging), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    manifest = {}
    for domain_dir in sorted(p for p in src.iterdir() if p.is_dir()):
        shards = sorted(domain_dir.glob("shard_*.bin"))
        if not shards:
            continue
        meta = json.loads(shards[0].with_suffix(".meta.json").read_text())
        target = out / f"train.{domain_dir.name}.bin"
        total = 0
        with target.open("wb") as f:
            for shard in shards:
                f.write(shard.read_bytes())
                total += int(json.loads(shard.with_suffix(".meta.json").read_text())["tokens"])
        target.with_suffix(".meta.json").write_text(json.dumps({"dtype": meta["dtype"], "vocab_size": meta["vocab_size"],
                                                                 "tokens": total, "shards": len(shards)}, indent=2))
        manifest[domain_dir.name] = target.name
        print(f"{domain_dir.name}: {total:,} tokens from {len(shards)} shards", flush=True)
    (out / "domains.json").write_text(json.dumps(manifest, indent=2))
    print(f"manifest: {out / 'domains.json'}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="stream, filter, dedup, decontaminate, tokenise into shards")
    r.add_argument("sources", help="JSON list of sources; see scripts/corpus/sources.example.json")
    r.add_argument("--out", required=True)
    r.add_argument("--bench", nargs="*", default=[], help="benchmark files to decontaminate against (default: data/raw)")
    r.add_argument("--shard-tokens", type=int, default=SHARD_TOKENS)
    r.add_argument("--max-tokens-per-source", type=float, default=20e9)
    r.set_defaults(fn=run)
    c = sub.add_parser("concat", help="join a staging folder's shards into train.<domain>.bin files and a manifest")
    c.add_argument("staging")
    c.add_argument("--out", required=True)
    c.set_defaults(fn=concat)
    a = p.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
