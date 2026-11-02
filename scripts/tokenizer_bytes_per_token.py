#!/usr/bin/env python3
"""Measure how many bytes each tokenizer packs into a token, per training domain.

    python scripts/tokenizer_bytes_per_token.py [--docs-per-domain 400] [--train-sizes 32000 48000 64000]

Compares the GPT-2 vocabulary GhostLM trains with today against the project's earlier
32K BPE and against fresh byte-level BPEs trained here on a sample of the same mix.
Bytes per token is the number that matters for a fixed token budget: a tokenizer that
scores 10 percent higher lets the same 150B-token run read 10 percent more text. Writes
the table to docs/tokenizer_bytes_per_token.md for the decision before the TPU corpus
is tokenised.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from data.collect import SOURCE_DOMAINS, domain_of as source_domain  # noqa: E402
from ghostlm.tokenizer import GhostTokenizer  # noqa: E402

CORPUS = ROOT / "data/processed/train.jsonl"
OUT_MD = ROOT / "docs/tokenizer_bytes_per_token.md"
EXISTING = {"gpt2_50k (current)": None, "v1_32k": ROOT / "data/tokenizer/v1/tokenizer.json",
            "v05_32k": ROOT / "data/tokenizer_v05/tokenizer.json"}
# Rough share of each domain in the planned TPU mix, for the weighted average.
MIX = {"general_web": 0.45, "code": 0.25, "cybersec": 0.12, "math": 0.08, "knowledge": 0.07, "instruction": 0.03}


def sample_docs(per_domain: int, seed: int, scan_limit: int) -> dict[str, list[str]]:
    """Reservoir-sample documents per domain from the first ``scan_limit`` corpus lines."""
    rng = random.Random(seed)
    seen: dict[str, int] = defaultdict(int)
    picked: dict[str, list[str]] = defaultdict(list)
    with CORPUS.open(encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= scan_limit:
                break
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            text = rec.get("text") or ""
            if len(text) < 200:
                continue
            domain = rec.get("domain") or SOURCE_DOMAINS.get(rec.get("source", "")) or source_domain(rec.get("source", ""))
            seen[domain] += 1
            if len(picked[domain]) < per_domain:
                picked[domain].append(text[:20000])
            else:
                j = rng.randrange(seen[domain])
                if j < per_domain:
                    picked[domain][j] = text[:20000]
    return picked


def train_bpe(texts: list[str], vocab_size: int):
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
    tok = Tokenizer(models.BPE(byte_fallback=False))
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(vocab_size=vocab_size, min_frequency=2, show_progress=False,
                                  initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
                                  special_tokens=["<|bos|>", "<|eos|>", "<|pad|>", "<|unk|>", "<|user|>", "<|assistant|>", "<|end|>"])
    tok.train_from_iterator(texts, trainer)
    return tok


def count_tokens(name: str, tok, texts: list[str]) -> int:
    if name.startswith("gpt2"):
        return sum(len(tok.encode(t)) for t in texts)
    return sum(len(tok.encode(t).ids) for t in texts)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--docs-per-domain", type=int, default=400)
    p.add_argument("--scan-limit", type=int, default=300_000, help="corpus lines to scan when sampling")
    p.add_argument("--train-sizes", type=int, nargs="*", default=[32000, 48000, 64000])
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()

    docs = sample_docs(a.docs_per_domain, a.seed, a.scan_limit)
    domains = [d for d in MIX if docs.get(d)] + sorted(d for d in docs if d not in MIX)
    print("sampled:", {d: len(docs[d]) for d in domains}, flush=True)
    bytes_per = {d: sum(len(t.encode("utf-8")) for t in docs[d]) for d in domains}

    tokenizers: dict[str, object] = {}
    from tokenizers import Tokenizer
    for name, path in EXISTING.items():
        if path is None:
            tokenizers[name] = GhostTokenizer()
        elif path.exists():
            tokenizers[name] = Tokenizer.from_file(str(path))
    train_texts = [t for d in domains for t in docs[d]]
    rng = random.Random(a.seed)
    rng.shuffle(train_texts)
    held_out = {d: docs[d][: max(1, len(docs[d]) // 5)] for d in domains}
    fit_texts = [t for d in domains for t in docs[d][max(1, len(docs[d]) // 5):]]
    for size in a.train_sizes:
        print(f"training byte-level BPE {size} on {sum(len(t) for t in fit_texts) / 1e6:.0f} MB", flush=True)
        tokenizers[f"new_{size // 1000}k"] = train_bpe(fit_texts, size)

    # Score every tokenizer on the same held-out fifth, so the fresh ones get no credit for memorising.
    rows = []
    for name, tok in tokenizers.items():
        per_domain = {}
        for d in domains:
            texts = held_out[d]
            per_domain[d] = sum(len(t.encode("utf-8")) for t in texts) / max(1, count_tokens(name, tok, texts))
        weighted = sum(MIX.get(d, 0) * per_domain[d] for d in domains) / sum(MIX.get(d, 0) for d in domains)
        rows.append((name, per_domain, weighted))
        print(f"{name:20s} weighted {weighted:.2f} bytes/token", flush=True)

    base = next(r for r in rows if r[0].startswith("gpt2"))[2]
    lines = ["# Bytes per token by tokenizer and domain", "",
             f"Held-out fifth of {a.docs_per_domain} sampled documents per domain from data/processed/train.jsonl "
             f"({sum(bytes_per.values()) / 1e6:.0f} MB sampled). Higher is better: more text per token. "
             "The weighted column uses the planned TPU mix "
             + ", ".join(f"{d} {int(w * 100)}%" for d, w in MIX.items()) + ".", "",
             "| Tokenizer | " + " | ".join(domains) + " | Weighted | vs current |",
             "|---|" + "---:|" * (len(domains) + 2)]
    for name, per_domain, weighted in rows:
        lines.append(f"| {name} | " + " | ".join(f"{per_domain[d]:.2f}" for d in domains)
                     + f" | {weighted:.2f} | {(weighted / base - 1) * 100:+.1f}% |")
    lines += ["", "A fresh vocabulary trained on this sample only sees a few hundred megabytes; one trained on the "
              "real mix will do slightly better than the rows above suggest."]
    OUT_MD.write_text("\n".join(lines) + "\n")
    print(f"wrote {OUT_MD}")


if __name__ == "__main__":
    main()
