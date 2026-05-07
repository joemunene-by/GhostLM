#!/usr/bin/env python3
"""Free-form fact-recall benchmark v2 with boundary-aware grading.

The v1 bench at ``data/raw/fact_recall_bench.jsonl`` (n=50) graded
matches via plain substring search. That has two known false-positive
modes that this v2 bench tightens up:

  1. **Token-boundary leakage.** Substring "256" matches inside
     "SHA-256". If a question is "what hash function produces a
     32-byte digest?" and its answer is "256" (referring to bits, in
     the alternate "SHA-256"), the model can echo "SHA-256" from the
     question prompt and the v1 grader credits a hit. The fix is
     ``boundary_match=True`` (default), which only credits a match
     when the alternate appears on word boundaries.

  2. **Question echoing.** Some questions mention key terms that the
     answer also contains. The v1 grader can't tell whether the model
     actually answered or just rephrased the question. v2 introduces
     ``disqualifiers``: if any of a question's listed disqualifier
     phrases appears in the completion, no match credit is given for
     that question regardless of which alternate matches. Used
     sparingly for questions where echoing is a real risk.

  3. **Multi-fragment answers.** Some answers require two distinct
     facts to be present (e.g. "RFC 7519 / JWT": you want the model
     to surface both the RFC number and the protocol name, not just
     one). v2 introduces ``must_appear``: a list of substrings that
     ALL must appear, with boundary matching, for credit.

Output schema (one record per line, ``data/raw/fact_recall_bench_v2.jsonl``):

    {
      "id":          "fr2-001",
      "topic":       "cve|mitre|cwe|owasp|crypto|protocol|tool|misc",
      "prompt":      "What is the CVE for ...?",
      "answer":      "CVE-2017-0144",
      "alternates":  ["CVE-2017-0144", "MS17-010"],
      "boundary_match": true,
      "disqualifiers": ["echoing-risk phrase 1", "echoing-risk phrase 2"],
      "must_appear":  ["RFC 7519", "JWT"]
    }

``alternates`` is "any of these counts as a match" (OR semantics);
``must_appear`` is "all of these must be present" (AND semantics);
the two are mutually exclusive per record (questions either have
one synonymous fact, or one composite-fact requirement).

Grading: a record is a hit iff one of its alternates matches with
boundary_match honored, AND none of its disqualifiers appear. For
must_appear records, hit iff every must_appear phrase is found
(boundary-matched) AND no disqualifier appears.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import fields
from pathlib import Path
from typing import List, Optional, Tuple

import torch

# Allow running from any cwd without PYTHONPATH=. by adding the repo root.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from ghostlm.config import GhostLMConfig
from ghostlm.model import GhostLM
from ghostlm.tokenizer import GhostTokenizer


# A "word boundary" for our purposes is any non-alphanumeric character
# (or start/end of string). re.search with \b is too permissive for
# multi-word phrases with hyphens or numbers (\b treats CVE-2017-0144
# as multiple words at the dashes). We compile our own anchor.
WORD_CHARS = re.compile(r"[A-Za-z0-9]")


def _normalize(s: str) -> str:
    """Lowercase + collapse runs of whitespace. Punctuation is preserved
    so things like "CVE-2017-0144" don't get mangled."""
    return re.sub(r"\s+", " ", s.lower()).strip()


def _appears_with_boundary(needle: str, haystack: str) -> bool:
    """True iff needle appears in haystack delimited by non-word chars
    (or string ends) on both sides. Both inputs are pre-normalized."""
    n = needle
    h = haystack
    if not n:
        return False
    start = 0
    while True:
        i = h.find(n, start)
        if i == -1:
            return False
        before_ok = (i == 0) or not WORD_CHARS.match(h[i - 1])
        after_idx = i + len(n)
        after_ok = (after_idx == len(h)) or not WORD_CHARS.match(h[after_idx])
        if before_ok and after_ok:
            return True
        start = i + 1


def _appears(needle: str, haystack: str, boundary: bool) -> bool:
    """Either substring or boundary-respecting match, depending on
    the per-record ``boundary_match`` flag."""
    if boundary:
        return _appears_with_boundary(needle, haystack)
    return needle in haystack


def grade_record(rec: dict, completion: str) -> Tuple[bool, str]:
    """Return (hit, reason). ``reason`` is a short string explaining
    the grader's decision; useful in the per-row log to spot
    spurious-looking hits and false-negative misses."""
    norm = _normalize(completion)
    boundary = rec.get("boundary_match", True)

    # Disqualifier check first: if any disqualifier appears, no credit
    # regardless of what else matches.
    for dq in rec.get("disqualifiers", []) or []:
        ndq = _normalize(dq)
        if _appears(ndq, norm, boundary):
            return False, f"disqualifier hit: {dq!r}"

    # Composite-fact requirement: every must_appear phrase must match.
    must = rec.get("must_appear") or []
    if must:
        missing = []
        for phrase in must:
            if not _appears(_normalize(phrase), norm, boundary):
                missing.append(phrase)
        if missing:
            return False, f"missing required phrase(s): {missing}"
        return True, "all must_appear phrases matched"

    # Standard alternates check (OR semantics).
    alternates = [rec["answer"]] + list(rec.get("alternates", []) or [])
    for alt in alternates:
        if _appears(_normalize(alt), norm, boundary):
            return True, f"alternate matched: {alt!r}"
    return False, "no alternate matched"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoints", nargs="+", required=True,
                   help="One or more LABEL=PATH specs (label appears in the "
                        "report). Example: v0.9-chat=checkpoints/phase19_chat_v09/best_model.pt")
    p.add_argument("--bench", default="data/raw/fact_recall_bench_v2.jsonl")
    p.add_argument("--max-tokens", type=int, default=120,
                   help="Per-question generation budget. Short on purpose: a "
                        "model that knows the fact surfaces it within the first "
                        "60-120 tokens; longer answers just inflate hit rates "
                        "via topic-tangent prose.")
    p.add_argument("--temperature", type=float, default=0.0,
                   help="Default greedy. Set non-zero for sampling variance "
                        "studies; the canonical numbers should be greedy.")
    p.add_argument("--top-k", type=int, default=0,
                   help="0 = no top-k filtering")
    p.add_argument("--device", default="auto")
    p.add_argument("--logs-dir", default="logs/fact_recall_v2",
                   help="Per-row JSONL log lands here; one file per checkpoint")
    # RAG (retrieval-augmented generation) optional path. When --rag-dir
    # points to a directory containing index.npy + chunks.jsonl, every
    # question is augmented with the top-K most similar passages before
    # the model sees it. Same retrieval flow as scripts/rag_chat.py and
    # the demo Space's chat_fn. Lets us measure whether retrieval lifts
    # v0.9 chat off the fact-recall floor without retraining.
    p.add_argument("--rag-dir", default=None,
                   help="Optional path to a directory with index.npy + "
                        "chunks.jsonl + meta.json. If set, each question "
                        "is augmented with the top-K most similar corpus "
                        "passages before generation.")
    p.add_argument("--rag-top-k", type=int, default=4,
                   help="Number of passages to retrieve per question")
    p.add_argument("--rag-embedder", default="BAAI/bge-small-en-v1.5",
                   help="HF model id for the retrieval embedder. Must "
                        "match the embedder the index was built with.")
    return p.parse_args()


def parse_specs(specs: List[str]) -> List[Tuple[str, Path]]:
    out = []
    for s in specs:
        if "=" not in s:
            raise SystemExit(f"Bad checkpoint spec (need LABEL=PATH): {s}")
        label, path = s.split("=", 1)
        out.append((label.strip(), Path(path.strip())))
    return out


def load_bench(path: Path) -> List[dict]:
    out: List[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out


def resolve_device(arg: str) -> str:
    if arg == "auto":
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch, "backends") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    return arg


def load_model(path: Path, device: str) -> Tuple[GhostLM, GhostLMConfig]:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    saved = ckpt["config"]
    cfg = GhostLMConfig(**{
        f.name: saved[f.name]
        for f in fields(GhostLMConfig)
        if f.name in saved
    })
    model = GhostLM(cfg)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model.to(device), cfg


def load_rag_state(rag_dir: Path, embedder_name: str, device: str):
    """Load the retrieval index + embedder. Returns a dict with
    ``index`` (np.ndarray, fp32, L2-normalized), ``chunks`` (list of
    dicts), ``embed_tok``, ``embed_model``. Raises on any failure
    (caller is expected to gate on --rag-dir presence)."""
    import json as _json
    import numpy as np
    idx = np.load(rag_dir / "index.npy")
    if idx.dtype != np.float32:
        idx = idx.astype(np.float32)
    chunks: List[dict] = []
    with (rag_dir / "chunks.jsonl").open("r", encoding="utf-8") as f:
        for line in f:
            chunks.append(_json.loads(line))
    from transformers import AutoModel, AutoTokenizer
    e_tok = AutoTokenizer.from_pretrained(embedder_name)
    # Force the embedder to CPU. BGE-small on MPS produces nan/inf in
    # some PyTorch / Mac driver combinations; on CUDA it would be fine
    # but we can't tell here. Embedding cost is one short query at a
    # time so CPU is plenty fast (~50-200 ms per query). The main
    # GhostLM model stays on whatever device the caller asked for.
    e_model = AutoModel.from_pretrained(embedder_name).to("cpu").eval()
    print(f"  RAG: {len(chunks)} chunks, dim {idx.shape[1]}, embedder {embedder_name} on CPU")
    return {"index": idx, "chunks": chunks, "embed_tok": e_tok, "embed_model": e_model}


def rag_augmented_prompt(question: str, rag, top_k: int, device: str) -> str:
    """Embed the question, retrieve top-K passages, return a single
    prompt string with the passages prepended in the same shape the
    Space's chat_fn uses. Same recipe as scripts/rag_chat.py."""
    import numpy as np
    text = "Represent this sentence for searching relevant passages: " + question
    # Embedder is pinned to CPU (see load_rag_state). Move inputs to
    # CPU regardless of what `device` the main model is on, so the
    # forward pass is deterministic across host devices.
    enc = rag["embed_tok"](
        text, padding=True, truncation=True, max_length=512, return_tensors="pt",
    ).to("cpu")
    with torch.no_grad():
        out = rag["embed_model"](**enc)
    emb = out.last_hidden_state[:, 0]
    emb = torch.nn.functional.normalize(emb, p=2, dim=-1)
    q_vec = emb.cpu().to(torch.float32).numpy().reshape(-1)
    scores = rag["index"] @ q_vec
    idxs = np.argsort(-scores)[: max(1, top_k)]
    refs = []
    for i, j in enumerate(idxs):
        ch = rag["chunks"][int(j)]
        snippet = (ch.get("text") or "")[:400]
        if len(ch.get("text") or "") > 400:
            snippet = snippet.rsplit(" ", 1)[0] + "..."
        refs.append(f"[{i+1}] ({ch.get('source', '?')} {ch.get('ref', '')}) {snippet}")
    return (
        "Reference passages from the cybersecurity corpus:\n\n"
        + "\n\n".join(refs)
        + "\n\nUse the reference passages above to answer the question. If the "
          "passages don't contain the answer, say so rather than guessing.\n\n"
          f"Question: {question}"
    )


def generate(model: GhostLM, tokenizer: GhostTokenizer, prompt: str,
             *, max_tokens: int, temperature: float, top_k: int,
             device: str) -> str:
    """Greedy-or-sampled generation. Stops at <|ghost_end|> if the
    model emits it. Decodes only the new tokens (no prompt echo)."""
    end_id = tokenizer._special_tokens[tokenizer.END]
    turns = [{"role": "user", "content": prompt}]
    prompt_ids = tokenizer.format_chat_prompt(turns)
    ids = torch.tensor(prompt_ids, dtype=torch.long, device=device).unsqueeze(0)
    new_ids: List[int] = []
    ctx = model.config.context_length
    with torch.no_grad():
        for _ in range(max_tokens):
            cond = ids[:, -ctx:]
            logits, _ = model(cond)
            logits = logits[:, -1, :].squeeze(0)
            if temperature > 0.0:
                logits = logits / temperature
                if top_k and top_k > 0:
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < v[-1]] = float("-inf")
                probs = torch.softmax(logits, dim=-1)
                tok = int(torch.multinomial(probs, 1).item())
            else:
                tok = int(logits.argmax().item())
            if tok == end_id:
                break
            new_ids.append(tok)
            ids = torch.cat([ids, torch.tensor([[tok]], device=device)], dim=1)
    return tokenizer.decode(new_ids).strip()


def main() -> int:
    args = parse_args()
    device = resolve_device(args.device)
    bench = load_bench(Path(args.bench))
    print(f"Bench: {args.bench}  ({len(bench)} records)")
    print(f"Device: {device}")
    print()

    Path(args.logs_dir).mkdir(parents=True, exist_ok=True)
    tokenizer = GhostTokenizer()
    summary_rows = []

    rag = None
    if args.rag_dir:
        rag_dir = Path(args.rag_dir)
        if not rag_dir.is_dir():
            raise SystemExit(f"--rag-dir does not exist: {rag_dir}")
        print(f"Loading RAG index from {rag_dir}")
        rag = load_rag_state(rag_dir, args.rag_embedder, device)
        print()

    for label, ckpt_path in parse_specs(args.checkpoints):
        print(f"=== {label}  ({ckpt_path}) ===")
        model, _ = load_model(ckpt_path, device)
        log_path = Path(args.logs_dir) / f"{label}.jsonl"
        log_fh = log_path.open("w", encoding="utf-8")
        hits = 0
        topic_total: dict = {}
        topic_hits: dict = {}
        for rec in bench:
            # If RAG is loaded, prepend retrieved passages to the prompt.
            # Otherwise pass the bare question through. Same prompt shape
            # the demo Space and scripts/rag_chat.py use.
            if rag is not None:
                prompt = rag_augmented_prompt(rec["prompt"], rag, args.rag_top_k, device)
            else:
                prompt = rec["prompt"]
            completion = generate(model, tokenizer, prompt,
                                  max_tokens=args.max_tokens,
                                  temperature=args.temperature, top_k=args.top_k,
                                  device=device)
            ok, reason = grade_record(rec, completion)
            t = rec.get("topic", "misc")
            topic_total[t] = topic_total.get(t, 0) + 1
            if ok:
                hits += 1
                topic_hits[t] = topic_hits.get(t, 0) + 1
            log_fh.write(json.dumps({
                "id": rec.get("id"),
                "topic": t,
                "prompt": rec["prompt"],
                "completion": completion,
                "hit": ok,
                "reason": reason,
            }, ensure_ascii=False) + "\n")
        log_fh.close()
        rate = hits / max(1, len(bench))
        print(f"  hits: {hits}/{len(bench)}  ({100*rate:.1f}%)")
        print("  per-topic:")
        for t in sorted(topic_total):
            print(f"    {t:10s}  {topic_hits.get(t, 0):3d}/{topic_total[t]:3d}")
        summary_rows.append({
            "label": label,
            "checkpoint": str(ckpt_path),
            "hits": hits, "total": len(bench),
            "rate": rate,
            "per_topic": {t: (topic_hits.get(t, 0), topic_total[t]) for t in topic_total},
        })

        # Free model before loading the next checkpoint.
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        elif hasattr(torch, "backends") and torch.backends.mps.is_available():
            torch.mps.empty_cache()
        print()

    # Print final cross-checkpoint table.
    print("=== Summary ===")
    print(f"{'checkpoint':40s} {'hits':>10s} {'rate':>8s}")
    for r in summary_rows:
        print(f"{r['label']:40s} {r['hits']:>4d}/{r['total']:>3d}  {100*r['rate']:>6.1f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
