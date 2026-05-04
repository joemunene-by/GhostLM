#!/usr/bin/env python3
"""Generate high-fact-density Q&A pairs from cybersec source documents
via Qwen-14B (Ollama).

The empirical work in docs/ctibench_bias_finding.md shows that all our
chat-tunes top out at ~30% real capability on CTIBench MCQ regardless
of architecture or recipe. Live testing confirmed the model is a
"cybersec parrot" — knows vocabulary patterns but lacks fact storage.
The diagnosis (per 5 independent AI sources): our 60M-token corpus is
too low-fact-density. CTF writeups teach style, not retrievable facts.

This script fixes the data-density problem. It feeds source records
(MITRE techniques, MITRE full STIX, CAPEC, CISA KEV, NVD CVE samples)
into Qwen-14B with a structured prompt that extracts 4-8 atomic
factual Q&A pairs per record. The output is a fact-dense JSONL that
can be mixed into pretraining at a high oversample ratio.

Output schema (one Q&A per line — pretrain documents, not chat):

  {"id": "T1059#fact1", "source": "fact_qa",
   "text": "Q: What is MITRE ATT&CK technique T1059?\\n\\n
            A: T1059 is Command and Scripting Interpreter, a technique
               adversaries use to execute commands or scripts via shells
               like cmd.exe, PowerShell, or bash."}

Resume-safe — re-running picks up where it left off (skips records
already in the output file). Single pass; restarts on Ollama failure.

Prerequisite: Ollama running locally with qwen2.5:14b cached.
Throughput on M4: ~15-20 output tokens/sec, so 8000 facts at ~80
tokens each ≈ 8 hours of overnight compute.
"""

from __future__ import annotations

import argparse
import json
import re
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Dict, List, Optional


OLLAMA_URL = "http://localhost:11434/api/generate"

PROMPT_TEMPLATE = """You are a cybersecurity expert building a fact-dense
training corpus for a small language model. Given the source document below,
extract {n_pairs} discrete factual question/answer pairs.

Rules:
- Each Q&A must be answerable from the source text alone.
- Questions must be SPECIFIC and FACTUAL (e.g., "What CVE is associated with EternalBlue?", "What port does SMB use by default?", "What is MITRE technique T1059?"). Avoid vague questions.
- Answers must be CONCISE and FACTUAL (1-3 sentences). Include specific identifiers, dates, ports, names where they appear.
- Vary the question forms (definition, association, attribute, mechanism).
- Output EXACTLY {n_pairs} pairs in this format, nothing else:

Q1: <question>
A1: <answer>

Q2: <question>
A2: <answer>

...continue through Q{n_pairs}.

Source document:
{text}

Now output the {n_pairs} Q&A pairs:"""

QA_REGEX = re.compile(r"Q\d+:\s*(.+?)\s*\n\s*A\d+:\s*(.+?)(?=\n\s*Q\d+:|\Z)", re.DOTALL)


def parse_args() -> argparse.Namespace:
    """CLI args."""
    p = argparse.ArgumentParser(description="Generate fact-dense Q&A pairs from cybersec sources via Qwen")
    p.add_argument("--sources", nargs="+", default=[
        "data/raw/mitre_attack.jsonl",
        "data/raw/mitre_full.jsonl",
        "data/raw/capec.jsonl",
        "data/raw/cisa_kev.jsonl",
    ], help="Source JSONL files to extract facts from")
    p.add_argument("--nvd-jsonl", default="data/raw/cve.jsonl",
                   help="NVD JSONL (sampled, since full is too large)")
    p.add_argument("--nvd-sample", type=int, default=2000,
                   help="Number of NVD records to sample")
    p.add_argument("--out", default="data/raw/fact_qa.jsonl")
    p.add_argument("--model", default="qwen2.5:14b")
    p.add_argument("--n-pairs-per-doc", type=int, default=5,
                   help="Q&A pairs to extract per source document")
    p.add_argument("--temperature", type=float, default=0.3,
                   help="Lower = more focused, factual answers")
    p.add_argument("--limit", type=int, default=None,
                   help="Cap total source records (smoke testing)")
    p.add_argument("--max-text-chars", type=int, default=2000,
                   help="Truncate source text to this many chars before Qwen")
    return p.parse_args()


def call_ollama(model: str, prompt: str, temperature: float, timeout: int = 180) -> Optional[str]:
    """POST to Ollama's /api/generate and return the response text, or None on error."""
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": temperature, "num_predict": 1024},
    }
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(OLLAMA_URL, data=data,
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = json.loads(resp.read())
            return body.get("response", "").strip()
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError) as e:
        print(f"  ollama error: {e}")
        return None


def parse_qa_pairs(raw: str) -> List[Dict[str, str]]:
    """Pull Q/A pairs out of Qwen's response."""
    if not raw:
        return []
    matches = QA_REGEX.findall(raw)
    pairs = []
    for q, a in matches:
        q = q.strip()
        a = a.strip()
        if len(q) < 8 or len(a) < 8:
            continue
        pairs.append({"q": q, "a": a})
    return pairs


def render_qa_record(rec_id: str, q: str, a: str, idx: int) -> Dict:
    """Format one Q&A as a pretrain text record (single document)."""
    text = f"Q: {q}\n\nA: {a}"
    return {"id": f"{rec_id}#fact{idx}", "source": "fact_qa", "text": text}


def load_sources(paths: List[str], nvd_path: str, nvd_sample: int, limit: Optional[int]) -> List[Dict]:
    """Load source records from MITRE/CAPEC/KEV/NVD into a flat list."""
    import random
    rng = random.Random(42)
    out: List[Dict] = []

    for path in paths:
        p = Path(path)
        if not p.exists():
            print(f"  skip (missing): {path}")
            continue
        with p.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                rec["_src"] = p.stem
                out.append(rec)
        print(f"  loaded {len(out)} records after {path}")

    nvd_p = Path(nvd_path)
    if nvd_p.exists():
        nvd_all: List[Dict] = []
        with nvd_p.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                nvd_all.append(json.loads(line))
        sample = rng.sample(nvd_all, min(nvd_sample, len(nvd_all)))
        for rec in sample:
            rec["_src"] = "nvd"
            out.append(rec)
        print(f"  sampled {len(sample)} NVD records (from {len(nvd_all)} total)")

    if limit:
        out = out[:limit]
    return out


def main() -> None:
    """Generate Q&A pairs from sources, resume-safe."""
    args = parse_args()
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    seen_ids: set = set()
    if out_path.exists():
        with out_path.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    rec = json.loads(line)
                    src_id = rec.get("id", "").split("#")[0]
                    if src_id:
                        seen_ids.add(src_id)
        print(f"  resume: {len(seen_ids)} source records already processed")

    sources = load_sources(args.sources, args.nvd_jsonl, args.nvd_sample, args.limit)
    print(f"\nTotal source records: {len(sources)}")
    print(f"Already processed:    {len(seen_ids)}")
    print(f"To process:           {len(sources) - len(seen_ids)}")
    print()

    out_fh = out_path.open("a", encoding="utf-8", buffering=1)
    written = 0
    failed = 0
    t0 = time.time()

    for i, rec in enumerate(sources):
        rec_id = str(rec.get("id") or rec.get("ref") or f"{rec['_src']}_{i}")
        if rec_id in seen_ids:
            continue

        text = (rec.get("text") or "").strip()
        if not text or len(text) < 80:
            continue
        if len(text) > args.max_text_chars:
            text = text[: args.max_text_chars].rsplit("\n", 1)[0]

        prompt = PROMPT_TEMPLATE.format(
            n_pairs=args.n_pairs_per_doc, text=text,
        )
        raw = call_ollama(args.model, prompt, args.temperature)
        pairs = parse_qa_pairs(raw or "")

        if not pairs:
            failed += 1
            continue

        for j, pair in enumerate(pairs):
            out_fh.write(json.dumps(
                render_qa_record(rec_id, pair["q"], pair["a"], j),
                ensure_ascii=False,
            ) + "\n")
            written += 1
        out_fh.flush()
        seen_ids.add(rec_id)

        if (len(seen_ids) - (len(seen_ids) - 1)) % 25 == 0 or i % 25 == 0:
            elapsed = time.time() - t0
            rate = (i + 1 - len(seen_ids) + (len(seen_ids) - sum(1 for _ in []))) / max(1, elapsed) * 60
            print(f"  [{i + 1}/{len(sources)}] written={written} failed={failed} "
                  f"rate~{rate:.1f} src/min")

    out_fh.close()
    print()
    print(f"Done. Wrote {written} Q&A records to {out_path}")
    if failed:
        print(f"  Failed (no parseable pairs) on {failed} source records")


if __name__ == "__main__":
    main()
