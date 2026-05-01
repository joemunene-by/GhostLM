#!/usr/bin/env python3
"""Build the GhostLM chat-tuning dataset from the existing pretrain corpus.

The pretrain corpus at ``data/processed/train.jsonl`` is structured by source
(nvd, exploitdb, mitre_attack, capec, ctftime, arxiv, synthetic). For each
source we apply per-source templates that turn a single document into one
synthetic User/Assistant turn — "What is CVE-2024-X?" → first paragraph of
the description, etc. Combined with the hand-written ``small_talk.jsonl``
seed, this produces a ~10K-pair instruction dataset ready for SFT.

Output: ``data/processed/chat_train.jsonl`` and ``chat_val.jsonl`` —
JSONL where each line is ``{"turns": [...], "source": ...}``.

Determinism: a fixed seed (``--seed``) controls all sampling and template
choice, so the dataset is reproducible.
"""

import argparse
import hashlib
import json
import random
import re
from collections import Counter
from pathlib import Path
from typing import Iterable, List, Optional


# ---------------------------------------------------------------------------
# Per-source template libraries
# ---------------------------------------------------------------------------

NVD_QUESTIONS = [
    "What is {id}?",
    "Explain {id}.",
    "Tell me about {id}.",
    "Describe the vulnerability {id}.",
    "What does {id} affect, and how?",
    "Walk me through {id}.",
    "What's the impact of {id}?",
    "Summarize {id}.",
]

MITRE_QUESTIONS = [
    "What is {id}?",
    "Explain MITRE technique {id}.",
    "Tell me about {id} ({name}).",
    "What does the {name} technique do?",
    "How is {name} used by adversaries?",
    "Summarize the {name} technique.",
    "Describe MITRE ATT&CK {id}.",
]

CAPEC_QUESTIONS = [
    "What is {id}?",
    "Explain the {name} attack pattern.",
    "Tell me about {id} ({name}).",
    "Describe the {name} CAPEC.",
    "What does the {name} attack pattern look like?",
    "Summarize {id}.",
]

EXPLOITDB_QUESTIONS = [
    "Tell me about Exploit-DB #{id}.",
    "What's in Exploit-DB entry #{id}?",
    "Summarize Exploit-DB {id}.",
    "What vulnerability does Exploit-DB #{id} target?",
]

CTFTIME_QUESTIONS = [
    "How was the '{task}' challenge solved?",
    "Walk me through '{task}' from {event}.",
    "What was the approach to '{task}'?",
    "Describe the '{task}' CTF challenge.",
    "Summarize the '{task}' writeup.",
]

SYNTHETIC_QUESTIONS = [
    "How do you exploit {topic}?",
    "Walk me through a {topic} attack.",
    "Explain {topic} in a CTF context.",
    "What does a typical {topic} exploitation look like?",
    "Talk me through {topic}.",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _sha_pick(items: List[str], key: str) -> str:
    """Deterministically pick an item from ``items`` keyed by ``key``."""
    h = int(hashlib.sha1(key.encode("utf-8")).hexdigest(), 16)
    return items[h % len(items)]


def _trim_to_paragraphs(text: str, max_chars: int = 1200) -> str:
    """Trim ``text`` to whole paragraphs, capped near ``max_chars``."""
    text = text.strip()
    if len(text) <= max_chars:
        return text
    paragraphs = text.split("\n\n")
    out: List[str] = []
    total = 0
    for p in paragraphs:
        if total + len(p) > max_chars and out:
            break
        out.append(p)
        total += len(p) + 2
    if not out:
        return text[:max_chars].rsplit(" ", 1)[0]
    return "\n\n".join(out).strip()


def _strip_md_badges(text: str) -> str:
    """Drop common markdown badge / image patterns that hurt readability."""
    text = re.sub(r"!\[[^\]]*\]\([^)]+\)", "", text)
    text = re.sub(r"\[!\[[^\]]*\]\([^)]+\)\]\([^)]+\)", "", text)
    return text


# ---------------------------------------------------------------------------
# Per-source builders — each returns list[dict] of chat records
# ---------------------------------------------------------------------------


def build_nvd(records: List[dict], target: int, rng: random.Random) -> List[dict]:
    """Sample NVD records and template Q&A pairs from each."""
    pool = [r for r in records if r.get("id", "").startswith("CVE-")]
    pool = rng.sample(pool, min(target, len(pool)))
    out: List[dict] = []
    for r in pool:
        cve = r["id"]
        text = r["text"].strip()
        # NVD descriptions are typically one paragraph already; cap by char.
        answer = _trim_to_paragraphs(text, max_chars=900)
        question = _sha_pick(NVD_QUESTIONS, cve).format(id=cve)
        out.append({
            "turns": [
                {"role": "user", "content": question},
                {"role": "assistant", "content": answer},
            ],
            "source": "nvd",
            "ref": cve,
        })
    return out


def build_mitre(records: List[dict], rng: random.Random) -> List[dict]:
    """Template every MITRE ATT&CK technique into one Q&A pair."""
    out: List[dict] = []
    for r in records:
        text = r["text"].strip()
        # First line: "MITRE ATT&CK Technique T1578.004: Revert Cloud Instance"
        m = re.match(r"MITRE ATT&CK Technique (T[\d.]+):\s*(.+)", text)
        if not m:
            continue
        tid, tname = m.group(1), m.group(2).strip()
        body = "\n".join(text.splitlines()[1:]).strip()
        body = _trim_to_paragraphs(body, max_chars=1200)
        if not body:
            continue
        # Build a name-aware answer that re-states the technique up front.
        answer = f"{tid} — {tname}.\n\n{body}"
        question = _sha_pick(MITRE_QUESTIONS, tid).format(id=tid, name=tname)
        out.append({
            "turns": [
                {"role": "user", "content": question},
                {"role": "assistant", "content": answer},
            ],
            "source": "mitre_attack",
            "ref": tid,
        })
    return out


def build_capec(records: List[dict], rng: random.Random) -> List[dict]:
    """Template each CAPEC pattern into one Q&A pair."""
    out: List[dict] = []
    for r in records:
        text = r["text"].strip()
        m = re.match(r"(CAPEC-\d+):\s*(.+)", text)
        if not m:
            continue
        cid, cname = m.group(1), m.group(2).strip()
        body = "\n".join(text.splitlines()[1:]).strip()
        body = _trim_to_paragraphs(body, max_chars=1200)
        if not body:
            continue
        answer = f"{cid} — {cname}.\n\n{body}"
        question = _sha_pick(CAPEC_QUESTIONS, cid).format(id=cid, name=cname)
        out.append({
            "turns": [
                {"role": "user", "content": question},
                {"role": "assistant", "content": answer},
            ],
            "source": "capec",
            "ref": cid,
        })
    return out


def build_exploitdb(records: List[dict], target: int, rng: random.Random) -> List[dict]:
    """Sample Exploit-DB records and produce vulnerability-summary Q&A."""
    pool = [r for r in records if r.get("id")]
    pool = rng.sample(pool, min(target, len(pool)))
    out: List[dict] = []
    for r in pool:
        eid = str(r["id"])
        text = r["text"].strip()
        # First line is "Exploit-DB #51569: Gila CMS 1.10.9 - Remote Code Execution (RCE) (Authenticated)"
        first, _, rest = text.partition("\n")
        m = re.match(r"Exploit-DB #(\d+):\s*(.+)", first)
        if not m:
            continue
        title = m.group(2).strip()
        # Drop raw exploit code — keep header lines and any narrative.
        narrative_lines: List[str] = []
        for line in rest.splitlines():
            if line.startswith("```") or line.startswith("#!/"):
                break
            narrative_lines.append(line)
        narrative = "\n".join(narrative_lines).strip()
        narrative = _trim_to_paragraphs(narrative, max_chars=1100)
        if not narrative:
            narrative = "(no narrative available — see the linked exploit code for details)"
        platform = r.get("platform", "")
        date = r.get("date", "")
        meta_bits = [b for b in (platform, date) if b]
        meta_line = f" Platform/date: {', '.join(meta_bits)}." if meta_bits else ""
        answer = f"Exploit-DB #{eid} — {title}.{meta_line}\n\n{narrative}"
        question = _sha_pick(EXPLOITDB_QUESTIONS, eid).format(id=eid)
        out.append({
            "turns": [
                {"role": "user", "content": question},
                {"role": "assistant", "content": answer},
            ],
            "source": "exploitdb",
            "ref": eid,
        })
    return out


def build_ctftime(records: List[dict], rng: random.Random) -> List[dict]:
    """Template CTFtime writeups into 'how was X solved' Q&A pairs."""
    out: List[dict] = []
    for r in records:
        task = (r.get("task_name") or "").strip()
        event = (r.get("event_name") or "").strip()
        if not task:
            continue
        text = _strip_md_badges(r.get("text", "")).strip()
        text = _trim_to_paragraphs(text, max_chars=1500)
        if not text:
            continue
        question = _sha_pick(CTFTIME_QUESTIONS, f"{task}|{event}").format(
            task=task, event=event or "the CTF",
        )
        out.append({
            "turns": [
                {"role": "user", "content": question},
                {"role": "assistant", "content": text},
            ],
            "source": "ctftime",
            "ref": str(r.get("writeup_id") or task),
        })
    return out


def build_synthetic(records: List[dict], rng: random.Random) -> List[dict]:
    """Template synthetic CTF-style writeups with topic-keyed questions."""
    out: List[dict] = []
    for r in records:
        topic = (r.get("subtopic") or r.get("topic") or "").strip()
        if not topic:
            continue
        text = r.get("text", "").strip()
        text = _trim_to_paragraphs(text, max_chars=1500)
        if not text:
            continue
        question = _sha_pick(SYNTHETIC_QUESTIONS, topic + r.get("id", "")).format(topic=topic)
        out.append({
            "turns": [
                {"role": "user", "content": question},
                {"role": "assistant", "content": text},
            ],
            "source": "synthetic",
            "ref": r.get("id", topic),
        })
    return out


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def load_jsonl(path: Path) -> List[dict]:
    """Load a JSONL file into a list of dicts."""
    out: List[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out


def write_jsonl(records: Iterable[dict], path: Path) -> int:
    """Write records to a JSONL file, returning the count written."""
    n = 0
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            n += 1
    return n


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    p = argparse.ArgumentParser(description="Build GhostLM chat-tuning dataset")
    p.add_argument("--corpus", default="data/processed/train.jsonl",
                   help="Source pretrain corpus")
    p.add_argument("--small-talk", default="data/raw/chat/small_talk.jsonl",
                   help="Hand-written small-talk seed JSONL")
    p.add_argument("--out-train", default="data/processed/chat_train.jsonl")
    p.add_argument("--out-val", default="data/processed/chat_val.jsonl")
    p.add_argument("--val-frac", type=float, default=0.05,
                   help="Fraction of synthetic Q&A held out for validation")
    p.add_argument("--nvd-target", type=int, default=3500,
                   help="Number of NVD records to sample (out of 64k)")
    p.add_argument("--exploitdb-target", type=int, default=2000,
                   help="Number of Exploit-DB records to sample")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main() -> None:
    """Build chat_train.jsonl and chat_val.jsonl from corpus + small-talk seed."""
    args = parse_args()
    rng = random.Random(args.seed)

    corpus = load_jsonl(Path(args.corpus))
    by_source: dict = {}
    for r in corpus:
        by_source.setdefault(r.get("source", "unknown"), []).append(r)

    print("=== Source counts in pretrain corpus ===")
    for s, rs in sorted(by_source.items(), key=lambda kv: -len(kv[1])):
        print(f"  {s}: {len(rs):,}")

    pairs: List[dict] = []
    pairs.extend(build_nvd(by_source.get("nvd", []), args.nvd_target, rng))
    pairs.extend(build_mitre(by_source.get("mitre_attack", []), rng))
    pairs.extend(build_capec(by_source.get("capec", []), rng))
    pairs.extend(build_exploitdb(by_source.get("exploitdb", []), args.exploitdb_target, rng))
    pairs.extend(build_ctftime(by_source.get("ctftime", []), rng))
    pairs.extend(build_synthetic(by_source.get("synthetic", []), rng))

    print()
    print("=== Generated Q&A pairs by source ===")
    for s, c in Counter(p["source"] for p in pairs).most_common():
        print(f"  {s}: {c:,}")
    print(f"  total: {len(pairs):,}")

    # Held-out validation split from the synthetic Q&A only (small-talk all goes to train).
    rng.shuffle(pairs)
    split = max(1, int(len(pairs) * args.val_frac))
    val_pairs = pairs[:split]
    train_pairs = pairs[split:]

    # Small-talk: always train, never val (we want every greeting / identity
    # answer baked into the model).
    small_talk = load_jsonl(Path(args.small_talk))
    print(f"  small_talk: {len(small_talk):,} (all to train)")
    train_pairs.extend(small_talk)

    rng.shuffle(train_pairs)

    n_train = write_jsonl(train_pairs, Path(args.out_train))
    n_val = write_jsonl(val_pairs, Path(args.out_val))
    print()
    print(f"Wrote {n_train:,} → {args.out_train}")
    print(f"Wrote {n_val:,} → {args.out_val}")


if __name__ == "__main__":
    main()
