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

MITRE_FULL_QUESTIONS_GENERIC = [
    "What is {id}?",
    "Tell me about {id}.",
    "Describe MITRE ATT&CK {id}.",
    "Summarize {id}.",
    "Explain {id}.",
]

MITRE_FULL_QUESTIONS_NAMED = [
    "What is {id} ({name})?",
    "Tell me about the {name} {type_lower}.",
    "Describe the {name} {type_lower}.",
    "Summarize {id} — {name}.",
    "What does {name} do?",
]

CISA_KEV_QUESTIONS = [
    "What is {cve}?",
    "Tell me about {cve}.",
    "Has {cve} been exploited in the wild?",
    "Why is {cve} on the CISA KEV catalog?",
    "Describe {cve}.",
    "What product does {cve} affect?",
    "Summarize {cve}.",
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


def build_mitre_full(records: List[dict], rng: random.Random) -> List[dict]:
    """Template MITRE full STIX records (mitigations, groups, malware, tools,
    tactics, campaigns) into Q&A pairs. The records share the same
    "MITRE ATT&CK <Type> <ID>: <Name>\\n" head as build_mitre, just with
    different type prefixes."""
    out: List[dict] = []
    head_re = re.compile(r"MITRE ATT&CK ([\w()/ -]+?)\s+([A-Z]+\d+(?:\.\d+)?):\s*(.+)")
    for r in records:
        text = r["text"].strip()
        first, _, rest = text.partition("\n")
        m = head_re.match(first)
        if not m:
            continue
        type_label = m.group(1).strip()
        rid = m.group(2).strip()
        name = m.group(3).strip()
        body = rest.strip()
        body = _trim_to_paragraphs(body, max_chars=1500)
        if not body:
            continue
        answer = f"{rid} — {name}.\n\n{body}"

        # Half the records use a generic id-keyed question; half use the
        # name to teach name → description retrieval.
        use_named = (int(hashlib.sha1(rid.encode("utf-8")).hexdigest(), 16) & 1) == 0
        if use_named:
            template = _sha_pick(MITRE_FULL_QUESTIONS_NAMED, rid)
            question = template.format(id=rid, name=name, type_lower=type_label.lower())
        else:
            template = _sha_pick(MITRE_FULL_QUESTIONS_GENERIC, rid)
            question = template.format(id=rid, name=name)

        out.append({
            "turns": [
                {"role": "user", "content": question},
                {"role": "assistant", "content": answer},
            ],
            "source": "mitre_full",
            "ref": rid,
        })
    return out


def build_cisa_kev(records: List[dict], rng: random.Random) -> List[dict]:
    """Template CISA KEV records (actively exploited CVEs) into Q&A pairs."""
    out: List[dict] = []
    head_re = re.compile(r"CISA KEV\s+—\s+(CVE-[\w-]+)")
    for r in records:
        text = r["text"].strip()
        m = head_re.search(text)
        if not m:
            continue
        cve = m.group(1).strip()
        body = _trim_to_paragraphs(text, max_chars=1500)
        question = _sha_pick(CISA_KEV_QUESTIONS, cve).format(cve=cve)
        out.append({
            "turns": [
                {"role": "user", "content": question},
                {"role": "assistant", "content": body},
            ],
            "source": "cisa_kev",
            "ref": cve,
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
    p.add_argument("--small-talk-multiplier", type=int, default=30,
                   help="How many copies of each small_talk pair to inject. "
                        "v1 used 1× and the model never learned to follow "
                        "instructions because cybersec swamped chat-shape signal.")
    p.add_argument("--small-talk-val-frac", type=float, default=0.1,
                   help="Fraction of small_talk pairs held out for validation "
                        "(BEFORE oversampling — keeps val pairs unique)")
    p.add_argument("--mcq-jsonl", default="data/raw/chat/mcq.jsonl",
                   help="Optional MCQ-format chat data to mix in (built by "
                        "scripts/build_mcq_data.py). Set to '' to skip.")
    p.add_argument("--mcq-multiplier", type=int, default=2,
                   help="Copies of the MCQ set to inject. 2× brings ~3.5K "
                        "MCQ-format examples into the mix and trains the "
                        "model to output a single letter after Answer:.")
    p.add_argument("--mcq-val-frac", type=float, default=0.05,
                   help="Held-out fraction of MCQs for validation.")
    p.add_argument("--mcq-cot-jsonl", default="",
                   help="Optional second MCQ source — CoT-templated records "
                        "(letter answer + 1-2 sentence justification, built by "
                        "scripts/build_mcq_cot_data.py). Mixed alongside the "
                        "raw letter-only MCQs to give the model both the "
                        "shortcut signal and the reasoning supervision.")
    p.add_argument("--mcq-cot-multiplier", type=int, default=2,
                   help="Copies of the CoT MCQ set to inject. Hybrid recipe "
                        "keeps raw MCQs at high mult and CoT at low mult.")
    p.add_argument("--exclude-sources", nargs="*", default=[],
                   help="Source names to skip (e.g. mitre_full cisa_kev) — "
                        "used to A/B-test which sources help vs hurt.")
    p.add_argument("--general-knowledge",
                   default="data/raw/chat/general_knowledge.jsonl",
                   help="Hand-written general-knowledge seed JSONL "
                        "(math, science, programming, geography, "
                        "etymology, refusal/uncertainty patterns). "
                        "Set to '' to skip.")
    p.add_argument("--general-knowledge-multiplier", type=int, default=5,
                   help="Copies of general_knowledge to inject. Default 5x "
                        "brings the bank to roughly five percent of "
                        "training pairs, which teaches 'GhostLM is not "
                        "pure cybersec' without swamping the security "
                        "signal.")
    p.add_argument("--general-knowledge-val-frac", type=float, default=0.1,
                   help="Held-out fraction of general_knowledge for val.")
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
    excluded = set(args.exclude_sources)
    if "mitre_full" not in excluded:
        pairs.extend(build_mitre_full(by_source.get("mitre_full", []), rng))
    if "cisa_kev" not in excluded:
        pairs.extend(build_cisa_kev(by_source.get("cisa_kev", []), rng))

    print()
    print("=== Generated Q&A pairs by source ===")
    for s, c in Counter(p["source"] for p in pairs).most_common():
        print(f"  {s}: {c:,}")
    print(f"  total: {len(pairs):,}")

    # Held-out validation split from the synthetic Q&A.
    rng.shuffle(pairs)
    split = max(1, int(len(pairs) * args.val_frac))
    val_pairs = pairs[:split]
    train_pairs = pairs[split:]

    # Small-talk handling — split out a small held-out set first (so val
    # measures generalization on chat-shape, not just memorization), then
    # oversample the training portion. We oversample because v1 had small_talk
    # at 1.6% of training and the resulting model emitted cybersec answers
    # regardless of the user prompt. ~30× brings small_talk to ~30% of pairs.
    small_talk_all = load_jsonl(Path(args.small_talk))
    rng.shuffle(small_talk_all)
    n_st_val = max(1, int(len(small_talk_all) * args.small_talk_val_frac))
    small_talk_val = small_talk_all[:n_st_val]
    small_talk_train_unique = small_talk_all[n_st_val:]
    small_talk_train_oversampled = small_talk_train_unique * args.small_talk_multiplier

    print(f"  small_talk: unique={len(small_talk_all):,} "
          f"(val={len(small_talk_val):,}, train_unique={len(small_talk_train_unique):,}, "
          f"train_after_×{args.small_talk_multiplier}={len(small_talk_train_oversampled):,})")

    train_pairs.extend(small_talk_train_oversampled)
    val_pairs.extend(small_talk_val)

    # MCQ-format examples — separate stream, mixed in same way as small_talk.
    # We oversample because the goal is to teach a single new behavior
    # (output a letter after "Answer:"), and a 2× multiplier on ~1.8K
    # examples lands the model on that signal across multiple epochs.
    if args.mcq_jsonl and Path(args.mcq_jsonl).exists():
        mcq_all = load_jsonl(Path(args.mcq_jsonl))
        rng.shuffle(mcq_all)
        n_mcq_val = max(1, int(len(mcq_all) * args.mcq_val_frac))
        mcq_val = mcq_all[:n_mcq_val]
        mcq_train_unique = mcq_all[n_mcq_val:]
        mcq_train_oversampled = mcq_train_unique * args.mcq_multiplier
        print(f"  mcq: unique={len(mcq_all):,} "
              f"(val={len(mcq_val):,}, train_unique={len(mcq_train_unique):,}, "
              f"train_after_×{args.mcq_multiplier}={len(mcq_train_oversampled):,})")
        train_pairs.extend(mcq_train_oversampled)
        val_pairs.extend(mcq_val)

    if args.mcq_cot_jsonl and Path(args.mcq_cot_jsonl).exists():
        cot_all = load_jsonl(Path(args.mcq_cot_jsonl))
        rng.shuffle(cot_all)
        n_cot_val = max(1, int(len(cot_all) * args.mcq_val_frac))
        cot_val = cot_all[:n_cot_val]
        cot_train_unique = cot_all[n_cot_val:]
        cot_train_oversampled = cot_train_unique * args.mcq_cot_multiplier
        print(f"  mcq_cot: unique={len(cot_all):,} "
              f"(val={len(cot_val):,}, train_unique={len(cot_train_unique):,}, "
              f"train_after_×{args.mcq_cot_multiplier}={len(cot_train_oversampled):,})")
        train_pairs.extend(cot_train_oversampled)
        val_pairs.extend(cot_val)

    # General-knowledge seed (math, science, programming, refusals,
    # cross-domain identity) — mixed at ~5% of training to teach the
    # model that GhostLM responds outside cybersec without swamping
    # the security signal.
    if args.general_knowledge and Path(args.general_knowledge).exists():
        gk_all = load_jsonl(Path(args.general_knowledge))
        rng.shuffle(gk_all)
        n_gk_val = max(1, int(len(gk_all)
                                * args.general_knowledge_val_frac))
        gk_val = gk_all[:n_gk_val]
        gk_train_unique = gk_all[n_gk_val:]
        gk_train_over = (gk_train_unique
                          * args.general_knowledge_multiplier)
        print(f"  general_knowledge: unique={len(gk_all):,} "
              f"(val={len(gk_val):,}, train_unique="
              f"{len(gk_train_unique):,}, train_after_"
              f"×{args.general_knowledge_multiplier}="
              f"{len(gk_train_over):,})")
        train_pairs.extend(gk_train_over)
        val_pairs.extend(gk_val)

    rng.shuffle(train_pairs)
    rng.shuffle(val_pairs)

    pct_st = 100.0 * len(small_talk_train_oversampled) / max(1, len(train_pairs))
    print(f"  train mix: small_talk={pct_st:.1f}%, cybersec+mcq={100 - pct_st:.1f}%")

    n_train = write_jsonl(train_pairs, Path(args.out_train))
    n_val = write_jsonl(val_pairs, Path(args.out_val))
    print()
    print(f"Wrote {n_train:,} → {args.out_train}")
    print(f"Wrote {n_val:,} → {args.out_val}")


if __name__ == "__main__":
    main()
