#!/usr/bin/env python3
"""Templated synthesis of provenance-augmented tool-use traces (bet 9).

Bet 9 ([docs/differentiation.md](differentiation.md) §"Bet 9:
operator-grade reasoning + provenance") trains ghost-base to *cite*
the tool response that justifies each factual claim. The trace
format extends the bet 1 four-message shape with a ``<|cite|>...
<|/cite|>`` tag in the assistant's final answer, attached to each
factual claim:

    USER:      <question>
    ASSISTANT: <|tool_call|>{"name": "<TOOL>", "args": {...}}<|/tool_call|>
    TOOL:      <|tool_response|>{"cve": "X", "description": "Y", ...}<|/tool_response|>
    ASSISTANT: X is described as Y <|cite|>nvd:X<|/cite|>. CVSS 8.1
               <|cite|>nvd:X#cvss<|/cite|>.

The cite scheme is ``{source_type}:{id}[#field]``, where source_type
is the tool name (or 'nvd' / 'mitre' / 'cwe' / 'rag' for the four
canonical sources), id is a unique identifier inside that source,
and the optional #field disambiguates which field of the source the
citation refers to.

Why this earns big-company attention: in a SOC context, wrong-but-
confident is worse than honest-uncertain. Operators need to defend
their analysis to leadership, which means every claim should be
traceable to its source. Big general-purpose models do not do this
consistently; their RLHF reward favours fluency over auditability.
A small from-scratch LM trained day-one on cite-mandatory traces
is a demonstrably different deployment artifact.

Run:

    PYTHONPATH=. python3 scripts/synth_tool_use_provenance.py \\
        --cve data/raw/cve_full.jsonl \\
        --cwe data/raw/cwe.jsonl \\
        --out data/processed/synth_tool_use_provenance.jsonl \\
        --max-cve 200 --max-cwe 100 --max-rag 100

Stacks on top of the 424 plain traces from synth_tool_use.py for a
combined ~800-record SFT corpus.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Dict, Iterator, List, Optional

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.distill_tool_use import trace_quality_ok, TOOLS  # noqa: E402
from scripts.synth_tool_use import (  # noqa: E402
    MITRE_BANK, _CVSS_RE, _REJECT_PREFIXES,
    stream_cve, stream_cwe, stream_rag_seeds,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def build_record(seed_source: str, seed_id: str, trace_text: str
                 ) -> Dict[str, str]:
    h = hashlib.sha1(
        f"{seed_source}\n{seed_id}\n{trace_text}".encode("utf-8")
    ).hexdigest()[:10]
    return {
        "id": f"synth_tool_use_provenance#{seed_id}#{h}",
        "source": "synth_tool_use_provenance",
        "teacher": "templated",
        "seed_source": seed_source,
        "seed_id": seed_id,
        "text": trace_text,
    }


def _format_trace(question: str, tool_name: str, tool_args: Dict,
                  tool_response: Dict, answer: str) -> str:
    """Emit the 4-message trace with the answer carrying cite tags."""
    return (
        f"USER: {question}\n"
        f"ASSISTANT: <|tool_call|>"
        f"{json.dumps({'name': tool_name, 'args': tool_args}, ensure_ascii=False)}"
        f"<|/tool_call|>\n"
        f"TOOL: <|tool_response|>"
        f"{json.dumps(tool_response, ensure_ascii=False)}"
        f"<|/tool_response|>\n"
        f"ASSISTANT: {answer}\n"
    )


def _cite(source_type: str, source_id: str, field: Optional[str] = None) -> str:
    """Format one ``<|cite|>...<|/cite|>`` tag."""
    body = f"{source_type}:{source_id}"
    if field:
        body += f"#{field}"
    return f"<|cite|>{body}<|/cite|>"


CITE_TAG_RE = re.compile(r"<\|cite\|>([^<]+)<\|/cite\|>")


def trace_with_cites_quality_ok(text: str) -> bool:
    """Quality filter for cite-augmented traces. Inherits the four-tag
    + JSON-tool-call check from trace_quality_ok and adds:

      - At least one ``<|cite|>...<|/cite|>`` tag must appear in the
        assistant's final answer (i.e. AFTER the closing
        ``<|/tool_response|>`` tag).
      - Cite tag bodies must match the ``source_type:source_id`` shape;
        a tag with empty content fails."""
    if not trace_quality_ok(text):
        return False
    end_resp = text.find("<|/tool_response|>")
    if end_resp < 0:
        return False
    answer_part = text[end_resp:]
    cites = CITE_TAG_RE.findall(answer_part)
    if not cites:
        return False
    for c in cites:
        c = c.strip()
        if not c or ":" not in c:
            return False
    return True


# ---------------------------------------------------------------------------
# search_cve_nvd with cites
# ---------------------------------------------------------------------------


def synth_cve_trace_cite(rec: Dict, idx: int) -> Optional[Dict[str, str]]:
    cid = rec["id"]
    text = (rec.get("text") or "")[:600]
    is_not_found = (idx % 10 == 9)
    question = f"What is {cid} about?"
    tool_args = {"q": cid}
    if is_not_found:
        tool_response = {"cve": cid, "found": False, "matches": []}
        # Even no-data answers benefit from a citation that points at
        # the failed lookup itself; this teaches the model that
        # 'absence of evidence' is also a citable fact.
        answer = (
            f"The lookup for {cid} returned no matches in NVD "
            f"{_cite('nvd', cid, 'lookup')}. I don't know what this "
            f"CVE is about based on this search alone. Possible "
            f"reasons: rejected / merged identifier, stale local "
            f"snapshot, or malformed id. Re-issue the lookup against "
            f"the canonical NVD endpoint to confirm."
        )
    else:
        cvss = None
        m = _CVSS_RE.search(text)
        if m:
            cvss = m.group(1) or m.group(2)
        tool_response = {
            "cve": cid, "description": text,
            "cvss": cvss, "source": "nvd",
        }
        first_sentence = text.split(".")[0][:300]
        if cvss:
            answer = (
                f"{cid} is described as: {first_sentence} "
                f"{_cite('nvd', cid, 'description')}. CVSS score "
                f"reported as {cvss} {_cite('nvd', cid, 'cvss')}."
            )
        else:
            answer = (
                f"{cid} is described as: {first_sentence} "
                f"{_cite('nvd', cid, 'description')}. No CVSS score "
                f"in the lookup result {_cite('nvd', cid, 'cvss')}."
            )
    trace = _format_trace(question, "search_cve_nvd", tool_args,
                          tool_response, answer)
    if not trace_with_cites_quality_ok(trace):
        return None
    return build_record("search_cve_nvd", cid, trace)


# ---------------------------------------------------------------------------
# lookup_mitre_technique with cites
# ---------------------------------------------------------------------------


def synth_mitre_traces_cite() -> Iterator[Dict[str, str]]:
    for idx, (tcode, name, tactic, platform, summary) in enumerate(MITRE_BANK):
        is_not_found = (idx % 10 == 9)
        question = f"What does ATT&CK technique {tcode} do?"
        tool_args = {"technique_id": tcode}
        if is_not_found:
            tool_response = {"technique_id": tcode, "found": False}
            answer = (
                f"The lookup for ATT&CK technique {tcode} returned "
                f"no matching entry in the catalog "
                f"{_cite('mitre', tcode, 'lookup')}. I don't know "
                f"what this technique is about based on this lookup "
                f"alone. Possible reasons: a sub-technique that "
                f"resolves under a different parent T-code, or a "
                f"stale local catalog snapshot. Re-issue the lookup "
                f"against attack.mitre.org to confirm."
            )
        else:
            tool_response = {
                "id": tcode, "name": name, "tactic": tactic,
                "platform": platform, "summary": summary,
                "url": f"https://attack.mitre.org/techniques/{tcode.replace('.', '/')}/",
            }
            answer = (
                f"{tcode} ({name}) {_cite('mitre', tcode, 'name')} "
                f"is a {tactic} technique on {platform} "
                f"{_cite('mitre', tcode, 'tactic')}: {summary} "
                f"{_cite('mitre', tcode, 'summary')}."
            )
        trace = _format_trace(question, "lookup_mitre_technique",
                              tool_args, tool_response, answer)
        if not trace_with_cites_quality_ok(trace):
            continue
        yield build_record("lookup_mitre_technique", tcode, trace)


# ---------------------------------------------------------------------------
# lookup_cwe with cites
# ---------------------------------------------------------------------------


def synth_cwe_trace_cite(rec: Dict, idx: int) -> Optional[Dict[str, str]]:
    cid = rec["id"]
    text = (rec.get("text") or "")[:500]
    is_not_found = (idx % 10 == 9)
    question = f"What is {cid}?"
    tool_args = {"cwe_id": cid}
    if is_not_found:
        tool_response = {"cwe_id": cid, "found": False}
        answer = (
            f"The lookup for {cid} returned no entry in the CWE "
            f"catalog {_cite('cwe', cid, 'lookup')}. I don't know "
            f"what this weakness is about based on this lookup alone. "
            f"Possible reasons: deprecated or merged entry, or stale "
            f"local snapshot. Re-issue the lookup against "
            f"cwe.mitre.org to confirm the current status of {cid}."
        )
    else:
        first_line = text.split("\n")[0]
        name_part = first_line.split(":", 1)[-1].strip() if ":" in first_line \
                    else first_line
        tool_response = {
            "id": cid, "name": name_part[:80], "description": text,
            "url": f"https://cwe.mitre.org/data/definitions/{cid.replace('CWE-', '')}.html",
        }
        answer = (
            f"{cid}: {name_part[:200]} "
            f"{_cite('cwe', cid, 'name')}. The full description "
            f"begins: {text[:200]} "
            f"{_cite('cwe', cid, 'description')}."
        )
    trace = _format_trace(question, "lookup_cwe", tool_args,
                          tool_response, answer)
    if not trace_with_cites_quality_ok(trace):
        return None
    return build_record("lookup_cwe", cid, trace)


# ---------------------------------------------------------------------------
# rag_retrieve with cites
# ---------------------------------------------------------------------------


def synth_rag_trace_cite(rec: Dict, idx: int) -> Optional[Dict[str, str]]:
    seed_id = rec.get("id") or rec.get("_seed_path") or f"rag_{idx}"
    text = (rec.get("text") or rec.get("content") or "")[:1500]
    is_not_found = (idx % 10 == 9)
    topic_hint = (text.split(".")[0][:80]).strip()
    question = (
        f"From the cybersec corpus, what does this say about "
        f"{topic_hint[:60].lower()}?"
    )
    tool_args = {"query": topic_hint, "k": 4}
    if is_not_found:
        tool_response = {"query": topic_hint, "passages": []}
        answer = (
            f"The corpus retrieval returned no relevant passages for "
            f"'{topic_hint[:50]}...' {_cite('rag', f'q:{topic_hint[:30]}', 'no_match')}. "
            f"I don't know based on this lookup alone. Possible "
            f"reasons: corpus index gap, narrow phrasing, or "
            f"sub-threshold relevance. Try a broader query."
        )
    else:
        chunks = []
        chunks.append({
            "id": "passage_0",
            "text": text[:600],
            "source": rec.get("_seed_path", "corpus"),
            "score": 0.87,
        })
        if len(text) > 700:
            chunks.append({
                "id": "passage_1",
                "text": text[600:1100],
                "source": rec.get("_seed_path", "corpus"),
                "score": 0.74,
            })
        tool_response = {"query": topic_hint, "passages": chunks}
        first_passage = chunks[0]["text"][:300].strip()
        answer = (
            f"Based on the retrieved passages, {first_passage} "
            f"{_cite('rag', 'passage_0', chunks[0]['source'])}."
        )
    trace = _format_trace(question, "rag_retrieve", tool_args,
                          tool_response, answer)
    if not trace_with_cites_quality_ok(trace):
        return None
    return build_record("rag_retrieve", str(seed_id), trace)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cve", default="data/raw/cve_full.jsonl")
    p.add_argument("--cwe", default="data/raw/cwe.jsonl")
    p.add_argument("--rag-seeds", default=None)
    p.add_argument("--out",
                   default="data/processed/synth_tool_use_provenance.jsonl")
    p.add_argument("--max-cve", type=int, default=200)
    p.add_argument("--max-cwe", type=int, default=100)
    p.add_argument("--max-rag", type=int, default=100)
    p.add_argument("--skip-cve", action="store_true")
    p.add_argument("--skip-mitre", action="store_true")
    p.add_argument("--skip-cwe", action="store_true")
    p.add_argument("--skip-rag", action="store_true")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    out_path = REPO_ROOT / args.out if not Path(args.out).is_absolute() \
               else Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    counts: Dict[str, int] = {}
    rejects: Dict[str, int] = {}

    with out_path.open("w", encoding="utf-8") as fout:
        if not args.skip_cve:
            cve_path = REPO_ROOT / args.cve if not Path(args.cve).is_absolute() \
                       else Path(args.cve)
            if not cve_path.exists():
                print(f"  [cve] seed missing at {cve_path}; skipping")
            else:
                for idx, cve in enumerate(stream_cve(cve_path, args.max_cve)):
                    rec = synth_cve_trace_cite(cve, idx)
                    if rec is None:
                        rejects["search_cve_nvd"] = rejects.get("search_cve_nvd", 0) + 1
                        continue
                    fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    counts["search_cve_nvd"] = counts.get("search_cve_nvd", 0) + 1
                print(f"  [search_cve_nvd] {counts.get('search_cve_nvd', 0)} accepted, "
                      f"{rejects.get('search_cve_nvd', 0)} rejected")

        if not args.skip_mitre:
            for rec in synth_mitre_traces_cite():
                fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                counts["lookup_mitre_technique"] = counts.get("lookup_mitre_technique", 0) + 1
            print(f"  [lookup_mitre_technique] {counts.get('lookup_mitre_technique', 0)} accepted")

        if not args.skip_cwe:
            cwe_path = REPO_ROOT / args.cwe if not Path(args.cwe).is_absolute() \
                       else Path(args.cwe)
            if not cwe_path.exists():
                print(f"  [cwe] seed missing at {cwe_path}; skipping")
            else:
                for idx, cwe in enumerate(stream_cwe(cwe_path, args.max_cwe)):
                    rec = synth_cwe_trace_cite(cwe, idx)
                    if rec is None:
                        rejects["lookup_cwe"] = rejects.get("lookup_cwe", 0) + 1
                        continue
                    fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    counts["lookup_cwe"] = counts.get("lookup_cwe", 0) + 1
                print(f"  [lookup_cwe] {counts.get('lookup_cwe', 0)} accepted, "
                      f"{rejects.get('lookup_cwe', 0)} rejected")

        if not args.skip_rag:
            if args.rag_seeds:
                rag_paths = [Path(s.strip()) for s in args.rag_seeds.split(",") if s.strip()]
            else:
                rag_paths = [REPO_ROOT / p for p in TOOLS["rag_retrieve"]["seeds"]]
            for idx, seed in enumerate(stream_rag_seeds(rag_paths, args.max_rag)):
                rec = synth_rag_trace_cite(seed, idx)
                if rec is None:
                    rejects["rag_retrieve"] = rejects.get("rag_retrieve", 0) + 1
                    continue
                fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                counts["rag_retrieve"] = counts.get("rag_retrieve", 0) + 1
            print(f"  [rag_retrieve] {counts.get('rag_retrieve', 0)} accepted, "
                  f"{rejects.get('rag_retrieve', 0)} rejected")

    n_total = sum(counts.values())
    print(f"\nWrote {n_total} cite-augmented traces to {out_path}")
    print(f"  by tool: {counts}")
    print(f"  rejects: {rejects}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
