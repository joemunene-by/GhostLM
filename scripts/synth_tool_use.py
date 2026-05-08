#!/usr/bin/env python3
"""Templated synthesis of bet 1 (tool-use SFT) training records.

The bet 1 hypothesis is that a small model trained to ISSUE TOOL
CALLS for factual lookups beats a same-size model trained to
MEMORIZE the same facts. The canonical pipeline is
``scripts/distill_tool_use.py`` calling an LLM teacher (Anthropic
~$200, free Ollama).

This script is the parallel deterministic-template path: produces
real tool-use traces with no LLM spend and no GPU. Same quality
filter (``trace_quality_ok`` from ``distill_tool_use``) so the
templated records are evaluated by exactly the same correctness
bar as the LLM-distilled flow.

Each emitted trace has the four-message shape the bet 1 SFT data
expects:

    USER: <question>
    ASSISTANT: <|tool_call|>{"name": "<TOOL>", "args": {...}}<|/tool_call|>
    TOOL: <|tool_response|>{...}<|/tool_response|>
    ASSISTANT: <answer that uses ONLY the tool response>

Coverage:

  search_cve_nvd
    Seed: data/raw/cve_full.jsonl (CVE entries 2020+, non-rejected).
    Volume: --max-cve traces (default 200).
    Question template: 'What is <CVE-id> about?'
    Tool response: {"cve": "<id>", "description": "<text[:600]>",
                    "cvss": null} (CVSS extracted via regex when
                    present).

  lookup_mitre_technique
    Seed: hand-curated bank of 30 ATT&CK techniques (same shape as
          synth_format_aware's SIGMA_TECHNIQUES, drawing on the
          tactic + platform fields).
    Volume: 30 traces (one per technique).

  lookup_cwe
    Seed: data/raw/cwe.jsonl.
    Volume: --max-cwe traces (default 100).
    Question template: 'What is <CWE-id>?'.

  rag_retrieve
    Seed: assorted cybersec shards (owasp_top10, security_blogs,
          rfcs, etc.).
    Volume: --max-rag traces (default 100).
    Question template: 'According to <source>, <natural question
                        derived from seed text>'.

Plus a ~10% "not found" injection (matching the bet 1 spec): every
~10th trace per tool returns an empty tool response and an answer
that acknowledges the lookup failed rather than confabulating.
That trains the model to say 'I don't know based on this lookup.'

Run:

    PYTHONPATH=. python3 scripts/synth_tool_use.py \\
        --cve data/raw/cve_full.jsonl \\
        --cwe data/raw/cwe.jsonl \\
        --out data/processed/synth_tool_use.jsonl \\
        --max-cve 200 --max-cwe 100 --max-rag 100

Output JSONL with ``DistillRecord``-shaped entries
(source = 'synth_tool_use'), drops into the SFT data identically to
the LLM-distilled flow.
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


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def build_record(seed_source: str, seed_id: str, trace_text: str) -> Dict[str, str]:
    """DistillRecord-shaped dict ready to write."""
    h = hashlib.sha1(
        f"{seed_source}\n{seed_id}\n{trace_text}".encode("utf-8")
    ).hexdigest()[:10]
    return {
        "id": f"synth_tool_use#{seed_id}#{h}",
        "source": "synth_tool_use",
        "teacher": "templated",
        "seed_source": seed_source,
        "seed_id": seed_id,
        "text": trace_text,
    }


def _format_trace(question: str, tool_name: str, tool_args: Dict,
                  tool_response: Dict, answer: str) -> str:
    """Emit the 4-message trace string in the literal format the bet 1
    quality filter expects."""
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


# ---------------------------------------------------------------------------
# search_cve_nvd
# ---------------------------------------------------------------------------


_REJECT_PREFIXES = ("Rejected reason", "** REJECT **")
_CVSS_RE = re.compile(r"CVSS:\s*([\d.]+)|cvss[\s_-]*v?[2-3]?\s*[\s:]?\s*([\d.]+)",
                      re.IGNORECASE)


def stream_cve(path: Path, max_records: int) -> Iterator[Dict]:
    n = 0
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            cid = rec.get("id", "")
            text = rec.get("text", "") or ""
            if not cid.startswith("CVE-202"):
                continue
            if any(text.startswith(p) for p in _REJECT_PREFIXES):
                continue
            if "DO NOT USE" in text[:80]:
                continue
            yield rec
            n += 1
            if n >= max_records:
                break


def synth_cve_trace(rec: Dict, idx: int) -> Optional[Dict[str, str]]:
    """One templated search_cve_nvd trace."""
    cid = rec["id"]
    text = (rec.get("text") or "")[:600]
    is_not_found = (idx % 10 == 9)  # ~10% "not found" injection
    question = f"What is {cid} about?"
    tool_args = {"q": cid}
    if is_not_found:
        tool_response = {"cve": cid, "found": False, "matches": []}
        answer = (
            f"The lookup for {cid} returned no matches in NVD. I don't "
            f"know what this CVE is about based on this search alone. "
            f"Possible reasons: the CVE has been rejected or merged with "
            f"another, the NVD index has not yet been updated, or the "
            f"identifier is malformed. Try a different CVE id or use "
            f"a free-text query to locate the underlying advisory."
        )
    else:
        cvss = None
        m = _CVSS_RE.search(text)
        if m:
            cvss = m.group(1) or m.group(2)
        tool_response = {
            "cve": cid,
            "description": text,
            "cvss": cvss,
            "source": "nvd",
        }
        # Use first sentence of description for the answer to keep it
        # bounded to facts in the tool response.
        first_sentence = text.split(".")[0][:300]
        answer = f"{cid}: {first_sentence}."
        if cvss:
            answer += f" CVSS score reported as {cvss}."
        else:
            answer += " No CVSS score in the lookup result."
    trace = _format_trace(question, "search_cve_nvd", tool_args,
                          tool_response, answer)
    if not trace_quality_ok(trace):
        return None
    return build_record("search_cve_nvd", cid, trace)


# ---------------------------------------------------------------------------
# lookup_mitre_technique
# ---------------------------------------------------------------------------


# 30 hand-curated techniques: (T-code, name, tactic, platform, summary).
MITRE_BANK = [
    ("T1059.001", "Command and Scripting Interpreter: PowerShell",
     "Execution", "windows",
     "Adversaries use PowerShell for execution; commonly via -EncodedCommand or remote download."),
    ("T1059.003", "Command and Scripting Interpreter: Windows Command Shell",
     "Execution", "windows",
     "cmd.exe used to execute and chain commands; common in initial-access loaders."),
    ("T1059.004", "Command and Scripting Interpreter: Unix Shell",
     "Execution", "linux",
     "bash/sh used by adversaries; curl|sh patterns common in initial-access scripts."),
    ("T1003.001", "OS Credential Dumping: LSASS Memory",
     "Credential Access", "windows",
     "Reading lsass.exe process memory to extract credentials; mimikatz, procdump."),
    ("T1078", "Valid Accounts",
     "Defense Evasion", "windows",
     "Use of stolen credentials to authenticate; blends with normal logon traffic."),
    ("T1133", "External Remote Services",
     "Initial Access", "windows",
     "RDP / VPN / SSH abuse for initial entry; LogonType 10 + public IP."),
    ("T1547.001", "Boot or Logon Autostart Execution: Registry Run Keys",
     "Persistence", "windows",
     "Registry Run / RunOnce values added for autostart on logon."),
    ("T1543.003", "Create or Modify System Process: Windows Service",
     "Persistence", "windows",
     "Adversaries install services for persistence; HKLM\\\\System\\\\Services\\\\."),
    ("T1071.001", "Application Layer Protocol: Web Protocols",
     "Command and Control", "network",
     "C2 over HTTP(S); blends with normal browsing traffic."),
    ("T1486", "Data Encrypted for Impact",
     "Impact", "windows",
     "File contents encrypted with attacker-controlled key; ransomware."),
    ("T1485", "Data Destruction",
     "Impact", "windows",
     "Files / backups deleted to prevent recovery; .bak files targeted."),
    ("T1083", "File and Directory Discovery",
     "Discovery", "windows",
     "where, dir, tree commands enumerated by adversaries scoping access."),
    ("T1018", "Remote System Discovery",
     "Discovery", "windows",
     "net view, nltest /domain_trusts to map AD."),
    ("T1021.001", "Remote Services: Remote Desktop Protocol",
     "Lateral Movement", "windows",
     "RDP from compromised host to other machines; LogonType 10."),
    ("T1021.002", "Remote Services: SMB/Windows Admin Shares",
     "Lateral Movement", "windows",
     "Admin shares used for file movement and lateral execution."),
    ("T1027", "Obfuscated Files or Information",
     "Defense Evasion", "windows",
     "Base64, XOR, or pack/encrypt payloads to evade signatures."),
    ("T1090", "Proxy",
     "Command and Control", "network",
     "Tor, SOCKS, or relay infrastructure to hide C2 origin."),
    ("T1497", "Virtualization/Sandbox Evasion",
     "Defense Evasion", "windows",
     "WMI / API queries to detect VMs and bail out in sandbox."),
    ("T1106", "Native API",
     "Execution", "windows",
     "rundll32 -> cmd.exe / powershell.exe spawn chains via API."),
    ("T1218.011", "System Binary Proxy Execution: Rundll32",
     "Defense Evasion", "windows",
     "rundll32 javascript: or mshtml,RunHTMLApplication to proxy execution."),
    ("T1218.005", "System Binary Proxy Execution: Mshta",
     "Defense Evasion", "windows",
     "mshta loading remote .hta or http URL for execution."),
    ("T1505.003", "Server Software Component: Web Shell",
     "Persistence", "windows",
     "ASP/PHP/JSP web shells dropped under wwwroot or inetpub."),
    ("T1190", "Exploit Public-Facing Application",
     "Initial Access", "network",
     "Web app exploits (SQLi, path traversal, RCE) for initial access."),
    ("T1110", "Brute Force",
     "Credential Access", "windows",
     "Failed-logon bursts; EventID 4625 spikes."),
    ("T1222", "File and Directory Permissions Modification",
     "Defense Evasion", "linux",
     "chmod / chown to make payloads executable or hide changes."),
    ("T1053.005", "Scheduled Task/Job: Scheduled Task",
     "Persistence", "windows",
     "schtasks /create for persistence or privilege escalation."),
    ("T1112", "Modify Registry",
     "Defense Evasion", "windows",
     "Modify HKLM\\\\Software\\\\Microsoft\\\\Windows Defender to disable AV."),
    ("T1574.002", "Hijack Execution Flow: DLL Side-Loading",
     "Defense Evasion", "windows",
     "Drop unsigned DLL next to a signed exe to be loaded by it."),
    ("T1140", "Deobfuscate/Decode Files or Information",
     "Defense Evasion", "windows",
     "certutil -decode / -decodehex used for staged payloads."),
    ("T1566.001", "Phishing: Spearphishing Attachment",
     "Initial Access", "windows",
     "Macro-enabled .docm/.xlsm dropped by phishing emails."),
]


def synth_mitre_traces() -> Iterator[Dict[str, str]]:
    for idx, (tcode, name, tactic, platform, summary) in enumerate(MITRE_BANK):
        is_not_found = (idx % 10 == 9)
        question = f"What does ATT&CK technique {tcode} do?"
        tool_args = {"technique_id": tcode}
        if is_not_found:
            tool_response = {"technique_id": tcode, "found": False}
            answer = (
                f"The lookup for ATT&CK technique {tcode} returned no "
                f"matching entry in the catalog. I don't know what this "
                f"technique is about based on this lookup alone. Possible "
                f"reasons: the identifier may be a sub-technique that "
                f"resolves under a different parent T-code, or the local "
                f"MITRE catalog snapshot may be out of date. Re-issue "
                f"the lookup against the canonical attack.mitre.org "
                f"endpoint to confirm."
            )
        else:
            tool_response = {
                "id": tcode,
                "name": name,
                "tactic": tactic,
                "platform": platform,
                "summary": summary,
                "url": f"https://attack.mitre.org/techniques/{tcode.replace('.', '/')}/",
            }
            answer = (
                f"{tcode} ({name}) is a {tactic} technique on {platform}: "
                f"{summary}"
            )
        trace = _format_trace(question, "lookup_mitre_technique",
                              tool_args, tool_response, answer)
        if not trace_quality_ok(trace):
            continue
        yield build_record("lookup_mitre_technique", tcode, trace)


# ---------------------------------------------------------------------------
# lookup_cwe
# ---------------------------------------------------------------------------


def stream_cwe(path: Path, max_records: int) -> Iterator[Dict]:
    n = 0
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            cid = rec.get("id", "")
            text = (rec.get("text") or "").strip()
            if not cid.startswith("CWE-") or not text:
                continue
            yield rec
            n += 1
            if n >= max_records:
                break


def synth_cwe_trace(rec: Dict, idx: int) -> Optional[Dict[str, str]]:
    cid = rec["id"]
    text = (rec.get("text") or "")[:500]
    is_not_found = (idx % 10 == 9)
    question = f"What is {cid}?"
    tool_args = {"cwe_id": cid}
    if is_not_found:
        tool_response = {"cwe_id": cid, "found": False}
        answer = (
            f"The lookup for {cid} returned no entry in the CWE "
            f"catalog. I don't know what this weakness is about "
            f"based on this lookup alone. Possible reasons: the "
            f"identifier may be a deprecated or merged entry, or "
            f"the local CWE snapshot may be out of date. Re-issue "
            f"the lookup against the canonical cwe.mitre.org "
            f"endpoint to confirm the current status of {cid}."
        )
    else:
        # Pull the first descriptive sentence.
        first_line = text.split("\n")[0]
        name_part = first_line.split(":", 1)[-1].strip() if ":" in first_line else first_line
        tool_response = {
            "id": cid,
            "name": name_part[:80],
            "description": text,
            "url": f"https://cwe.mitre.org/data/definitions/{cid.replace('CWE-', '')}.html",
        }
        answer = f"{cid}: {name_part[:200]}."
    trace = _format_trace(question, "lookup_cwe", tool_args,
                          tool_response, answer)
    if not trace_quality_ok(trace):
        return None
    return build_record("lookup_cwe", cid, trace)


# ---------------------------------------------------------------------------
# rag_retrieve
# ---------------------------------------------------------------------------


def stream_rag_seeds(paths: List[Path], max_records: int) -> Iterator[Dict]:
    n = 0
    for p in paths:
        if not p.exists():
            continue
        with p.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                text = (rec.get("text") or rec.get("content") or "").strip()
                if not text or len(text) < 200:
                    continue
                rec["_seed_path"] = str(p.name)
                yield rec
                n += 1
                if n >= max_records:
                    return


def synth_rag_trace(rec: Dict, idx: int) -> Optional[Dict[str, str]]:
    seed_id = rec.get("id") or rec.get("_seed_path") or f"rag_{idx}"
    text = (rec.get("text") or rec.get("content") or "")[:1500]
    is_not_found = (idx % 10 == 9)
    # Question template: take the first ~80 chars of the seed text as
    # the topic, ask for definition or detail.
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
            f"'{topic_hint[:50]}...'. I don't know based on this lookup "
            f"alone. Possible reasons: the corpus index may not cover "
            f"this specific topic at the requested granularity, the "
            f"query phrasing may be too narrow, or the embedding model "
            f"may have ranked all candidates below the relevance "
            f"threshold. Try a broader query or a different keyword set."
        )
    else:
        # 1-2 passages drawn from the seed text.
        chunks = []
        chunks.append({
            "text": text[:600],
            "source": rec.get("_seed_path", "corpus"),
            "score": 0.87,
        })
        if len(text) > 700:
            chunks.append({
                "text": text[600:1100],
                "source": rec.get("_seed_path", "corpus"),
                "score": 0.74,
            })
        tool_response = {"query": topic_hint, "passages": chunks}
        first_passage = chunks[0]["text"][:300].strip()
        answer = (
            f"Based on the retrieved passages, {first_passage}"
            + ("." if not first_passage.endswith(".") else "")
        )
    trace = _format_trace(question, "rag_retrieve", tool_args,
                          tool_response, answer)
    if not trace_quality_ok(trace):
        return None
    return build_record("rag_retrieve", str(seed_id), trace)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cve", default="data/raw/cve_full.jsonl")
    p.add_argument("--cwe", default="data/raw/cwe.jsonl")
    p.add_argument("--rag-seeds", default=None,
                   help="Comma-separated list of jsonl paths to seed "
                        "rag_retrieve traces from. Defaults to the four "
                        "seeds listed in TOOLS['rag_retrieve'].")
    p.add_argument("--out", default="data/processed/synth_tool_use.jsonl")
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
                    rec = synth_cve_trace(cve, idx)
                    if rec is None:
                        rejects["search_cve_nvd"] = rejects.get("search_cve_nvd", 0) + 1
                        continue
                    fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    counts["search_cve_nvd"] = counts.get("search_cve_nvd", 0) + 1
                print(f"  [search_cve_nvd] {counts.get('search_cve_nvd', 0)} accepted, "
                      f"{rejects.get('search_cve_nvd', 0)} rejected")

        if not args.skip_mitre:
            for rec in synth_mitre_traces():
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
                    rec = synth_cwe_trace(cwe, idx)
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
                rec = synth_rag_trace(seed, idx)
                if rec is None:
                    rejects["rag_retrieve"] = rejects.get("rag_retrieve", 0) + 1
                    continue
                fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                counts["rag_retrieve"] = counts.get("rag_retrieve", 0) + 1
            print(f"  [rag_retrieve] {counts.get('rag_retrieve', 0)} accepted, "
                  f"{rejects.get('rag_retrieve', 0)} rejected")

    n_total = sum(counts.values())
    print(f"\nWrote {n_total} traces to {out_path}")
    print(f"  by tool: {counts}")
    print(f"  rejects: {rejects}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
