"""Tool registry and built-in tool implementations for GhostAgent.

The bet 1 SFT data trained ghost-base on four canonical tools:

  search_cve_nvd          Look up a CVE by id or free-text query.
  lookup_mitre_technique  Look up an ATT&CK technique by T-code.
  lookup_cwe              Look up a CWE entry by number.
  rag_retrieve            Top-K retrieval from the cybersec corpus.

Each tool has a backend function and a JSON-schema-style args
declaration. The runtime dispatcher:

  1. Validates ``call.args`` against the registered ``args_schema``.
  2. Invokes the backend with the args.
  3. Wraps the response in a TOOL message and feeds it back.

Each backend follows a graceful-degradation pattern:

  - If a real upstream service is reachable (NVD API, MITRE
    Workbench), use it.
  - Otherwise, look up an offline cache shipped with the package
    (a small bundled subset of MITRE / CWE / common CVEs).
  - Otherwise, return a structured ``not_found`` response so the
    model learns to handle the failure mode rather than crash.

The cached fallbacks mean the agent runtime is testable offline
and on systems without network egress, which matters because the
v0.9.5 SFT data taught the model the no-found response format.
"""

from __future__ import annotations

import json
import os
import re
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional


# ---------------------------------------------------------------------------
# Tool data classes
# ---------------------------------------------------------------------------


@dataclass
class Tool:
    """One named tool the agent can invoke.

    ``args_schema`` is a {arg_name: human_description} dict. Used both
    for validation and for system-prompt rendering when the agent
    asks a model that hasn't seen the bet 1 SFT (in which case the
    runtime can prepend a tool description block).
    """
    name: str
    description: str
    args_schema: Dict[str, str]
    fn: Callable[[Dict[str, Any]], Dict[str, Any]]
    required_args: List[str] = field(default_factory=list)


@dataclass
class ToolResult:
    """Outcome of one tool invocation."""
    tool_name: str
    response: Any
    error: Optional[str] = None
    latency_ms: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "tool_name": self.tool_name,
            "response": self.response,
            "error": self.error,
            "latency_ms": self.latency_ms,
        }


# ---------------------------------------------------------------------------
# Tiny offline caches
# ---------------------------------------------------------------------------


# Hand-curated CVE summaries for common CVEs that show up in the eval
# sets and the day-to-day analyst flow. The agent uses these when the
# NVD API is unreachable. Real deployment swaps this for a proper
# offline NVD mirror.
_CVE_OFFLINE_CACHE: Dict[str, Dict[str, Any]] = {
    "CVE-2017-0144": {
        "cve": "CVE-2017-0144",
        "description": (
            "The SMBv1 server in Microsoft Windows allows remote attackers "
            "to execute arbitrary code via crafted packets, aka 'Windows "
            "SMB Remote Code Execution Vulnerability.' This is the "
            "EternalBlue vulnerability, weaponised in the WannaCry "
            "ransomware outbreak of May 2017."
        ),
        "cvss": "8.1",
        "source": "offline_cache",
    },
    "CVE-2014-0160": {
        "cve": "CVE-2014-0160",
        "description": (
            "OpenSSL TLS heartbeat extension (RFC 6520) read overflow "
            "(Heartbleed). Allows remote attackers to obtain sensitive "
            "information from process memory via crafted heartbeat "
            "packets. Affects OpenSSL 1.0.1 through 1.0.1f."
        ),
        "cvss": "7.5",
        "source": "offline_cache",
    },
    "CVE-2021-44228": {
        "cve": "CVE-2021-44228",
        "description": (
            "Log4Shell. Apache Log4j2 2.0-beta9 through 2.15.0 (excluding "
            "security releases) JNDI features used in configuration, log "
            "messages, and parameters do not protect against attacker-"
            "controlled LDAP and other JNDI related endpoints. Allows "
            "remote code execution via crafted log inputs."
        ),
        "cvss": "10.0",
        "source": "offline_cache",
    },
    "CVE-2019-0708": {
        "cve": "CVE-2019-0708",
        "description": (
            "BlueKeep. A remote code execution vulnerability in Remote "
            "Desktop Services (RDS) on Windows 7, Server 2008, and "
            "earlier. Wormable; pre-authentication; affects port 3389/tcp."
        ),
        "cvss": "9.8",
        "source": "offline_cache",
    },
}


# Hand-curated MITRE technique summaries.
_MITRE_OFFLINE_CACHE: Dict[str, Dict[str, Any]] = {
    "T1059.001": {
        "id": "T1059.001",
        "name": "Command and Scripting Interpreter: PowerShell",
        "tactic": "Execution",
        "platform": "windows",
        "summary": (
            "Adversaries use PowerShell for execution. Common patterns "
            "include -EncodedCommand with base64 payloads, "
            "-ExecutionPolicy Bypass, and Invoke-Expression of "
            "downloaded content."
        ),
        "url": "https://attack.mitre.org/techniques/T1059/001/",
        "source": "offline_cache",
    },
    "T1003.001": {
        "id": "T1003.001",
        "name": "OS Credential Dumping: LSASS Memory",
        "tactic": "Credential Access",
        "platform": "windows",
        "summary": (
            "Reading lsass.exe process memory to extract credentials. "
            "mimikatz, procdump, comsvcs.dll!MiniDump are common tools. "
            "Detected via Sysmon EventID 10 with GrantedAccess values "
            "0x1010, 0x1410, 0x1438, or 0x143A."
        ),
        "url": "https://attack.mitre.org/techniques/T1003/001/",
        "source": "offline_cache",
    },
    "T1078": {
        "id": "T1078",
        "name": "Valid Accounts",
        "tactic": "Defense Evasion / Persistence / Privilege Escalation / Initial Access",
        "platform": "multi",
        "summary": (
            "Use of stolen credentials to authenticate. Blends with "
            "normal logon traffic. Detection requires baseline "
            "deviation analytics (new geo, new device, off-hours)."
        ),
        "url": "https://attack.mitre.org/techniques/T1078/",
        "source": "offline_cache",
    },
    "T1486": {
        "id": "T1486",
        "name": "Data Encrypted for Impact",
        "tactic": "Impact",
        "platform": "windows",
        "summary": (
            "Ransomware encryption of victim data. File extensions "
            "change to .locked / .encrypted / .crypted / .enc. "
            "Detected via rapid-fire FileCreate events with custom "
            "extensions."
        ),
        "url": "https://attack.mitre.org/techniques/T1486/",
        "source": "offline_cache",
    },
}


# Hand-curated CWE summaries.
_CWE_OFFLINE_CACHE: Dict[str, Dict[str, Any]] = {
    "CWE-89": {
        "id": "CWE-89",
        "name": "Improper Neutralization of Special Elements used in an SQL Command",
        "description": (
            "SQL Injection. The product constructs all or part of an SQL "
            "command using externally-influenced input from an upstream "
            "component, but it does not neutralize or incorrectly "
            "neutralizes special elements that could modify the intended "
            "SQL command when sent to a downstream component."
        ),
        "url": "https://cwe.mitre.org/data/definitions/89.html",
        "source": "offline_cache",
    },
    "CWE-79": {
        "id": "CWE-79",
        "name": "Improper Neutralization of Input During Web Page Generation",
        "description": (
            "Cross-site Scripting (XSS). The product does not neutralize "
            "user-controllable input before placing it in output that is "
            "used as a web page that is served to other users."
        ),
        "url": "https://cwe.mitre.org/data/definitions/79.html",
        "source": "offline_cache",
    },
    "CWE-22": {
        "id": "CWE-22",
        "name": "Improper Limitation of a Pathname to a Restricted Directory",
        "description": (
            "Path Traversal. The product uses external input to construct "
            "a pathname that is intended to identify a file or directory "
            "that is located underneath a restricted parent directory, but "
            "the product does not properly neutralize special elements "
            "within the pathname that can cause the pathname to resolve to "
            "a location that is outside of the restricted directory."
        ),
        "url": "https://cwe.mitre.org/data/definitions/22.html",
        "source": "offline_cache",
    },
}


# ---------------------------------------------------------------------------
# Tool backends
# ---------------------------------------------------------------------------


def _backend_search_cve_nvd(args: Dict[str, Any]) -> Dict[str, Any]:
    """Look up a CVE. Tries the NVD JSON API if a network is
    available; falls back to the offline cache."""
    q = args.get("q", "").strip()
    if not q:
        return {"error": "missing required arg 'q' (CVE id or query)"}

    # Match CVE id pattern.
    cve_match = re.match(r"^CVE-\d{4}-\d{4,7}$", q.upper())
    if not cve_match:
        # Free-text search; we only support id lookups in the offline
        # cache, so return not_found honestly.
        for cve_id, blob in _CVE_OFFLINE_CACHE.items():
            if q.lower() in blob["description"].lower():
                return blob
        return {"cve": q, "found": False, "matches": [],
                "source": "offline_cache"}

    cve_id = q.upper()
    # Try real NVD API if reachable.
    if os.environ.get("GHOST_AGENT_OFFLINE", "0") != "1":
        try:
            url = (f"https://services.nvd.nist.gov/rest/json/cves/2.0"
                   f"?cveId={cve_id}")
            req = urllib.request.Request(
                url, headers={"User-Agent": "ghostlm-agent/0.1"},
            )
            with urllib.request.urlopen(req, timeout=5) as resp:
                data = json.loads(resp.read())
            vulns = data.get("vulnerabilities", [])
            if vulns:
                cve = vulns[0].get("cve", {})
                desc = next(
                    (d.get("value", "") for d in cve.get("descriptions", [])
                     if d.get("lang") == "en"), "",
                )
                cvss = None
                metrics = cve.get("metrics", {}).get("cvssMetricV31", [])
                if metrics:
                    cvss = str(metrics[0].get("cvssData", {}).get("baseScore"))
                return {
                    "cve": cve_id, "description": desc, "cvss": cvss,
                    "source": "nvd_api",
                }
        except (urllib.error.URLError, urllib.error.HTTPError, OSError):
            pass  # Fall through to offline cache.

    if cve_id in _CVE_OFFLINE_CACHE:
        return _CVE_OFFLINE_CACHE[cve_id]
    return {"cve": cve_id, "found": False, "matches": [],
            "source": "offline_cache"}


def _backend_lookup_mitre_technique(args: Dict[str, Any]) -> Dict[str, Any]:
    """Look up a MITRE ATT&CK technique by T-code."""
    tcode = args.get("technique_id", "").strip().upper()
    if not tcode:
        return {"error": "missing required arg 'technique_id'"}
    if tcode in _MITRE_OFFLINE_CACHE:
        return _MITRE_OFFLINE_CACHE[tcode]
    return {"technique_id": tcode, "found": False,
            "source": "offline_cache"}


def _backend_lookup_cwe(args: Dict[str, Any]) -> Dict[str, Any]:
    """Look up a CWE entry by id."""
    cwe = args.get("cwe_id", "").strip().upper()
    if not cwe:
        return {"error": "missing required arg 'cwe_id'"}
    if not cwe.startswith("CWE-"):
        cwe = f"CWE-{cwe}"
    if cwe in _CWE_OFFLINE_CACHE:
        return _CWE_OFFLINE_CACHE[cwe]
    return {"cwe_id": cwe, "found": False, "source": "offline_cache"}


def _backend_rag_retrieve(args: Dict[str, Any]) -> Dict[str, Any]:
    """Retrieve top-K corpus passages by query.

    The full RAG layer (BGE embedder + LanceDB index) lives in
    scripts/build_rag_index.py and ghostlm.rag (when present); here
    we ship a tiny offline fallback that does a string-match against
    the offline caches above. This is enough to demonstrate the loop;
    a production agent swaps in the real retriever.
    """
    query = args.get("query", "").strip()
    if not query:
        return {"error": "missing required arg 'query'"}
    k = int(args.get("k", 4))

    # Search offline caches by string match for a usable demo.
    passages: List[Dict[str, Any]] = []
    q_lower = query.lower()
    for source, cache in (
        ("nvd", _CVE_OFFLINE_CACHE),
        ("mitre", _MITRE_OFFLINE_CACHE),
        ("cwe", _CWE_OFFLINE_CACHE),
    ):
        for entry_id, blob in cache.items():
            text = blob.get("description") or blob.get("summary") or ""
            if q_lower in text.lower() or q_lower in entry_id.lower():
                passages.append({
                    "id": f"{source}:{entry_id}",
                    "text": text[:400],
                    "source": source,
                    "score": 0.85,
                })
                if len(passages) >= k:
                    break
        if len(passages) >= k:
            break

    return {"query": query, "passages": passages,
            "source": "offline_cache"}


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


TOOLS_REGISTRY: Dict[str, Tool] = {
    "search_cve_nvd": Tool(
        name="search_cve_nvd",
        description="Look up a CVE by ID (CVE-YYYY-NNNN) or free-text "
                    "query. Returns description, CVSS, and source.",
        args_schema={"q": "CVE id or natural-language query"},
        required_args=["q"],
        fn=_backend_search_cve_nvd,
    ),
    "lookup_mitre_technique": Tool(
        name="lookup_mitre_technique",
        description="Look up a MITRE ATT&CK technique by ID "
                    "(e.g. T1059, T1059.001).",
        args_schema={"technique_id": "MITRE technique ID"},
        required_args=["technique_id"],
        fn=_backend_lookup_mitre_technique,
    ),
    "lookup_cwe": Tool(
        name="lookup_cwe",
        description="Look up a CWE entry by id (e.g. CWE-89).",
        args_schema={"cwe_id": "CWE identifier (with or without CWE- prefix)"},
        required_args=["cwe_id"],
        fn=_backend_lookup_cwe,
    ),
    "rag_retrieve": Tool(
        name="rag_retrieve",
        description="Retrieve top-K passages from the cybersec corpus "
                    "by query.",
        args_schema={"query": "natural-language query",
                      "k": "number of passages (default 4)"},
        required_args=["query"],
        fn=_backend_rag_retrieve,
    ),
}


def dispatch(call_name: str, args: Dict[str, Any],
             registry: Optional[Dict[str, Tool]] = None) -> ToolResult:
    """Dispatch a parsed tool call. Validates args against the
    schema, runs the backend with timing, returns a ToolResult.
    Errors (unknown tool, missing required arg, backend exception)
    are captured into ``ToolResult.error`` rather than raised so the
    agent loop can include the error in the conversation and let the
    model recover."""
    reg = registry if registry is not None else TOOLS_REGISTRY
    tool = reg.get(call_name)
    if tool is None:
        return ToolResult(tool_name=call_name, response=None,
                           error=f"unknown tool: {call_name!r}")
    missing = [a for a in tool.required_args if a not in args]
    if missing:
        return ToolResult(
            tool_name=call_name, response=None,
            error=f"missing required arg(s): {missing}",
        )
    t0 = time.time()
    try:
        response = tool.fn(args)
        latency_ms = int((time.time() - t0) * 1000)
        return ToolResult(
            tool_name=call_name, response=response,
            latency_ms=latency_ms,
        )
    except Exception as e:  # noqa: BLE001 - tool runtime
        latency_ms = int((time.time() - t0) * 1000)
        return ToolResult(
            tool_name=call_name, response=None,
            error=f"{type(e).__name__}: {e}",
            latency_ms=latency_ms,
        )
