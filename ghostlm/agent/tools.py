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
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
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
# CISA Known Exploited Vulnerabilities (KEV)
#
# CISA publishes a public JSON feed of CVEs that are actively exploited
# in the wild, with required-action deadlines for federal agencies. The
# feed is at https://www.cisa.gov/sites/default/files/feeds/...
# Real upstream is reachable without an API key, so this tool follows the
# same try-real-then-cache pattern as search_cve_nvd.
# ---------------------------------------------------------------------------


_KEV_OFFLINE_CACHE: Dict[str, Dict[str, Any]] = {
    "CVE-2017-0144": {
        "cve": "CVE-2017-0144", "vendor": "Microsoft",
        "product": "SMBv1", "vulnerabilityName": "EternalBlue",
        "shortDescription": "SMBv1 RCE weaponised in WannaCry.",
        "requiredAction": "Apply MS17-010; disable SMBv1 where unused.",
        "knownRansomwareCampaignUse": "Known", "source": "offline_cache",
    },
    "CVE-2021-44228": {
        "cve": "CVE-2021-44228", "vendor": "Apache",
        "product": "Log4j2", "vulnerabilityName": "Log4Shell",
        "shortDescription": "JNDI lookup in log strings allows RCE.",
        "requiredAction": "Upgrade to Log4j 2.17.1+.",
        "knownRansomwareCampaignUse": "Known", "source": "offline_cache",
    },
    "CVE-2019-0708": {
        "cve": "CVE-2019-0708", "vendor": "Microsoft",
        "product": "Remote Desktop Services", "vulnerabilityName": "BlueKeep",
        "shortDescription": "Pre-auth RCE in RDP; wormable.",
        "requiredAction": "Apply patches; disable port 3389 where unused.",
        "knownRansomwareCampaignUse": "Unknown", "source": "offline_cache",
    },
    "CVE-2020-1472": {
        "cve": "CVE-2020-1472", "vendor": "Microsoft",
        "product": "Netlogon", "vulnerabilityName": "Zerologon",
        "shortDescription": "Crypto flaw in Netlogon allows DC takeover.",
        "requiredAction": "Apply August 2020 cumulative updates.",
        "knownRansomwareCampaignUse": "Known", "source": "offline_cache",
    },
    "CVE-2024-3094": {
        "cve": "CVE-2024-3094", "vendor": "xz / liblzma",
        "product": "xz-utils", "vulnerabilityName": "xz-utils backdoor",
        "shortDescription": "Supply-chain backdoor in liblzma 5.6.0/5.6.1.",
        "requiredAction": "Downgrade liblzma to 5.4.x; rotate SSH keys.",
        "knownRansomwareCampaignUse": "Unknown", "source": "offline_cache",
    },
    "CVE-2023-44487": {
        "cve": "CVE-2023-44487", "vendor": "Multiple",
        "product": "HTTP/2 implementations",
        "vulnerabilityName": "HTTP/2 Rapid Reset",
        "shortDescription": "DDoS via repeated stream resets.",
        "requiredAction": "Apply vendor patches; rate-limit RST_STREAM.",
        "knownRansomwareCampaignUse": "Unknown", "source": "offline_cache",
    },
}


def _backend_lookup_cisa_kev(args: Dict[str, Any]) -> Dict[str, Any]:
    """Check whether a CVE is on CISA's Known Exploited Vulnerabilities
    list. Returns the KEV entry with required-action data when present."""
    cve = args.get("cve_id", "").strip().upper()
    if not cve:
        return {"error": "missing required arg 'cve_id'"}

    if os.environ.get("GHOST_AGENT_OFFLINE", "0") != "1":
        try:
            url = ("https://www.cisa.gov/sites/default/files/feeds/"
                   "known_exploited_vulnerabilities.json")
            req = urllib.request.Request(
                url, headers={"User-Agent": "ghostlm-agent/0.1"},
            )
            with urllib.request.urlopen(req, timeout=8) as resp:
                data = json.loads(resp.read())
            for vuln in data.get("vulnerabilities", []):
                if vuln.get("cveID", "").upper() == cve:
                    return {
                        "cve": cve,
                        "vendor": vuln.get("vendorProject"),
                        "product": vuln.get("product"),
                        "vulnerabilityName": vuln.get("vulnerabilityName"),
                        "dateAdded": vuln.get("dateAdded"),
                        "shortDescription": vuln.get("shortDescription"),
                        "requiredAction": vuln.get("requiredAction"),
                        "dueDate": vuln.get("dueDate"),
                        "knownRansomwareCampaignUse":
                            vuln.get("knownRansomwareCampaignUse"),
                        "source": "cisa_kev_api",
                    }
            return {"cve": cve, "found": False, "source": "cisa_kev_api"}
        except (urllib.error.URLError, urllib.error.HTTPError, OSError):
            pass

    if cve in _KEV_OFFLINE_CACHE:
        return _KEV_OFFLINE_CACHE[cve]
    return {"cve": cve, "found": False, "source": "offline_cache"}


# ---------------------------------------------------------------------------
# GreyNoise IP reputation
#
# GreyNoise classifies internet-wide scanners and benign noise vs targeted
# attacks. Free community API exists at api.greynoise.io but rate-limited.
# We try the API if reachable and a key is set; otherwise fall back to a
# hand-curated cache of well-known scanners + RFC documentation IPs.
# ---------------------------------------------------------------------------


_GREYNOISE_OFFLINE_CACHE: Dict[str, Dict[str, Any]] = {
    "192.0.2.1": {
        "ip": "192.0.2.1", "classification": "documentation",
        "name": "RFC 5737 documentation prefix",
        "noise": False, "riot": False,
        "source": "offline_cache",
    },
    "198.51.100.1": {
        "ip": "198.51.100.1", "classification": "documentation",
        "name": "RFC 5737 documentation prefix",
        "noise": False, "riot": False,
        "source": "offline_cache",
    },
    "8.8.8.8": {
        "ip": "8.8.8.8", "classification": "benign",
        "name": "Google Public DNS", "noise": False, "riot": True,
        "source": "offline_cache",
    },
    "1.1.1.1": {
        "ip": "1.1.1.1", "classification": "benign",
        "name": "Cloudflare DNS", "noise": False, "riot": True,
        "source": "offline_cache",
    },
}


def _backend_lookup_greynoise(args: Dict[str, Any]) -> Dict[str, Any]:
    """Look up an IP's GreyNoise classification (noise / benign /
    malicious / unknown). Tries the community API if a key is set;
    otherwise returns the offline cache or unknown."""
    ip = args.get("ip", "").strip()
    if not ip:
        return {"error": "missing required arg 'ip'"}

    api_key = os.environ.get("GREYNOISE_API_KEY", "")
    if (api_key and
            os.environ.get("GHOST_AGENT_OFFLINE", "0") != "1"):
        try:
            url = f"https://api.greynoise.io/v3/community/{ip}"
            req = urllib.request.Request(
                url, headers={
                    "key": api_key,
                    "User-Agent": "ghostlm-agent/0.1",
                },
            )
            with urllib.request.urlopen(req, timeout=5) as resp:
                data = json.loads(resp.read())
            return {
                "ip": data.get("ip", ip),
                "classification": data.get("classification"),
                "name": data.get("name"),
                "noise": data.get("noise"),
                "riot": data.get("riot"),
                "last_seen": data.get("last_seen"),
                "source": "greynoise_api",
            }
        except (urllib.error.URLError, urllib.error.HTTPError, OSError):
            pass

    if ip in _GREYNOISE_OFFLINE_CACHE:
        return _GREYNOISE_OFFLINE_CACHE[ip]
    return {"ip": ip, "classification": "unknown",
            "found": False, "source": "offline_cache"}


# ---------------------------------------------------------------------------
# VirusTotal file hash lookup (offline-first)
#
# Real VT API requires an API key; we expose the integration shape but
# default to the offline cache. Hand-curated entries below cover the
# canonical demo / test cases (EICAR test file, well-known historical
# malware family hashes from public reports).
# ---------------------------------------------------------------------------


_VT_HASH_OFFLINE_CACHE: Dict[str, Dict[str, Any]] = {
    # EICAR antivirus test file (intentionally non-malicious).
    "275a021bbfb6489e54d471899f7db9d1663fc695ec2fe2a2c4538aabf651fd0f": {
        "sha256": "275a021bbfb6489e54d471899f7db9d1663fc695ec2fe2a2c4538aabf651fd0f",
        "type": "EICAR test file", "malicious": 0, "suspicious": 0,
        "harmless": 70, "undetected": 0,
        "popular_threat_classification": "test/eicar",
        "note": "Intentional AV test file, not real malware.",
        "source": "offline_cache",
    },
    # WannaCry SMB worm canonical hash (public, in many reports).
    "ed01ebfbc9eb5bbea545af4d01bf5f1071661840480439c6e5babe8e080e41aa": {
        "sha256": "ed01ebfbc9eb5bbea545af4d01bf5f1071661840480439c6e5babe8e080e41aa",
        "type": "WannaCry / WCry ransomware loader",
        "malicious": 64, "suspicious": 2,
        "popular_threat_classification": "trojan.wannacry/wcry",
        "first_seen": "2017-05-12",
        "source": "offline_cache",
    },
}


def _backend_lookup_virustotal_hash(args: Dict[str, Any]) -> Dict[str, Any]:
    """Look up a file hash's VirusTotal reputation. Real VT lookups
    require a paid API key; without one this returns the offline cache
    or unknown."""
    h = args.get("hash", "").strip().lower()
    if not h:
        return {"error": "missing required arg 'hash'"}
    if not re.match(r"^[a-f0-9]{32,64}$", h):
        return {"error": f"invalid hash format: {h!r} "
                          f"(expected MD5/SHA1/SHA256 hex)"}

    api_key = os.environ.get("VIRUSTOTAL_API_KEY", "")
    if (api_key and
            os.environ.get("GHOST_AGENT_OFFLINE", "0") != "1"):
        try:
            url = f"https://www.virustotal.com/api/v3/files/{h}"
            req = urllib.request.Request(
                url, headers={
                    "x-apikey": api_key,
                    "User-Agent": "ghostlm-agent/0.1",
                },
            )
            with urllib.request.urlopen(req, timeout=8) as resp:
                data = json.loads(resp.read())
            attr = data.get("data", {}).get("attributes", {})
            stats = attr.get("last_analysis_stats", {}) or {}
            return {
                "sha256": attr.get("sha256", h),
                "type": attr.get("type_description"),
                "malicious": stats.get("malicious"),
                "suspicious": stats.get("suspicious"),
                "harmless": stats.get("harmless"),
                "undetected": stats.get("undetected"),
                "popular_threat_classification":
                    (attr.get("popular_threat_classification", {}) or {})
                    .get("suggested_threat_label"),
                "source": "virustotal_api",
            }
        except (urllib.error.URLError, urllib.error.HTTPError, OSError):
            pass

    if h in _VT_HASH_OFFLINE_CACHE:
        return _VT_HASH_OFFLINE_CACHE[h]
    return {"sha256": h, "found": False, "source": "offline_cache"}


# ---------------------------------------------------------------------------
# Shodan service enumeration (offline-only)
#
# Real Shodan needs a paid API key; we ship a tiny offline cache that
# documents the response shape so the agent can be tested end-to-end
# without network. Production deployments wire SHODAN_API_KEY and
# replace _backend_lookup_shodan with the live path.
# ---------------------------------------------------------------------------


_SHODAN_OFFLINE_CACHE: Dict[str, Dict[str, Any]] = {
    "8.8.8.8": {
        "ip": "8.8.8.8",
        "hostnames": ["dns.google"],
        "country": "US",
        "org": "Google LLC",
        "ports": [53, 443],
        "services": [
            {"port": 53, "transport": "udp",
             "module": "dns", "product": "Google DNS"},
            {"port": 443, "transport": "tcp",
             "module": "https", "product": "GFE"},
        ],
        "source": "offline_cache",
    },
    "1.1.1.1": {
        "ip": "1.1.1.1",
        "hostnames": ["one.one.one.one"],
        "country": "AU", "org": "Cloudflare, Inc.",
        "ports": [53, 80, 443],
        "services": [
            {"port": 53, "transport": "udp", "module": "dns"},
            {"port": 443, "transport": "tcp",
             "module": "https", "product": "Cloudflare"},
        ],
        "source": "offline_cache",
    },
}


def _backend_lookup_shodan(args: Dict[str, Any]) -> Dict[str, Any]:
    """Look up an IP's Shodan service profile (open ports, banners).

    Real Shodan requires a paid SHODAN_API_KEY. Without one (and in
    offline mode) returns the offline cache or unknown. The response
    shape mirrors Shodan's JSON output so a model trained on this
    tool can read either source."""
    ip = args.get("ip", "").strip()
    if not ip:
        return {"error": "missing required arg 'ip'"}

    api_key = os.environ.get("SHODAN_API_KEY", "")
    if (api_key and
            os.environ.get("GHOST_AGENT_OFFLINE", "0") != "1"):
        try:
            # Shodan's REST API only accepts the key as a query param
            # (no header auth). Be aware the key can surface in proxy /
            # access logs on the egress path.
            url = f"https://api.shodan.io/shodan/host/{ip}?key={api_key}"
            req = urllib.request.Request(
                url, headers={"User-Agent": "ghostlm-agent/0.1"})
            with urllib.request.urlopen(req, timeout=8) as resp:
                data = json.loads(resp.read())
            return {
                "ip": data.get("ip_str", ip),
                "hostnames": data.get("hostnames", []),
                "country": data.get("country_code"),
                "org": data.get("org"),
                "ports": data.get("ports", []),
                "services": [
                    {"port": s.get("port"),
                     "transport": s.get("transport"),
                     "module": s.get("_shodan", {}).get("module"),
                     "product": s.get("product")}
                    for s in data.get("data", []) or []
                ],
                "source": "shodan_api",
            }
        except (urllib.error.URLError, urllib.error.HTTPError, OSError):
            pass

    if ip in _SHODAN_OFFLINE_CACHE:
        return _SHODAN_OFFLINE_CACHE[ip]
    return {"ip": ip, "found": False, "source": "offline_cache"}


# ---------------------------------------------------------------------------
# AlienVault OTX IOC pulse search (offline-only)
#
# OTX requires a free API key for live lookups. We ship a small offline
# cache of public APT-group pulse summaries so the tool's response
# shape is exercised in tests + demos. Production wires OTX_API_KEY.
# ---------------------------------------------------------------------------


_OTX_OFFLINE_CACHE: Dict[str, Dict[str, Any]] = {
    "lazarus": {
        "indicator": "lazarus",
        "pulse_count": 50,
        "summaries": [
            {"name": "Lazarus Group: Operation North Star",
             "tags": ["apt", "north-korea", "financial"],
             "tlp": "white"},
            {"name": "Lazarus Group: 3CX Supply Chain",
             "tags": ["supply-chain", "apt38"],
             "tlp": "white"},
        ],
        "source": "offline_cache",
    },
    "apt28": {
        "indicator": "apt28",
        "pulse_count": 35,
        "summaries": [
            {"name": "APT28 / Fancy Bear: GRU 26165 activity",
             "tags": ["apt", "russia", "espionage"],
             "tlp": "white"},
        ],
        "source": "offline_cache",
    },
}


def _backend_lookup_alienvault_otx(args: Dict[str, Any]) -> Dict[str, Any]:
    """Search AlienVault OTX for IOC pulses matching a given indicator
    (IP, domain, hash, APT name).

    Real OTX requires OTX_API_KEY for live lookups. Without one returns
    the offline cache or not_found."""
    indicator = args.get("indicator", "").strip().lower()
    if not indicator:
        return {"error": "missing required arg 'indicator'"}

    api_key = os.environ.get("OTX_API_KEY", "")
    if (api_key and
            os.environ.get("GHOST_AGENT_OFFLINE", "0") != "1"):
        try:
            url = (f"https://otx.alienvault.com/api/v1/search/pulses"
                   f"?q={urllib.parse.quote(indicator)}")
            req = urllib.request.Request(
                url, headers={
                    "X-OTX-API-KEY": api_key,
                    "User-Agent": "ghostlm-agent/0.1",
                },
            )
            with urllib.request.urlopen(req, timeout=8) as resp:
                data = json.loads(resp.read())
            results = data.get("results", []) or []
            return {
                "indicator": indicator,
                "pulse_count": data.get("count", len(results)),
                "summaries": [
                    {"name": p.get("name"),
                     "tags": p.get("tags", []),
                     "tlp": p.get("TLP")}
                    for p in results[:5]
                ],
                "source": "otx_api",
            }
        except (urllib.error.URLError, urllib.error.HTTPError, OSError):
            pass

    if indicator in _OTX_OFFLINE_CACHE:
        return _OTX_OFFLINE_CACHE[indicator]
    return {"indicator": indicator, "found": False, "source": "offline_cache"}


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
    "lookup_cisa_kev": Tool(
        name="lookup_cisa_kev",
        description="Check whether a CVE is on CISA's Known Exploited "
                    "Vulnerabilities list. Returns required-action and "
                    "ransomware-use status when present.",
        args_schema={"cve_id": "CVE identifier (e.g. CVE-2021-44228)"},
        required_args=["cve_id"],
        fn=_backend_lookup_cisa_kev,
    ),
    "lookup_greynoise": Tool(
        name="lookup_greynoise",
        description="Look up an IP address's GreyNoise classification "
                    "(noise / benign / malicious / unknown). "
                    "Distinguishes internet-wide scanners from "
                    "targeted threats.",
        args_schema={"ip": "IPv4 or IPv6 address"},
        required_args=["ip"],
        fn=_backend_lookup_greynoise,
    ),
    "lookup_virustotal_hash": Tool(
        name="lookup_virustotal_hash",
        description="Look up a file hash's VirusTotal reputation "
                    "(MD5/SHA1/SHA256). Returns malicious/suspicious/"
                    "harmless detection counts and threat label.",
        args_schema={"hash": "file hash in hex (32, 40, or 64 chars)"},
        required_args=["hash"],
        fn=_backend_lookup_virustotal_hash,
    ),
    "lookup_shodan": Tool(
        name="lookup_shodan",
        description="Look up an IP address's Shodan service profile "
                    "(open ports, hostnames, organization, banners).",
        args_schema={"ip": "IPv4 address"},
        required_args=["ip"],
        fn=_backend_lookup_shodan,
    ),
    "lookup_alienvault_otx": Tool(
        name="lookup_alienvault_otx",
        description="Search AlienVault OTX for IOC pulses matching an "
                    "indicator (IP, domain, hash, or APT name).",
        args_schema={"indicator": "IOC string or APT group name"},
        required_args=["indicator"],
        fn=_backend_lookup_alienvault_otx,
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
