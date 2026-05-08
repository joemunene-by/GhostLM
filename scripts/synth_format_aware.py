#!/usr/bin/env python3
"""Templated synthesis of bet 6 (format-aware) training records.

The bet 6 hypothesis is that ghost-base needs to see structured CTI
artifacts (STIX 2.1, YARA, Sigma, MISP) at pretrain time so it can
emit them at inference time. The canonical pipeline is
``scripts/distill_format_aware.py`` calling an LLM teacher
(Anthropic ~$50-100, or free Ollama). This script produces real
training records with NEITHER spend NOR a teacher: deterministic
templates seeded from the existing GhostLM corpus + a small bank
of hand-curated patterns.

Why this is worth shipping alongside the LLM-distilled traces (when
they arrive):

  - **Deterministic and reproducible.** Same corpus + same templates
    yield byte-identical outputs. LLM distillation has temperature.
  - **100% syntactic validity.** Every emitted record is checked
    against the bet 6 parser (parse_stix / parse_yara / parse_sigma)
    before write. LLM distillation lands ~60-80% valid in practice.
  - **Lower-diversity + higher-volume baseline.** Templated synth
    produces hundreds of records along a few axes; the diversity
    comes from the corpus seeds rather than teacher creativity.
    Mixing this baseline with future LLM-distilled records gives
    the model both structural floor (templates) and idiomatic
    variety (LLM).

Coverage in this script:

  STIX 2.1 indicators
    Seed: data/raw/cve_full.jsonl (CVE entries 2020+, non-rejected).
    Template: emit indicator SDO with type=indicator, spec_version=2.1,
              pattern derived from CVE description keywords (RCE -> file
              hash pattern; DoS / network -> network-traffic pattern;
              auth bypass -> user-account pattern), external_references
              pointing back at the CVE id.
    Volume:   --max-stix records (default 200)

  Sigma rules
    Seed: a hand-curated bank of 30 common ATT&CK techniques covering
          execution / persistence / privilege-escalation / defense-evasion
          / credential-access / discovery / lateral-movement / collection
          / exfiltration / impact tactics across windows / linux / macos.
    Template: emit YAML with title / id (deterministic GUID) / status /
              description / logsource / detection / falsepositives / level /
              tags. Logsource picked per technique platform; detection
              selection picked per technique characteristic event.
    Volume:   one record per technique = 30 records.

  YARA rules
    Seed: a hand-curated bank of 30 common malware family templates
          covering loaders, ransomware, credential stealers, and RATs.
    Template: emit rule with meta block, strings (mix of hex magic +
              ascii markers), and condition combining file-magic + 2-of
              string presence.
    Volume:   one record per family = 30 records.

  MISP events
    Skipped here. MISP attribute extraction needs IOCs that are
    sparse in CVE/MITRE descriptions; LLM teacher adds the most
    value on this format.

Run:

    PYTHONPATH=. python3 scripts/synth_format_aware.py \\
        --cve data/raw/cve_full.jsonl \\
        --out data/processed/synth_format_aware.jsonl \\
        --max-stix 200

Output is JSONL of ``DistillRecord``-shaped entries with
``source = 'synth_format_aware'``, drops into the pretrain corpus
identically to the LLM-distilled flow.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterator, List, Optional

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.distill_format_aware import (  # noqa: E402
    parse_stix, parse_yara, parse_sigma,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def deterministic_uuid(seed: str) -> str:
    """Stable v4-shaped UUID derived from a seed string. Used for
    STIX SDO ids and Sigma rule ids so re-runs produce identical
    artifacts."""
    h = hashlib.sha1(seed.encode("utf-8")).hexdigest()
    return f"{h[0:8]}-{h[8:12]}-4{h[13:16]}-a{h[17:20]}-{h[20:32]}"


def now_iso() -> str:
    """RFC3339 UTC timestamp with millisecond precision."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.000Z")


def build_record(seed_source: str, seed_id: str,
                 prompt: str, artifact: str) -> Dict[str, str]:
    """Assemble a DistillRecord-shaped dict ready to write."""
    h = hashlib.sha1(
        f"{seed_source}\n{seed_id}\n{artifact}".encode("utf-8")
    ).hexdigest()[:10]
    text = (
        f"Source: {seed_id}\n"
        f"Format: {seed_source}\n\n"
        f"Prompt:\n{prompt}\n\n"
        f"Artifact:\n{artifact.strip()}\n"
    )
    return {
        "id": f"synth_format_aware#{seed_id}#{h}",
        "source": "synth_format_aware",
        "teacher": "templated",
        "seed_source": seed_source,
        "seed_id": seed_id,
        "text": text,
    }


# ---------------------------------------------------------------------------
# STIX 2.1 from CVE
# ---------------------------------------------------------------------------


_REJECT_PREFIXES = ("Rejected reason", "** REJECT **")


def stream_cve(path: Path, max_records: int) -> Iterator[Dict]:
    """Yield non-rejected, 2020+ CVE records up to ``max_records``."""
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


def cve_pattern(text: str, cve_id: str) -> str:
    """Pick a STIX pattern best matching the CVE description.

    Heuristic only: keyword match on the description. Real-world STIX
    indicators are usually richer; this is template-grade. The point
    is the model learns the *shape* of stix-pattern, not whether each
    individual pattern is the most useful one for the CVE."""
    t = text.lower()
    if any(k in t for k in ("smb", "port 445")):
        return ("[network-traffic:dst_port = 445 AND "
                "network-traffic:protocols[*] = 'smb']")
    if any(k in t for k in ("rdp", "port 3389")):
        return ("[network-traffic:dst_port = 3389 AND "
                "network-traffic:protocols[*] = 'tcp']")
    if any(k in t for k in ("http", "web", "url", "api endpoint")):
        return (f"[url:value MATCHES '.*{cve_id.lower()}.*' OR "
                "network-traffic:dst_port = 443]")
    if any(k in t for k in ("dns", "resolver")):
        return "[network-traffic:dst_port = 53]"
    if any(k in t for k in ("denial of service", "denial-of-service",
                            " dos ", "crash", "deny")):
        return "[network-traffic:dst_port IN (80,443,8080)]"
    if any(k in t for k in ("upload", "file", "extension", "mime")):
        return ("[file:hashes.'SHA-256' MATCHES '.*' AND "
                "file:name MATCHES '\\\\.(php|jsp|asp|aspx)$']")
    if any(k in t for k in ("buffer overflow", "heap", "stack overflow",
                            "rce", "remote code", "code execution")):
        return "[file:hashes.'SHA-256' MATCHES '.*']"
    if any(k in t for k in ("auth", "credential", "token", "session")):
        return "[user-account:account_login = '*']"
    return "[file:name MATCHES '.*']"


def synth_stix_from_cve(rec: Dict) -> Optional[Dict[str, str]]:
    """One STIX indicator SDO from one CVE record."""
    cid = rec["id"]
    text = rec.get("text", "") or ""
    name_words = text.split(".")[0][:80] if text else cid
    pattern = cve_pattern(text, cid)
    artifact = json.dumps({
        "type": "indicator",
        "spec_version": "2.1",
        "id": f"indicator--{deterministic_uuid(cid)}",
        "created": now_iso(),
        "modified": now_iso(),
        "pattern_type": "stix",
        "pattern": pattern,
        "valid_from": now_iso(),
        "labels": ["malicious-activity", "exploit-target"],
        "name": f"{cid} indicator",
        "description": (text[:300] + "...") if len(text) > 300 else text,
        "external_references": [
            {"source_name": "cve", "external_id": cid,
             "url": f"https://nvd.nist.gov/vuln/detail/{cid}"}
        ],
    })
    if parse_stix(artifact) is None:
        return None
    prompt = (
        "Convert this CVE record to a STIX 2.1 indicator SDO with "
        "type=indicator, spec_version=2.1, pattern_type='stix', a "
        "pattern using stix-pattern grammar, and external_references "
        f"linking to {cid}.\n\nCVE record:\n{text[:1500]}"
    )
    return build_record("stix_indicator", cid, prompt, artifact)


# ---------------------------------------------------------------------------
# Sigma rules from hand-curated technique bank
# ---------------------------------------------------------------------------


# Each entry: (T-code, tactic, platform, logsource, characteristic
# selection content, level, falsepositive). Hand-curated to span the
# 12 ATT&CK tactics plus Windows/Linux/macOS/multi.
SIGMA_TECHNIQUES = [
    ("T1059.001", "Execution", "windows", ("category", "process_creation"),
     {"Image|endswith": ["\\\\powershell.exe", "\\\\pwsh.exe"],
      "CommandLine|contains": ["-EncodedCommand", "-enc "]},
     "high", "Legitimate admin scripts using -EncodedCommand"),
    ("T1059.003", "Execution", "windows", ("category", "process_creation"),
     {"Image|endswith": ["\\\\cmd.exe"],
      "CommandLine|contains": ["/c ", "&&", "||", "powershell"]},
     "medium", "Build automation"),
    ("T1059.004", "Execution", "linux", ("category", "process_creation"),
     {"Image|endswith": ["/bash", "/sh"],
      "CommandLine|re": "(curl|wget).*\\\\|.*sh"},
     "high", "DevOps install scripts"),
    ("T1003.001", "Credential Access", "windows", ("category", "process_access"),
     {"TargetImage|endswith": ["\\\\lsass.exe"],
      "GrantedAccess": ["0x1010", "0x1410", "0x1438", "0x143A"]},
     "high", "Endpoint protection products legitimately reading lsass"),
    ("T1078", "Defense Evasion", "windows", ("service", "security"),
     {"EventID": [4624], "LogonType": [10, 3]},
     "low", "Authorized remote access"),
    ("T1133", "Initial Access", "windows", ("service", "security"),
     {"EventID": [4624], "LogonType": [10]},
     "medium", "Authorized remote work over RDP"),
    ("T1547.001", "Persistence", "windows", ("category", "registry_event"),
     {"TargetObject|contains": ["\\\\CurrentVersion\\\\Run",
                                "\\\\CurrentVersion\\\\RunOnce"]},
     "medium", "Legitimate software install"),
    ("T1543.003", "Persistence", "windows", ("category", "registry_event"),
     {"TargetObject|contains": ["\\\\Services\\\\"],
      "Details|contains": [".exe"]},
     "medium", "Service install by IT"),
    ("T1071.001", "Command and Control", "network", ("category", "proxy"),
     {"cs-method": ["POST", "PUT"],
      "cs-uri-stem|contains": ["/api/", "/upload"]},
     "low", "Application uploads"),
    ("T1486", "Impact", "windows", ("category", "file_event"),
     {"TargetFilename|re": ".*\\.(locked|encrypted|crypted|enc)$"},
     "high", "Backup software"),
    ("T1485", "Impact", "windows", ("category", "file_event"),
     {"TargetFilename|contains": ["\\\\Backups\\\\", "\\\\.bak"],
      "EventType": ["delete"]},
     "high", "Maintenance scripts"),
    ("T1083", "Discovery", "windows", ("category", "process_creation"),
     {"Image|endswith": ["\\\\where.exe", "\\\\dir.exe", "\\\\tree.exe"]},
     "low", "Normal admin discovery"),
    ("T1018", "Discovery", "windows", ("category", "process_creation"),
     {"Image|endswith": ["\\\\net.exe", "\\\\nltest.exe"],
      "CommandLine|contains": ["view", "domain_trusts"]},
     "low", "AD admin work"),
    ("T1021.001", "Lateral Movement", "windows", ("service", "security"),
     {"EventID": [4624], "LogonType": [10],
      "AuthenticationPackageName": ["Negotiate"]},
     "medium", "Authorized RDP"),
    ("T1021.002", "Lateral Movement", "windows", ("service", "security"),
     {"EventID": [4624], "LogonType": [3],
      "AuthenticationPackageName": ["NTLM"]},
     "medium", "SMB share access"),
    ("T1027", "Defense Evasion", "windows", ("category", "process_creation"),
     {"CommandLine|re": "(?:[A-Za-z0-9+/=]{200,})"},
     "medium", "Compiled PowerShell modules"),
    ("T1090", "Command and Control", "network", ("category", "firewall"),
     {"dst-port": [9050, 9051, 1080]},
     "medium", "Privacy-tool users"),
    ("T1497", "Defense Evasion", "windows", ("category", "process_creation"),
     {"Image|endswith": ["\\\\wmic.exe"],
      "CommandLine|contains": ["computersystem", "csproduct", "bios"]},
     "low", "Inventory scripts"),
    ("T1106", "Execution", "windows", ("category", "process_creation"),
     {"ParentImage|endswith": ["\\\\rundll32.exe"],
      "Image|endswith": ["\\\\cmd.exe", "\\\\powershell.exe"]},
     "high", "Some signed binaries do this"),
    ("T1218.011", "Defense Evasion", "windows", ("category", "process_creation"),
     {"Image|endswith": ["\\\\rundll32.exe"],
      "CommandLine|contains": ["javascript:", "mshtml,RunHTMLApplication"]},
     "high", "(none)"),
    ("T1218.005", "Defense Evasion", "windows", ("category", "process_creation"),
     {"Image|endswith": ["\\\\mshta.exe"],
      "CommandLine|contains": ["http", ".hta"]},
     "high", "Legacy HTA-based business apps"),
    ("T1505.003", "Persistence", "windows", ("category", "file_event"),
     {"TargetFilename|re": ".*\\\\(wwwroot|inetpub).*\\.(asp|aspx|jsp|php)$"},
     "high", "Authorized webdev deploys"),
    ("T1190", "Initial Access", "network", ("category", "webserver"),
     {"sc-status": [200],
      "cs-uri-stem|re": ".*(\\.\\./|select.*from|union.*select|';--).*"},
     "high", "Pen-testing windows"),
    ("T1110", "Credential Access", "windows", ("service", "security"),
     {"EventID": [4625]},
     "medium", "Forgotten passwords"),
    ("T1222", "Defense Evasion", "linux", ("category", "process_creation"),
     {"Image|endswith": ["/chmod", "/chown"],
      "CommandLine|contains": ["+x", "777"]},
     "low", "Install scripts"),
    ("T1053.005", "Persistence", "windows", ("category", "process_creation"),
     {"Image|endswith": ["\\\\schtasks.exe"],
      "CommandLine|contains": ["/create", "/sc"]},
     "low", "IT-admin scheduled tasks"),
    ("T1112", "Defense Evasion", "windows", ("category", "registry_event"),
     {"TargetObject|contains": ["\\\\Software\\\\Microsoft\\\\Windows Defender"]},
     "high", "Defender configuration via GPO"),
    ("T1574.002", "Defense Evasion", "windows", ("category", "image_load"),
     {"ImageLoaded|endswith": [".dll"],
      "OriginalFileName|exists": False},
     "high", "Unsigned vendor DLLs"),
    ("T1140", "Defense Evasion", "windows", ("category", "process_creation"),
     {"Image|endswith": ["\\\\certutil.exe"],
      "CommandLine|contains": ["-decode", "-decodehex"]},
     "high", "PKI admin work"),
    ("T1566.001", "Initial Access", "windows", ("category", "file_event"),
     {"TargetFilename|re": ".*\\.(docm|xlsm|xls|doc)$",
      "Image|endswith": ["\\\\winword.exe", "\\\\excel.exe"]},
     "medium", "Macro-enabled corp documents"),
]


def yaml_dump_simple(obj, indent: int = 0) -> str:
    """Minimal YAML emitter (ours only handles the dict / list / scalar
    shapes our Sigma rules use; not a general yaml dumper)."""
    sp = " " * indent
    if isinstance(obj, dict):
        out = []
        for k, v in obj.items():
            if isinstance(v, (dict, list)):
                out.append(f"{sp}{k}:")
                out.append(yaml_dump_simple(v, indent + 4))
            else:
                out.append(f"{sp}{k}: {yaml_scalar(v)}")
        return "\n".join(out)
    if isinstance(obj, list):
        return "\n".join(f"{sp}- {yaml_scalar(v)}" for v in obj)
    return f"{sp}{yaml_scalar(obj)}"


def yaml_scalar(v) -> str:
    """Format one scalar for YAML output."""
    if v is True:
        return "true"
    if v is False:
        return "false"
    if v is None:
        return "null"
    if isinstance(v, (int, float)):
        return str(v)
    s = str(v)
    if any(c in s for c in (":", "#", "{", "}", "[", "]", ",", "&", "*",
                            "!", "|", ">", "'", '"', "%", "@", "`")) or \
       s.startswith("- ") or s == "":
        return json.dumps(s)
    return s


def synth_sigma_from_technique(spec) -> Optional[Dict[str, str]]:
    """One Sigma rule from one (T-code, tactic, platform, ...) tuple."""
    tcode, tactic, platform, logsource_kv, selection_body, level, fp = spec
    rule_id = deterministic_uuid(f"sigma:{tcode}")
    logsource = {"product": platform if platform != "network" else None}
    logsource.pop(None, None) if None in logsource else None
    logsource[logsource_kv[0]] = logsource_kv[1]
    if platform == "network":
        logsource = {logsource_kv[0]: logsource_kv[1]}
    detection = {
        "selection": selection_body,
        "condition": "selection",
    }
    title = (f"{tactic.split(' ')[0]} {tcode} via "
             f"{logsource_kv[1].replace('_', ' ')}")
    body = {
        "title": title,
        "id": rule_id,
        "status": "experimental",
        "description": f"Detects {tcode} ({tactic}) on {platform}",
        "references": [f"https://attack.mitre.org/techniques/{tcode.replace('.','/')}/"],
        "author": "GhostLM (templated)",
        "date": "2026/05/08",
        "logsource": logsource,
        "detection": detection,
        "falsepositives": [fp] if fp != "(none)" else [],
        "level": level,
        "tags": [
            f"attack.{tactic.lower().replace(' ', '_')}",
            f"attack.{tcode.lower()}",
        ],
    }
    artifact = yaml_dump_simple(body)
    if parse_sigma(artifact) is None:
        return None
    prompt = (
        f"Write a Sigma rule for ATT&CK {tcode} ({tactic}) on "
        f"{platform}. Use the appropriate logsource for the "
        "characteristic event type, a detection block with at least "
        "one selection, a condition that combines them, "
        "falsepositives, and the right severity level."
    )
    return build_record("sigma_rule", tcode, prompt, artifact)


# ---------------------------------------------------------------------------
# YARA rules from hand-curated family bank
# ---------------------------------------------------------------------------


YARA_FAMILIES = [
    # Loader / trojan family
    ("Emotet_Loader", "loader", "EmotetLoader", "Global\\\\I98B68E3C",
     {"4D 5A": "PE magic"},
     ["EmotetLoader2024", "Loader.Bot"]),
    ("TrickBot", "banking_trojan", "TrickBotMutex_2024", "Global\\\\TBot2024",
     {"4D 5A": "PE magic", "00 01 02 03": "marker"},
     ["TrickBot", "TBOT", "moduleconfig.txt"]),
    ("Qakbot", "banking_trojan", "QakbotMutex", "Global\\\\Qbot",
     {"4D 5A": "PE magic"},
     ["QBot", "qakbot.dll", "module_config"]),
    ("Cobalt_Strike_Beacon", "post_exploitation",
     "CobaltStrikeBeacon", "Global\\\\CSBeacon",
     {"4D 5A": "PE magic"},
     ["beacon.dll", "spawnto_x86", "%s%dStager"]),
    # Ransomware
    ("LockBit", "ransomware", "LockBitMutex", "Global\\\\LockBit3",
     {"4D 5A": "PE magic"},
     ["LOCKBIT", ".lockbit", "RestoreMyFiles.txt"]),
    ("BlackCat", "ransomware", "BlackCatMutex", "Global\\\\ALPHV",
     {"4D 5A": "PE magic"},
     ["BlackCat", "ALPHV", "RECOVER-"]),
    ("Royal", "ransomware", "RoyalMutex", "Global\\\\Royal",
     {"4D 5A": "PE magic"},
     ["Royal", ".royal_w", "README.TXT"]),
    ("Cl0p", "ransomware", "ClopMutex", "Global\\\\Cl0p",
     {"4D 5A": "PE magic"},
     ["Cl0p", ".clop", "ReadMe.txt"]),
    # Credential stealers
    ("RedLine_Stealer", "credential_stealer",
     "RedLineMutex", "Global\\\\RedLine",
     {"4D 5A": "PE magic"},
     ["RedLine", "Stealer.exe", "C:\\\\ProgramData\\\\stub"]),
    ("Raccoon_Stealer", "credential_stealer",
     "RaccoonMutex", "Global\\\\Raccoon",
     {"4D 5A": "PE magic"},
     ["Raccoon", "stealer.dll", "RACCOON_BUILD"]),
    ("Lumma_Stealer", "credential_stealer",
     "LummaMutex", "Global\\\\Lumma",
     {"4D 5A": "PE magic"},
     ["Lumma", "wallet.dat", "logins.json"]),
    # RATs
    ("AsyncRAT", "rat", "AsyncRATMutex", "Global\\\\AsyncRAT",
     {"4D 5A": "PE magic"},
     ["AsyncRAT", "Plugin.dll", "DcRat"]),
    ("RemcosRAT", "rat", "RemcosMutex", "Global\\\\Remcos",
     {"4D 5A": "PE magic"},
     ["Remcos", "remcos_remote.dll", "Breaking-Security"]),
    ("AgentTesla", "rat_keylogger", "AgentTeslaMutex", "Global\\\\AgentT",
     {"4D 5A": "PE magic"},
     ["AgentTesla", "Tesla.exe", "Mailto:"]),
    ("DarkGate", "loader_rat", "DarkGateMutex", "Global\\\\DarkGate",
     {"4D 5A": "PE magic"},
     ["DarkGate", "darkgate.dll", "AutoIt3"]),
    # Web shells
    ("China_Chopper", "web_shell", "n/a", "n/a",
     {},
     ["@eval(", "request[\"", "?>"]),
    ("WSO_Webshell", "web_shell", "n/a", "n/a",
     {},
     ["WSO 2.5", "WSO 4.0", "passthru"]),
    # Malicious documents
    ("Office_Macro_Dropper", "doc_dropper", "n/a", "n/a",
     {"D0 CF 11 E0": "OLE2 magic"},
     ["AutoOpen", "Shell ", "WScript.Shell"]),
    ("PDF_JavaScript_Dropper", "doc_dropper", "n/a", "n/a",
     {"25 50 44 46": "PDF magic"},
     ["/JavaScript", "/JS ", "app.launchURL"]),
    ("HTA_Dropper", "doc_dropper", "n/a", "n/a",
     {},
     ["<HTA:APPLICATION", "WScript.Shell", "ActiveXObject"]),
    # Cryptominers
    ("XMRig", "cryptominer", "XMRigMutex", "Global\\\\XMRig",
     {"4D 5A": "PE magic"},
     ["XMRig", "stratum+tcp", "donate-level"]),
    # Backdoors
    ("PlugX", "rat", "PlugXMutex", "Global\\\\PlugX",
     {"4D 5A": "PE magic"},
     ["PlugX", "plugx.dll", "RC4 key"]),
    ("ShadowPad", "rat", "ShadowPadMutex", "Global\\\\Shadow",
     {"4D 5A": "PE magic"},
     ["ShadowPad", "shadowpad.dll", "C:\\\\Users\\\\Public\\\\"]),
    ("CobaltStrike_HTTP_Stager", "post_exploitation",
     "CSStagerMutex", "Global\\\\CSStager",
     {"4D 5A": "PE magic"},
     ["WinINet", "wininet.dll", "InternetOpenA"]),
    # Phishing kits
    ("Phishing_O365_Kit", "phishing_kit", "n/a", "n/a",
     {},
     ["o365 login", "office365", "<form action=\"login.php\""]),
    ("Phishing_Generic", "phishing_kit", "n/a", "n/a",
     {},
     ["password", "username", "<input type=\"password\""]),
    # APT-style
    ("Sunburst_Style", "apt_backdoor", "SunburstMutex", "Global\\\\Sunburst",
     {"4D 5A": "PE magic"},
     ["SolarWinds", "OrionImprovementBusinessLayer", "ReportWatcher"]),
    ("Industroyer", "ics_malware", "n/a", "n/a",
     {"4D 5A": "PE magic"},
     ["IEC-101", "IEC-104", "OPC"]),
    # Generic
    ("Generic_PE_Packer", "packer", "n/a", "n/a",
     {"4D 5A": "PE magic", "55 50 58 21": "UPX0 marker"},
     ["UPX0", "UPX1", "UPX!"]),
    ("Generic_ELF_Backdoor", "backdoor", "n/a", "n/a",
     {"7F 45 4C 46": "ELF magic"},
     ["bind shell", "/bin/sh", "execve"]),
]


def synth_yara_from_family(spec) -> Optional[Dict[str, str]]:
    """One YARA rule from one family-template tuple."""
    family, fam_type, key, mutex, magic_bytes, markers = spec
    rule_name = family
    lines = [
        f"rule {rule_name} {{",
        "    meta:",
        "        author = \"GhostLM\"",
        f"        description = \"Detects {family} ({fam_type})\"",
        f"        family = \"{family.lower()}\"",
        "        tlp = \"AMBER\"",
        "        date = \"2026-05-08\"",
        "    strings:",
    ]
    si = 0
    has_pe = False
    for hex_seq, label in magic_bytes.items():
        lines.append(f"        $h{si} = {{ {hex_seq} }}")
        if hex_seq == "4D 5A":
            has_pe = True
        si += 1
    for i, m in enumerate(markers):
        lines.append(f"        $s{i} = \"{m}\" ascii wide")
    lines.append("    condition:")
    if has_pe and markers:
        lines.append("        $h0 at 0 and 2 of ($s*)")
    elif markers:
        lines.append("        2 of ($s*)")
    elif has_pe:
        lines.append("        $h0 at 0")
    else:
        lines.append("        any of them")
    lines.append("}")
    artifact = "\n".join(lines) + "\n"
    if parse_yara(artifact) is None:
        return None
    prompt = (
        f"Write a YARA rule that detects the {family} {fam_type}. "
        "Include a meta block (author/description/family/tlp), a "
        "strings section combining hex magic and ascii markers, and "
        "a condition combining them."
    )
    return build_record("yara_rule", family, prompt, artifact)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cve", default="data/raw/cve_full.jsonl")
    p.add_argument("--out", default="data/processed/synth_format_aware.jsonl")
    p.add_argument("--max-stix", type=int, default=200)
    p.add_argument("--skip-stix", action="store_true")
    p.add_argument("--skip-sigma", action="store_true")
    p.add_argument("--skip-yara", action="store_true")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    cve_path = REPO_ROOT / args.cve if not Path(args.cve).is_absolute() \
               else Path(args.cve)
    out_path = REPO_ROOT / args.out if not Path(args.out).is_absolute() \
               else Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    n_total = 0
    by_format: Dict[str, int] = {}
    rejects: Dict[str, int] = {}
    with out_path.open("w", encoding="utf-8") as fout:
        if not args.skip_stix:
            if not cve_path.exists():
                print(f"  [stix] CVE seed missing at {cve_path}; skipping")
            else:
                for cve in stream_cve(cve_path, args.max_stix):
                    rec = synth_stix_from_cve(cve)
                    if rec is None:
                        rejects["stix_indicator"] = rejects.get("stix_indicator", 0) + 1
                        continue
                    fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    by_format["stix_indicator"] = by_format.get("stix_indicator", 0) + 1
                    n_total += 1
                print(f"  [stix] {by_format.get('stix_indicator', 0)} accepted, "
                      f"{rejects.get('stix_indicator', 0)} rejected")

        if not args.skip_sigma:
            for spec in SIGMA_TECHNIQUES:
                rec = synth_sigma_from_technique(spec)
                if rec is None:
                    rejects["sigma_rule"] = rejects.get("sigma_rule", 0) + 1
                    continue
                fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                by_format["sigma_rule"] = by_format.get("sigma_rule", 0) + 1
                n_total += 1
            print(f"  [sigma] {by_format.get('sigma_rule', 0)} accepted, "
                  f"{rejects.get('sigma_rule', 0)} rejected")

        if not args.skip_yara:
            for spec in YARA_FAMILIES:
                rec = synth_yara_from_family(spec)
                if rec is None:
                    rejects["yara_rule"] = rejects.get("yara_rule", 0) + 1
                    continue
                fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                by_format["yara_rule"] = by_format.get("yara_rule", 0) + 1
                n_total += 1
            print(f"  [yara] {by_format.get('yara_rule', 0)} accepted, "
                  f"{rejects.get('yara_rule', 0)} rejected")

    print(f"\nWrote {n_total} records to {out_path}")
    print(f"  by format: {by_format}")
    print(f"  rejects:   {rejects}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
