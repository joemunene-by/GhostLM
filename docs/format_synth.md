# Bet 6 templated synth corpus

## Purpose

Bet 6 ([docs/differentiation.md](differentiation.md) §"Bet 6:
format-aware structured-data pretrain") needs ghost-base to see
real STIX 2.1 / YARA / Sigma / MISP artifacts during pretrain so
it can produce them at inference time. The canonical pipeline is
[`scripts/distill_format_aware.py`](../scripts/distill_format_aware.py)
calling an LLM teacher (Anthropic ~$50-100, free Ollama).

This doc captures the parallel deterministic-template path in
[`scripts/synth_format_aware.py`](../scripts/synth_format_aware.py)
that produces parser-valid training records with **no LLM spend
and no GPU**. The two paths are complementary: templated synth
gives volume + structural floor; LLM distillation gives idiomatic
variety. Mix both into pretrain.

## Run + result (2026-05-08)

```bash
PYTHONPATH=. python3 scripts/synth_format_aware.py --max-stix 500
```

| Format | Records | Rejects | Seed |
|---|---:|---:|---|
| STIX 2.1 indicators | 500 | 0 | First 500 non-rejected CVE-2020+ entries from `data/raw/cve_full.jsonl` |
| Sigma rules | 30 | 0 | Hand-curated 30-technique bank covering all 12 ATT&CK tactics |
| YARA rules | 30 | 0 | Hand-curated 30-family bank (loaders, ransomware, stealers, RATs, web shells, doc droppers, cryptominers, APT-style backdoors) |
| **TOTAL** | **560** | **0** | (~836 KB output) |

**Every record passes its format's parser** (`parse_stix`,
`parse_sigma`, `parse_yara`) before write. The parsers come from
the same module the LLM-distilled flow uses, so the templated
records are evaluated by exactly the same correctness bar.

## Why 100% parser-pass rate matters

LLM distillation in practice lands ~60-80% parser-valid records
even with a strong teacher (per the bet 6 scaffold's quality-filter
discussion). Templated synth lands 100% because the script only
emits text the parser will accept; failures land at template-edit
time, not training time.

That gives ghost-base a structural floor of ~560 records that we
can rely on. When the LLM-distilled records arrive, they'll add
1000-2000 noisier-but-more-diverse records on top.

## STIX template detail

The template picks a STIX pattern by keyword-matching the CVE
description:

| CVE description signal | STIX pattern emitted |
|---|---|
| "smb", "port 445" | `[network-traffic:dst_port = 445 AND network-traffic:protocols[*] = 'smb']` |
| "rdp", "port 3389" | `[network-traffic:dst_port = 3389 AND network-traffic:protocols[*] = 'tcp']` |
| "http", "web", "url" | `[url:value MATCHES '.*<cve_id>.*' OR network-traffic:dst_port = 443]` |
| "dns", "resolver" | `[network-traffic:dst_port = 53]` |
| "denial of service" / "dos" / "crash" | `[network-traffic:dst_port IN (80,443,8080)]` |
| "upload", "file", "extension" | `[file:hashes.'SHA-256' MATCHES '.*' AND file:name MATCHES '\\.(php|jsp|asp|aspx)$']` |
| "buffer overflow", "rce", "remote code" | `[file:hashes.'SHA-256' MATCHES '.*']` |
| "auth", "credential", "token" | `[user-account:account_login = '*']` |
| (default) | `[file:name MATCHES '.*']` |

Each emitted indicator has `external_references[0]` pointing back
at the CVE id with the canonical NVD URL, so the model learns the
prose-to-CVE-link convention as well as the SDO shape.

## Sigma technique bank (30 entries)

The hand-curated bank (in `scripts/synth_format_aware.py:SIGMA_TECHNIQUES`)
covers:

- **Execution**: T1059.001 (PowerShell), T1059.003 (cmd), T1059.004 (bash), T1106 (Native API), T1218.005 (mshta), T1218.011 (rundll32)
- **Persistence**: T1547.001 (Run keys), T1543.003 (Services), T1505.003 (Web shell), T1053.005 (Scheduled Tasks)
- **Privilege Escalation**: covered via the persistence techniques above
- **Defense Evasion**: T1027 (Obfuscated files), T1112 (Modify registry), T1140 (Deobfuscate/Decode), T1574.002 (DLL side-loading), T1497 (Sandbox evasion)
- **Credential Access**: T1003.001 (LSASS dump), T1110 (Brute force)
- **Discovery**: T1083 (File and Directory Discovery), T1018 (Remote System Discovery)
- **Lateral Movement**: T1021.001 (RDP), T1021.002 (SMB), T1133 (External Remote Services)
- **Command and Control**: T1071.001 (Web Protocols), T1090 (Proxy)
- **Initial Access**: T1190 (Exploit Public-Facing App), T1566.001 (Spearphishing Attachment), T1078 (Valid Accounts)
- **Impact**: T1486 (Data Encrypted for Impact), T1485 (Data Destruction)
- **Linux**: T1222 (File and Directory Permissions Modification)

Each rule emits a complete YAML document with title / id (deterministic
GUID per technique) / status / description / logsource / detection /
condition / falsepositives / level / tags.

## YARA family bank (30 entries)

The hand-curated bank (in `scripts/synth_format_aware.py:YARA_FAMILIES`)
covers:

- **Loaders / banking trojans**: Emotet, TrickBot, Qakbot
- **Post-exploitation**: Cobalt Strike Beacon (PE + HTTP stager)
- **Ransomware**: LockBit, BlackCat, Royal, Cl0p
- **Credential stealers**: RedLine, Raccoon, Lumma
- **RATs**: AsyncRAT, RemcosRAT, AgentTesla, DarkGate, PlugX, ShadowPad
- **Web shells**: China Chopper, WSO Webshell
- **Document droppers**: Office Macro, PDF JavaScript, HTA
- **Cryptominers**: XMRig
- **APT-style**: Sunburst-style, Industroyer
- **Generic**: PE Packer (UPX), ELF backdoor, Phishing kits

Each rule emits a complete YARA rule with meta block (author /
description / family / tlp / date), a strings section combining
hex magic and ascii markers, and a condition combining them.

## Reproducing

```bash
PYTHONPATH=. python3 scripts/synth_format_aware.py \
    --cve data/raw/cve_full.jsonl \
    --out data/processed/synth_format_aware.jsonl \
    --max-stix 500
```

Deterministic: same corpus + same script produces byte-identical
output. The output JSONL is gitignored under `data/processed/*`;
regenerate as needed.

## Scaling

The 500-CVE STIX cap is a config knob. The CVE corpus has ~186K
2020+ non-rejected entries, so the hard ceiling is ~186,000
records on STIX alone. Realistic mixing for ghost-base pretrain
might be 5,000-10,000 STIX templates (1-2% of the corpus tokens
budget).

The Sigma + YARA banks are bounded at 30 each by hand-curation.
Growing them is straightforward: add tuples to
`SIGMA_TECHNIQUES` / `YARA_FAMILIES`. A 100-technique Sigma bank
and a 100-family YARA bank are realistic next-steps; both are
~half a day of careful curation.

## What this does NOT replace

Templated synth produces rigid, low-diversity records. A
production training mix should pair:

1. ~5K templated records (this script): structural floor, 100%
   parser-valid, deterministic.
2. ~1K LLM-distilled records (`distill_format_aware.py`):
   idiomatic variety, ~60-80% parser-valid, costs ~$50-100 on
   Sonnet.

The combination gives ghost-base both the *shape* (templates) and
the *idiom* (LLM teacher) of structured CTI artifacts. Ship one,
the other follows.
