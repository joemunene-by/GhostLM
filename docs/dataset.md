# GhostLM Dataset Guide

## Overview
GhostLM is trained on cybersecurity-domain text from multiple sources.

## Current Sources

### NVD CVE Descriptions (Real Data)
- Source: National Vulnerability Database REST API v2.0
- Records: ~9,925 CVE descriptions
- Format: CVE ID + English description
- Example: "CVE-2023-1234: A buffer overflow vulnerability in..."
- Quality: High — official government security database

### Security Research Papers (Synthetic)
- Source: Curated synthetic abstracts covering 10 cybersecurity research areas
- Records: 500
- Topics: vulnerability detection, adversarial ML, network intrusion, cryptographic protocols, fuzzing, side-channel attacks, ransomware detection, supply chain security, memory safety, WAF evasion
- Quality: Medium — synthetic but domain-accurate

### CTF Writeups (Synthetic)
- Source: Synthetic writeups covering 10 CTF challenge types
- Records: 500
- Topics: SQL injection, XSS, buffer overflow, privilege escalation, reverse engineering, cryptography, network forensics, steganography, web vulnerabilities, binary exploitation
- Quality: Medium — synthetic but realistic methodology descriptions

## Dataset Statistics
| Split | Records | Tokens |
|---|---|---|
| Train | 10,378 | ~490,532 |
| Validation | 547 | ~25,214 |
| Total | 10,925 | ~515,746 |

## Adding New Data Sources
Edit data/collect.py and add a new collect_* function following this pattern:
```python
def collect_my_source(output_path="data/raw/my_source.jsonl", max_records=5000):
    records = []
    # ... collect data ...
    for item in data:
        cleaned = clean_text(item["text"])
        if len(cleaned) >= 50:
            records.append({"id": item["id"], "text": cleaned, "source": "my_source"})
    save_jsonl(records, output_path)
```
Then add the path to merge_datasets() in main().

## Planned Data Sources
- exploit-db.com exploit descriptions
- OWASP documentation
- Real arXiv cs.CR papers (large download required)
- Metasploit module descriptions
- HackTheBox/TryHackMe writeups
