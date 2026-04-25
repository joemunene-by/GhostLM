"""GhostLM data collection — downloads and preprocesses cybersecurity training data from public sources."""

import csv
import datetime
import hashlib
import json
import os
import random
import re
import subprocess
import tempfile
import time
import unicodedata
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Optional

import requests
from tqdm import tqdm


def clean_text(text: str) -> str:
    """Clean and normalize raw text for training.

    Strips excessive whitespace, removes non-printable characters,
    and normalizes unicode to ASCII where possible.

    Args:
        text: Raw input text string.

    Returns:
        Cleaned and normalized text string.
    """
    if not isinstance(text, str):
        text = str(text)

    # Normalize unicode to ASCII where possible
    text = unicodedata.normalize("NFKD", text)
    text = text.encode("ascii", "ignore").decode("ascii")

    # Remove non-printable characters (keep newlines and tabs)
    text = "".join(
        ch for ch in text
        if ch in ("\n", "\t") or ch.isprintable()
    )

    # Strip excessive blank lines (more than 2 consecutive newlines)
    text = re.sub(r"\n{3,}", "\n\n", text)

    # Strip leading/trailing whitespace per line
    lines = [line.strip() for line in text.splitlines()]
    text = "\n".join(lines)

    # Collapse multiple spaces into one
    text = re.sub(r" {2,}", " ", text)

    return text.strip()


def save_jsonl(records: List[Dict], path: str) -> None:
    """Save a list of dictionaries as a JSONL file.

    Each dictionary is written as a single JSON object on its own line.

    Args:
        records: List of dictionaries to serialize.
        path: Output file path (will be created with parent directories).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"  Saved {len(records)} records to {path}")


def load_jsonl(path: str) -> List[Dict]:
    """Load a JSONL file and return a list of dictionaries.

    Args:
        path: Path to the .jsonl file.

    Returns:
        List of dictionaries parsed from each line.
    """
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def collect_cve_descriptions(
    output_path: str = "data/raw/cve.jsonl",
    max_records: int = 10000,
    start_year: int = 1999,
    end_year: Optional[int] = None,
    per_year_cap: int = 500,
    append: bool = False,
) -> None:
    """Fetch CVE descriptions from the NVD REST API v2.0, balanced across years.

    NVD caps any ``pubStartDate``/``pubEndDate`` range at 120 consecutive days,
    so each year is fetched in 120-day windows (Jan–Apr, May–Aug, Sep–Dec).
    Reads ``NVD_API_KEY`` from the environment for a higher rate limit.

    Args:
        output_path: Destination path for the output JSONL file.
        max_records: Total cap across all years.
        start_year: First year window to query (inclusive).
        end_year: Last year window (defaults to current UTC year).
        per_year_cap: Max records accepted per year (prevents single-year dominance).
        append: If True, load existing records at ``output_path`` and merge
            (dedup-by-id), rather than overwriting.
    """
    print("Collecting CVE descriptions from NVD API (120-day windowed)...")
    base_url = "https://services.nvd.nist.gov/rest/json/cves/2.0"
    api_key = os.environ.get("NVD_API_KEY")
    delay = 0.7 if api_key else 6.0
    headers = {"apiKey": api_key} if api_key else {}

    if end_year is None:
        end_year = datetime.datetime.utcnow().year

    existing: Dict[str, Dict] = {}
    if append and Path(output_path).exists():
        for rec in load_jsonl(output_path):
            existing[rec.get("id", "")] = rec
        print(f"  append mode: {len(existing)} existing records loaded")

    # NVD caps date ranges at 120 days and returns HTTP 404 when exceeded
    # (reason surfaced in the `message` response header). Using 119-day
    # chunks leaves a safety margin and covers leap years cleanly.
    windows = []
    for year in range(start_year, end_year + 1):
        day = datetime.date(year, 1, 1)
        year_end = datetime.date(year, 12, 31)
        while day <= year_end:
            window_end = min(day + datetime.timedelta(days=118), year_end)
            start_str = day.strftime("%Y-%m-%dT00:00:00.000")
            end_str = window_end.strftime("%Y-%m-%dT23:59:59.999")
            windows.append((start_str, end_str, year))
            day = window_end + datetime.timedelta(days=1)

    year_counts: Dict[int, int] = {}
    new_records: List[Dict] = []

    for pub_start, pub_end, year in windows:
        if len(new_records) + len(existing) >= max_records:
            break
        if year_counts.get(year, 0) >= per_year_cap:
            continue
        params = {
            "pubStartDate": pub_start,
            "pubEndDate": pub_end,
            "resultsPerPage": 2000,
            "startIndex": 0,
        }
        try:
            resp = requests.get(base_url, params=params, headers=headers, timeout=60)
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            print(f"  {pub_start[:10]}..{pub_end[:10]}: fetch failed ({e})")
            time.sleep(delay)
            continue

        added = 0
        for item in data.get("vulnerabilities", []):
            try:
                cve_id = item.get("cve", {}).get("id", "")
                if not cve_id or cve_id in existing:
                    continue
                descriptions = item.get("cve", {}).get("descriptions", [])
                description = ""
                for desc in descriptions:
                    if isinstance(desc, dict) and desc.get("lang") == "en":
                        description = desc.get("value", "")
                        break
                if not description and descriptions:
                    description = descriptions[0].get("value", "")
                cleaned = clean_text(description)
                if len(cleaned) < 50:
                    continue
                rec = {"id": cve_id, "text": cleaned, "source": "nvd"}
                new_records.append(rec)
                existing[cve_id] = rec
                year_counts[year] = year_counts.get(year, 0) + 1
                added += 1
                if year_counts[year] >= per_year_cap or len(new_records) + len(existing) >= max_records:
                    break
            except Exception:
                continue
        total = len(existing)
        print(f"  {pub_start[:10]}..{pub_end[:10]}: +{added}  (total {total})")
        time.sleep(delay)

    all_records = list(existing.values())
    if all_records:
        save_jsonl(all_records, output_path)
        years_seen = sorted({int(CVE_YEAR_RE.search(r["id"]).group(1))
                             for r in all_records
                             if CVE_YEAR_RE.search(r.get("id", ""))})
        if years_seen:
            print(f"  year span: {years_seen[0]}–{years_seen[-1]} ({len(years_seen)} years)")
    else:
        print("  Warning: No valid CVE records collected.")


CVE_YEAR_RE = re.compile(r"CVE-(\d{4})-\d+")


def collect_cve_full(
    output_path: str = "data/raw/cve_full.jsonl",
    start_year: int = 1999,
    end_year: Optional[int] = None,
    page_size: int = 2000,
    flush_every: int = 5000,
    request_timeout: int = 60,
) -> None:
    """Fetch the full NVD CVE corpus (uncapped, properly paginated).

    Differences from ``collect_cve_descriptions``:
      * No per-year cap, no global max — pulls every CVE in the date range.
      * Paginates within each 119-day window via ``startIndex``, so dense
        windows (recent years with >2000 CVEs per quarter) are not silently
        truncated at the first page.
      * Flushes to ``output_path`` every ``flush_every`` newly-collected
        records so a long pull can be killed and resumed without losing
        most of the work. Always runs in append-merge mode.

    Reads ``NVD_API_KEY`` from the environment for the higher rate limit
    (50 req / 30s vs 5 req / 30s anonymous). Without a key this is a
    multi-hour pull; with a key, on the order of minutes.

    Args:
        output_path: Destination JSONL path (kept distinct from
            ``cve.jsonl`` so the v0.3.0 corpus stays reproducible).
        start_year: First year to query (inclusive).
        end_year: Last year (defaults to current UTC year).
        page_size: ``resultsPerPage`` per request (NVD max is 2000).
        flush_every: Flush merged records to disk after this many new
            records have been added since the last flush.
        request_timeout: Per-request timeout in seconds.
    """
    print("Collecting full NVD CVE corpus (paginated, uncapped)...")
    base_url = "https://services.nvd.nist.gov/rest/json/cves/2.0"
    api_key = os.environ.get("NVD_API_KEY")
    delay = 0.7 if api_key else 6.0
    headers = {"apiKey": api_key} if api_key else {}
    if not api_key:
        print("  NVD_API_KEY not set — using anonymous rate limit (~5 req / 30s).")

    if end_year is None:
        end_year = datetime.datetime.utcnow().year

    existing: Dict[str, Dict] = {}
    if Path(output_path).exists():
        for rec in load_jsonl(output_path):
            existing[rec.get("id", "")] = rec
        print(f"  resume mode: {len(existing)} existing records loaded from {output_path}")

    # 119-day windows leave a safety margin under NVD's 120-day cap.
    windows = []
    for year in range(start_year, end_year + 1):
        day = datetime.date(year, 1, 1)
        year_end = datetime.date(year, 12, 31)
        while day <= year_end:
            window_end = min(day + datetime.timedelta(days=118), year_end)
            start_str = day.strftime("%Y-%m-%dT00:00:00.000")
            end_str = window_end.strftime("%Y-%m-%dT23:59:59.999")
            windows.append((start_str, end_str, year))
            day = window_end + datetime.timedelta(days=1)

    new_since_flush = 0
    total_new = 0

    def flush():
        save_jsonl(list(existing.values()), output_path)

    for pub_start, pub_end, year in windows:
        start_index = 0
        window_added = 0
        while True:
            params = {
                "pubStartDate": pub_start,
                "pubEndDate": pub_end,
                "resultsPerPage": page_size,
                "startIndex": start_index,
            }
            try:
                resp = requests.get(base_url, params=params, headers=headers, timeout=request_timeout)
                resp.raise_for_status()
                data = resp.json()
            except Exception as e:
                print(f"  {pub_start[:10]}..{pub_end[:10]} idx={start_index}: fetch failed ({e})")
                time.sleep(delay)
                break

            vulns = data.get("vulnerabilities", [])
            total_results = data.get("totalResults", 0)

            page_added = 0
            for item in vulns:
                try:
                    cve_id = item.get("cve", {}).get("id", "")
                    if not cve_id or cve_id in existing:
                        continue
                    descriptions = item.get("cve", {}).get("descriptions", [])
                    description = ""
                    for desc in descriptions:
                        if isinstance(desc, dict) and desc.get("lang") == "en":
                            description = desc.get("value", "")
                            break
                    if not description and descriptions:
                        description = descriptions[0].get("value", "")
                    cleaned = clean_text(description)
                    if len(cleaned) < 50:
                        continue
                    rec = {"id": cve_id, "text": cleaned, "source": "nvd"}
                    existing[cve_id] = rec
                    page_added += 1
                except Exception:
                    continue

            window_added += page_added
            new_since_flush += page_added
            total_new += page_added
            start_index += len(vulns)

            time.sleep(delay)

            # End of window — break when we've consumed all results, or when
            # the page came back empty (shouldn't happen if totalResults
            # was honest, but defensive).
            if start_index >= total_results or not vulns:
                break

        print(
            f"  {pub_start[:10]}..{pub_end[:10]}: +{window_added} "
            f"(window total {total_results}, corpus {len(existing)})"
        )

        if new_since_flush >= flush_every:
            flush()
            print(f"  flushed {len(existing)} records to {output_path}")
            new_since_flush = 0

    flush()
    years_seen = sorted({
        int(CVE_YEAR_RE.search(r["id"]).group(1))
        for r in existing.values()
        if CVE_YEAR_RE.search(r.get("id", ""))
    })
    print(f"\n  Done. Added {total_new} new records, total {len(existing)}.")
    if years_seen:
        print(f"  Year span: {years_seen[0]}–{years_seen[-1]} ({len(years_seen)} years)")


def collect_security_papers(
    output_path: str = "data/raw/papers.jsonl",
    max_records: int = 2000,
) -> None:
    """Fetch real cs.CR paper abstracts from the arXiv API.

    Queries arXiv's Atom export endpoint for the cs.CR (cryptography and
    security) category in descending date order. Falls back to a small
    curated synthetic set only if the API is unreachable.

    Args:
        output_path: Destination path for the output JSONL file.
        max_records: Maximum number of paper records to collect.
    """
    print("Collecting security papers from arXiv cs.CR...")
    records: List[Dict] = []
    batch = 200
    ns = {"atom": "http://www.w3.org/2005/Atom"}

    for start in range(0, max_records, batch):
        url = (
            "http://export.arxiv.org/api/query"
            f"?search_query=cat:cs.CR"
            f"&start={start}"
            f"&max_results={min(batch, max_records - start)}"
            "&sortBy=submittedDate&sortOrder=descending"
        )
        try:
            resp = requests.get(url, timeout=60)
            resp.raise_for_status()
            root = ET.fromstring(resp.text)
        except Exception as e:
            print(f"  arXiv fetch failed at start={start}: {e}")
            break
        entries = root.findall("atom:entry", ns)
        if not entries:
            break
        for entry in entries:
            id_el = entry.find("atom:id", ns)
            title_el = entry.find("atom:title", ns)
            summary_el = entry.find("atom:summary", ns)
            if summary_el is None or not summary_el.text:
                continue
            arxiv_id = id_el.text.strip() if id_el is not None and id_el.text else ""
            title = title_el.text.strip() if title_el is not None and title_el.text else ""
            summary = summary_el.text.strip()
            text = f"{title}\n\n{summary}" if title else summary
            cleaned = clean_text(text)
            if len(cleaned) >= 100:
                records.append({"id": arxiv_id, "text": cleaned, "source": "arxiv"})
        print(f"  arXiv start={start}: +{len(entries)} entries ({len(records)} kept)")
        # arXiv asks for one request per 3 seconds
        time.sleep(3)

    if records:
        save_jsonl(records, output_path)
        return

    print("  arXiv unavailable — falling back to curated synthetic abstracts (small set, no padding).")
    synthetic_papers = [
        {"title": "Automated Vulnerability Detection Using Deep Learning", "abstract": "We present a deep learning approach to automatically detect security vulnerabilities in source code. Our model achieves 94% precision on the NIST NVD dataset, outperforming traditional static analysis tools. The approach combines abstract syntax tree analysis with transformer-based sequence modeling to identify common vulnerability patterns including buffer overflows, SQL injection, and cryptographic weaknesses."},
        {"title": "Adversarial Machine Learning in Cybersecurity", "abstract": "This paper surveys adversarial attacks against machine learning models deployed in security-critical systems. We analyze evasion attacks, poisoning attacks, and model extraction techniques targeting intrusion detection systems and malware classifiers. Our findings indicate that ensemble defenses and adversarial training significantly improve robustness against adaptive attackers."},
        {"title": "Network Intrusion Detection Using Graph Neural Networks", "abstract": "We propose a graph neural network architecture for network intrusion detection that models network traffic as dynamic graphs. By capturing temporal dependencies between packets and flows, our approach detects sophisticated multi-stage attacks including APTs and lateral movement that evade traditional signature-based detection systems."},
        {"title": "Formal Verification of Cryptographic Protocols", "abstract": "We apply formal verification methods to analyze the security properties of modern cryptographic protocols. Using model checking and theorem proving, we identify subtle flaws in protocol specifications that could enable authentication bypass and key recovery attacks. Our analysis covers TLS 1.3, Signal Protocol, and several blockchain consensus mechanisms."},
        {"title": "Fuzzing for Vulnerability Discovery in Binary Programs", "abstract": "Coverage-guided fuzzing has emerged as one of the most effective techniques for discovering security vulnerabilities in binary programs. We present enhancements to AFL++ that improve path exploration through symbolic execution integration and machine learning-guided mutation strategies, achieving 3x improvement in vulnerability discovery rate on standard benchmarks."},
        {"title": "Side-Channel Attacks on Hardware Security Modules", "abstract": "Hardware security modules are assumed to provide tamper-resistant cryptographic operations, but physical side-channel attacks can extract secret keys through power analysis and electromagnetic emissions. We demonstrate practical attacks against commercial HSMs and propose countermeasures including constant-time implementation and noise injection to mitigate information leakage."},
        {"title": "Ransomware Detection Through Behavioral Analysis", "abstract": "Modern ransomware employs sophisticated evasion techniques to bypass signature-based antivirus solutions. We develop a behavioral analysis system that monitors file system operations, registry modifications, and network activity to detect ransomware before significant data loss occurs. Our system achieves 99.2% detection rate with less than 0.1% false positives on a dataset of 10,000 ransomware samples."},
        {"title": "Supply Chain Security in Software Development", "abstract": "Software supply chain attacks have become a significant threat vector, compromising trusted development pipelines to distribute malicious code. We analyze recent supply chain incidents including SolarWinds and XZ Utils, identifying common attack patterns and proposing automated detection mechanisms based on code signing, dependency pinning, and behavioral monitoring of build processes."},
        {"title": "Memory Safety Vulnerabilities in Systems Programming Languages", "abstract": "Memory safety bugs including buffer overflows, use-after-free, and null pointer dereferences remain a primary source of security vulnerabilities in systems software. We conduct a longitudinal study of CVEs in C and C++ projects, analyzing root causes and evaluating the effectiveness of sanitizers, static analysis, and memory-safe language migration as mitigation strategies."},
        {"title": "Web Application Firewall Evasion Techniques", "abstract": "Web application firewalls serve as a critical defense layer against injection attacks and web-based exploits. We systematically evaluate evasion techniques including encoding variations, SQL comment injection, and HTTP protocol manipulation against commercial and open-source WAF solutions. Our results demonstrate significant gaps in detection coverage and propose improved signature generation methods."},
    ]

    for i, paper in enumerate(synthetic_papers):
        combined = f"{paper['title']}\n\n{paper['abstract']}"
        cleaned = clean_text(combined)
        if len(cleaned) >= 100:
            records.append({"id": i, "text": cleaned, "source": "papers"})

    if records:
        save_jsonl(records, output_path)
    else:
        print("  Warning: No paper records generated.")


def collect_ctf_writeups(output_path: str = "data/raw/ctf.jsonl") -> None:
    """Collect CTF (Capture The Flag) writeup texts from HuggingFace datasets.

    Attempts to load the 0xJustin/Dungeons-and-Hackers dataset first,
    then falls back to ethz-spylab/ctf-dataset. If both fail, generates
    500 synthetic CTF-style training records covering common security
    challenge topics.

    Args:
        output_path: Destination path for the output JSONL file.
    """
    print("Collecting CTF writeups...")
    records = []

    # Try primary dataset (lazy import — keeps CVE/paper collectors working without `datasets`)
    try:
        from datasets import load_dataset
    except ImportError:
        print("  `datasets` not installed — skipping HuggingFace sources, using synthetic fallback.")
        records = _generate_synthetic_ctf_data()
        if records:
            save_jsonl(records, output_path)
        return

    try:
        dataset = load_dataset("0xJustin/Dungeons-and-Hackers", split="train")
        for i, item in tqdm(enumerate(dataset), desc="CTF writeups", total=len(dataset)):
            try:
                writeup = item.get("text") or item.get("writeup") or item.get("content", "")

                if isinstance(writeup, list):
                    writeup = " ".join(str(w) for w in writeup)

                cleaned = clean_text(str(writeup))

                if len(cleaned) >= 200:
                    records.append({
                        "id": i,
                        "text": cleaned,
                        "source": "ctf",
                    })
            except Exception:
                continue
    except Exception as e:
        print(f"  Warning: Could not load 0xJustin/Dungeons-and-Hackers: {e}")

        # Try fallback dataset
        try:
            dataset = load_dataset("ethz-spylab/ctf-dataset", split="train")
            for i, item in tqdm(enumerate(dataset), desc="CTF writeups", total=len(dataset)):
                try:
                    writeup = item.get("text") or item.get("writeup") or item.get("content", "")

                    if isinstance(writeup, list):
                        writeup = " ".join(str(w) for w in writeup)

                    cleaned = clean_text(str(writeup))

                    if len(cleaned) >= 200:
                        records.append({
                            "id": i,
                            "text": cleaned,
                            "source": "ctf",
                        })
                except Exception:
                    continue
        except Exception as e2:
            print(f"  Warning: Could not load ethz-spylab/ctf-dataset: {e2}")
            print("  Generating synthetic CTF training data...")
            records = _generate_synthetic_ctf_data()

    if records:
        save_jsonl(records, output_path)
    else:
        print("  Warning: No valid CTF writeup records collected.")


def _generate_synthetic_ctf_data(count: int = 500) -> List[Dict]:
    """Generate synthetic CTF-style training records — unique texts only, no padding.

    Emits each template text at most once. The ``count`` argument is treated
    as an upper bound. The record total will be the number of distinct
    templates, not ``count``, so the corpus never reports duplicates as new
    examples. For real CTF writeups, point ``collect_ctf_writeups`` at a
    HuggingFace dataset.

    Args:
        count: Maximum number of synthetic records to emit.

    Returns:
        List of dicts with id, text, and source fields.
    """
    topics = [
        {
            "topic": "SQL injection",
            "texts": [
                "This CTF challenge involved exploiting a SQL injection vulnerability in a web application's login form. The input field was not properly sanitized, allowing us to inject UNION SELECT statements to extract data from the database. We used sqlmap to automate the process and discovered the admin credentials hidden in the users table. The flag was stored in a separate configuration table that required chaining multiple injection techniques to access.",
                "The SQL injection challenge required bypassing a WAF filter that blocked common keywords like SELECT and UNION. We used double URL encoding and case variation to evade the filter rules. After identifying the injection point through error-based testing, we crafted a blind SQL injection payload using boolean-based techniques to enumerate the database schema and extract the flag character by character.",
                "In this advanced SQL injection challenge, the application used parameterized queries for the main login but had a secondary search function that was vulnerable. We discovered the vulnerability by fuzzing all input parameters and noticed a timing difference when injecting sleep functions. Using time-based blind SQL injection, we extracted the database version, table names, and ultimately the flag from a hidden admin_notes table.",
            ],
        },
        {
            "topic": "XSS",
            "texts": [
                "The cross-site scripting challenge involved finding a reflected XSS vulnerability in a comment submission system. The application filtered script tags but missed img tags with onerror handlers. We crafted a payload using an img tag with a broken source URL and an onerror attribute that executed our JavaScript. The payload stole the admin session cookie and sent it to our listener, allowing us to access the admin panel and retrieve the flag.",
                "This DOM-based XSS challenge required understanding how the client-side JavaScript processed URL parameters. The application used innerHTML to render content from the hash fragment without proper sanitization. We constructed a malicious URL that, when visited by the admin bot, executed our JavaScript payload in the context of the target origin. The payload read the flag from a hidden DOM element and exfiltrated it via a DNS lookup.",
            ],
        },
        {
            "topic": "buffer overflow",
            "texts": [
                "The buffer overflow challenge involved a simple C program that used gets() to read user input into a 64-byte buffer. We used GDB to determine the exact offset needed to overwrite the return address, then crafted a payload with NOP sled, shellcode, and the return address pointing back into our buffer. The binary had no stack canaries or ASLR enabled, making the exploitation straightforward. We spawned a shell and read the flag from the home directory.",
                "This advanced buffer overflow challenge required bypassing both ASLR and NX bit protections. We used a return-oriented programming approach, chaining together ROP gadgets found in the binary to call mprotect and make the stack executable. After leaking a libc address through a format string vulnerability, we calculated the base address and constructed our ROP chain to execute shellcode and retrieve the flag.",
            ],
        },
        {
            "topic": "privilege escalation",
            "texts": [
                "The privilege escalation challenge started with a low-privilege user account on a Linux system. We enumerated the system using linpeas and discovered a SUID binary with a known vulnerability. By exploiting a path traversal in the binary, we were able to execute commands as root. The flag was located in the root directory and required reading a file with restricted permissions that our escalated privileges allowed us to access.",
                "This Windows privilege escalation challenge required identifying a misconfigured service running with SYSTEM privileges. We found that the service binary was writable by our user account, so we replaced it with a reverse shell payload. When the service restarted, we gained a SYSTEM-level shell and could access the flag stored in the Administrator's protected folder.",
            ],
        },
        {
            "topic": "reverse engineering",
            "texts": [
                "The reverse engineering challenge provided a compiled ELF binary that required us to find the correct input string to unlock the flag. Using Ghidra, we decompiled the binary and traced the validation logic through multiple obfuscation layers including XOR encoding and byte shuffling. We wrote a Python script to reverse the transformation and generate the correct input, which when provided to the binary printed the flag to stdout.",
                "This crackme challenge involved a packed binary that used anti-debugging techniques to prevent analysis. We used UPX to unpack the binary and then set breakpoints on common anti-debug checks to bypass them. Static analysis revealed a custom hash function that validated the serial key. We brute-forced the key space using a distributed approach and found the valid serial that revealed the flag.",
            ],
        },
        {
            "topic": "cryptography",
            "texts": [
                "The cryptography challenge involved breaking a custom encryption scheme that combined AES-CBC with a weak key derivation function. The key was derived from a short password using a single round of MD5, making it vulnerable to dictionary attacks. We used hashcat with a custom ruleset to recover the password, then decrypted the ciphertext to reveal the flag embedded in the plaintext.",
                "This RSA challenge provided a public key with a small exponent and a ciphertext encrypted with the same message under multiple keys. We applied Hastad's broadcast attack to recover the plaintext without factoring any of the moduli. The attack worked because the same message was encrypted under three different public keys with exponent e=3, allowing us to use the Chinese Remainder Theorem to recover the cube root of the message.",
            ],
        },
        {
            "topic": "network forensics",
            "texts": [
                "The network forensics challenge provided a PCAP file containing captured network traffic from a compromised system. We used Wireshark to filter and analyze the traffic, identifying suspicious DNS queries that contained exfiltrated data encoded in subdomain labels. By reconstructing the DNS tunnel traffic and decoding the base64-encoded payloads, we recovered the stolen credentials and the flag hidden in the exfiltrated data stream.",
                "This packet analysis challenge required identifying a man-in-the-middle attack within a large PCAP capture. We noticed ARP poisoning attempts followed by SSL stripping attacks. By following the TCP streams of the downgraded HTTP connections, we found the victim's login credentials being transmitted in plaintext. The flag was embedded in one of the intercepted web requests as a custom header value.",
            ],
        },
        {
            "topic": "steganography",
            "texts": [
                "The steganography challenge provided an image file that appeared to be a normal photograph. Using binwalk, we discovered a hidden ZIP archive appended to the end of the PNG file. Extracting the archive revealed a text file with a base64-encoded string that decoded to the flag. Additionally, the image contained LSB-encoded data in the least significant bits of the blue channel that provided hints for solving the challenge.",
                "This audio steganography challenge required analyzing a WAV file for hidden data. Using a spectrogram viewer, we discovered text rendered into the frequency domain of the audio signal. The visible text was a partial flag, and the remaining portion was hidden using phase encoding in the audio samples. We wrote a Python script using scipy to extract the phase-encoded bits and reconstruct the complete flag.",
            ],
        },
        {
            "topic": "web vulnerabilities",
            "texts": [
                "The web vulnerability challenge involved a file upload feature that accepted image files but had a flawed validation routine. By crafting a polyglot file that was both a valid PNG and a PHP script, we bypassed the file type check. The server stored uploads in a web-accessible directory, allowing us to execute our PHP payload by visiting the uploaded file URL. The script read the flag from the server's configuration directory.",
                "This server-side request forgery challenge allowed us to make the application fetch URLs on our behalf. We used SSRF to access the cloud instance metadata service and retrieve temporary IAM credentials. With those credentials, we accessed an S3 bucket that contained the application's secrets, including the flag stored in a configuration file that was not meant to be publicly accessible.",
            ],
        },
        {
            "topic": "binary exploitation",
            "texts": [
                "The binary exploitation challenge involved a heap-based vulnerability where a use-after-free condition allowed us to control a function pointer. We crafted a series of allocations and frees to place our controlled data at the location of the freed chunk, then triggered the dangling pointer to redirect execution to our shellcode. The challenge required understanding glibc's malloc implementation and heap chunk metadata.",
                "This format string vulnerability challenge required leaking the stack canary value and then overwriting a GOT entry to gain code execution. We used the format string to read arbitrary memory addresses, identified the canary position through trial and error, and then crafted a second payload that overwrote the printf GOT entry with the address of our shellcode. The binary's partial RELRO made GOT overwriting possible.",
            ],
        },
    ]

    records: List[Dict] = []
    seen = set()
    rng = random.Random(42)
    flat = [(t["topic"], txt) for t in topics for txt in t["texts"]]
    rng.shuffle(flat)
    for i, (topic, text) in enumerate(flat):
        if len(records) >= count:
            break
        key = hashlib.md5(text.strip().lower().encode("utf-8")).hexdigest()
        if key in seen:
            continue
        seen.add(key)
        records.append({"id": i, "text": text, "source": "synthetic", "topic": topic})

    return records


def collect_mitre_attack(
    output_path: str = "data/raw/mitre_attack.jsonl",
    max_records: int = 5000,
) -> None:
    """Fetch MITRE ATT&CK technique descriptions from the STIX 2.1 enterprise dataset.

    Downloads the ATT&CK Enterprise STIX bundle and extracts technique
    names, descriptions, and tactic phases.

    Args:
        output_path: Destination path for the output JSONL file.
        max_records: Maximum number of records to collect.
    """
    print("Collecting MITRE ATT&CK techniques...")
    records = []

    stix_url = "https://raw.githubusercontent.com/mitre/cti/master/enterprise-attack/enterprise-attack.json"

    try:
        resp = requests.get(stix_url, timeout=60)
        resp.raise_for_status()
        bundle = resp.json()
    except Exception as e:
        print(f"  Warning: Failed to fetch ATT&CK STIX bundle: {e}")
        print("  Generating synthetic ATT&CK data as fallback...")
        records = _generate_synthetic_attack_data()
        if records:
            save_jsonl(records[:max_records], output_path)
        return

    for obj in tqdm(bundle.get("objects", []), desc="ATT&CK techniques", leave=False):
        if obj.get("type") != "attack-pattern":
            continue
        if obj.get("revoked", False) or obj.get("x_mitre_deprecated", False):
            continue

        name = obj.get("name", "")
        description = obj.get("description", "")
        if not description:
            continue

        # Extract tactic phases
        phases = []
        for kcp in obj.get("kill_chain_phases", []):
            if kcp.get("kill_chain_name") == "mitre-attack":
                phases.append(kcp.get("phase_name", ""))

        # Extract external ID (e.g. T1059)
        ext_id = ""
        for ref in obj.get("external_references", []):
            if ref.get("source_name") == "mitre-attack":
                ext_id = ref.get("external_id", "")
                break

        phase_str = ", ".join(phases) if phases else "unknown"
        text = f"MITRE ATT&CK Technique {ext_id}: {name}\nTactic: {phase_str}\n\n{description}"
        cleaned = clean_text(text)

        if len(cleaned) >= 100:
            records.append({
                "id": ext_id or name,
                "text": cleaned,
                "source": "mitre_attack",
            })

        if len(records) >= max_records:
            break

    if records:
        save_jsonl(records, output_path)
    else:
        print("  Warning: No ATT&CK records collected.")


def collect_exploitdb(
    output_path: str = "data/raw/exploitdb.jsonl",
    max_records: int = 5000,
) -> None:
    """Fetch exploit texts from a shallow-cloned Exploit-DB repository.

    Shallow clones the Exploit-DB repository, reads `files_exploits.csv`,
    prepends metadata headers to exploit files, and stores cleaned records.

    Args:
        output_path: Destination path for the output JSONL file.
        max_records: Maximum number of records to collect.
    """
    print("Collecting Exploit-DB records...")
    records = []

    with tempfile.TemporaryDirectory() as tmp:
        repo_dir = Path(tmp) / "exploitdb"
        try:
            subprocess.run(
                [
                    "git",
                    "clone",
                    "--depth",
                    "1",
                    "https://gitlab.com/exploit-database/exploitdb.git",
                    str(repo_dir),
                ],
                check=True,
                capture_output=True,
                text=True,
            )
        except Exception as e:
            print(f"  Warning: Failed to clone Exploit-DB repository: {e}")
            return

        csv_path = repo_dir / "files_exploits.csv"
        if not csv_path.exists():
            print("  Warning: files_exploits.csv not found in cloned repository.")
            return

        try:
            with open(csv_path, "r", encoding="utf-8", errors="ignore", newline="") as f:
                rows = list(csv.DictReader(f))
        except Exception as e:
            print(f"  Warning: Failed to parse Exploit-DB CSV: {e}")
            return

        for row in tqdm(rows, desc="Exploit-DB files", leave=False):
            file_path = (row.get("file") or "").strip()
            if not file_path:
                continue

            exploit_path = repo_dir / file_path
            if not exploit_path.exists():
                continue

            try:
                with open(exploit_path, "r", encoding="utf-8", errors="ignore") as ef:
                    exploit_content = ef.read()
            except Exception:
                continue

            header = (
                f"Exploit-DB #{row.get('id', '')}: {row.get('description', '')}\n"
                f"Platform: {row.get('platform', '')} / {row.get('type', '')}"
            )
            codes = (row.get("codes") or "").strip()
            if codes:
                header += f"\nCVE: {codes}"

            cleaned = clean_text(f"{header}\n\n{exploit_content}")
            if 100 <= len(cleaned) <= 8000:
                records.append({
                    "id": f"edb-{row.get('id', '')}",
                    "text": cleaned,
                    "source": "exploitdb",
                })

            if len(records) >= max_records:
                break

    if records:
        save_jsonl(records, output_path)
    else:
        print("  Warning: No Exploit-DB records collected.")


def _generate_synthetic_attack_data(count: int = 200) -> List[Dict]:
    """Generate synthetic MITRE ATT&CK-style technique descriptions as fallback."""
    techniques = [
        {"id": "T1059", "name": "Command and Scripting Interpreter", "tactic": "execution",
         "desc": "Adversaries may abuse command and script interpreters to execute commands, scripts, or binaries. These interfaces and languages provide ways of interacting with computer systems and are a common feature across many platforms. Attackers use PowerShell, Bash, Python, and other interpreters to execute malicious payloads, download additional tools, and maintain persistence on compromised systems."},
        {"id": "T1078", "name": "Valid Accounts", "tactic": "defense-evasion, persistence, privilege-escalation, initial-access",
         "desc": "Adversaries may obtain and abuse credentials of existing accounts as a means of gaining Initial Access, Persistence, Privilege Escalation, or Defense Evasion. Compromised credentials may be used to bypass access controls and may even be used for persistent access to remote systems. Using valid accounts allows adversaries to blend in with normal activity, making detection more difficult."},
        {"id": "T1190", "name": "Exploit Public-Facing Application", "tactic": "initial-access",
         "desc": "Adversaries may attempt to exploit a weakness in an Internet-facing host or system to initially access a network. The weakness in the system can be a software bug, a temporary glitch, or a misconfiguration. Common targets include web servers, database servers, and network services exposed to the internet such as VPNs and remote access gateways."},
        {"id": "T1071", "name": "Application Layer Protocol", "tactic": "command-and-control",
         "desc": "Adversaries may communicate using OSI application layer protocols to avoid detection or network filtering by blending in with existing traffic. Commands to the remote system and the results of those commands will be embedded within the protocol traffic between the client and server. Common protocols abused include HTTP, HTTPS, DNS, and SMTP."},
        {"id": "T1486", "name": "Data Encrypted for Impact", "tactic": "impact",
         "desc": "Adversaries may encrypt data on target systems or on large numbers of systems in a network to interrupt availability to system and network resources. This is commonly associated with ransomware, where the adversary encrypts files using strong cryptographic algorithms and demands payment for the decryption key."},
        {"id": "T1053", "name": "Scheduled Task/Job", "tactic": "execution, persistence, privilege-escalation",
         "desc": "Adversaries may abuse task scheduling functionality to facilitate initial or recurring execution of malicious code. Utilities exist within all major operating systems to schedule programs or scripts to be executed at a specified date and time. Adversaries use cron jobs, Windows Task Scheduler, and systemd timers to maintain persistence."},
        {"id": "T1055", "name": "Process Injection", "tactic": "defense-evasion, privilege-escalation",
         "desc": "Adversaries may inject code into processes in order to evade process-based defenses as well as possibly elevate privileges. Process injection is a method of executing arbitrary code in the address space of a separate live process. Running code in the context of another process may allow access to the process's memory, system or network resources, and possibly elevated privileges."},
        {"id": "T1027", "name": "Obfuscated Files or Information", "tactic": "defense-evasion",
         "desc": "Adversaries may attempt to make an executable or file difficult to discover or analyze by encrypting, encoding, or otherwise obfuscating its contents on the system or in transit. This is common behavior for malware authors who use packers, crypters, and steganography to hide malicious payloads from security tools and analysts."},
    ]

    records = []
    for i in range(count):
        t = techniques[i % len(techniques)]
        text = f"MITRE ATT&CK Technique {t['id']}: {t['name']}\nTactic: {t['tactic']}\n\n{t['desc']}"
        records.append({"id": t["id"], "text": clean_text(text), "source": "mitre_attack"})
    return records


def collect_cwe_descriptions(
    output_path: str = "data/raw/cwe.jsonl",
    max_records: int = 5000,
) -> None:
    """Fetch CWE (Common Weakness Enumeration) descriptions.

    Downloads CWE data from the MITRE CWE REST API or falls back
    to synthetic data covering the most common weaknesses.

    Args:
        output_path: Destination path for the output JSONL file.
        max_records: Maximum number of CWE records to collect.
    """
    print("Collecting CWE descriptions...")
    records = []

    # CWE top entries — curated from the CWE Top 25 and related
    cwe_entries = [
        {"id": "CWE-79", "name": "Improper Neutralization of Input During Web Page Generation (Cross-site Scripting)",
         "desc": "The product does not neutralize or incorrectly neutralizes user-controllable input before it is placed in output that is used as a web page that is served to other users. Cross-site scripting (XSS) vulnerabilities occur when an application includes untrusted data in a new web page without proper validation or escaping, or updates an existing web page with user-supplied data using a browser API that can create HTML or JavaScript. XSS allows attackers to execute scripts in the victim's browser which can hijack user sessions, deface web sites, or redirect the user to malicious sites. There are three main types: Reflected XSS, Stored XSS, and DOM-based XSS."},
        {"id": "CWE-787", "name": "Out-of-bounds Write",
         "desc": "The product writes data past the end, or before the beginning, of the intended buffer. This typically occurs when the pointer or its index is incremented or decremented to a position beyond the bounds of the buffer or when pointer arithmetic results in a position outside of the valid memory location. Out-of-bounds writes can result in corruption of data, a crash, or code execution. The software may modify an index or perform pointer arithmetic that references a memory location that is outside of the boundaries of the buffer, causing a write to an unexpected memory location."},
        {"id": "CWE-89", "name": "Improper Neutralization of Special Elements used in an SQL Command (SQL Injection)",
         "desc": "The product constructs all or part of an SQL command using externally-influenced input from an upstream component, but it does not neutralize or incorrectly neutralizes special elements that could modify the intended SQL command when it is sent to a downstream component. Without sufficient removal or quoting of SQL syntax in user-controllable inputs, the generated SQL query can cause those inputs to be interpreted as SQL instead of ordinary user data. Attackers can use SQL injection to read sensitive data, modify database data, execute administration operations, and in some cases issue commands to the operating system."},
        {"id": "CWE-416", "name": "Use After Free",
         "desc": "Referencing memory after it has been freed can cause a program to crash, use unexpected values, or execute code. The use of previously-freed memory can have any number of adverse consequences, ranging from the corruption of valid data to the execution of arbitrary code, depending on the instantiation and timing of the flaw. When memory is freed, the contents are not cleared and can be reused. If the freed memory is referenced again, the program may use data that has been altered by a different part of the program or by an attacker."},
        {"id": "CWE-78", "name": "Improper Neutralization of Special Elements used in an OS Command (OS Command Injection)",
         "desc": "The product constructs all or part of an OS command using externally-influenced input from an upstream component, but it does not neutralize or incorrectly neutralizes special elements that could modify the intended OS command when it is sent to a downstream component. This could allow attackers to execute unexpected, dangerous commands directly on the operating system. This weakness can lead to a vulnerability in environments in which the attacker does not have direct access to the operating system, such as in web applications."},
        {"id": "CWE-20", "name": "Improper Input Validation",
         "desc": "The product receives input or data, but it does not validate or incorrectly validates that the input has the properties that are required to process the data safely and correctly. Input validation is a frequently-used technique for checking potentially dangerous inputs in order to ensure that the inputs are safe for processing within the code, or when communicating with other components. When software does not validate input properly, an attacker is able to craft the input in a form that is not expected by the rest of the application."},
        {"id": "CWE-125", "name": "Out-of-bounds Read",
         "desc": "The product reads data past the end, or before the beginning, of the intended buffer. This typically occurs when the pointer or its index is decremented or incremented to a position outside the bounds of the buffer. An out-of-bounds read can allow attackers to read sensitive information from other memory locations or cause a crash. The typical result is a crash, which can lead to denial of service. In some cases, this allows the attacker to read sensitive data such as cryptographic keys or passwords from adjacent memory."},
        {"id": "CWE-22", "name": "Improper Limitation of a Pathname to a Restricted Directory (Path Traversal)",
         "desc": "The product uses external input to construct a pathname that is intended to identify a file or directory that is located underneath a restricted parent directory, but the product does not properly neutralize special elements within the pathname that can cause the pathname to resolve to a location that is outside of the restricted directory. Many file operations are intended to take place within a restricted directory. By using special path elements such as .. and /, attackers can escape outside of the restricted location to access files or directories elsewhere on the system."},
        {"id": "CWE-352", "name": "Cross-Site Request Forgery (CSRF)",
         "desc": "The web application does not, or cannot, sufficiently verify whether a well-formed, valid, consistent request was intentionally provided by the user who submitted the request. When a web server is designed to receive a request from a client without any mechanism for verifying that it was intentionally sent, then it might be possible for an attacker to trick a client into making an unintentional request to the web server which will be treated as an authentic request."},
        {"id": "CWE-434", "name": "Unrestricted Upload of File with Dangerous Type",
         "desc": "The product allows the upload of files without checking the file type against a list of acceptable types, or only checking the file type against a list of unacceptable types. This can result in the upload of executable server-side scripts that give the attacker code execution capability on the server. Web shells, backdoors, and other malicious files can be uploaded to achieve remote code execution if the uploaded files are accessible via the web server."},
        {"id": "CWE-862", "name": "Missing Authorization",
         "desc": "The product does not perform an authorization check when an actor attempts to access a resource or perform an action. Without access control checks, users can access data or perform actions that they should not be allowed to perform. This can lead to a wide range of problems including information exposure, denial of service, and arbitrary code execution."},
        {"id": "CWE-476", "name": "NULL Pointer Dereference",
         "desc": "A NULL pointer dereference occurs when the application dereferences a pointer that it expects to be valid, but is NULL, typically causing a crash or exit. NULL pointer dereference issues can occur through a number of flaws, including race conditions, and simple programming omissions. In some cases, attackers can use this to cause denial of service conditions."},
        {"id": "CWE-190", "name": "Integer Overflow or Wraparound",
         "desc": "The product performs a calculation that can produce an integer overflow or wraparound, when the logic assumes that the resulting value will always be larger than the original value. This can introduce other weaknesses when the calculation is used for resource management or execution control. An integer overflow occurs when the result of an arithmetic operation exceeds the maximum value that can be stored in the associated representation."},
        {"id": "CWE-502", "name": "Deserialization of Untrusted Data",
         "desc": "The application deserializes untrusted data without sufficiently verifying that the resulting data will be valid. It is often convenient to serialize and deserialize objects for communication or storage. However, deserialized data or code can often be modified without using the provided accessor functions if it does not use cryptographic safeguards to protect integrity. Attackers can exploit this to achieve remote code execution, denial of service, or authentication bypass."},
        {"id": "CWE-287", "name": "Improper Authentication",
         "desc": "When an actor claims to have a given identity, the product does not prove or insufficiently proves that the claim is correct. If the product does not sufficiently prove that an actor has a given identity, an attacker may be able to impersonate another user, gain unintended access to data, or perform actions that require authentication without proper credentials."},
    ]

    for entry in cwe_entries:
        text = f"{entry['id']}: {entry['name']}\n\n{entry['desc']}"
        cleaned = clean_text(text)
        if len(cleaned) >= 100:
            records.append({
                "id": entry["id"],
                "text": cleaned,
                "source": "cwe",
            })

    # Repeat to fill quota
    if records and len(records) < max_records:
        base = list(records)
        while len(records) < max_records:
            records.extend(base)
        records = records[:max_records]

    if records:
        save_jsonl(records, output_path)
    else:
        print("  Warning: No CWE records generated.")


def collect_owasp(
    output_path: str = "data/raw/owasp.jsonl",
    max_records: int = 2000,
) -> None:
    """Collect OWASP Top 10 and related security guidance content.

    Uses curated OWASP Top 10 (2021) descriptions covering the most
    critical web application security risks.

    Args:
        output_path: Destination path for the output JSONL file.
        max_records: Maximum number of records to collect.
    """
    print("Collecting OWASP content...")
    records = []

    owasp_top10 = [
        {"id": "A01:2021", "name": "Broken Access Control",
         "desc": "Access control enforces policy such that users cannot act outside of their intended permissions. Failures typically lead to unauthorized information disclosure, modification, or destruction of all data or performing a business function outside the user's limits. Common access control vulnerabilities include violation of the principle of least privilege, bypassing access control checks by modifying the URL, internal application state, or the HTML page, permitting viewing or editing someone else's account, elevation of privilege, metadata manipulation such as replaying or tampering with JWT access control tokens, and CORS misconfiguration allowing API access from unauthorized origins."},
        {"id": "A02:2021", "name": "Cryptographic Failures",
         "desc": "The first thing is to determine the protection needs of data in transit and at rest. For example, passwords, credit card numbers, health records, personal information, and business secrets require extra protection. For all such data: Is any data transmitted in clear text? This concerns protocols such as HTTP, SMTP, FTP also using TLS upgrades like STARTTLS. Are any old or weak cryptographic algorithms or protocols used either by default or in older code? Are default crypto keys in use, weak crypto keys generated or re-used, or is proper key management or rotation missing?"},
        {"id": "A03:2021", "name": "Injection",
         "desc": "An application is vulnerable to attack when user-supplied data is not validated, filtered, or sanitized by the application, dynamic queries or non-parameterized calls without context-aware escaping are used directly in the interpreter, hostile data is used within object-relational mapping search parameters to extract additional sensitive records, and hostile data is directly used or concatenated. Some of the more common injections are SQL, NoSQL, OS command, Object Relational Mapping, LDAP, and Expression Language or Object Graph Navigation Library injection. The concept is identical among all interpreters."},
        {"id": "A04:2021", "name": "Insecure Design",
         "desc": "Insecure design is a broad category representing different weaknesses, expressed as missing or ineffective control design. Insecure design is not the source for all other Top 10 risk categories. There is a difference between insecure design and insecure implementation. A secure design can still have implementation defects leading to vulnerabilities. An insecure design cannot be fixed by a perfect implementation as by definition, needed security controls were never created to defend against specific attacks. Threat modeling, secure design patterns, and reference architectures are needed."},
        {"id": "A05:2021", "name": "Security Misconfiguration",
         "desc": "The application might be vulnerable if the application is missing appropriate security hardening across any part of the application stack or improperly configured permissions on cloud services. Unnecessary features are enabled or installed such as unnecessary ports, services, pages, accounts, or privileges. Default accounts and their passwords are still enabled and unchanged. Error handling reveals stack traces or other overly informative error messages to users. For upgraded systems, the latest security features are disabled or not configured securely."},
        {"id": "A06:2021", "name": "Vulnerable and Outdated Components",
         "desc": "You are likely vulnerable if you do not know the versions of all components you use, both client-side and server-side. This includes components you directly use as well as nested dependencies. If the software is vulnerable, unsupported, or out of date, this includes the OS, web or application server, database management system, applications, APIs and all components, runtime environments, and libraries. If you do not scan for vulnerabilities regularly and subscribe to security bulletins related to the components you use."},
        {"id": "A07:2021", "name": "Identification and Authentication Failures",
         "desc": "Confirmation of the user's identity, authentication, and session management is critical to protect against authentication-related attacks. There may be authentication weaknesses if the application permits automated attacks such as credential stuffing where the attacker has a list of valid usernames and passwords. Permits brute force or other automated attacks. Permits default, weak, or well-known passwords. Uses weak or ineffective credential recovery and forgot-password processes. Uses plain text, encrypted, or weakly hashed passwords data stores."},
        {"id": "A08:2021", "name": "Software and Data Integrity Failures",
         "desc": "Software and data integrity failures relate to code and infrastructure that does not protect against integrity violations. An example of this is where an application relies upon plugins, libraries, or modules from untrusted sources, repositories, and content delivery networks. An insecure CI/CD pipeline can introduce the potential for unauthorized access, malicious code, or system compromise. Many applications now include auto-update functionality, where updates are downloaded without sufficient integrity verification and applied to the previously trusted application."},
        {"id": "A09:2021", "name": "Security Logging and Monitoring Failures",
         "desc": "This category is to help detect, escalate, and respond to active breaches. Without logging and monitoring, breaches cannot be detected. Insufficient logging, detection, monitoring, and active response occurs any time: Auditable events such as logins, failed logins, and high-value transactions are not logged. Warnings and errors generate no, inadequate, or unclear log messages. Logs of applications and APIs are not monitored for suspicious activity. Logs are only stored locally. Appropriate alerting thresholds and response escalation processes are not in place or effective."},
        {"id": "A10:2021", "name": "Server-Side Request Forgery (SSRF)",
         "desc": "SSRF flaws occur whenever a web application is fetching a remote resource without validating the user-supplied URL. It allows an attacker to coerce the application to send a crafted request to an unexpected destination, even when protected by a firewall, VPN, or another type of network access control list. As modern web applications provide end-users with convenient features, fetching a URL becomes a common scenario. As a result, the incidence of SSRF is increasing. Also, the severity of SSRF is becoming higher due to cloud services and the complexity of architectures."},
    ]

    for entry in owasp_top10:
        text = f"OWASP {entry['id']} — {entry['name']}\n\n{entry['desc']}"
        cleaned = clean_text(text)
        if len(cleaned) >= 100:
            records.append({
                "id": entry["id"],
                "text": cleaned,
                "source": "owasp",
            })

    # Repeat to fill quota
    if records and len(records) < max_records:
        base = list(records)
        while len(records) < max_records:
            records.extend(base)
        records = records[:max_records]

    if records:
        save_jsonl(records, output_path)
    else:
        print("  Warning: No OWASP records generated.")


def collect_capec(
    output_path: str = "data/raw/capec.jsonl",
    max_records: int = 3000,
) -> None:
    """Fetch CAPEC (Common Attack Pattern Enumeration and Classification) data.

    Downloads CAPEC STIX data from the MITRE repository and extracts
    attack pattern descriptions.

    Args:
        output_path: Destination path for the output JSONL file.
        max_records: Maximum number of records to collect.
    """
    print("Collecting CAPEC attack patterns...")
    records = []

    capec_url = "https://raw.githubusercontent.com/mitre/cti/master/capec/2.1/stix-capec.json"

    try:
        resp = requests.get(capec_url, timeout=60)
        resp.raise_for_status()
        bundle = resp.json()
    except Exception as e:
        print(f"  Warning: Failed to fetch CAPEC STIX bundle: {e}")
        print("  Generating synthetic CAPEC data as fallback...")
        records = _generate_synthetic_capec_data()
        if records:
            save_jsonl(records[:max_records], output_path)
        return

    for obj in tqdm(bundle.get("objects", []), desc="CAPEC patterns", leave=False):
        if obj.get("type") != "attack-pattern":
            continue

        name = obj.get("name", "")
        description = obj.get("description", "")
        if not description:
            continue

        # Extract CAPEC ID
        capec_id = ""
        for ref in obj.get("external_references", []):
            if ref.get("source_name") == "capec":
                capec_id = ref.get("external_id", "")
                break

        text = f"CAPEC {capec_id}: {name}\n\n{description}"
        cleaned = clean_text(text)

        if len(cleaned) >= 100:
            records.append({
                "id": capec_id or name,
                "text": cleaned,
                "source": "capec",
            })

        if len(records) >= max_records:
            break

    if records:
        save_jsonl(records, output_path)
    else:
        print("  Warning: No CAPEC records collected.")


def _generate_synthetic_capec_data(count: int = 200) -> List[Dict]:
    """Generate synthetic CAPEC-style attack pattern descriptions as fallback."""
    patterns = [
        {"id": "CAPEC-66", "name": "SQL Injection", "desc": "This attack exploits target software that constructs SQL statements based on user input. An attacker crafts input strings so that when the target software constructs SQL statements based on the input, the resulting SQL statement performs actions other than those the application intended. SQL Injection results from failure of the application to appropriately validate input."},
        {"id": "CAPEC-86", "name": "XSS Through HTTP Headers", "desc": "An attacker exploits web applications that use HTTP headers to pass data to web applications by embedding malicious content in header data that is not properly validated. When the application processes the header values, the malicious content executes as part of the web application."},
        {"id": "CAPEC-100", "name": "Overflow Buffers", "desc": "An adversary may try to overflow a buffer in the target software to gain control of execution or cause denial of service. Buffer overflow attacks target improper or missing bounds checking on buffer operations, typically triggered by input injected by an adversary."},
        {"id": "CAPEC-112", "name": "Brute Force", "desc": "In this attack, some asset like a credential, key, or passphrase is protected by a finite secret value. The attacker attempts to gain access to this asset by using trial-and-error to exhaustively explore all the possible secret values in the hope of finding the secret. If the secret is not extremely large, the attacker may be able to explore the entire space within the available time and computing resources."},
        {"id": "CAPEC-125", "name": "Flooding", "desc": "An adversary consumes the resources of a target by rapidly engaging in a large number of interactions with the target. This type of attack generally exposes a weakness in rate limiting or flow control in the target's processing of communication requests. The adversary's goal is to deny service by using up all available resources."},
    ]

    records = []
    for i in range(count):
        p = patterns[i % len(patterns)]
        text = f"{p['id']}: {p['name']}\n\n{p['desc']}"
        records.append({"id": p["id"], "text": clean_text(text), "source": "capec"})
    return records


def deduplicate_records(records: List[Dict], key: str = "text") -> List[Dict]:
    """Remove duplicate records based on text content hash.

    Uses a normalized hash of the text field to identify and remove
    exact and near-exact duplicates.

    Args:
        records: List of record dictionaries.
        key: Field name to deduplicate on.

    Returns:
        Deduplicated list of records.
    """
    import hashlib

    seen = set()
    unique = []
    for record in records:
        text = record.get(key, "").strip().lower()
        # Normalize whitespace for near-dedup
        text_norm = re.sub(r"\s+", " ", text)
        h = hashlib.md5(text_norm.encode("utf-8")).hexdigest()
        if h not in seen:
            seen.add(h)
            unique.append(record)

    removed = len(records) - len(unique)
    if removed > 0:
        print(f"  Deduplication: removed {removed} duplicates ({len(unique)} unique records)")
    return unique


def merge_datasets(
    input_paths: List[str],
    output_path: str = "data/processed/train.jsonl",
    val_split: float = 0.05,
    shuffle: bool = True,
    seed: int = 42,
) -> None:
    """Merge multiple JSONL datasets and split into train/validation sets.

    Dedupes across all inputs by content hash, then assigns each record to
    train or val deterministically by hashing its text — identical texts
    always land in the same split, so val cannot leak into train.

    Args:
        input_paths: List of paths to JSONL files to merge.
        output_path: Destination path for the training split JSONL.
        val_split: Fraction of data to reserve for validation (0.0 to 1.0).
        shuffle: Whether to shuffle train records (affects order only).
        seed: Random seed for shuffling.
    """
    print("Merging datasets...")
    all_records: List[Dict] = []

    for path in input_paths:
        if os.path.exists(path):
            records = load_jsonl(path)
            all_records.extend(records)
            print(f"  Loaded {len(records)} records from {path}")
        else:
            print(f"  Warning: {path} not found, skipping.")

    if not all_records:
        print("  Warning: No records to merge.")
        return

    all_records = deduplicate_records(all_records)

    val_bucket_max = max(1, int(round(val_split * 100)))
    train_records: List[Dict] = []
    val_records: List[Dict] = []
    for r in all_records:
        text_norm = re.sub(r"\s+", " ", r.get("text", "").strip().lower())
        bucket = int(hashlib.md5(text_norm.encode("utf-8")).hexdigest(), 16) % 100
        if bucket < val_bucket_max:
            val_records.append(r)
        else:
            train_records.append(r)

    if shuffle:
        random.seed(seed)
        random.shuffle(train_records)
        random.shuffle(val_records)

    val_path = str(Path(output_path).with_name(Path(output_path).stem.replace("train", "val") + ".jsonl"))
    save_jsonl(train_records, output_path)
    save_jsonl(val_records, val_path)

    # Post-split leakage check — deterministic split should guarantee 0, verify anyway.
    val_texts = {r.get("text", "") for r in val_records}
    leakage = sum(1 for r in train_records if r.get("text", "") in val_texts)

    print(f"\n  Dataset stats:")
    print(f"    Total records: {len(all_records)}")
    print(f"    Train: {len(train_records)}")
    print(f"    Validation: {len(val_records)}")
    print(f"    Leakage check: {leakage} val texts found in train (expected 0)")


def main() -> None:
    """Run the full GhostLM data collection pipeline.

    Creates necessary directories, downloads data from all configured
    sources, and merges them into train/validation splits.
    """
    print("=" * 50)
    print("GhostLM Data Collection Pipeline")
    print("=" * 50)

    # Ensure directories exist
    Path("data/raw").mkdir(parents=True, exist_ok=True)
    Path("data/processed").mkdir(parents=True, exist_ok=True)

    # Collect from all sources
    cve_path = "data/raw/cve.jsonl"
    papers_path = "data/raw/papers.jsonl"
    ctf_path = "data/raw/ctf.jsonl"
    attack_path = "data/raw/mitre_attack.jsonl"
    cwe_path = "data/raw/cwe.jsonl"
    owasp_path = "data/raw/owasp.jsonl"
    capec_path = "data/raw/capec.jsonl"
    exploitdb_path = "data/raw/exploitdb.jsonl"

    collect_cve_descriptions(output_path=cve_path)
    collect_security_papers(output_path=papers_path)
    collect_ctf_writeups(output_path=ctf_path)
    collect_mitre_attack(output_path=attack_path)
    collect_cwe_descriptions(output_path=cwe_path)
    collect_owasp(output_path=owasp_path)
    collect_capec(output_path=capec_path)
    collect_exploitdb(output_path=exploitdb_path)

    # Merge into train/val splits (with deduplication)
    merge_datasets(
        input_paths=[
            cve_path, papers_path, ctf_path,
            attack_path, cwe_path, owasp_path, capec_path, exploitdb_path,
        ],
        output_path="data/processed/train.jsonl",
    )

    print("\n" + "=" * 50)
    print("Data collection complete.")
    print("=" * 50)


if __name__ == "__main__":
    main()
