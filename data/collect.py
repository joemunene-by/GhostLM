"""GhostLM data collection — downloads and preprocesses cybersecurity training data from public sources."""

import json
import os
import random
import re
import time
import unicodedata
from pathlib import Path
from typing import Dict, List, Optional

import requests
from datasets import load_dataset
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
) -> None:
    """Fetch CVE descriptions from the NVD REST API v2.0.

    Paginates through the NVD API in batches of 2000 records,
    extracting CVE IDs and descriptions. Respects rate limits
    with a 1-second delay between requests.

    Args:
        output_path: Destination path for the output JSONL file.
        max_records: Maximum number of CVE records to collect.
    """
    print("Collecting CVE descriptions from NVD API...")
    records = []
    base_url = "https://services.nvd.nist.gov/rest/json/cves/2.0"
    page_size = 2000
    max_pages = 5

    for page in range(max_pages):
        start_index = page * page_size
        url = f"{base_url}?resultsPerPage={page_size}&startIndex={start_index}"

        try:
            resp = requests.get(url, timeout=30)
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            print(f"  Warning: Failed to fetch NVD page {page + 1}: {e}")
            break

        vulnerabilities = data.get("vulnerabilities", [])
        if not vulnerabilities:
            print(f"  No more CVE records at page {page + 1}.")
            break

        for item in tqdm(vulnerabilities, desc=f"CVE page {page + 1}", leave=False):
            try:
                cve_id = item.get("cve", {}).get("id", "")
                descriptions = item.get("cve", {}).get("descriptions", [])

                description = ""
                for desc in descriptions:
                    if isinstance(desc, dict) and desc.get("lang") == "en":
                        description = desc.get("value", "")
                        break
                if not description and descriptions:
                    description = descriptions[0].get("value", "")

                cleaned = clean_text(description)

                if len(cleaned) >= 50:
                    records.append({
                        "id": cve_id,
                        "text": cleaned,
                        "source": "nvd",
                    })

                if len(records) >= max_records:
                    break
            except Exception:
                continue

        if len(records) >= max_records:
            break

        time.sleep(1)

    if records:
        save_jsonl(records, output_path)
    else:
        print("  Warning: No valid CVE records collected.")


def collect_security_papers(
    output_path: str = "data/raw/papers.jsonl",
    max_records: int = 5000,
) -> None:
    """Download and extract cybersecurity research paper abstracts from arXiv.

    Loads the gfissore/arxiv-abstracts-2021 dataset from HuggingFace and
    filters for papers in the cs.CR (cryptography and security) category.
    Combines title and abstract into a single text field, cleans it,
    and saves as JSONL.

    Args:
        output_path: Destination path for the output JSONL file.
        max_records: Maximum number of paper records to collect.
    """
    print("Collecting security papers from arXiv...")
    records = []

    try:
        dataset = load_dataset("gfissore/arxiv-abstracts-2021", split="train")
    except Exception as e:
        print(f"  Warning: Could not load gfissore/arxiv-abstracts-2021: {e}")
        print("  Skipping security papers collection.")
        return

    for i, item in tqdm(enumerate(dataset), desc="Papers", total=min(len(dataset), max_records)):
        try:
            categories = item.get("categories", "") or ""

            if "cs.CR" not in categories:
                continue

            title = item.get("title", "") or ""
            abstract = item.get("abstract", "") or ""

            combined = f"{title}\n\n{abstract}"
            cleaned = clean_text(combined)

            if len(cleaned) >= 100:
                records.append({
                    "id": i,
                    "text": cleaned,
                    "source": "arxiv",
                })

            if len(records) >= max_records:
                break
        except Exception:
            continue

    if records:
        save_jsonl(records, output_path)
    else:
        print("  Warning: No valid security paper records collected.")


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

    # Try primary dataset
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
    """Generate synthetic CTF-style training records covering security topics.

    Creates realistic training text covering common CTF challenge types
    including SQL injection, XSS, buffer overflow, privilege escalation,
    reverse engineering, cryptography, network forensics, steganography,
    web vulnerabilities, and binary exploitation.

    Args:
        count: Number of synthetic records to generate.

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

    records = []
    for i in range(count):
        topic_data = topics[i % len(topics)]
        text_options = topic_data["texts"]
        text = text_options[i % len(text_options)]
        records.append({
            "id": i,
            "text": text,
            "source": "synthetic",
        })

    return records


def merge_datasets(
    input_paths: List[str],
    output_path: str = "data/processed/train.jsonl",
    val_split: float = 0.05,
    shuffle: bool = True,
    seed: int = 42,
) -> None:
    """Merge multiple JSONL datasets and split into train/validation sets.

    Loads all specified input JSONL files, optionally shuffles them,
    then splits into training and validation subsets.

    Args:
        input_paths: List of paths to JSONL files to merge.
        output_path: Destination path for the training split JSONL.
        val_split: Fraction of data to reserve for validation (0.0 to 1.0).
        shuffle: Whether to shuffle records before splitting.
        seed: Random seed for shuffling.
    """
    print("Merging datasets...")
    all_records = []

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

    if shuffle:
        random.seed(seed)
        random.shuffle(all_records)

    split_idx = int(len(all_records) * (1 - val_split))
    train_records = all_records[:split_idx]
    val_records = all_records[split_idx:]

    val_path = str(Path(output_path).with_name(Path(output_path).stem.replace("train", "val") + ".jsonl"))

    save_jsonl(train_records, output_path)
    save_jsonl(val_records, val_path)

    print(f"\n  Dataset stats:")
    print(f"    Total records: {len(all_records)}")
    print(f"    Train: {len(train_records)}")
    print(f"    Validation: {len(val_records)}")


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

    collect_cve_descriptions(output_path=cve_path)
    collect_security_papers(output_path=papers_path)
    collect_ctf_writeups(output_path=ctf_path)

    # Merge into train/val splits
    merge_datasets(
        input_paths=[cve_path, papers_path, ctf_path],
        output_path="data/processed/train.jsonl",
    )

    print("\n" + "=" * 50)
    print("Data collection complete.")
    print("=" * 50)


if __name__ == "__main__":
    main()
