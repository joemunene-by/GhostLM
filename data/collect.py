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
    """Generate curated synthetic cybersecurity paper content for training.

    Uses a set of high-quality synthetic paper abstracts covering key
    cybersecurity research areas, repeated to reach the desired dataset size.

    Args:
        output_path: Destination path for the output JSONL file.
        max_records: Maximum number of paper records to collect.
    """
    print("Collecting security papers (using curated synthetic cybersecurity content)...")

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
    ] * 50  # repeat to get 500 records

    records = []
    for i, paper in enumerate(synthetic_papers[:max_records]):
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


def collect_mitre_attack(
    output_path: str = "data/raw/mitre_attack.jsonl",
    max_records: int = 5000,
) -> None:
    """Fetch MITRE ATT&CK technique descriptions from the STIX/TAXII public API.

    Downloads the Enterprise ATT&CK STIX bundle and extracts technique
    names and descriptions. Falls back to a comprehensive curated list
    if the API is unreachable.

    Args:
        output_path: Destination path for the output JSONL file.
        max_records: Maximum number of records to collect.
    """
    print("Collecting MITRE ATT&CK technique descriptions...")
    records = []

    # Try fetching from MITRE's public STIX data on GitHub
    stix_url = (
        "https://raw.githubusercontent.com/mitre/cti/master/"
        "enterprise-attack/enterprise-attack.json"
    )
    try:
        resp = requests.get(stix_url, timeout=60)
        resp.raise_for_status()
        bundle = resp.json()

        for obj in bundle.get("objects", []):
            if obj.get("type") == "attack-pattern":
                name = obj.get("name", "")
                desc = obj.get("description", "")
                ext_refs = obj.get("external_references", [])
                technique_id = ""
                for ref in ext_refs:
                    if ref.get("source_name") == "mitre-attack":
                        technique_id = ref.get("external_id", "")
                        break

                cleaned = clean_text(f"{technique_id} - {name}\n\n{desc}")
                if len(cleaned) >= 80:
                    records.append({
                        "id": technique_id or f"attack-{len(records)}",
                        "text": cleaned,
                        "source": "mitre_attack",
                    })
                if len(records) >= max_records:
                    break

    except Exception as e:
        print(f"  Warning: Could not fetch MITRE ATT&CK STIX bundle: {e}")
        print("  Using curated MITRE ATT&CK technique list...")
        records = _curated_mitre_attack_techniques()

    if records:
        save_jsonl(records, output_path)
    else:
        print("  Warning: No MITRE ATT&CK records collected.")


def _curated_mitre_attack_techniques() -> List[Dict]:
    """Return a curated list of MITRE ATT&CK technique descriptions as fallback data.

    Covers the most common ATT&CK techniques across the kill chain including
    initial access, execution, persistence, privilege escalation, defense
    evasion, credential access, discovery, lateral movement, collection,
    exfiltration, and command and control.

    Returns:
        List of dicts with id, text, and source fields.
    """
    techniques = [
        {"id": "T1566", "name": "Phishing", "desc": "Adversaries may send phishing messages to gain access to victim systems. All forms of phishing are electronically delivered social engineering. Phishing can be targeted, known as spearphishing. In spearphishing, a specific individual, company, or industry will be targeted by the adversary. Adversaries may send victims emails containing malicious attachments or links, typically to execute malicious code on victim systems or to gather credentials for use in other techniques."},
        {"id": "T1059", "name": "Command and Scripting Interpreter", "desc": "Adversaries may abuse command and script interpreters to execute commands, scripts, or binaries. These interfaces and languages provide ways of interacting with computer systems and are a common feature across many platforms. Most systems come with some built-in command-line interface and scripting capabilities such as PowerShell on Windows, Bash/sh on Unix, and Python or AppleScript on macOS."},
        {"id": "T1053", "name": "Scheduled Task/Job", "desc": "Adversaries may abuse task scheduling functionality to facilitate initial or recurring execution of malicious code. Utilities exist within all major operating systems to schedule programs or scripts to be executed at a specified date and time. A task can also be scheduled on a remote system, provided the proper authentication is met such as RPC and file and printer sharing in Windows environments."},
        {"id": "T1078", "name": "Valid Accounts", "desc": "Adversaries may obtain and abuse credentials of existing accounts as a means of gaining Initial Access, Persistence, Privilege Escalation, or Defense Evasion. Compromised credentials may be used to bypass access controls placed on various resources on systems within the network and may even be used for persistent access to remote systems and externally available services."},
        {"id": "T1547", "name": "Boot or Logon Autostart Execution", "desc": "Adversaries may configure system settings to automatically execute a program during system boot or logon to maintain persistence or gain higher-level privileges on compromised systems. Operating systems may have mechanisms for automatically running a program on system boot or account logon, including the Windows Registry run keys, Linux init scripts, and macOS login items."},
        {"id": "T1055", "name": "Process Injection", "desc": "Adversaries may inject code into processes in order to evade process-based defenses as well as possibly elevate privileges. Process injection is a method of executing arbitrary code in the address space of a separate live process. Running code in the context of another process may allow access to the process's memory, system/network resources, and possibly elevated privileges."},
        {"id": "T1003", "name": "OS Credential Dumping", "desc": "Adversaries may attempt to dump credentials to obtain account login and credential material, normally in the form of a hash or a clear text password, from the operating system and software. Credentials can then be used to perform Lateral Movement and access restricted information. Tools like Mimikatz, secretsdump, and hashdump are commonly used for credential extraction from LSASS memory, SAM database, and domain controller NTDS.dit files."},
        {"id": "T1021", "name": "Remote Services", "desc": "Adversaries may use Valid Accounts to log into a service specifically designed to accept remote connections, such as telnet, SSH, RDP, and VNC. The adversary may then perform actions as the logged-on user. In an enterprise environment, servers and workstations can be organized into domains providing centralized identity management through Active Directory."},
        {"id": "T1071", "name": "Application Layer Protocol", "desc": "Adversaries may communicate using application layer protocols to avoid detection/network filtering by blending in with existing traffic. Commands to the remote system, and often the results of those commands, will be embedded within the protocol traffic between the client and server. Adversaries may use HTTP, HTTPS, DNS, SMTP, and other protocols for command and control communications."},
        {"id": "T1486", "name": "Data Encrypted for Impact", "desc": "Adversaries may encrypt data on target systems or on large numbers of systems in a network to interrupt availability to system and network resources. This is commonly associated with ransomware operations where victims' files are encrypted and a ransom demand is made for decryption keys. Encryption algorithms such as AES and RSA are typically used."},
        {"id": "T1190", "name": "Exploit Public-Facing Application", "desc": "Adversaries may attempt to take advantage of a weakness in an Internet-facing computer or program using software, data, or commands in order to cause unintended or unanticipated behavior. The weakness in the system can be a bug, a glitch, or a design vulnerability. These applications are often websites, but can include databases (like SQL), standard services (like SMB or SSH), and any other applications with Internet accessible open sockets."},
        {"id": "T1105", "name": "Ingress Tool Transfer", "desc": "Adversaries may transfer tools or other files from an external system into a compromised environment. Files may be copied from an external adversary controlled system through the command and control channel to bring tools into the victim network or through alternate protocols with another tool such as FTP or curl. Tools can also be downloaded from legitimate hosting services and CDNs."},
        {"id": "T1070", "name": "Indicator Removal", "desc": "Adversaries may delete or alter generated artifacts on a host system, including logs and potentially captured files such as quarantined malware. Locations and format of logs are platform or product specific, however standard operating system logs are captured as Windows events or Linux/macOS files such as Bash History and /var/log/*."},
        {"id": "T1027", "name": "Obfuscated Files or Information", "desc": "Adversaries may attempt to make an executable or file difficult to discover or analyze by encrypting, encoding, or otherwise obfuscating its contents on the system or in transit. This is common behavior that can be used across different platforms and the network to evade defenses. Techniques include base64 encoding, XOR encryption, code signing, steganography, and packing."},
        {"id": "T1048", "name": "Exfiltration Over Alternative Protocol", "desc": "Adversaries may steal data by exfiltrating it over a different protocol than that of the existing command and control channel. The data may also be sent to an alternate network location from the main command and control server. Alternate protocols include FTP, SMTP, HTTP/S, DNS, SMB, or any other network protocol not being monitored for exfiltration."},
        {"id": "T1562", "name": "Impair Defenses", "desc": "Adversaries may maliciously modify components of a victim environment in order to hinder or disable defensive mechanisms. This encompasses a variety of techniques such as disabling security tools, modifying firewall rules, reducing log levels, and removing evidence of compromise. Adversaries may also impair command history logging, tamper with security agents, or disable Windows Event Logging."},
        {"id": "T1219", "name": "Remote Access Software", "desc": "An adversary may use legitimate desktop support and remote access software, such as Team Viewer, AnyDesk, Go2Assist, LogMein, and others, to establish an interactive command and control channel to target systems within networks. These services are commonly used as legitimate technical support software and may be allowed by application control within a target environment."},
        {"id": "T1574", "name": "Hijack Execution Flow", "desc": "Adversaries may execute their own malicious payloads by hijacking the way operating systems run programs. Hijacking execution flow can be for the purposes of persistence since this hijacked execution may reoccur over time. Techniques include DLL search order hijacking, DLL side-loading, dylib hijacking, executable installer file permissions weakness, and PATH environment variable modification."},
        {"id": "T1098", "name": "Account Manipulation", "desc": "Adversaries may manipulate accounts to maintain access to victim systems. Account manipulation may consist of any action that preserves adversary access to a compromised account, such as modifying credentials or permission groups. These actions could also include account activity designed to subvert security policies, such as performing iterative password updates to bypass password duration policies."},
        {"id": "T1218", "name": "System Binary Proxy Execution", "desc": "Adversaries may bypass process and/or signature-based defenses by proxying execution of malicious content with signed binaries. Binaries signed with trusted digital certificates can execute on Windows systems protected by digital signature validation. Several Microsoft signed binaries that are default on Windows installations can be used to proxy execution of other files, including mshta, rundll32, regsvr32, and certutil."},
    ]

    records = []
    for t in techniques:
        text = clean_text(f"{t['id']} - {t['name']}\n\n{t['desc']}")
        records.append({"id": t["id"], "text": text, "source": "mitre_attack"})

    # Repeat to create a larger training set
    expanded = []
    for i in range(25):
        for r in records:
            expanded.append({
                "id": f"{r['id']}_v{i}",
                "text": r["text"],
                "source": "mitre_attack",
            })
    return expanded[:5000]


def collect_owasp_top10(output_path: str = "data/raw/owasp.jsonl") -> None:
    """Generate training data from the OWASP Top 10 (2021) vulnerability categories.

    Uses hardcoded descriptions of the 10 categories since OWASP Top 10
    is a stable, well-known reference. Each entry includes the category
    name, risk description, and mitigation guidance.

    Args:
        output_path: Destination path for the output JSONL file.
    """
    print("Collecting OWASP Top 10 descriptions...")

    owasp_entries = [
        {
            "id": "A01:2021",
            "name": "Broken Access Control",
            "desc": (
                "Access control enforces policy such that users cannot act outside of their intended permissions. "
                "Failures typically lead to unauthorized information disclosure, modification, or destruction of all "
                "data or performing a business function outside the user's limits. Common vulnerabilities include "
                "violation of the principle of least privilege, bypassing access control checks by modifying the URL, "
                "internal application state, or the HTML page, or modifying the API request. Insecure direct object "
                "references (IDOR) allow attackers to access other users' data by changing the resource identifier. "
                "Missing access controls for POST, PUT, and DELETE operations in APIs are also prevalent. Mitigations "
                "include denying access by default, implementing access control mechanisms once and reusing them, "
                "enforcing record ownership, disabling web server directory listing, and logging access control failures."
            ),
        },
        {
            "id": "A02:2021",
            "name": "Cryptographic Failures",
            "desc": (
                "Previously known as Sensitive Data Exposure, this category focuses on failures related to cryptography "
                "which often lead to exposure of sensitive data. This includes use of weak or deprecated cryptographic "
                "algorithms such as MD5, SHA1, DES, and RC4, hard-coded or default cryptographic keys, lack of proper "
                "key management and rotation, transmission of data in clear text (HTTP, SMTP, FTP), use of obsolete "
                "protocols such as SSL and early TLS versions, and improper certificate validation. Mitigations include "
                "classifying data processed, stored, or transmitted by an application, encrypting all sensitive data at "
                "rest using strong algorithms like AES-256, encrypting data in transit with TLS 1.2+, using authenticated "
                "encryption modes, and ensuring proper key generation and management."
            ),
        },
        {
            "id": "A03:2021",
            "name": "Injection",
            "desc": (
                "An application is vulnerable to injection when user-supplied data is not validated, filtered, or "
                "sanitized, dynamic queries or non-parameterized calls without context-aware escaping are used directly "
                "in the interpreter, or hostile data is used within ORM search parameters to extract additional records. "
                "SQL injection, NoSQL injection, OS command injection, LDAP injection, and Expression Language injection "
                "are common variants. Cross-site scripting (XSS) is also a form of injection where malicious scripts are "
                "injected into trusted websites. Server-side template injection (SSTI) and CRLF injection are newer "
                "attack vectors. Mitigations include using parameterized queries and prepared statements, input validation "
                "using server-side allowlists, escaping special characters, and using LIMIT and other SQL controls to "
                "prevent mass disclosure of records."
            ),
        },
        {
            "id": "A04:2021",
            "name": "Insecure Design",
            "desc": (
                "Insecure design is a broad category representing different weaknesses expressed as missing or ineffective "
                "control design. Insecure design is not the source for all other Top 10 risk categories. There is a "
                "difference between insecure design and insecure implementation. A secure design can still have "
                "implementation defects leading to vulnerabilities. An insecure design cannot be fixed by a perfect "
                "implementation as by definition, needed security controls were never created to defend against specific "
                "attacks. Threat modeling, secure design patterns, and reference architectures are key practices. "
                "Mitigations include establishing a secure development lifecycle, using threat modeling for critical "
                "authentication, access control, and business logic flows, writing unit and integration tests for "
                "security-relevant flows, and segregating tenant data."
            ),
        },
        {
            "id": "A05:2021",
            "name": "Security Misconfiguration",
            "desc": (
                "The application might be vulnerable if it is missing appropriate security hardening across any part of "
                "the application stack, has improperly configured permissions on cloud services, has unnecessary features "
                "enabled or installed (e.g., unnecessary ports, services, pages, accounts, or privileges), default "
                "accounts and passwords are still enabled and unchanged, error handling reveals stack traces or overly "
                "informative error messages, or security settings in frameworks and libraries are not set to secure "
                "values. XML External Entity (XXE) processing vulnerabilities also fall in this category. Mitigations "
                "include a repeatable hardening process, a minimal platform without unnecessary features, reviewing and "
                "updating configurations as part of patch management, segmented application architecture, and automated "
                "verification of configurations."
            ),
        },
        {
            "id": "A06:2021",
            "name": "Vulnerable and Outdated Components",
            "desc": (
                "Components such as libraries, frameworks, and other software modules run with the same privileges as the "
                "application. If a vulnerable component is exploited, such an attack can facilitate serious data loss or "
                "server takeover. Applications and APIs using components with known vulnerabilities may undermine "
                "application defenses and enable various attacks and impacts. You are likely vulnerable if you do not know "
                "the versions of all components you use, if the software is vulnerable or unsupported, if you do not scan "
                "for vulnerabilities regularly, or if you do not fix or upgrade the underlying platform in a timely "
                "fashion. Mitigations include removing unused dependencies, continuously inventorying component versions, "
                "monitoring sources like CVE and NVD, using software composition analysis tools, and subscribing to "
                "email alerts for security vulnerabilities."
            ),
        },
        {
            "id": "A07:2021",
            "name": "Identification and Authentication Failures",
            "desc": (
                "Confirmation of the user's identity, authentication, and session management is critical to protect "
                "against authentication-related attacks. Weaknesses include permitting brute force or credential stuffing "
                "attacks, allowing default, weak, or well-known passwords, using weak or ineffective credential recovery "
                "and forgot-password processes, using plain text or weakly hashed passwords, having missing or ineffective "
                "multi-factor authentication, and exposing session identifiers in the URL. Session fixation attacks, "
                "session hijacking through predictable session tokens, and improper session invalidation after logout or "
                "inactivity are also common issues. Mitigations include implementing multi-factor authentication, not "
                "deploying with default credentials, implementing password strength checks, limiting failed login "
                "attempts, and using server-side secure session managers."
            ),
        },
        {
            "id": "A08:2021",
            "name": "Software and Data Integrity Failures",
            "desc": (
                "Software and data integrity failures relate to code and infrastructure that does not protect against "
                "integrity violations. This includes using software update mechanisms without integrity verification, "
                "using untrusted CDNs or libraries without subresource integrity checks, insecure CI/CD pipelines that "
                "can introduce unauthorized code changes, and auto-update functionality that downloads updates without "
                "sufficient integrity verification. Insecure deserialization is a specific sub-category where untrusted "
                "data is used to abuse the logic of an application, inflate a denial of service attack, or execute "
                "arbitrary code. Mitigations include using digital signatures to verify software and data integrity, "
                "ensuring libraries and dependencies are consumed from trusted repositories, using software supply chain "
                "security tools, and implementing proper code review processes for changes."
            ),
        },
        {
            "id": "A09:2021",
            "name": "Security Logging and Monitoring Failures",
            "desc": (
                "Without logging and monitoring, breaches cannot be detected. Insufficient logging, detection, monitoring, "
                "and active response occurs when auditable events such as logins, failed logins, and high-value "
                "transactions are not logged, warnings and errors generate no or inadequate log messages, logs of "
                "applications and APIs are not monitored for suspicious activity, logs are only stored locally, appropriate "
                "alerting thresholds and response escalation processes are not in place, and penetration testing and scans "
                "by DAST tools do not trigger alerts. Mitigations include ensuring all login, access control, and "
                "server-side input validation failures are logged with sufficient user context, ensuring log data is "
                "encoded correctly to prevent injections, establishing effective monitoring and alerting, and adopting an "
                "incident response and recovery plan."
            ),
        },
        {
            "id": "A10:2021",
            "name": "Server-Side Request Forgery (SSRF)",
            "desc": (
                "SSRF flaws occur whenever a web application is fetching a remote resource without validating the "
                "user-supplied URL. It allows an attacker to coerce the application to send a crafted request to an "
                "unexpected destination, even when protected by a firewall, VPN, or another type of network access control "
                "list. Attackers can use SSRF to access internal services behind the firewall, scan internal ports, "
                "read local files, access cloud provider metadata services (such as AWS IMDS at 169.254.169.254), and "
                "perform remote code execution. As modern web applications provide end-users with convenient features and "
                "the frequency of SSRF is increasing due to cloud services and the complexity of architectures. "
                "Mitigations include segmenting remote resource access, enforcing URL schemas and ports, disabling HTTP "
                "redirections, and not sending raw responses to clients."
            ),
        },
    ]

    records = []
    record_id = 0
    for entry in owasp_entries:
        text = clean_text(f"OWASP {entry['id']} - {entry['name']}\n\n{entry['desc']}")
        # Create multiple training records with slight variations in framing
        framings = [
            text,
            f"Security Vulnerability: {entry['name']}\n\n{entry['desc']}",
            f"What is {entry['name']}? {entry['desc']}",
            f"Explain the security risk of {entry['name']}. {entry['desc']}",
            f"OWASP Top 10 Risk: {entry['name']}\n\nCategory: {entry['id']}\n\n{entry['desc']}",
        ]
        for framing in framings:
            cleaned = clean_text(framing)
            if len(cleaned) >= 80:
                records.append({
                    "id": f"owasp-{record_id}",
                    "text": cleaned,
                    "source": "owasp",
                })
                record_id += 1

    # Repeat to build a larger training set
    expanded = []
    for i in range(20):
        for r in records:
            expanded.append({
                "id": f"{r['id']}_v{i}",
                "text": r["text"],
                "source": "owasp",
            })

    if expanded:
        save_jsonl(expanded, output_path)
    else:
        print("  Warning: No OWASP records generated.")


def collect_synthetic_security_writeups(
    output_path: str = "data/raw/security_writeups.jsonl",
    max_records: int = 5000,
) -> None:
    """Generate synthetic security writeups covering modern attack surfaces.

    Creates training data across cloud security (AWS/Azure misconfigurations),
    container security (Docker/Kubernetes escapes), API security, OAuth
    vulnerabilities, and JWT attacks.

    Args:
        output_path: Destination path for the output JSONL file.
        max_records: Maximum number of records to generate.
    """
    print("Generating synthetic security writeups (cloud, container, API, OAuth, JWT)...")

    writeups = [
        # --- Cloud Security: AWS Misconfigurations ---
        {
            "topic": "AWS S3 Bucket Misconfiguration",
            "text": (
                "Amazon S3 bucket misconfigurations remain one of the most common cloud security issues. When bucket "
                "policies or ACLs are set to allow public access, sensitive data such as customer records, backups, and "
                "application secrets can be exposed to the internet. Attackers use tools like bucket-finder, AWSBucketDump, "
                "and GrayhatWarfare to enumerate publicly accessible buckets. A common misconfiguration is granting "
                "s3:GetObject to the AllUsers principal or setting the bucket ACL to public-read. AWS now provides "
                "S3 Block Public Access settings at the account level, but legacy configurations may still expose data. "
                "Detection involves checking bucket policies for wildcard principals, monitoring AWS CloudTrail for "
                "unauthorized access patterns, and using AWS Config rules to enforce bucket encryption and access controls."
            ),
        },
        {
            "topic": "AWS IAM Privilege Escalation",
            "text": (
                "AWS Identity and Access Management (IAM) privilege escalation occurs when an attacker with limited IAM "
                "permissions discovers a path to obtain higher privileges. Common escalation vectors include: creating new "
                "IAM policies with AdministratorAccess and attaching them to the attacker's user, assuming roles with "
                "overly permissive trust policies, exploiting Lambda functions that run with elevated permissions, using "
                "iam:PassRole to pass a high-privilege role to a service, and leveraging iam:CreateLoginProfile to create "
                "console access for existing users. Tools like Pacu and pmapper map out potential escalation paths. "
                "Mitigations include applying least privilege policies, using IAM Access Analyzer to identify overly "
                "permissive resources, enabling MFA for all privileged operations, and monitoring CloudTrail for suspicious "
                "IAM API calls like CreatePolicy, AttachUserPolicy, and AssumeRole."
            ),
        },
        {
            "topic": "AWS EC2 IMDS Exploitation",
            "text": (
                "The EC2 Instance Metadata Service (IMDS) at 169.254.169.254 provides temporary credentials and instance "
                "information. When an SSRF vulnerability exists in a web application running on EC2, attackers can retrieve "
                "the IAM role credentials from http://169.254.169.254/latest/meta-data/iam/security-credentials/. These "
                "credentials can then be used externally to access AWS services. IMDSv2 mitigates this by requiring a "
                "session token obtained through a PUT request with a TTL header, making SSRF exploitation significantly "
                "harder. Organizations should enforce IMDSv2-only through the HttpTokens=required instance metadata option "
                "and monitor for unusual metadata service access patterns in VPC flow logs."
            ),
        },
        # --- Cloud Security: Azure Misconfigurations ---
        {
            "topic": "Azure Blob Storage Exposure",
            "text": (
                "Azure Blob Storage containers can be misconfigured with public access levels that expose data to "
                "unauthenticated users. The Container access level allows listing and reading all blobs, while the Blob "
                "access level allows reading individual blobs if the URL is known. Attackers enumerate Azure storage "
                "accounts using DNS brute-forcing of the .blob.core.windows.net namespace. Common exposures include "
                "database backups, application configurations containing connection strings, and source code repositories. "
                "Azure provides storage account-level settings to disable public blob access entirely. Security teams "
                "should use Azure Policy to enforce private container access, enable Azure Defender for Storage to detect "
                "anomalous access patterns, and require Shared Access Signatures (SAS) with short expiration times."
            ),
        },
        {
            "topic": "Azure AD Misconfiguration",
            "text": (
                "Azure Active Directory misconfigurations can lead to unauthorized access across an organization's cloud "
                "estate. Common issues include overly permissive application registrations that allow any user to consent "
                "to app permissions, guest accounts with access to directory data, misconfigured conditional access policies "
                "that exclude high-privilege accounts, and legacy authentication protocols that bypass MFA. Attackers use "
                "tools like ROADtools and AADInternals to enumerate Azure AD configurations, extract tokens, and escalate "
                "privileges. The Global Administrator role in Azure AD provides unrestricted access and should be protected "
                "with Privileged Identity Management (PIM) requiring just-in-time activation and approval workflows."
            ),
        },
        # --- Container Security: Docker ---
        {
            "topic": "Docker Container Escape via Privileged Mode",
            "text": (
                "Running Docker containers in privileged mode (--privileged flag) disables all security protections and "
                "grants the container full access to the host's devices and kernel capabilities. An attacker inside a "
                "privileged container can escape to the host by mounting the host filesystem (mount /dev/sda1 /mnt), "
                "loading kernel modules, accessing raw hardware devices, and manipulating cgroups. Even without full "
                "privileged mode, excessive Linux capabilities like CAP_SYS_ADMIN, CAP_SYS_PTRACE, or CAP_NET_ADMIN can "
                "enable container escapes. Mitigations include never running containers as privileged in production, "
                "dropping all unnecessary capabilities, using seccomp profiles to restrict system calls, enabling "
                "AppArmor/SELinux profiles, and running containers with read-only root filesystems."
            ),
        },
        {
            "topic": "Docker Socket Exposure",
            "text": (
                "Mounting the Docker socket (/var/run/docker.sock) inside a container effectively grants root access to "
                "the host system. An attacker who gains code execution inside such a container can use the Docker API to "
                "create new privileged containers that mount the host filesystem, effectively escaping the container. "
                "This is a common misconfiguration in CI/CD pipelines where Docker-in-Docker is used for building images. "
                "Detection involves scanning container configurations for volume mounts of the Docker socket, using "
                "admission controllers in Kubernetes to prevent such mounts, and implementing Docker socket proxies with "
                "restricted API access like Tecnativa's docker-socket-proxy."
            ),
        },
        # --- Container Security: Kubernetes ---
        {
            "topic": "Kubernetes RBAC Misconfiguration",
            "text": (
                "Kubernetes Role-Based Access Control (RBAC) misconfigurations are a leading cause of privilege escalation "
                "in cluster environments. Common issues include granting cluster-admin ClusterRole to service accounts, "
                "using overly broad wildcard permissions in Roles (verbs: ['*'], resources: ['*']), binding powerful roles "
                "to the default service account in a namespace, and allowing pod creation with escalation-enabling specs. "
                "An attacker with create pods permission can mount service account tokens, run privileged containers, "
                "or access the node's filesystem. Tools like kubectl-who-can, rakkess, and rbac-lookup help audit RBAC "
                "configurations. Mitigations include following least-privilege principles, disabling automounting of "
                "service account tokens, using OPA Gatekeeper to enforce pod security standards, and regularly auditing "
                "RBAC bindings for excessive permissions."
            ),
        },
        {
            "topic": "Kubernetes Pod Escape via Host Namespaces",
            "text": (
                "Kubernetes pods configured with hostPID, hostNetwork, or hostIPC share the respective Linux namespace "
                "with the host node, significantly weakening container isolation. A pod with hostPID: true can see and "
                "signal all processes on the node, potentially injecting code via /proc/<pid>/mem or nsenter. A pod with "
                "hostNetwork: true bypasses network policies and can sniff traffic on the node's network interfaces. "
                "Combined with a privileged security context, these settings allow complete node compromise. Pod Security "
                "Standards (PSS) at the Restricted level prevent these configurations. Admission controllers like "
                "Kyverno and OPA Gatekeeper should enforce policies that reject pods requesting host namespaces."
            ),
        },
        {
            "topic": "Kubernetes etcd Exposure",
            "text": (
                "etcd is the key-value store that holds all Kubernetes cluster state, including Secrets in base64 encoding. "
                "If etcd is exposed without authentication, an attacker can read all cluster secrets, modify workloads, "
                "and escalate to full cluster admin. By default, etcd listens on port 2379/2380 and some deployments "
                "inadvertently expose it to the network. Even with TLS, if client certificate authentication is not "
                "enforced, anonymous access may be possible. Mitigations include enabling etcd TLS with mutual "
                "authentication, encrypting secrets at rest using a KMS provider, restricting network access to etcd to "
                "only the API server nodes, and enabling audit logging for etcd access patterns."
            ),
        },
        # --- API Security ---
        {
            "topic": "Broken Object Level Authorization (BOLA)",
            "text": (
                "Broken Object Level Authorization, also known as Insecure Direct Object Reference (IDOR), is the most "
                "common API security vulnerability. It occurs when an API endpoint accepts an object identifier from the "
                "client (e.g., /api/users/123/orders) without verifying that the authenticated user has permission to "
                "access that specific object. Attackers enumerate IDs to access other users' data, orders, transactions, "
                "or personal information. APIs are particularly susceptible because they tend to expose more endpoints and "
                "object identifiers than traditional web applications. Mitigations include implementing authorization checks "
                "for every object access, using non-sequential UUIDs instead of auto-incrementing integers, implementing "
                "access control at the data layer, and adding rate limiting to prevent mass enumeration."
            ),
        },
        {
            "topic": "API Rate Limiting and Resource Exhaustion",
            "text": (
                "APIs without proper rate limiting are vulnerable to denial-of-service attacks, brute-force credential "
                "stuffing, and data scraping. Attackers exploit the lack of rate controls to send thousands of requests "
                "per second to authentication endpoints, enumerate valid user accounts, exhaust server resources, or "
                "scrape entire databases through paginated list endpoints. GraphQL APIs are particularly susceptible to "
                "resource exhaustion through deeply nested queries or batch query attacks that amplify a single HTTP "
                "request into thousands of database operations. Mitigations include implementing rate limiting per client "
                "IP and per API key, using query complexity analysis for GraphQL, setting maximum pagination limits, "
                "implementing request throttling, and deploying API gateways with built-in DDoS protection."
            ),
        },
        {
            "topic": "Mass Assignment Vulnerability in APIs",
            "text": (
                "Mass assignment occurs when an API automatically binds client-provided data to internal object properties "
                "without proper filtering. If a user registration endpoint accepts a JSON body, an attacker may add "
                "unexpected fields like 'role': 'admin' or 'isVerified': true that the backend framework blindly assigns "
                "to the user model. Modern frameworks like Ruby on Rails, Django, Express, and Spring Boot are all "
                "susceptible if developers don't explicitly whitelist allowed fields. This vulnerability is especially "
                "dangerous in APIs because they typically accept structured data formats. Mitigations include explicitly "
                "defining allowed fields for each operation, using separate DTOs for input and internal models, and "
                "implementing schema validation that rejects unknown properties."
            ),
        },
        # --- OAuth Vulnerabilities ---
        {
            "topic": "OAuth 2.0 Authorization Code Interception",
            "text": (
                "OAuth 2.0 authorization code flow can be vulnerable to code interception attacks where an attacker "
                "captures the authorization code before the legitimate client exchanges it for an access token. In mobile "
                "applications using custom URI schemes, a malicious app can register the same scheme and receive the "
                "redirect. The PKCE (Proof Key for Code Exchange) extension mitigates this by requiring the client to "
                "prove it initiated the authorization request using a code_verifier and code_challenge. Without PKCE, "
                "attackers can also exploit open redirectors in the authorization endpoint to redirect the code to an "
                "attacker-controlled domain. OAuth implementations should enforce PKCE for all clients, validate redirect "
                "URIs using exact string matching, and use short-lived authorization codes with one-time use enforcement."
            ),
        },
        {
            "topic": "OAuth Token Theft and Replay",
            "text": (
                "OAuth access tokens and refresh tokens are high-value targets for attackers. Token theft can occur through "
                "cross-site scripting (XSS) in the client application, insecure token storage in browser localStorage, "
                "token leakage in server logs or referrer headers, or man-in-the-middle attacks on non-HTTPS connections. "
                "Once stolen, bearer tokens can be replayed from any client without additional authentication. Sender-"
                "constrained tokens (DPoP - Demonstrating Proof of Possession) bind tokens to a cryptographic key, "
                "preventing replay from other clients. Mitigations include storing tokens in httpOnly secure cookies, "
                "implementing token rotation for refresh tokens, using short-lived access tokens (5-15 minutes), detecting "
                "token reuse (refresh token rotation with reuse detection), and implementing mutual TLS for API access."
            ),
        },
        {
            "topic": "OAuth Scope Escalation",
            "text": (
                "OAuth scope escalation occurs when an application obtains access to more resources or permissions than the "
                "user intended to authorize. This can happen when the authorization server doesn't properly validate "
                "requested scopes, when a client requests broad scopes that are rubber-stamped by users, or when token "
                "exchange processes don't properly downscope tokens. In multi-tenant environments, scope escalation can "
                "lead to cross-tenant data access. Attackers may also exploit the token exchange grant type (RFC 8693) to "
                "upgrade token permissions. Mitigations include implementing granular scopes, presenting clear consent "
                "screens showing specific permissions requested, validating scopes at the resource server level, using "
                "resource indicators (RFC 8707), and auditing granted scopes against actual API usage."
            ),
        },
        # --- JWT Attacks ---
        {
            "topic": "JWT Algorithm Confusion Attack",
            "text": (
                "The JWT algorithm confusion (or key confusion) attack exploits implementations that trust the 'alg' "
                "header in the token to determine the verification algorithm. When a server is configured to accept tokens "
                "signed with RS256 (asymmetric RSA), an attacker can change the algorithm to HS256 (symmetric HMAC) and "
                "sign the token using the server's RSA public key as the HMAC secret. Since the public key is often "
                "available, the attacker can forge valid tokens. The attack works because the verification code uses the "
                "same key for both RSA public key verification and HMAC secret comparison. Mitigations include explicitly "
                "specifying the expected algorithm during verification (never trusting the token header), using separate "
                "code paths for different algorithms, and migrating to libraries that require algorithm specification."
            ),
        },
        {
            "topic": "JWT None Algorithm Attack",
            "text": (
                "The JWT 'none' algorithm attack exploits implementations that accept tokens with 'alg': 'none' in the "
                "header, which indicates the token is unsigned. An attacker can take a valid JWT, decode it, modify the "
                "payload (e.g., changing the user ID or role to admin), set the algorithm to 'none', remove the signature, "
                "and submit the unsigned token. Vulnerable libraries accept this token as valid because the 'none' "
                "algorithm explicitly means no signature verification is required. Variations include 'None', 'NONE', and "
                "'nOnE' to bypass case-sensitive blacklists. Mitigations include explicitly rejecting the 'none' algorithm "
                "in token verification, using an allowlist of accepted algorithms, and ensuring the JWT library does not "
                "accept unsigned tokens by default."
            ),
        },
        {
            "topic": "JWT Claim Injection and jku/x5u Header Attacks",
            "text": (
                "JWT headers can contain a 'jku' (JSON Web Key Set URL) or 'x5u' (X.509 URL) parameter that specifies "
                "where to fetch the public key for signature verification. An attacker can point these URLs to their own "
                "server hosting a key pair they control, sign the token with their private key, and the victim server will "
                "fetch and use the attacker's public key for verification. Similarly, the 'kid' (Key ID) parameter can be "
                "injected with path traversal sequences or SQL injection payloads if used unsafely in key lookup logic. "
                "Mitigations include ignoring jku and x5u headers and using locally configured keys only, validating kid "
                "values against an allowlist, never using kid in database queries without sanitization, and implementing "
                "a static JWKS endpoint that the application controls."
            ),
        },
        {
            "topic": "JWT Token Lifetime and Revocation Issues",
            "text": (
                "JWTs are stateless by design, which means once issued they are valid until expiration regardless of "
                "whether the user's session has been terminated, their account has been disabled, or their permissions have "
                "changed. Long-lived JWTs (hours to days) create a window where compromised tokens remain usable. Unlike "
                "session-based authentication where the server can immediately invalidate a session, JWT revocation "
                "requires maintaining a token blocklist (negating the stateless benefit) or using short-lived tokens with "
                "refresh token rotation. Best practices include setting short expiration times (5-15 minutes), implementing "
                "refresh token rotation with reuse detection, maintaining a revocation list for compromised tokens using "
                "Redis or a similar fast store, including a jti (JWT ID) claim for tracking, and implementing token "
                "binding to prevent token theft replay."
            ),
        },
    ]

    records = []
    for i, wu in enumerate(writeups):
        full_text = f"{wu['topic']}\n\n{wu['text']}"
        cleaned = clean_text(full_text)
        if len(cleaned) >= 100:
            records.append({
                "id": f"secwriteup-{i}",
                "text": cleaned,
                "source": "security_writeups",
            })

    # Expand by repeating to build larger training volume
    expanded = []
    for rep in range(max_records // max(len(records), 1)):
        for r in records:
            expanded.append({
                "id": f"{r['id']}_v{rep}",
                "text": r["text"],
                "source": "security_writeups",
            })
    # Add remaining
    remainder = max_records - len(expanded)
    if remainder > 0:
        for r in records[:remainder]:
            expanded.append({
                "id": f"{r['id']}_extra",
                "text": r["text"],
                "source": "security_writeups",
            })

    if expanded:
        save_jsonl(expanded[:max_records], output_path)
    else:
        print("  Warning: No security writeup records generated.")


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
    mitre_path = "data/raw/mitre_attack.jsonl"
    owasp_path = "data/raw/owasp.jsonl"
    writeups_path = "data/raw/security_writeups.jsonl"

    collect_cve_descriptions(output_path=cve_path)
    collect_security_papers(output_path=papers_path)
    collect_ctf_writeups(output_path=ctf_path)
    collect_mitre_attack(output_path=mitre_path)
    collect_owasp_top10(output_path=owasp_path)
    collect_synthetic_security_writeups(output_path=writeups_path)

    # Merge into train/val splits
    merge_datasets(
        input_paths=[
            cve_path, papers_path, ctf_path,
            mitre_path, owasp_path, writeups_path,
        ],
        output_path="data/processed/train.jsonl",
    )

    print("\n" + "=" * 50)
    print("Data collection complete.")
    print("=" * 50)


if __name__ == "__main__":
    main()
