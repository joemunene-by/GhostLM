#!/usr/bin/env python3
"""Generate synthetic cybersecurity writeups via local Ollama (Qwen2.5-Coder-14B).

Purpose: produce diverse, procedural, in-domain training data to rebalance the
GhostLM corpus (currently ~75% templated NVD descriptions). Output is appended
to data/raw/ctf_synth.jsonl in the existing schema and is resume-safe.

Usage:
    ollama pull qwen2.5-coder:14b
    python3 scripts/gen_synthetic.py

Tune TARGET_COUNT down (e.g. 20) for a smoke test before committing to the full run.
"""

import json
import random
import re
import time
import urllib.request
from pathlib import Path

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "qwen2.5-coder:14b"
OUTPUT_PATH = Path("data/raw/ctf_synth.jsonl")
TARGET_COUNT = 3000
SEED = 42


TOPICS = {
    "web": [
        "SQL injection via UNION-based extraction on a vulnerable login page",
        "blind time-based SQL injection inferring data one bit at a time",
        "SQLi WAF bypass via double URL encoding and case variation",
        "second-order SQL injection in a profile-update flow",
        "NoSQL injection against MongoDB via $ne and $regex operators",
        "reflected XSS in a search parameter with a basic filter",
        "stored XSS via mutation XSS (mXSS) in an innerHTML sink",
        "DOM-based XSS through URL hash and document.write",
        "CSP bypass via an exploitable JSONP endpoint",
        "classic SSRF reaching AWS IMDSv1 for instance credentials",
        "blind SSRF detected via out-of-band DNS",
        "SSRF via the gopher:// scheme to trigger a Redis RCE",
        "JWT algorithm confusion attack (HS256 signed with the RSA public key)",
        "OAuth redirect_uri bypass via parameter pollution",
        "session fixation leading to account takeover",
        "file upload bypass via PNG/PHP polyglot",
        "ZIP slip in an archive-import feature",
        "Java deserialization with ysoserial CommonsCollections gadget",
        "PHP unserialize exploitation via the phar:// wrapper",
        "Python pickle deserialization abuse of a cache file",
        "Jinja2 SSTI escaping the sandbox to reach os.popen",
        "prototype pollution in a JavaScript object-merge helper",
        "HTTP request smuggling (CL.TE) on a reverse proxy",
        "XXE retrieving /etc/passwd via file://",
        "CSRF against a SameSite=Lax misconfiguration",
        "CORS misconfiguration allowing credential theft",
        "business-logic flaw in a multi-step checkout flow",
        "path traversal in a file-download handler",
        "TOCTOU race condition in a withdrawal endpoint",
    ],
    "pwn": [
        "stack buffer overflow with no canary and NX disabled",
        "stack overflow bypassing canary via a format-string leak",
        "ret2libc against an ASLR-enabled binary after a libc leak",
        "heap use-after-free overwriting a function pointer",
        "heap double-free via tcache poisoning on glibc 2.31",
        "house of force on older glibc",
        "unsafe unlink via overlapping chunks",
        "format string: leaking the stack canary and libc base",
        "format string: overwriting a GOT entry for control-flow hijack",
        "ROP chain calling mprotect then pivoting to shellcode",
        "SROP on a static binary",
        "integer overflow causing a short heap allocation and corruption",
        "off-by-one poison-null-byte exploitation on glibc heap",
        "ret2dlresolve to invoke execve without leaking libc",
    ],
    "re": [
        "reverse engineering an ELF crackme in Ghidra",
        "unpacking a UPX-packed binary and analyzing the dumped code",
        "bypassing ptrace-based anti-debug by patching the check",
        "defeating timing anti-debug checks with Frida",
        "static analysis of an obfuscated .NET binary using dnSpy",
        "Android APK reversing with jadx and SSL-pinning bypass via Frida",
        "symbolic execution of a validation routine with angr",
    ],
    "crypto": [
        "low-exponent RSA attack (e=3) via cube root",
        "Hastad broadcast attack across three RSA public keys",
        "Wiener's attack against small private-exponent RSA",
        "AES-ECB pattern disclosure via chosen plaintext",
        "AES-CBC padding oracle attack using POET",
        "hash length extension attack on SHA-256",
        "ECDSA nonce reuse leading to private key recovery",
        "XOR stream-cipher key reuse solved by crib dragging",
        "predicting Mersenne Twister state from observed outputs",
        "JWT alg=none bypass on a hand-rolled verifier",
    ],
    "forensics": [
        "PCAP analysis of a DNS-tunneling exfil channel in Wireshark",
        "extracting transmitted files from TCP streams",
        "memory forensics with Volatility identifying process injection",
        "disk carving a suspicious image with foremost",
        "Windows Event Log analysis reconstructing lateral movement",
        "ARP spoofing + SSL stripping: reconstructing captured credentials",
        "WPA2 handshake capture and cracking with hashcat",
    ],
    "stego": [
        "extracting an LSB-hidden message from a PNG with zsteg",
        "detecting zero-width Unicode characters hiding text",
        "decoding an audio spectrogram puzzle in Audacity",
        "polyglot PNG+ZIP carved apart with binwalk",
        "EXIF metadata containing an encoded payload",
    ],
    "privesc": [
        "Linux SUID abuse via a custom setuid binary that calls system()",
        "sudo misconfig: exploiting tar --checkpoint-action as root",
        "Linux capabilities abuse (cap_dac_read_search) to read /etc/shadow",
        "cron job PATH hijacking for root escalation",
        "Windows service with unquoted path and a writable intermediate dir",
        "Windows service binary replacement via weak DACL",
        "DLL hijacking against a service that loads from a writable dir",
        "SeImpersonatePrivilege abuse via a Juicy-Potato-style attack",
        "AlwaysInstallElevated registry abuse",
        "Docker container escape via a mounted docker.sock",
    ],
    "cloud": [
        "AWS IAM role chaining via sts:AssumeRole misconfig",
        "EC2 IMDSv1 abuse for role credential theft",
        "S3 bucket enumeration uncovering a world-readable sensitive object",
        "Kubernetes RBAC misconfig letting a pod reach cluster-admin",
        "Terraform state file in a public S3 containing plaintext secrets",
    ],
    "recon": [
        "GitHub dorking uncovering leaked API keys in commit history",
        "subdomain enumeration with amass and DNS brute forcing",
        "certificate transparency log mining for hidden infrastructure",
        "Shodan query chaining to fingerprint a specific appliance",
    ],
}


REGISTERS = [
    ("first_person_retrospective", 0.40,
     "First-person retrospective writeup, past tense. Varied openings; example: "
     "'The binary exploitation challenge involved a heap UAF where...'"),
    ("third_person_explanatory", 0.25,
     "Third-person explanatory, present tense. Example: 'A format string "
     "vulnerability occurs when user input is passed as the format argument to "
     "printf...'"),
    ("second_person_walkthrough", 0.20,
     "Second-person walkthrough, imperative. Example: 'To exploit this, first "
     "identify the injection point by...'"),
    ("qa_format", 0.10,
     "Q&A format. Start with 'Q: <specific question>\\n\\nA: <detailed answer>'."),
    ("scenario", 0.05,
     "Scenario-driven. Open by describing the lab/environment, then walk through "
     "the attack path."),
]


PROMPT = """You are generating training data for a small cybersecurity language model.

Produce ONE self-contained technical writeup on the topic and in the register specified below.

Requirements:
- 150 to 500 words. Single block of prose. No markdown headers, no bullet lists.
- Technically accurate. Name real tools where relevant (Ghidra, sqlmap, Burp Suite, pwntools, Volatility, Wireshark, Metasploit, hashcat, Frida, gdb/pwndbg, radare2, angr, binwalk, impacket, BloodHound).
- Use precise vocabulary: actual function names, real CLI flags, concrete exploit primitives.
- Do NOT invent CVE numbers. Do NOT invent VDB-XXXXX identifiers. Do NOT write NVD-style boilerplate such as "allows remote attackers to cause a denial of service via a crafted request".
- Well-known named CVEs are OK: Log4Shell (CVE-2021-44228), EternalBlue (MS17-010 / CVE-2017-0144), Heartbleed (CVE-2014-0160), Shellshock (CVE-2014-6271), Dirty COW (CVE-2016-5195), PrintNightmare (CVE-2021-34527).
- Vary openings; do NOT begin with "This challenge" every time.
- Do NOT reference fictional CTF event names or specific fictional flag strings.
- Frame offensive content as CTF retrospective, defensive/educational explanation, or lab walkthrough. Never as targeting a real system.

Topic: {topic}
Register: {register}

Output only the writeup. No preamble, no meta-commentary, no disclaimers."""


BANNED = [
    "allows remote attackers to",
    "i cannot",
    "i'm sorry",
    "i will not",
    "i can't help",
    "as an ai",
    "ethical guidelines",
    "i am unable",
]
VDB_RE = re.compile(r"\bVDB-\d+\b")
CVE_RE = re.compile(r"\bCVE-\d{4}-\d{4,}\b")
OK_CVES = {
    "CVE-2021-44228",
    "CVE-2014-0160",
    "CVE-2014-6271",
    "CVE-2016-5195",
    "CVE-2021-34527",
    "CVE-2017-0144",
}


def accept(text: str, prior_grams: set) -> tuple[bool, str]:
    wc = len(text.split())
    if wc < 120:
        return False, f"short({wc})"
    if wc > 600:
        return False, f"long({wc})"
    low = text.lower()
    for b in BANNED:
        if b in low:
            return False, f"banned:{b}"
    if VDB_RE.search(text):
        return False, "fake_vdb"
    for m in CVE_RE.findall(text):
        if m not in OK_CVES:
            return False, f"fake_cve:{m}"
    toks = [t.lower() for t in re.findall(r"[A-Za-z]+", text)]
    grams = {tuple(toks[i : i + 4]) for i in range(len(toks) - 3)}
    if grams and prior_grams:
        overlap = len(grams & prior_grams) / max(1, len(grams))
        if overlap > 0.5:
            return False, f"similar({overlap:.0%})"
    return True, "ok"


def ollama_gen(prompt: str, temp: float = 0.85) -> str:
    data = json.dumps(
        {
            "model": MODEL,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": temp, "top_p": 0.9, "num_predict": 400},
        }
    ).encode()
    req = urllib.request.Request(
        OLLAMA_URL, data=data, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=300) as r:
        return json.loads(r.read()).get("response", "").strip()


def pick_register() -> tuple[str, str]:
    r = random.random()
    cum = 0.0
    for name, w, desc in REGISTERS:
        cum += w
        if r < cum:
            return name, desc
    return REGISTERS[-1][0], REGISTERS[-1][2]


def main() -> None:
    random.seed(SEED)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    prior_grams: set = set()
    accepted = 0
    if OUTPUT_PATH.exists():
        with OUTPUT_PATH.open() as f:
            for line in f:
                d = json.loads(line)
                accepted += 1
                toks = [t.lower() for t in re.findall(r"[A-Za-z]+", d["text"])]
                prior_grams.update(
                    tuple(toks[i : i + 4]) for i in range(len(toks) - 3)
                )
        print(f"Resuming from {accepted} existing entries.")

    flat = [(c, t) for c, ts in TOPICS.items() for t in ts]
    t0 = time.time()
    rej = 0
    nid = accepted

    while accepted < TARGET_COUNT:
        cat, topic = random.choice(flat)
        reg_name, reg_desc = pick_register()
        try:
            text = ollama_gen(PROMPT.format(topic=topic, register=reg_desc))
        except Exception as e:
            print(f"  gen error: {e}; sleeping 5s")
            time.sleep(5)
            continue

        ok, reason = accept(text, prior_grams)
        if not ok:
            rej += 1
            if rej % 25 == 0:
                print(f"  rejected {rej} (last: {reason})")
            continue

        rec = {
            "id": f"synthetic-{nid}",
            "text": text,
            "source": "synthetic",
            "topic": cat,
            "subtopic": topic,
            "register": reg_name,
        }
        with OUTPUT_PATH.open("a") as f:
            f.write(json.dumps(rec) + "\n")

        toks = [t.lower() for t in re.findall(r"[A-Za-z]+", text)]
        prior_grams.update(tuple(toks[i : i + 4]) for i in range(len(toks) - 3))
        accepted += 1
        nid += 1

        if accepted % 10 == 0:
            el = time.time() - t0
            rate = accepted / max(1, el)
            eta = (TARGET_COUNT - accepted) / max(1e-9, rate) / 60
            print(
                f"  {accepted}/{TARGET_COUNT}  rej={rej}  "
                f"{rate * 60:.1f}/min  eta={eta:.0f}m"
            )

    print(f"\nDone. {accepted} writeups saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
