#!/usr/bin/env python3
"""Pull a curated set of security-relevant IETF RFCs as plain text.

The IETF publishes RFCs as plain ASCII text at rfc-editor.org. Many of
them are foundational security documents (TLS, OAuth, JWT, DNSSEC, HTTP
auth) that GhostLM has never seen. The protocol-spec register is
distinct from the writeup / advisory / textbook registers we already
have.

This script pulls a hand-picked list of ~50 high-value RFCs (TLS, JOSE,
OAuth, DNSSEC, HTTP Sec, IPsec, SSH, PKI, etc.). Public domain.

Output: ``data/raw/rfcs.jsonl`` with the standard
``{"id", "source", "text"}`` schema. Source is ``rfcs``.
"""

from __future__ import annotations

import argparse
import json
import time
import urllib.request
from pathlib import Path


# Curated security-relevant RFCs. Numbers are stable identifiers.
SECURITY_RFCS = [
    # TLS / SSL
    8446,  # TLS 1.3
    5246,  # TLS 1.2
    7457,  # TLS attack summary
    7525,  # TLS recommendations
    # JOSE / JWT / OAuth
    7515,  # JSON Web Signature (JWS)
    7516,  # JSON Web Encryption (JWE)
    7517,  # JSON Web Key (JWK)
    7518,  # JSON Web Algorithms (JWA)
    7519,  # JSON Web Token (JWT)
    6749,  # OAuth 2.0
    6750,  # OAuth 2.0 Bearer Token Usage
    7636,  # OAuth PKCE
    8252,  # OAuth for native apps
    8628,  # OAuth Device Flow
    9068,  # JWT Profile for OAuth Access Tokens
    # HTTP security
    6797,  # HSTS
    7034,  # X-Frame-Options
    7469,  # HPKP (Public Key Pinning)
    7615,  # HTTP authentication-info
    7616,  # HTTP Digest auth
    7617,  # HTTP Basic auth
    9110,  # HTTP semantics
    # DNS
    4033,  # DNSSEC introduction
    4034,  # DNSSEC resource records
    4035,  # DNSSEC protocol modifications
    7858,  # DNS over TLS
    8484,  # DNS over HTTPS
    # IPsec
    4301,  # IPsec architecture
    4302,  # AH
    4303,  # ESP
    7296,  # IKEv2
    # SSH
    4251,  # SSH protocol architecture
    4252,  # SSH authentication
    4253,  # SSH transport layer
    4254,  # SSH connection protocol
    # PKI / X.509
    5280,  # X.509 PKI
    6960,  # OCSP
    # Crypto primitives
    8439,  # ChaCha20 + Poly1305
    7748,  # Curve25519 / Curve448
    8032,  # EdDSA
    8017,  # PKCS#1 RSA
    # Email security
    6376,  # DKIM
    7208,  # SPF
    7489,  # DMARC
    # Misc auth + API
    7591,  # OAuth 2.0 Dynamic Client Registration
    8259,  # JSON
    9325,  # TLS 1.2 BCP (deprecates 7525)
    # Vulnerability disclosure
    9116,  # security.txt
]


def parse_args() -> argparse.Namespace:
    """CLI args."""
    p = argparse.ArgumentParser(description="Collect curated security RFCs")
    p.add_argument("--out", default="data/raw/rfcs.jsonl")
    p.add_argument("--request-delay", type=float, default=1.0,
                   help="Seconds between RFC fetches (be polite)")
    p.add_argument("--max-chars", type=int, default=20000,
                   help="Truncate long RFCs (TLS 1.3 spec is huge)")
    p.add_argument("--min-chars", type=int, default=500)
    return p.parse_args()


def fetch_rfc(num: int, timeout: int = 30) -> str:
    """Fetch one RFC plain text from rfc-editor.org."""
    url = f"https://www.rfc-editor.org/rfc/rfc{num}.txt"
    req = urllib.request.Request(url, headers={"User-Agent": "GhostLM-RFCFetcher/0.6"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read().decode("utf-8", errors="ignore")


def clean_rfc(text: str) -> str:
    """Strip RFC page-break / form-feed artifacts and the standard
    page header/footer noise."""
    # Form feeds and CRs
    text = text.replace("\r", "").replace("\f", "")
    # Drop "[Page N]" markers and page-header lines that follow
    lines = text.split("\n")
    out_lines = []
    skip_until_blank = False
    for line in lines:
        if "[Page " in line and line.rstrip().endswith("]"):
            skip_until_blank = True
            continue
        if skip_until_blank:
            if line.strip() == "":
                skip_until_blank = False
            continue
        out_lines.append(line.rstrip())
    cleaned = "\n".join(out_lines)
    # Collapse 3+ blank lines
    while "\n\n\n" in cleaned:
        cleaned = cleaned.replace("\n\n\n", "\n\n")
    return cleaned.strip()


def main() -> None:
    """Fetch and write all curated RFCs (resume-safe)."""
    args = parse_args()
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    seen: set = set()
    if out_path.exists():
        with out_path.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    rec = json.loads(line)
                    if rec.get("id"):
                        seen.add(rec["id"])
        print(f"  resume: {len(seen)} RFCs already done")

    out_fh = out_path.open("a", encoding="utf-8", buffering=1)
    written = 0
    skipped = 0
    failed = 0

    for i, num in enumerate(SECURITY_RFCS):
        rfc_id = f"RFC-{num}"
        if rfc_id in seen:
            continue
        try:
            raw = fetch_rfc(num)
        except Exception as e:
            print(f"  RFC {num}: fetch error {e}")
            failed += 1
            time.sleep(args.request_delay)
            continue

        cleaned = clean_rfc(raw)
        if len(cleaned) < args.min_chars:
            skipped += 1
            time.sleep(args.request_delay)
            continue
        if len(cleaned) > args.max_chars:
            cleaned = cleaned[: args.max_chars].rsplit("\n\n", 1)[0]

        # Use the first non-empty line as title (RFC top has the title)
        first_lines = [l for l in cleaned.split("\n")[:30] if l.strip()]
        title = first_lines[0] if first_lines else f"RFC {num}"

        rec = {
            "id": rfc_id,
            "source": "rfcs",
            "text": f"{title}\n\n{cleaned}",
        }
        out_fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        written += 1
        if (i + 1) % 10 == 0:
            print(f"  [{i + 1}/{len(SECURITY_RFCS)}] written={written} failed={failed}")
        time.sleep(args.request_delay)
    out_fh.close()

    print(f"\nDone. Wrote {written} RFCs to {out_path}")
    if skipped:
        print(f"  Skipped {skipped} too-short")
    if failed:
        print(f"  Failed {failed}")


if __name__ == "__main__":
    main()
