#!/usr/bin/env python3
"""Pull a curated set of NIST SP 800 publications as plain text.

NIST Special Publication 800 series is the canonical US-government
infosec reference. Risk Management Framework (SP 800-37), Security
and Privacy Controls (SP 800-53), Digital Identity Guidelines
(SP 800-63), Incident Handling (SP 800-61), Penetration Testing
(SP 800-115), Secure Software Development (SP 800-218) — these are
the documents every security program is benchmarked against, and
none of them are in the v0.9 corpus.

Each publication is a long PDF (50-500 pages). We use pymupdf
(fitz) to extract text, then chunk into ~12K-char records so a
single huge document doesn't dominate the corpus.

Source: nvlpubs.nist.gov direct PDFs. NIST publications are US
government works and in the public domain — safe to redistribute.

Output: ``data/raw/nist_sp800.jsonl`` with the standard
``{"id", "source", "text"}`` schema. Source is ``nist_sp800``.

Requires: ``pip install pymupdf`` (a.k.a. fitz).
"""

from __future__ import annotations

import argparse
import json
import re
import time
import urllib.request
from pathlib import Path


# Curated SP 800 publications. Each entry: (label, url).
# All hosted at nvlpubs.nist.gov in PDF form, US gov public domain.
SP800_PUBLICATIONS = [
    ("SP 800-30 r1 — Guide for Conducting Risk Assessments",
     "https://nvlpubs.nist.gov/nistpubs/Legacy/SP/nistspecialpublication800-30r1.pdf"),
    ("SP 800-37 r2 — Risk Management Framework",
     "https://nvlpubs.nist.gov/nistpubs/SpecialPublications/NIST.SP.800-37r2.pdf"),
    ("SP 800-53 r5 — Security and Privacy Controls",
     "https://nvlpubs.nist.gov/nistpubs/SpecialPublications/NIST.SP.800-53r5.pdf"),
    ("SP 800-53A r5 — Assessing Security and Privacy Controls",
     "https://nvlpubs.nist.gov/nistpubs/SpecialPublications/NIST.SP.800-53Ar5.pdf"),
    ("SP 800-53B — Control Baselines for Information Systems",
     "https://nvlpubs.nist.gov/nistpubs/SpecialPublications/NIST.SP.800-53B.pdf"),
    ("SP 800-61 r2 — Computer Security Incident Handling Guide",
     "https://nvlpubs.nist.gov/nistpubs/SpecialPublications/NIST.SP.800-61r2.pdf"),
    ("SP 800-63-3 — Digital Identity Guidelines",
     "https://nvlpubs.nist.gov/nistpubs/SpecialPublications/NIST.SP.800-63-3.pdf"),
    ("SP 800-63A — Enrollment and Identity Proofing",
     "https://nvlpubs.nist.gov/nistpubs/SpecialPublications/NIST.SP.800-63a.pdf"),
    ("SP 800-63B — Authentication and Lifecycle Management",
     "https://nvlpubs.nist.gov/nistpubs/SpecialPublications/NIST.SP.800-63b.pdf"),
    ("SP 800-63C — Federation and Assertions",
     "https://nvlpubs.nist.gov/nistpubs/SpecialPublications/NIST.SP.800-63c.pdf"),
    ("SP 800-92 — Guide to Computer Security Log Management",
     "https://nvlpubs.nist.gov/nistpubs/Legacy/SP/nistspecialpublication800-92.pdf"),
    ("SP 800-94 — Guide to Intrusion Detection and Prevention Systems",
     "https://nvlpubs.nist.gov/nistpubs/Legacy/SP/nistspecialpublication800-94.pdf"),
    ("SP 800-115 — Technical Guide to Information Security Testing and Assessment",
     "https://nvlpubs.nist.gov/nistpubs/Legacy/SP/nistspecialpublication800-115.pdf"),
    ("SP 800-126 r3 — Security Content Automation Protocol (SCAP)",
     "https://nvlpubs.nist.gov/nistpubs/SpecialPublications/NIST.SP.800-126r3.pdf"),
    ("SP 800-145 — Definition of Cloud Computing",
     "https://nvlpubs.nist.gov/nistpubs/Legacy/SP/nistspecialpublication800-145.pdf"),
    ("SP 800-146 — Cloud Computing Synopsis and Recommendations",
     "https://nvlpubs.nist.gov/nistpubs/Legacy/SP/nistspecialpublication800-146.pdf"),
    ("SP 800-150 — Guide to Cyber Threat Information Sharing",
     "https://nvlpubs.nist.gov/nistpubs/SpecialPublications/NIST.SP.800-150.pdf"),
    ("SP 800-160 v1 r1 — Engineering Trustworthy Secure Systems",
     "https://nvlpubs.nist.gov/nistpubs/SpecialPublications/NIST.SP.800-160v1r1.pdf"),
    ("SP 800-160 v2 r1 — Cyber-Resilient Systems",
     "https://nvlpubs.nist.gov/nistpubs/SpecialPublications/NIST.SP.800-160v2r1.pdf"),
    ("SP 800-161 r1 — Cybersecurity Supply Chain Risk Management",
     "https://nvlpubs.nist.gov/nistpubs/SpecialPublications/NIST.SP.800-161r1.pdf"),
    ("SP 800-171 r3 — Protecting CUI in Nonfederal Systems",
     "https://nvlpubs.nist.gov/nistpubs/SpecialPublications/NIST.SP.800-171r3.pdf"),
    ("SP 800-181 r1 — NICE Cybersecurity Workforce Framework",
     "https://nvlpubs.nist.gov/nistpubs/SpecialPublications/NIST.SP.800-181r1.pdf"),
    ("SP 800-184 — Guide for Cybersecurity Event Recovery",
     "https://nvlpubs.nist.gov/nistpubs/SpecialPublications/NIST.SP.800-184.pdf"),
    ("SP 800-207 — Zero Trust Architecture",
     "https://nvlpubs.nist.gov/nistpubs/SpecialPublications/NIST.SP.800-207.pdf"),
    ("SP 800-218 — Secure Software Development Framework",
     "https://nvlpubs.nist.gov/nistpubs/SpecialPublications/NIST.SP.800-218.pdf"),
    ("SP 800-228 — Guide to Operational Technology Security",
     "https://nvlpubs.nist.gov/nistpubs/SpecialPublications/NIST.SP.800-82r3.pdf"),
]


def parse_args() -> argparse.Namespace:
    """CLI args."""
    p = argparse.ArgumentParser(description="Pull NIST SP 800 publications")
    p.add_argument("--out", default="data/raw/nist_sp800.jsonl")
    p.add_argument("--cache-dir", default="data/raw/.nist_pdf_cache")
    p.add_argument("--chunk-chars", type=int, default=12000,
                   help="Split each PDF into chunks of this size for training")
    p.add_argument("--min-chunk-chars", type=int, default=500)
    p.add_argument("--request-delay", type=float, default=2.0,
                   help="Delay between PDF downloads")
    return p.parse_args()


def download_pdf(url: str, cache: Path, timeout: int = 120) -> Path | None:
    """Download (or read cached) PDF."""
    name = url.rstrip("/").split("/")[-1]
    p = cache / name
    if p.exists() and p.stat().st_size > 1000:
        return p
    cache.mkdir(parents=True, exist_ok=True)
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "GhostLM-NIST/0.9"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = resp.read()
        p.write_bytes(data)
        return p
    except Exception as e:
        print(f"  download failed: {e}")
        return None


def extract_text(pdf_path: Path) -> str:
    """Extract text from a PDF using pymupdf (fitz)."""
    import fitz  # type: ignore
    doc = fitz.open(pdf_path)
    parts = []
    for page in doc:
        parts.append(page.get_text("text"))
    doc.close()
    return "\n".join(parts)


def clean_text(text: str) -> str:
    """Strip page-header / footer noise and collapse whitespace."""
    # Strip "U.S. Department of Commerce" boilerplate, page numbers, etc.
    text = re.sub(r"\n[ \t]*Page \d+( of \d+)?[ \t]*\n", "\n", text)
    text = re.sub(r"\n[ \t]*\d+[ \t]*\n", "\n", text)  # naked page numbers
    # Collapse 3+ blank lines
    while "\n\n\n" in text:
        text = text.replace("\n\n\n", "\n\n")
    return text.strip()


def chunk_text(text: str, chunk_chars: int, min_chunk_chars: int) -> list[str]:
    """Split into ~chunk_chars chunks at paragraph boundaries."""
    if len(text) <= chunk_chars:
        return [text] if len(text) >= min_chunk_chars else []
    paragraphs = text.split("\n\n")
    chunks: list[str] = []
    cur = ""
    for para in paragraphs:
        if len(cur) + len(para) + 2 > chunk_chars and cur:
            if len(cur) >= min_chunk_chars:
                chunks.append(cur.strip())
            cur = para
        else:
            cur = (cur + "\n\n" + para) if cur else para
    if cur and len(cur) >= min_chunk_chars:
        chunks.append(cur.strip())
    return chunks


def main() -> None:
    """Pull every publication, extract text, chunk, write JSONL."""
    args = parse_args()
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cache = Path(args.cache_dir)

    seen: set = set()
    if out_path.exists():
        with out_path.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    rec = json.loads(line)
                    if rec.get("id"):
                        seen.add(rec["id"])
        print(f"  resume: {len(seen)} chunks already on disk")

    out_fh = out_path.open("a", encoding="utf-8", buffering=1)
    pubs_done = 0
    chunks_written = 0
    failed = 0
    for label, url in SP800_PUBLICATIONS:
        pub_id = label.split(" — ")[0].replace(" ", "_").replace(".", "")
        # Skip if all expected chunks already on disk
        already = sum(1 for s in seen if s.startswith(pub_id + "_"))
        if already > 0:
            print(f"  {label}: {already} chunks already on disk, skipping")
            pubs_done += 1
            continue
        print(f"\n  {label}")
        pdf = download_pdf(url, cache)
        if not pdf:
            failed += 1
            continue
        try:
            text = extract_text(pdf)
        except ImportError:
            print("  pymupdf not installed (pip install pymupdf); aborting")
            return
        except Exception as e:
            print(f"  extract failed: {e}")
            failed += 1
            continue
        text = clean_text(text)
        chunks = chunk_text(text, args.chunk_chars, args.min_chunk_chars)
        if not chunks:
            print(f"  no usable chunks (len={len(text)})")
            continue
        for i, chunk in enumerate(chunks):
            cid = f"{pub_id}_{i:03d}"
            rec = {
                "id": cid,
                "source": "nist_sp800",
                "text": f"{label}\n\n{chunk}",
                "publication": label,
                "chunk_idx": i,
                "url": url,
            }
            out_fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            chunks_written += 1
        print(f"  {len(chunks)} chunks written")
        pubs_done += 1
        time.sleep(args.request_delay)

    out_fh.close()
    print(f"\nDone. {pubs_done} publications, {chunks_written} chunks to {out_path}")
    if failed:
        print(f"  failed {failed}")


if __name__ == "__main__":
    main()
