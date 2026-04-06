"""GhostLM data collection — downloads and preprocesses cybersecurity training data from public sources."""

import json
import os
import random
import re
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
    """Download and extract CVE descriptions from the NVD dataset.

    Attempts to load the CVEProject/nvdcve dataset from HuggingFace,
    falling back to mitre/cve if unavailable. Extracts CVE IDs and
    description text, cleans them, and saves as JSONL.

    Args:
        output_path: Destination path for the output JSONL file.
        max_records: Maximum number of CVE records to collect.
    """
    print("Collecting CVE descriptions...")
    records = []

    try:
        dataset = load_dataset("CVEProject/nvdcve", split="train", trust_remote_code=True)
    except Exception as e:
        print(f"  Warning: Could not load CVEProject/nvdcve: {e}")
        try:
            dataset = load_dataset("mitre/cve", split="train", trust_remote_code=True)
        except Exception as e2:
            print(f"  Warning: Could not load mitre/cve: {e2}")
            print("  Skipping CVE collection.")
            return

    for item in tqdm(dataset, desc="CVE records", total=min(len(dataset), max_records)):
        try:
            cve_id = item.get("cveId") or item.get("cve_id") or item.get("id", "")
            description = item.get("descriptions") or item.get("description") or ""

            if isinstance(description, list):
                description = " ".join(
                    d.get("value", "") for d in description if isinstance(d, dict)
                )
            elif isinstance(description, dict):
                description = description.get("value", "")

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

    if records:
        save_jsonl(records, output_path)
    else:
        print("  Warning: No valid CVE records collected.")


def collect_security_papers(
    output_path: str = "data/raw/papers.jsonl",
    max_records: int = 5000,
) -> None:
    """Download and extract cybersecurity research paper abstracts.

    Attempts to load the Gaborandi/cybersecurity-papers dataset, falling
    back to allenai/peS2o. Combines title and abstract into a single
    text field, cleans it, and saves as JSONL.

    Args:
        output_path: Destination path for the output JSONL file.
        max_records: Maximum number of paper records to collect.
    """
    print("Collecting security papers...")
    records = []

    try:
        dataset = load_dataset("Gaborandi/cybersecurity-papers", split="train", trust_remote_code=True)
    except Exception as e:
        print(f"  Warning: Could not load Gaborandi/cybersecurity-papers: {e}")
        try:
            dataset = load_dataset("allenai/peS2o", split="train", trust_remote_code=True)
        except Exception as e2:
            print(f"  Warning: Could not load allenai/peS2o: {e2}")
            print("  Skipping security papers collection.")
            return

    for i, item in tqdm(enumerate(dataset), desc="Papers", total=min(len(dataset), max_records)):
        try:
            title = item.get("title", "") or ""
            abstract = item.get("abstract", "") or ""

            combined = f"{title}\n\n{abstract}"
            cleaned = clean_text(combined)

            if len(cleaned) >= 100:
                records.append({
                    "id": i,
                    "text": cleaned,
                    "source": "papers",
                })

            if len(records) >= max_records:
                break
        except Exception:
            continue

    if records:
        save_jsonl(records, output_path)
    else:
        print("  Warning: No valid paper records collected.")


def collect_ctf_writeups(output_path: str = "data/raw/ctf.jsonl") -> None:
    """Download and extract CTF (Capture The Flag) writeup texts.

    Attempts to load the mrcabbage972/ctf-writeups dataset from HuggingFace.
    Extracts writeup text, cleans it, and saves as JSONL.

    Args:
        output_path: Destination path for the output JSONL file.
    """
    print("Collecting CTF writeups...")
    records = []

    try:
        dataset = load_dataset("mrcabbage972/ctf-writeups", split="train", trust_remote_code=True)
    except Exception as e:
        print(f"  Warning: Could not load mrcabbage972/ctf-writeups: {e}")
        print("  Skipping CTF writeup collection.")
        return

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

    if records:
        save_jsonl(records, output_path)
    else:
        print("  Warning: No valid CTF writeup records collected.")


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
