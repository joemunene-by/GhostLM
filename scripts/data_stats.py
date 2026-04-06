"""GhostLM data statistics — analyzes token counts, vocabulary coverage, and source distribution."""

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path

try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


def parse_args():
    """Parse command-line arguments for data statistics analysis.

    Returns:
        argparse.Namespace with all parsed analysis arguments.
    """
    parser = argparse.ArgumentParser(description="Analyze GhostLM training data statistics")

    parser.add_argument(
        "--train",
        type=str,
        default="data/processed/train.jsonl",
        help="Path to training JSONL file",
    )
    parser.add_argument(
        "--val",
        type=str,
        default="data/processed/val.jsonl",
        help="Path to validation JSONL file",
    )
    parser.add_argument(
        "--raw-dir",
        type=str,
        default="data/raw",
        help="Directory containing raw data files",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Generate distribution charts",
    )

    return parser.parse_args()


def load_jsonl(path: str) -> list:
    """Load and return records from a JSONL file.

    Args:
        path: Path to the JSONL file.

    Returns:
        List of dictionaries parsed from each line.
    """
    records = []
    p = Path(path)
    if not p.exists():
        return records
    with open(p, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def analyze_records(records: list, name: str) -> dict:
    """Analyze a list of records and compute detailed statistics.

    Args:
        records: List of record dictionaries with "text" and "source" fields.
        name: Human-readable name for this split (e.g. "Train").

    Returns:
        Dictionary containing all computed statistics.
    """
    if not records:
        print(f"  ── {name} Split ──────────────────────")
        print("  No records found.")
        return {}

    lengths = [len(r.get("text", "")) for r in records]
    sources = Counter(r.get("source", "unknown") for r in records)
    total_chars = sum(lengths)
    estimated_tokens = total_chars // 4

    source_str = " | ".join(f"{k}: {v:,}" for k, v in sources.most_common())

    print(f"  ── {name} Split {'─' * (37 - len(name))}")
    print(f"  Records:        {len(records):,}")
    print(f"  Sources:        {source_str}")
    print(f"  Avg length:     {sum(lengths) // len(lengths):,} chars")
    print(f"  Total tokens:   ~{estimated_tokens:,} (estimated)")
    print(f"  Shortest:       {min(lengths):,} chars")
    print(f"  Longest:        {max(lengths):,} chars")
    print()

    return {
        "name": name,
        "records": len(records),
        "sources": dict(sources),
        "avg_length": sum(lengths) // len(lengths),
        "total_chars": total_chars,
        "estimated_tokens": estimated_tokens,
        "min_length": min(lengths),
        "max_length": max(lengths),
        "lengths": lengths,
    }


def analyze_vocabulary(records: list) -> dict:
    """Analyze word frequencies and cybersecurity term coverage.

    Args:
        records: List of record dictionaries with "text" field.

    Returns:
        Dictionary with vocabulary statistics.
    """
    all_text = " ".join(r.get("text", "") for r in records)
    words = all_text.lower().split()
    word_counts = Counter(words)

    unique_words = len(word_counts)
    top_20 = word_counts.most_common(20)

    security_terms = [
        "vulnerability", "attacker", "remote", "arbitrary", "overflow",
        "injection", "execute", "denial", "service", "exploit",
        "malware", "authentication", "privilege", "escalation", "bypass",
    ]

    print(f"  ── Vocabulary Analysis {'─' * 17}")
    print(f"  Unique words:     {unique_words:,}")
    print(f"  Total words:      {len(words):,}")
    print()
    print(f"  Top 10 words:")
    for word, count in top_20[:10]:
        print(f"    {word:<20} {count:>6,}")
    print()

    print(f"  Cybersecurity terms found:")
    for term in security_terms:
        count = word_counts.get(term, 0)
        if count > 0:
            print(f"    {term:<20} {count:>6,}")
    print()

    return {
        "unique_words": unique_words,
        "total_words": len(words),
        "top_20": top_20,
        "security_terms": {t: word_counts.get(t, 0) for t in security_terms},
    }


def main():
    """Run the full GhostLM data statistics analysis.

    Loads train and validation splits, computes record and vocabulary
    statistics, and optionally generates distribution charts.
    """
    args = parse_args()

    print("=" * 45)
    print("GhostLM Data Statistics")
    print("=" * 45)
    print()

    # Load splits
    train_records = load_jsonl(args.train)
    val_records = load_jsonl(args.val)

    # Analyze splits
    train_stats = analyze_records(train_records, "Train")
    val_stats = analyze_records(val_records, "Validation")

    # Analyze vocabulary
    vocab_stats = analyze_vocabulary(train_records)

    # Overall summary
    total_records = len(train_records) + len(val_records)
    total_tokens = train_stats.get("estimated_tokens", 0) + val_stats.get("estimated_tokens", 0)
    all_sources = set()
    all_sources.update(train_stats.get("sources", {}).keys())
    all_sources.update(val_stats.get("sources", {}).keys())

    train_exists = Path(args.train).exists()
    val_exists = Path(args.val).exists()
    ready = train_exists and val_exists and total_records > 0

    print(f"  ── Overall Summary {'─' * 22}")
    print(f"  Total dataset:    {total_records:,} records, ~{total_tokens:,} tokens")
    print(f"  Data sources:     {len(all_sources)} unique ({', '.join(sorted(all_sources))})")
    print(f"  Ready for training: {'YES' if ready else 'NO'}")
    print("=" * 45)

    # Generate plots if requested
    if args.plot and HAS_MATPLOTLIB and train_stats:
        print("\nGenerating distribution charts...")

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        # Pie chart of source distribution
        sources = train_stats["sources"]
        ax1.pie(
            sources.values(),
            labels=sources.keys(),
            autopct="%1.1f%%",
            startangle=90,
        )
        ax1.set_title("Source Distribution")

        # Histogram of text lengths
        ax2.hist(train_stats["lengths"], bins=50, color="#4A90D9", edgecolor="white", alpha=0.8)
        ax2.set_title("Text Length Distribution")
        ax2.set_xlabel("Characters")
        ax2.set_ylabel("Count")

        plt.tight_layout()

        output_path = Path("logs/data_stats.png")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"  Charts saved to {output_path}")
        plt.close()
    elif args.plot and not HAS_MATPLOTLIB:
        print("\nInstall matplotlib to generate charts: pip install matplotlib")


if __name__ == "__main__":
    main()
