"""GhostLM data audit — pre-training diagnostics: percentiles, dedup, leakage, token share."""

import argparse
import json
import re
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


CVE_ID_RE = re.compile(r"CVE-(\d{4})-\d+")

CTF_CATEGORIES = {
    "web": ["sql injection", "xss", "csrf", "ssrf", "jwt", "cookie", "http", "web app"],
    "pwn": ["buffer overflow", "rop", "shellcode", "pwntools", "gadget", "libc", "heap"],
    "crypto": ["rsa", "aes", "cipher", "encryption", "decrypt", "hash collision", "oracle"],
    "reverse": ["reverse engineer", "ghidra", "ida", "disassembl", "binary analysis"],
    "forensics": ["pcap", "wireshark", "memory dump", "volatility", "steganograph", "forensic"],
}


def parse_args():
    """Parse command-line arguments for the data audit."""
    parser = argparse.ArgumentParser(description="Audit GhostLM training corpus before training")
    parser.add_argument("--raw-dir", type=str, default="data/raw")
    parser.add_argument("--train", type=str, default="data/processed/train.jsonl")
    parser.add_argument("--val", type=str, default="data/processed/val.jsonl")
    parser.add_argument("--plot", action="store_true", help="Save audit charts to logs/data_audit.png")
    return parser.parse_args()


def load_jsonl(path):
    """Yield records from a JSONL file, flagging malformed lines."""
    p = Path(path)
    if not p.exists():
        return [], 0
    records, malformed = [], 0
    with open(p, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                malformed += 1
    return records, malformed


def pct(values, q):
    """Return the q-th percentile (0-100) of values, or 0 if empty."""
    if not values:
        return 0
    # statistics.quantiles gives n-1 cutpoints for n partitions
    cuts = statistics.quantiles(values, n=100, method="inclusive")
    # cuts[i] is the (i+1)th percentile; index 49 is p50, 89 is p90, etc.
    return int(cuts[q - 1]) if 1 <= q <= 99 else int(max(values))


def header(title):
    print(f"\n── {title} {'─' * max(1, 48 - len(title))}")


def audit_file(records, name):
    """Per-file audit: counts, length percentiles, empties, exact dups."""
    header(f"{name} ({len(records):,} records)")
    if not records:
        print("  (empty)")
        return {}

    texts = [r.get("text", "") for r in records]
    lengths = [len(t) for t in texts]
    empties = sum(1 for t in texts if not t.strip())

    dup_counts = Counter(texts)
    dup_groups = sum(1 for _, c in dup_counts.items() if c > 1)
    dup_records = sum(c for _, c in dup_counts.items() if c > 1) - dup_groups

    total_chars = sum(lengths)
    print(f"  empty text:     {empties}")
    print(f"  exact dups:     {dup_records:,} extra records across {dup_groups:,} groups "
          f"({(dup_records / len(records) * 100):.1f}%)")
    print(f"  total chars:    {total_chars:,}  (~{total_chars // 4:,} tokens est.)")
    print(f"  length p50:     {pct(lengths, 50):,}")
    print(f"  length p90:     {pct(lengths, 90):,}")
    print(f"  length p95:     {pct(lengths, 95):,}")
    print(f"  length p99:     {pct(lengths, 99):,}")
    print(f"  length max:     {max(lengths):,}")

    return {
        "lengths": lengths,
        "total_chars": total_chars,
        "empties": empties,
        "dup_records": dup_records,
        "texts": texts,
    }


def audit_cve(records):
    """CVE-specific: year distribution from CVE-YYYY-nnnn IDs."""
    header("CVE year distribution")
    years = Counter()
    missing = 0
    for r in records:
        m = CVE_ID_RE.search(r.get("id", ""))
        if m:
            years[int(m.group(1))] += 1
        else:
            missing += 1
    if not years:
        print("  (no parseable CVE IDs)")
        return years
    y_min, y_max = min(years), max(years)
    print(f"  span:           {y_min}–{y_max}  ({y_max - y_min + 1} years)")
    print(f"  missing IDs:    {missing}")
    print(f"  top 5 years:    " + ", ".join(f"{y}:{n}" for y, n in years.most_common(5)))
    # bucket by decade for a quick skew read
    decades = Counter()
    for y, n in years.items():
        decades[(y // 10) * 10] += n
    print("  by decade:      " + ", ".join(f"{d}s:{n:,}" for d, n in sorted(decades.items())))
    return years


def audit_ctf(records):
    """CTF: synthetic vs real split + keyword-inferred category share."""
    header("CTF composition")
    sources = Counter(r.get("source", "unknown") for r in records)
    print("  sources:        " + ", ".join(f"{k}:{v}" for k, v in sources.most_common()))

    cats = Counter()
    uncategorized = 0
    for r in records:
        text = r.get("text", "").lower()
        hits = [cat for cat, kws in CTF_CATEGORIES.items() if any(kw in text for kw in kws)]
        if not hits:
            uncategorized += 1
        else:
            for cat in hits:
                cats[cat] += 1
    total = len(records)
    print("  category share (keyword-inferred, multi-label):")
    for cat, n in cats.most_common():
        print(f"    {cat:<12} {n:>4,}  ({n / total * 100:.0f}%)")
    print(f"    {'none':<12} {uncategorized:>4,}  ({uncategorized / total * 100:.0f}%)")
    return cats


def audit_token_share(per_source_chars):
    """Cross-source: what fraction of training tokens each source contributes."""
    header("Token share (what the model actually sees)")
    total = sum(per_source_chars.values())
    if total == 0:
        print("  (no data)")
        return
    for src, chars in sorted(per_source_chars.items(), key=lambda x: -x[1]):
        print(f"  {src:<12} ~{chars // 4:>10,} tokens  ({chars / total * 100:5.1f}%)")


def audit_leakage(train_texts, val_texts):
    """Check for exact val texts appearing in train — contaminates eval."""
    header("Train/val leakage")
    if not train_texts or not val_texts:
        print("  (skipped — missing split)")
        return 0
    train_set = set(train_texts)
    leaked = sum(1 for t in val_texts if t in train_set)
    print(f"  val records in train: {leaked}  ({leaked / len(val_texts) * 100:.2f}% of val)")
    if leaked:
        print("  WARNING: remove these from val before training.")
    return leaked


def make_plots(raw_stats, cve_years, ctf_cats, token_share, out_path):
    """Save a 2x2 audit figure: lengths/year/token-share/ctf-cats."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    ax = axes[0][0]
    labels = [n for n, s in raw_stats.items() if s.get("lengths")]
    data = [raw_stats[n]["lengths"] for n in labels]
    if data:
        ax.boxplot(data, labels=labels, showfliers=False)
        ax.set_yscale("log")
    ax.set_title("Text length by source (log scale, no outliers)")
    ax.set_ylabel("chars")

    ax = axes[0][1]
    if cve_years:
        ys = sorted(cve_years)
        ax.bar(ys, [cve_years[y] for y in ys], color="#4A90D9")
        ax.set_title("CVE count by year")
        ax.set_xlabel("year")

    ax = axes[1][0]
    if token_share:
        items = sorted(token_share.items(), key=lambda x: -x[1])
        ax.bar([k for k, _ in items], [v // 4 for _, v in items], color="#D97A4A")
        ax.set_title("Token share by source (est.)")
        ax.set_ylabel("tokens")

    ax = axes[1][1]
    if ctf_cats:
        items = ctf_cats.most_common()
        ax.bar([k for k, _ in items], [v for _, v in items], color="#6FB76F")
        ax.set_title("CTF category (keyword-inferred)")
        ax.set_ylabel("records")

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nCharts saved to {out_path}")


def main():
    args = parse_args()
    raw = Path(args.raw_dir)
    print("=" * 50)
    print("GhostLM Data Audit")
    print("=" * 50)

    raw_files = {p.stem: p for p in sorted(raw.glob("*.jsonl"))}
    raw_stats = {}
    per_source_chars = defaultdict(int)
    all_raw_records = []

    for name, path in raw_files.items():
        records, malformed = load_jsonl(path)
        if malformed:
            print(f"\n[warn] {name}: {malformed} malformed lines skipped")
        stats = audit_file(records, name)
        raw_stats[name] = stats
        per_source_chars[name] = stats.get("total_chars", 0)
        all_raw_records.extend(records)

    cve_years = Counter()
    cve_key = "cve_full" if "cve_full" in raw_files else ("cve" if "cve" in raw_files else None)
    if cve_key:
        records, _ = load_jsonl(raw_files[cve_key])
        cve_years = audit_cve(records)

    ctf_cats = Counter()
    if "ctf" in raw_files:
        records, _ = load_jsonl(raw_files["ctf"])
        ctf_cats = audit_ctf(records)

    audit_token_share(per_source_chars)

    train_records, _ = load_jsonl(args.train)
    val_records, _ = load_jsonl(args.val)
    train_texts = [r.get("text", "") for r in train_records]
    val_texts = [r.get("text", "") for r in val_records]

    header(f"Processed splits")
    print(f"  train: {len(train_records):,}  val: {len(val_records):,}  "
          f"ratio: {(len(val_records) / max(1, len(train_records)) * 100):.1f}% val")
    audit_leakage(train_texts, val_texts)

    # Cross-file dup check
    header("Cross-source duplicates (raw)")
    all_texts = Counter(r.get("text", "") for r in all_raw_records)
    cross_dups = sum(c - 1 for c in all_texts.values() if c > 1)
    print(f"  duplicate raw records across all sources: {cross_dups:,}")

    print("\n" + "=" * 50)

    if args.plot and HAS_MATPLOTLIB:
        make_plots(raw_stats, cve_years, ctf_cats, per_source_chars, Path("logs/data_audit.png"))
    elif args.plot:
        print("matplotlib not installed — skipping charts")


if __name__ == "__main__":
    main()
