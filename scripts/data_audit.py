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

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.rebuild_corpus import select_corpus_sources


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


def audit_token_share(processed_chars, raw_chars=None):
    """Cross-source: fraction of training tokens each source contributes.

    Computed from the actual processed train+val splits, grouped by each
    record's ``source`` field — this is what the model will literally see
    during training, after any subsampling caps in ``rebuild_corpus.py``.

    When ``raw_chars`` is provided and differs materially from the
    processed totals (i.e. some source got subsampled or dropped), the
    raw size is shown alongside so the gap is visible.
    """
    header("Token share (what the model actually sees)")
    total = sum(processed_chars.values())
    if total == 0:
        print("  (no data)")
        return

    show_raw = bool(raw_chars) and any(
        raw_chars.get(s, 0) > c * 1.05 for s, c in processed_chars.items()
    )
    if show_raw:
        print(f"  {'source':<14} {'tokens':>14}  {'share':>7}    {'raw':>14}  {'kept':>5}")
    for src, chars in sorted(processed_chars.items(), key=lambda x: -x[1]):
        line = f"  {src:<14} ~{chars // 4:>12,}  {chars / total * 100:5.1f}%"
        if show_raw:
            raw = raw_chars.get(src, chars)
            kept_pct = (chars / raw * 100) if raw > 0 else 100.0
            line += f"    ~{raw // 4:>12,}  {kept_pct:4.0f}%"
        print(line)
    if show_raw:
        print()
        print("  Note: 'raw' is the on-disk size of the source; 'kept' is the share that")
        print("  survived rebuild_corpus.py (subsampling, dedup). When kept < 100% the")
        print("  source was capped by --max-cve-tokens or its duplicates collapsed.")


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


# ROADMAP-anchored token targets per scale rung. The lower bound is the
# minimum we should hit before progressing to that rung; the upper bound
# is the Chinchilla-comfortable target. ghost-tiny is "done" at any
# corpus size (it's the throwaway-rung educational artifact); the
# others gate ghost-small / ghost-base advancement.
V040_TOKEN_TARGETS = {
    "v0.4.0 — ghost-small (55M params)": (50_000_000, 100_000_000),
    "v0.5.0 — ghost-base (350M params)": (1_000_000_000, 7_000_000_000),
}

# What the project plans to add for v0.4.0 corpus volume per ROADMAP.md.
# Estimated upper bound is what the source would contribute at full
# pull. Currently-collected lets the audit print "X collected /
# Y planned" without requiring all sources to exist on disk yet.
V040_PLANNED_SOURCES = [
    ("nvd",            "NVD CVE descriptions (capped)",                 6_000_000),
    ("exploitdb",      "Exploit-DB PoCs (real code + advisories)",      8_000_000),
    ("ctftime",        "CTFtime real writeups (depth ≥ 3K)",            5_000_000),
    ("ctf_repos",      "GitHub CTF writeup repos (permissive only)",    4_000_000),
    ("arxiv",          "arXiv cs.CR (full-text PDFs, not abstracts)",  15_000_000),
    ("mitre_attack",   "MITRE ATT&CK (depth: techniques + groups)",     1_000_000),
    ("capec",          "CAPEC attack patterns (depth)",                   500_000),
    ("synthetic",      "Synthetic CTF (deprecating once real ≥ 2x)",    1_500_000),
    ("tool_docs",      "Tool docs: nmap, pwntools, scapy, impacket",    3_000_000),
]


def audit_v040_target(processed_chars):
    """Print progress toward the v0.4.0 / v0.5.0 token-volume targets.

    The audit_token_share section above shows the current corpus state.
    This section answers the next question: "how far are we from the
    next training rung, and which sources are likely to close the gap?"
    Token estimates come from ROADMAP.md and CORPUS.md.
    """
    header("v0.4.0 corpus target tracker")
    total_chars = sum(processed_chars.values())
    total_tokens = total_chars // 4
    print(f"  current: ~{total_tokens:,} tokens (from audit_token_share above)")
    print()

    for label, (lo, hi) in V040_TOKEN_TARGETS.items():
        gap_lo = max(0, lo - total_tokens)
        pct_lo = min(100, total_tokens / lo * 100) if lo else 0
        bar_w = 30
        filled = int(bar_w * pct_lo / 100)
        bar = "█" * filled + "░" * (bar_w - filled)
        print(f"  {label}")
        print(f"    target:   {lo:>15,} – {hi:,} tokens")
        print(f"    progress: [{bar}] {pct_lo:5.1f}% of lower bound")
        if gap_lo > 0:
            print(f"    gap:      {gap_lo:,} tokens to lower bound, "
                  f"{hi - total_tokens:,} to upper bound")
        else:
            print(f"    status:   lower bound met; "
                  f"{max(0, hi - total_tokens):,} tokens to upper bound")
        print()

    header("v0.4.0 source roadmap (planned upper-bound tokens)")
    print(f"  {'source':<14} {'collected':>14}  {'planned':>14}  {'pct':>6}  description")
    print(f"  {'-'*14} {'-'*14}  {'-'*14}  {'-'*6}  {'-'*44}")
    total_planned = 0
    total_collected = 0
    for src, desc, planned in V040_PLANNED_SOURCES:
        collected_chars = processed_chars.get(src, 0)
        # synthetic-CTF lives in raw as ctf.jsonl; its source field is
        # "synthetic" so the lookup matches; fall back to "ctf" if the
        # rebuild used the legacy field.
        if src == "synthetic" and collected_chars == 0:
            collected_chars = processed_chars.get("ctf", 0)
        collected = collected_chars // 4
        pct = collected / planned * 100 if planned else 0
        total_planned += planned
        total_collected += collected
        print(f"  {src:<14} {collected:>14,}  {planned:>14,}  {pct:5.1f}%  {desc}")
    print(f"  {'-'*14} {'-'*14}  {'-'*14}  {'-'*6}")
    overall_pct = total_collected / total_planned * 100 if total_planned else 0
    print(f"  {'TOTAL':<14} {total_collected:>14,}  {total_planned:>14,}  "
          f"{overall_pct:5.1f}%")


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

    selected_paths, _ = select_corpus_sources(raw, prefer_full_nvd=True)
    selected = {Path(p).stem: Path(p) for p in selected_paths}
    raw_files = {p.stem: p for p in sorted(raw.glob("*.jsonl"))}

    excluded = [name for name in raw_files if name not in selected]
    if excluded:
        print(f"\n[info] excluded from training merge (superseded): {', '.join(excluded)}")

    raw_stats = {}
    all_raw_records = []

    for name, path in selected.items():
        records, malformed = load_jsonl(path)
        if malformed:
            print(f"\n[warn] {name}: {malformed} malformed lines skipped")
        stats = audit_file(records, name)
        raw_stats[name] = stats
        all_raw_records.extend(records)

    cve_years = Counter()
    cve_key = "cve_full" if "cve_full" in selected else ("cve" if "cve" in selected else None)
    if cve_key:
        records, _ = load_jsonl(selected[cve_key])
        cve_years = audit_cve(records)

    ctf_cats = Counter()
    if "ctf" in selected:
        records, _ = load_jsonl(selected["ctf"])
        ctf_cats = audit_ctf(records)

    train_records, _ = load_jsonl(args.train)
    val_records, _ = load_jsonl(args.val)
    train_texts = [r.get("text", "") for r in train_records]
    val_texts = [r.get("text", "") for r in val_records]

    # Token share, grouped by record-level `source` field, computed from
    # the actual processed splits (post-subsample, post-dedup). The raw
    # totals are kept too so audit_token_share can flag when subsampling
    # has materially shrunk a source.
    raw_chars_by_source = defaultdict(int)
    for rec in all_raw_records:
        raw_chars_by_source[rec.get("source", "unknown")] += len(rec.get("text", ""))

    processed_chars_by_source = defaultdict(int)
    for rec in train_records + val_records:
        processed_chars_by_source[rec.get("source", "unknown")] += len(rec.get("text", ""))

    audit_token_share(processed_chars_by_source, raw_chars=raw_chars_by_source)

    header(f"Processed splits")
    print(f"  train: {len(train_records):,}  val: {len(val_records):,}  "
          f"ratio: {(len(val_records) / max(1, len(train_records)) * 100):.1f}% val")
    audit_leakage(train_texts, val_texts)

    audit_v040_target(processed_chars_by_source)

    # Cross-file dup check
    header("Cross-source duplicates (raw)")
    all_texts = Counter(r.get("text", "") for r in all_raw_records)
    cross_dups = sum(c - 1 for c in all_texts.values() if c > 1)
    print(f"  duplicate raw records across all sources: {cross_dups:,}")

    print("\n" + "=" * 50)

    if args.plot and HAS_MATPLOTLIB:
        make_plots(raw_stats, cve_years, ctf_cats, processed_chars_by_source, Path("logs/data_audit.png"))
    elif args.plot:
        print("matplotlib not installed — skipping charts")


if __name__ == "__main__":
    main()
