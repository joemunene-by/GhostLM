"""Rebuild the train/val splits after a corpus pull.

Re-runs the merge step against the current ``data/raw/*.jsonl`` files
without touching the collectors. Designed for the Phase 3 post-NVD-pull
workflow: once ``data/raw/cve_full.jsonl`` is on disk (from
``scripts/collect_nvd_full.py``), this swaps it in as the CVE source for
the merge and writes fresh ``data/processed/{train,val}.jsonl``.

The deterministic-hash split is preserved — identical texts always land
in the same bucket, so re-running this is idempotent.
"""

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from data.collect import merge_datasets


# Named corpus-mix profiles. Each maps a training domain (see
# ``data.collect.domain_of``) to a token cap; domains absent from a profile
# are left uncapped. ``cybersec`` is the legacy single-domain default. The
# ``generalist`` / ``balanced`` profiles cap cybersec so the general-domain
# sources (general_web, code, math, knowledge) carry real share — the lever
# that turns GhostLM from a cybersec-only model into a generalist with
# retained security depth.
#
# Budgets are token caps, not exact shares: a domain pulls in min(collected,
# cap) tokens, so the achieved mix depends on how much each collector pulled.
# Run a collection pass first, then `rebuild_corpus.py --profile <name>` and
# read the "Domain mix" report it prints to confirm the realized shares.
CORPUS_PROFILES = {
    # Legacy behaviour: cap only NVD via --max-cve-tokens, no domain caps.
    "cybersec": {},
    # Generalist: cybersec is the largest single specialty but a minority of
    # tokens; general web/code dominate; math/knowledge fill out breadth.
    "generalist": {
        "cybersec": 120_000_000,
        "general_web": 140_000_000,
        "code": 100_000_000,
        "math": 45_000_000,
        "knowledge": 45_000_000,
        "instruction": 20_000_000,
    },
    # Balanced: every domain capped to a similar budget for an even mix.
    "balanced": {
        "cybersec": 80_000_000,
        "general_web": 80_000_000,
        "code": 80_000_000,
        "math": 40_000_000,
        "knowledge": 40_000_000,
        "instruction": 20_000_000,
    },
}


def parse_domain_budget(items):
    """Parse repeated ``DOMAIN=TOKENS`` overrides into a dict.

    ``TOKENS`` accepts plain ints or ``k``/``m``/``b`` suffixes
    (``cybersec=120m``). Returns ``{}`` for an empty/None input.
    """
    budgets = {}
    for item in items or []:
        if "=" not in item:
            raise SystemExit(f"--domain-budget expects DOMAIN=TOKENS, got: {item!r}")
        domain, raw = item.split("=", 1)
        raw = raw.strip().lower()
        mult = 1
        if raw and raw[-1] in "kmb":
            mult = {"k": 1_000, "m": 1_000_000, "b": 1_000_000_000}[raw[-1]]
            raw = raw[:-1]
        budgets[domain.strip()] = int(float(raw) * mult)
    return budgets


# Filename globs that must never enter the training merge: held-out
# benchmark/eval sets and the SecQA/general-MCQ rulers. Without this, a
# rebuild would silently ingest its own eval data and contaminate every
# benchmark. Synthetic *training* banks (``*_patterns.jsonl``,
# ``*_seeds.jsonl``) are intentionally NOT excluded.
DEFAULT_EXCLUDE_GLOBS = (
    "*_eval.jsonl",
    "*_bench.jsonl",
    "*_bench_v2.jsonl",
    "secqa.jsonl",
    "general_mcq_bench.jsonl",
)


def select_corpus_sources(raw_dir, prefer_full_nvd=True, exclude_globs=DEFAULT_EXCLUDE_GLOBS):
    """Pick the JSONL files to feed into the merge from a raw/ directory.

    When both ``cve.jsonl`` (the v0.3.0 baseline corpus) and
    ``cve_full.jsonl`` (the post-Phase-3 NVD pull) are present, only one
    should go into the merge. By default ``cve_full.jsonl`` wins;
    ``prefer_full_nvd=False`` forces the legacy file for reproducibility.

    Args:
        raw_dir: Path to the directory containing ``*.jsonl`` raw sources.
        prefer_full_nvd: If True (default) and ``cve_full.jsonl`` exists,
            it is selected and ``cve.jsonl`` is excluded. If False (or
            ``cve_full.jsonl`` is absent), ``cve.jsonl`` is selected.

    Returns:
        ``(sources, cve_choice)`` where ``sources`` is a list of selected
        JSONL paths as strings and ``cve_choice`` is the Path to the CVE
        file that won the selection (or ``None`` if neither exists).
    """
    raw_dir = Path(raw_dir)
    candidates = sorted(raw_dir.glob("*.jsonl"))
    cve_full = raw_dir / "cve_full.jsonl"
    cve_legacy = raw_dir / "cve.jsonl"

    exclude_globs = tuple(exclude_globs or ())
    use_full = prefer_full_nvd and cve_full.exists()
    sources = []
    for p in candidates:
        if p.name == "cve.jsonl" and use_full:
            continue  # superseded by cve_full
        if p.name == "cve_full.jsonl" and not use_full:
            continue
        if any(p.match(g) for g in exclude_globs):
            continue  # held-out eval/benchmark set — never train on it
        sources.append(str(p))

    cve_choice = cve_full if use_full else (cve_legacy if cve_legacy.exists() else None)
    return sources, cve_choice


def parse_args():
    p = argparse.ArgumentParser(description="Rebuild train/val splits from data/raw/.")
    p.add_argument("--raw-dir", default="data/raw", help="Directory containing raw JSONL files.")
    p.add_argument("--output", default="data/processed/train.jsonl",
                   help="Output train path (val is sibling).")
    p.add_argument("--val-split", type=float, default=0.05, help="Validation fraction.")
    p.add_argument(
        "--prefer-full-nvd",
        action="store_true",
        default=True,
        help="If data/raw/cve_full.jsonl exists, use it instead of cve.jsonl. Default true.",
    )
    p.add_argument(
        "--no-prefer-full-nvd",
        dest="prefer_full_nvd",
        action="store_false",
        help="Force using the legacy cve.jsonl even if cve_full.jsonl exists.",
    )
    p.add_argument(
        "--max-cve-tokens",
        type=int,
        default=None,
        help="Cap NVD CVE contribution at this many tokens. Without this, NVD's ~27M "
             "tokens dominate the corpus (~90%% share) and dilute every other source. "
             "Sampling is deterministic by content hash so re-runs are reproducible. "
             "Default: no cap.",
    )
    p.add_argument(
        "--profile",
        choices=sorted(CORPUS_PROFILES.keys()),
        default="cybersec",
        help="Corpus-mix profile. 'cybersec' (default) applies no domain caps "
             "(legacy behaviour). 'generalist' / 'balanced' cap the cybersec domain "
             "so general web/code/math/knowledge carry real token share — the lever "
             "for de-specializing GhostLM. See CORPUS_PROFILES.",
    )
    p.add_argument(
        "--domain-budget",
        action="append",
        metavar="DOMAIN=TOKENS",
        help="Override a single domain's token cap (repeatable). Accepts k/m/b "
             "suffixes, e.g. --domain-budget cybersec=100m. Overrides the value from "
             "--profile for that domain.",
    )
    return p.parse_args()


def main():
    args = parse_args()
    raw = Path(args.raw_dir)
    if not raw.is_dir():
        sys.exit(f"raw dir not found: {raw}")

    # Resolve domain budgets: profile defaults, then per-domain overrides.
    domain_budgets = dict(CORPUS_PROFILES[args.profile])
    domain_budgets.update(parse_domain_budget(args.domain_budget))

    sources, cve_choice = select_corpus_sources(raw, prefer_full_nvd=args.prefer_full_nvd)
    print("Rebuild corpus")
    print(f"  raw dir:    {raw}")
    print(f"  CVE source: {cve_choice}")
    print(f"  profile:    {args.profile}")
    if args.max_cve_tokens is not None:
        print(f"  CVE cap:    {args.max_cve_tokens:,} tokens (deterministic subsample)")
    if domain_budgets:
        print(f"  domain budgets (tokens):")
        for d, b in sorted(domain_budgets.items(), key=lambda kv: -kv[1]):
            print(f"    - {d:14s} {b:>14,}")
    print(f"  sources ({len(sources)}):")
    for s in sources:
        print(f"    - {s}")

    if not sources:
        sys.exit("no raw sources found — did the collectors run?")

    merge_datasets(
        input_paths=sources,
        output_path=args.output,
        val_split=args.val_split,
        max_cve_tokens=args.max_cve_tokens,
        domain_token_budgets=domain_budgets or None,
    )


if __name__ == "__main__":
    main()
