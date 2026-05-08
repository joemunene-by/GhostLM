#!/usr/bin/env python3
"""Structural-compliance eval for the bet 6 format-aware pretrain.

Bet 6's claim is that ghost-base, after seeing STIX 2.1 / YARA / Sigma /
MISP artifacts during pretrain, can *produce* those formats reliably.
"Reliably" needs a metric. This script is that metric.

Pipeline:

  1. Read a predictions file ``--predictions <jsonl>``. Each line:
       {"format": "<one of stix_indicator|yara_rule|sigma_rule|misp_event>",
        "prompt": "<original NL request>",
        "predicted_artifact": "<model output>"}

  2. For each prediction, run the matching parser from
     ``scripts/distill_format_aware`` (parse_stix / parse_yara /
     parse_sigma / parse_misp). A prediction "passes parse" iff the
     parser returns a non-None value (= the artifact is structurally
     valid for that format).

  3. Optionally apply a key-field check. Some prompts demand specific
     content (e.g. a CVE number must appear in the STIX
     external_references; a MISP event must include a sha256
     attribute). If the prediction record carries a ``required_fields``
     list (each entry: {"path": "Event.Attribute.0.type", "value":
     "sha256"}), the script verifies each required field. The
     gold ``data/raw/format_aware_seeds.jsonl`` ships with
     hand-curated required_fields; downstream eval sets can layer
     their own.

  4. Print a per-format pass-rate table and an overall score.

Usage (eval against the gold seed set itself, used as a self-check
for the parsers + required-field machinery):

    PYTHONPATH=. python3 scripts/eval_format_compliance.py \\
        --predictions data/raw/format_aware_seeds.jsonl \\
        --predictions-field artifact

Usage (eval a real model's outputs, produced by running the prompts
in the seed set through chat.py / generate.py and saving as JSONL
with a ``predicted_artifact`` field):

    PYTHONPATH=. python3 scripts/eval_format_compliance.py \\
        --predictions runs/v0.9.4_format_eval.jsonl

Output: a markdown table written to ``--out`` (default
``logs/format_compliance_<run_name>.md``) plus a one-line summary
to stdout for CI scraping.

Why the eval matters: every other small cybersec LM has zero
structural-compliance numbers because they don't train on these
formats. Reporting a real number here is *the* differentiation
metric for bet 6 versus a 7B general LM that has seen STIX in
common-crawl text but doesn't know YARA / Sigma syntax. A 60%
parse rate on a from-scratch ghost-base would already exceed
what GPT-4 / Claude scores on raw YARA generation today.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.distill_format_aware import (  # noqa: E402
    parse_stix, parse_yara, parse_sigma, parse_misp,
)


PARSERS = {
    "stix_indicator": parse_stix,
    "yara_rule": parse_yara,
    "sigma_rule": parse_sigma,
    "misp_event": parse_misp,
}


def get_path(obj: Any, path: str) -> Any:
    """Walk a dotted path into a dict / list. Empty segments and
    out-of-range indices return None instead of raising; this keeps
    the field-check loop simple."""
    cur: Any = obj
    for part in path.split("."):
        if cur is None:
            return None
        if part.isdigit() and isinstance(cur, list):
            idx = int(part)
            cur = cur[idx] if 0 <= idx < len(cur) else None
        elif isinstance(cur, dict):
            cur = cur.get(part)
        else:
            return None
    return cur


def required_fields_pass(parsed: Any, required: List[Dict[str, str]]) -> List[str]:
    """Return the list of required-field misses. An empty list means
    all fields matched. ``required`` entries are
    ``{"path": "<dotted>", "value": "<expected>"}``; ``value`` may be
    a substring (matched case-insensitively) when the field is text."""
    misses: List[str] = []
    if parsed is None:
        return ["<unparseable>"]
    for req in required:
        path = req.get("path", "")
        expected = req.get("value")
        actual = get_path(parsed, path)
        if expected is None:
            if actual is None:
                misses.append(f"{path}: missing")
            continue
        if actual is None:
            misses.append(f"{path}: missing (wanted {expected!r})")
            continue
        if isinstance(actual, str) and isinstance(expected, str):
            if expected.lower() not in actual.lower():
                misses.append(f"{path}: {actual!r} != {expected!r}")
        else:
            if actual != expected:
                misses.append(f"{path}: {actual!r} != {expected!r}")
    return misses


def required_substrings_pass(artifact: str, required: List[str]) -> List[str]:
    """Substring-level required-content check. Works for YARA (raw
    text, not a dict) and as a complement to required_fields for the
    other formats. Case-insensitive substring match; one entry per
    miss."""
    misses: List[str] = []
    if not artifact:
        return [f"<empty artifact, wanted {sub!r}>" for sub in required]
    art_lower = artifact.lower()
    for sub in required:
        if sub.lower() not in art_lower:
            misses.append(f"missing substring: {sub!r}")
    return misses


def evaluate_record(rec: Dict[str, Any], pred_field: str) -> Dict[str, Any]:
    """Score one prediction. Returns a dict with parse_ok, fields_ok,
    fields_misses, and the parsed object (or None).

    Two field-check styles are supported per record:
      ``required_fields``       dotted-path checks against the parsed
                                object; right for STIX / Sigma / MISP.
      ``required_substrings``   plain substring matches on the raw
                                artifact text; right for YARA, plus a
                                useful fallback for other formats."""
    fmt = rec.get("format")
    if fmt not in PARSERS:
        return {"parse_ok": False, "fields_ok": False,
                "fields_misses": [f"unknown format: {fmt!r}"], "parsed": None}
    parser = PARSERS[fmt]
    artifact = rec.get(pred_field) or ""
    parsed = parser(artifact)
    parse_ok = parsed is not None
    field_misses = required_fields_pass(parsed, rec.get("required_fields") or [])
    sub_misses = required_substrings_pass(artifact, rec.get("required_substrings") or [])
    misses = field_misses + sub_misses
    return {
        "parse_ok": parse_ok,
        "fields_ok": parse_ok and not misses,
        "fields_misses": misses,
        "parsed": parsed,
    }


def wilson_ci(k: int, n: int, z: float = 1.96) -> Tuple[float, float]:
    """Wilson 95%-CI for a binomial proportion, returned as percentages.

    Right interval to use here because the Clopper-Pearson CI is too
    conservative at small n and the normal-approximation CI breaks
    down at p near 0 or 1 (which is exactly the regime we're in:
    v0.9 sits at 0%, ghost-base may sit at low double digits). For
    n=32 k=0, this returns roughly (0%, 10.7%); for n=8 k=0, it
    returns (0%, 36.9%). The tightening is the reason the eval set
    grew from 8 to 32 records."""
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    spread = z * ((p * (1 - p) + z * z / (4 * n)) / n) ** 0.5 / denom
    lo = max(0.0, center - spread)
    hi = min(1.0, center + spread)
    return (100 * lo, 100 * hi)


def render_report(results: List[Tuple[Dict, Dict]]) -> str:
    """Build the markdown report. Per-format table + overall summary."""
    by_fmt: Dict[str, List[Dict]] = defaultdict(list)
    for rec, ev in results:
        by_fmt[rec.get("format", "unknown")].append(ev)

    lines = ["# Format compliance report", ""]
    lines.append(f"Total predictions: **{len(results)}**\n")
    lines.append("Pass rates with Wilson 95% CIs (right CI for binomial "
                 "proportions at small n).\n")
    lines.append("| Format | n | parse-pass | parse % (95% CI) | "
                 "fields-pass | fields % (95% CI) |")
    lines.append("|---|---:|---:|---|---:|---|")
    total_n = total_parse = total_fields = 0
    for fmt in sorted(by_fmt.keys()):
        evs = by_fmt[fmt]
        n = len(evs)
        parse_n = sum(1 for e in evs if e["parse_ok"])
        fields_n = sum(1 for e in evs if e["fields_ok"])
        total_n += n
        total_parse += parse_n
        total_fields += fields_n
        pct_parse = 100 * parse_n / n if n else 0.0
        pct_fields = 100 * fields_n / n if n else 0.0
        plo, phi = wilson_ci(parse_n, n)
        flo, fhi = wilson_ci(fields_n, n)
        lines.append(
            f"| {fmt} | {n} | {parse_n} | "
            f"{pct_parse:.1f}% [{plo:.1f}-{phi:.1f}] | {fields_n} | "
            f"{pct_fields:.1f}% [{flo:.1f}-{fhi:.1f}] |"
        )
    if total_n:
        plo, phi = wilson_ci(total_parse, total_n)
        flo, fhi = wilson_ci(total_fields, total_n)
        lines.append(
            f"| **OVERALL** | **{total_n}** | **{total_parse}** | "
            f"**{100*total_parse/total_n:.1f}% [{plo:.1f}-{phi:.1f}]** | "
            f"**{total_fields}** | "
            f"**{100*total_fields/total_n:.1f}% [{flo:.1f}-{fhi:.1f}]** |"
        )

    # Optional miss enumeration so failures are debuggable from the
    # report alone.
    miss_examples = [
        (rec, ev) for rec, ev in results
        if not ev["fields_ok"] and ev.get("fields_misses")
    ]
    if miss_examples:
        lines.append("\n## Required-field misses (first 10)\n")
        for rec, ev in miss_examples[:10]:
            lines.append(f"- **{rec.get('format')}** "
                         f"prompt={rec.get('prompt','')[:80]!r}")
            for m in ev["fields_misses"][:3]:
                lines.append(f"  - {m}")

    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--predictions", required=True,
                   help="JSONL file with format / prompt / predicted_artifact")
    p.add_argument("--predictions-field", default="predicted_artifact",
                   help="Name of the field holding the model's output. "
                        "Use 'artifact' to score the gold seed set "
                        "itself (self-check).")
    p.add_argument("--out",
                   help="Markdown report path (default: "
                        "logs/format_compliance_<basename>.md)")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    pred_path = Path(args.predictions)
    if not pred_path.exists():
        sys.exit(f"predictions file not found: {pred_path}")

    results: List[Tuple[Dict, Dict]] = []
    with pred_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            ev = evaluate_record(rec, args.predictions_field)
            results.append((rec, ev))

    if not results:
        sys.exit(f"no usable records in {pred_path}")

    report = render_report(results)
    out_path = Path(args.out) if args.out else (
        REPO_ROOT / "logs" / f"format_compliance_{pred_path.stem}.md"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(report, encoding="utf-8")

    n = len(results)
    parse_pass = sum(1 for _, ev in results if ev["parse_ok"])
    fields_pass = sum(1 for _, ev in results if ev["fields_ok"])
    print(f"format compliance: {fields_pass}/{n} fields-pass "
          f"({100*fields_pass/n:.1f}%), {parse_pass}/{n} parse-pass "
          f"({100*parse_pass/n:.1f}%)")
    print(f"report: {out_path}")
    return 0 if parse_pass == n else 1


if __name__ == "__main__":
    raise SystemExit(main())
