#!/usr/bin/env python3
"""Bet 6: structured-format pretrain pipeline (STIX / YARA / Sigma / MISP).

Other small cybersec LMs train on prose: blog posts, RFCs, MITRE
descriptions, CVE summaries. They get OK at *talking about* threat
intel but can't *produce* the structured artifacts that real CTI
workflows consume: STIX 2.1 bundles, YARA rules, Sigma detection
queries, MISP event JSON. Those are the formats threat intel is
exchanged in. A model that can read AND emit them slots into
existing pipelines without a translator.

This script generates synthetic (natural_language ↔ structured_artifact)
training pairs across four format families, seeded from the existing
GhostLM corpus:

  STIX 2.1 indicators
    Seed: NVD/CVE entries + MITRE Att&ck techniques
    Templates: prose CVE → STIX `indicator` SDO with proper pattern
               grammar; STIX bundle → IOC list extraction; STIX
               `attack-pattern` ↔ MITRE technique round-trip.

  YARA rules
    Seed: malware family descriptions + hex-string-heavy corpus
          excerpts (collect_security_blogs, distill_malware_analysis).
    Templates: prose IOC → YARA rule with proper $string/condition
               sections; YARA rule → human-readable explanation;
               YARA rule → list of file types it matches.

  Sigma rules
    Seed: ATT&CK technique descriptions (T-codes have natural Sigma
          translations) + known-bad event log patterns.
    Templates: NL detection requirement → Sigma rule with proper
               logsource/detection/condition shape; Sigma rule →
               equivalent KQL/Splunk SPL conversion sketch.

  MISP events
    Seed: incident reports from collect_security_blogs +
          collect_vendor_research.
    Templates: prose incident summary → MISP event JSON with
               `Attribute` array of typed IOCs (ip-dst, hostname,
               sha256, etc.); MISP event → STIX bundle conversion.

Output: a single JSONL file with `DistillRecord`s tagged
`source="distill_format_aware"`, with sub-format encoded in
`seed_source` (e.g. `seed_source="stix_indicator"`). Drops into
``data/processed/train.jsonl`` like every other distill output.

Why bet 6 belongs in the differentiation strategy. The five original
bets (tool-use SFT, daily updates, custom BPE, long context, MoE) all
operate on the assumption that the model's job is *answering* cybersec
questions. Bet 6 is different: the model's job becomes *interfacing*
with cybersec tooling. That's a capability no general LLM has at
small parameter counts, and it directly addresses the analyst-workflow
audience GhostLM is aimed at. The bet 3 result already showed the
"recompress the same text" hypothesis only buys 1.6%; the structural
move (different *kinds* of text the model sees, not denser tokens of
the same kind) is where the bigger win likely sits.

Run (smoke test, 10 traces per format, free Ollama):

    PYTHONPATH=. python3 scripts/distill_format_aware.py \\
        --provider ollama --model qwen2.5:14b \\
        --max-traces-per-format 10 \\
        --out data/processed/distill_format_aware.jsonl

Run (production, ~$50-100 on Anthropic, all four formats, ~1K traces):

    ANTHROPIC_API_KEY=... PYTHONPATH=. python3 \\
        scripts/distill_format_aware.py \\
        --provider anthropic --model claude-sonnet-4-6 \\
        --max-traces-per-format 250 \\
        --out data/processed/distill_format_aware.jsonl

Cost estimate: 1K traces × ~1500 output tokens × $3/MT (Sonnet) ≈
$5; with system prompts and reroll on quality-filter failures, budget
~$50-100 to land 1K clean traces.

Quality filter: each format has a syntactic check (parse_stix,
parse_yara, parse_sigma, parse_misp) that rejects unparseable
generations. Generations are also passed through ``content_dedup``
from ``distill_common`` so the teacher model doesn't fill the corpus
with 50 copies of the same boilerplate STIX bundle.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.distill_common import (  # noqa: E402
    DistillRecord, ProviderConfig, ResumeIndex, StreamingWriter,
    call_provider, content_dedup, load_jsonl_source, quality_ok,
)


SYSTEM_PROMPT = (
    "You are a senior cyber threat intelligence engineer. Your job is "
    "to produce machine-parseable, syntactically-valid structured "
    "artifacts (STIX 2.1, YARA, Sigma, MISP) from natural-language "
    "descriptions of threats. Output ONLY the structured artifact "
    "wrapped in the format the user requests, no surrounding prose, "
    "no apologies, no markdown fences unless explicitly asked."
)


# ---------------------------------------------------------------------------
# STIX 2.1
# ---------------------------------------------------------------------------


def stix_prompt_from_cve(seed: Dict[str, str]) -> str:
    """Build the prompt that asks the teacher model to convert a CVE
    record into a STIX 2.1 indicator SDO. The template name-checks the
    fields STIX expects so the teacher doesn't drift into a STIX 1.x
    XML response."""
    return (
        "Given the following CVE record, produce a STIX 2.1 indicator "
        "Stixobject (SDO) as a single JSON object. Include: type, "
        "spec_version='2.1', id (indicator--<uuid>), created, modified, "
        "pattern_type='stix', pattern (using stix-pattern grammar), "
        "valid_from, labels, name. Keep the JSON minimal and valid "
        "(no prose around it).\n\n"
        f"CVE record:\n{seed['seed_text'][:1500]}"
    )


def parse_stix(blob: str) -> Optional[Dict]:
    """Validate that the blob is a parseable STIX 2.1 SDO. Returns the
    parsed object on success, None on failure. Uses a structural check
    (required fields per STIX 2.1 §3.1) rather than a full validator;
    full STIX validation needs the ``stix2`` library which isn't a
    repo-wide dep."""
    blob = blob.strip()
    if blob.startswith("```"):
        # Strip code-fence wrapping if the teacher slipped one in.
        blob = re.sub(r"^```(?:json)?", "", blob, count=1).rstrip("` \n")
    try:
        obj = json.loads(blob)
    except json.JSONDecodeError:
        return None
    if not isinstance(obj, dict):
        return None
    required = {"type", "spec_version", "id", "created", "modified"}
    if not required.issubset(obj.keys()):
        return None
    if obj.get("spec_version") != "2.1":
        return None
    return obj


# ---------------------------------------------------------------------------
# YARA
# ---------------------------------------------------------------------------


def yara_prompt_from_malware(seed: Dict[str, str]) -> str:
    """Build the prompt that asks the teacher to write a YARA rule
    matching the described malware family."""
    return (
        "Write a YARA rule that detects the malware family described "
        "below. Include: a meaningful rule name, a metadata block with "
        "author/description/family/tlp, a strings section using both "
        "$text_ and $hex_ identifiers (with at least 3 unique strings), "
        "and a condition combining file-magic + string presence. Output "
        "only the raw YARA rule (no surrounding markdown).\n\n"
        f"Malware description:\n{seed['seed_text'][:1500]}"
    )


YARA_RULE_RE = re.compile(
    r"^\s*rule\s+\w+\s*\{",
    re.MULTILINE,
)
YARA_STRINGS_RE = re.compile(r"strings\s*:\s*", re.MULTILINE)
YARA_CONDITION_RE = re.compile(r"condition\s*:\s*", re.MULTILINE)


def parse_yara(blob: str) -> Optional[str]:
    """Light-touch YARA validation: rule header + strings + condition.
    Returns the cleaned rule on success, None otherwise. A real
    validator would invoke the yara CLI; that's an opt-in step the
    operator can layer on (``yara -p 1 file rule``)."""
    blob = blob.strip()
    if blob.startswith("```"):
        blob = re.sub(r"^```(?:yara|yar)?", "", blob, count=1).rstrip("` \n")
    if not YARA_RULE_RE.search(blob):
        return None
    if not YARA_STRINGS_RE.search(blob):
        return None
    if not YARA_CONDITION_RE.search(blob):
        return None
    # Brace balance.
    if blob.count("{") != blob.count("}"):
        return None
    return blob


# ---------------------------------------------------------------------------
# Sigma
# ---------------------------------------------------------------------------


def sigma_prompt_from_attack(seed: Dict[str, str]) -> str:
    """Build the prompt that asks the teacher for a Sigma rule
    detecting the described ATT&CK technique."""
    return (
        "Write a Sigma rule (https://github.com/SigmaHQ/sigma) that "
        "detects the technique described below. Include: title, id "
        "(GUID), status (test|experimental|stable), description, "
        "logsource (with category/product), detection block with at "
        "least one selection and a `condition`, falsepositives, level. "
        "Output a single YAML document (no markdown wrapper).\n\n"
        f"Technique description:\n{seed['seed_text'][:1500]}"
    )


SIGMA_REQUIRED_FIELDS = ("title", "logsource", "detection")


def parse_sigma(blob: str) -> Optional[Dict]:
    """Light-touch Sigma validation: load YAML + check required keys.
    Falls back to a regex-based field probe when PyYAML isn't installed
    (the production loop normally has it; smoke tests on a fresh M4
    might not)."""
    blob = blob.strip()
    if blob.startswith("```"):
        blob = re.sub(r"^```(?:ya?ml)?", "", blob, count=1).rstrip("` \n")
    try:
        import yaml  # type: ignore[import-not-found]
        obj = yaml.safe_load(blob)
        if not isinstance(obj, dict):
            return None
        if not all(k in obj for k in SIGMA_REQUIRED_FIELDS):
            return None
        det = obj.get("detection")
        if not isinstance(det, dict) or "condition" not in det:
            return None
        return obj
    except ImportError:
        # Regex fallback (good enough to filter clearly broken outputs
        # in a smoke-test environment.
        for k in SIGMA_REQUIRED_FIELDS:
            if not re.search(rf"^\s*{k}\s*:", blob, re.MULTILINE):
                return None
        if not re.search(r"^\s*condition\s*:", blob, re.MULTILINE):
            return None
        return {"_unparsed": blob}


# ---------------------------------------------------------------------------
# MISP
# ---------------------------------------------------------------------------


def misp_prompt_from_incident(seed: Dict[str, str]) -> str:
    """Build the prompt that asks the teacher to convert an incident
    report into a MISP event JSON."""
    return (
        "Convert the incident report below into a MISP event JSON "
        "object (the shape MISP's REST API expects). The top-level "
        "object MUST have an `Event` key whose value contains: info, "
        "date, threat_level_id (1=high|2=medium|3=low|4=undefined), "
        "analysis (0|1|2), distribution, and an `Attribute` array of "
        "typed IOC entries (each with `type` from MISP's controlled "
        "vocab (ip-dst, hostname, domain, sha256, url, email-src, "
        "etc.) plus `value` and `category`). Output JSON only.\n\n"
        f"Incident:\n{seed['seed_text'][:1800]}"
    )


def parse_misp(blob: str) -> Optional[Dict]:
    """Validate that the blob has the required ``Event`` shell + a
    populated ``Attribute`` array of typed IOCs."""
    blob = blob.strip()
    if blob.startswith("```"):
        blob = re.sub(r"^```(?:json)?", "", blob, count=1).rstrip("` \n")
    try:
        obj = json.loads(blob)
    except json.JSONDecodeError:
        return None
    event = obj.get("Event") if isinstance(obj, dict) else None
    if not isinstance(event, dict):
        return None
    attrs = event.get("Attribute")
    if not isinstance(attrs, list) or not attrs:
        return None
    # At least one attribute must have type + value.
    if not all(
        isinstance(a, dict) and a.get("type") and a.get("value") for a in attrs
    ):
        return None
    return obj


# ---------------------------------------------------------------------------
# Format dispatch
# ---------------------------------------------------------------------------


FORMATS = {
    "stix_indicator": {
        "seed_source_path": "data/raw/nvd_full.jsonl",
        "prompt_fn": stix_prompt_from_cve,
        "parse_fn": parse_stix,
    },
    "yara_rule": {
        "seed_source_path": "data/raw/security_blogs.jsonl",
        "prompt_fn": yara_prompt_from_malware,
        "parse_fn": parse_yara,
    },
    "sigma_rule": {
        "seed_source_path": "data/raw/mitre_full.jsonl",
        "prompt_fn": sigma_prompt_from_attack,
        "parse_fn": parse_sigma,
    },
    "misp_event": {
        "seed_source_path": "data/raw/security_blogs.jsonl",
        "prompt_fn": misp_prompt_from_incident,
        "parse_fn": parse_misp,
    },
}


# ---------------------------------------------------------------------------
# Generation loop
# ---------------------------------------------------------------------------


def generate_one(cfg: ProviderConfig, fmt_name: str, seed: Dict[str, str],
                 prompt_fn, parse_fn) -> Optional[DistillRecord]:
    """Single (seed -> prompt -> teacher -> parsed artifact) round.
    Returns a typed DistillRecord on success, None on any failure
    (provider error, unparseable output, quality-filter rejection)."""
    prompt = prompt_fn(seed)
    raw = call_provider(cfg, prompt, system=SYSTEM_PROMPT)
    if not raw:
        return None
    parsed = parse_fn(raw)
    if parsed is None:
        return None
    if not quality_ok(raw, min_words=20, max_words=2000):
        return None
    # Format the training-time text as a paired prompt+answer so the
    # model learns the bidirectional mapping (NL <-> structured) in a
    # single shot. ``<|user|>``/``<|assistant|>`` aren't required at
    # pretrain time. They get reformatted at SFT time if needed.
    text = (
        f"Source: {seed.get('seed_id', 'unknown')}\n"
        f"Format: {fmt_name}\n\n"
        f"Prompt:\n{prompt}\n\n"
        f"Artifact:\n{raw.strip()}\n"
    )
    return DistillRecord.make(
        source="distill_format_aware",
        teacher=f"{cfg.name}/{cfg.model}",
        seed_source=fmt_name,
        seed_id=seed.get("seed_id", "unknown"),
        text=text,
    )


def run_format(cfg: ProviderConfig, fmt_name: str, fmt_spec: Dict,
               max_traces: int, writer: StreamingWriter,
               resume: ResumeIndex) -> int:
    """Loop over seed records for one format, generate up to
    ``max_traces`` clean records, write each as it lands. Returns the
    count of accepted records."""
    seed_path = REPO_ROOT / fmt_spec["seed_source_path"]
    seeds = load_jsonl_source(seed_path)
    if not seeds:
        print(f"  [{fmt_name}] no seed source at {seed_path}; skipping")
        return 0
    accepted = 0
    candidates: List[DistillRecord] = []
    for seed in seeds:
        if accepted >= max_traces:
            break
        if resume.already_done(fmt_name, seed["seed_id"]):
            continue
        rec = generate_one(cfg, fmt_name, seed,
                           fmt_spec["prompt_fn"], fmt_spec["parse_fn"])
        if rec is None:
            continue
        candidates.append(rec)
        accepted += 1
        if accepted % 10 == 0:
            print(f"  [{fmt_name}] accepted {accepted}/{max_traces}")
    # Dedup pass before writing. content_dedup is an Iterator so we
    # exhaust it and write the survivors.
    written = 0
    for rec in content_dedup(candidates):
        writer.write(rec)
        written += 1
    print(f"  [{fmt_name}] {written} records written ({accepted - written} dropped to dedup)")
    return written


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--provider", default="ollama",
                   choices=["ollama", "anthropic", "openai"])
    p.add_argument("--model", default="qwen2.5:14b")
    p.add_argument("--base-url")
    p.add_argument("--api-key-env")
    p.add_argument("--temperature", type=float, default=0.4)
    p.add_argument("--max-tokens", type=int, default=2000)
    p.add_argument("--out", default="data/processed/distill_format_aware.jsonl")
    p.add_argument("--max-traces-per-format", type=int, default=250,
                   help="Cap accepted records per format. With four "
                        "formats default 250 → up to 1000 records total.")
    p.add_argument("--formats", default="stix_indicator,yara_rule,sigma_rule,misp_event",
                   help="Comma-separated subset of formats to run")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    cfg = ProviderConfig(
        name=args.provider, model=args.model,
        base_url=args.base_url, api_key_env=args.api_key_env,
        temperature=args.temperature, max_tokens=args.max_tokens,
    )
    out_path = REPO_ROOT / args.out
    writer = StreamingWriter(out_path)
    resume = ResumeIndex(out_path)

    formats_to_run = [f.strip() for f in args.formats.split(",") if f.strip()]
    unknown = [f for f in formats_to_run if f not in FORMATS]
    if unknown:
        sys.exit(f"unknown formats: {unknown}; available: {list(FORMATS)}")

    print(f"distill_format_aware via {cfg.name}/{cfg.model}")
    print(f"  formats: {formats_to_run}")
    print(f"  max per format: {args.max_traces_per_format}")
    print(f"  output: {out_path}")
    print(f"  already-done seeds: {len(resume.seen)}")

    total = 0
    for fmt_name in formats_to_run:
        print(f"\n=== {fmt_name} ===")
        spec = FORMATS[fmt_name]
        total += run_format(cfg, fmt_name, spec,
                            args.max_traces_per_format, writer, resume)

    writer.close()
    print(f"\nDone. Wrote {total} records to {out_path}")
    print("Next: rebuild train.jsonl via scripts/build_chat_dataset.py "
          "(or merge in pretrain). Then ghost-base will see structured "
          "STIX/YARA/Sigma/MISP artifacts during pretrain.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
