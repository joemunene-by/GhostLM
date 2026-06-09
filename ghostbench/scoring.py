"""Scoring primitives for GhostBench.

The core abstraction is ``Score``, a per-prediction outcome with
named tiers. A tier is one of:

  ``parse``       Did the prediction parse as valid for the format?
                  STIX 2.1 bundle parses, YARA rule has rule + strings
                  + condition, Sigma is loadable YAML, MISP has
                  Event.Attribute array, provenance has well-formed
                  cite tags. Bets without a structural format
                  (code-security, binary-literacy) treat parse as
                  vacuously True.

  ``fields``      Do the dotted-path fields specified in
                  ``required_fields`` resolve to the expected values
                  in the parsed object? Right for STIX / Sigma /
                  MISP / provenance.

  ``substrings``  Do the substrings specified in
                  ``required_substrings`` appear in the raw artifact
                  text? Right for YARA, code-security, binary-
                  literacy, plus a useful complement for the others.

  ``behavioral``  Would a real downstream tool accept this artifact?
                  Implemented in v0.3: ghostbench.behavioral has
                  validators that lazy-import the canonical reference
                  parser (``stix2``, ``yara-python``, ``pysigma``,
                  ``jsonschema``) and fall back to enhanced-structural
                  checks when those libraries aren't installed.
                  Catches edge cases the parse tier doesn't (invalid
                  UUIDs in STIX ids, unbalanced parens in YARA
                  conditions, MISP attribute types outside the
                  controlled vocab). Requested by setting
                  ``behavioral: true`` on the eval record.

  ``semantic``    (Reserved.) An LLM-as-judge tier that scores the
                  prediction against the eval record on richer
                  criteria than substring match. Not implemented in
                  v0.3; the slot is reserved so future ``ghostbench``
                  versions can add it without breaking the API.

A ``Score`` records pass/fail for every tier the eval record asked
about. ``Score.passed`` is the strict-AND across requested tiers;
``Score.tier_pass(name)`` is the per-tier bool.

The free functions ``score_record`` and ``score_predictions``
turn these primitives into the operator-facing eval API.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Score and Report data classes
# ---------------------------------------------------------------------------


@dataclass
class Score:
    """One prediction's outcome across all configured tiers."""
    seed_id: str
    fmt: str
    requested_tiers: Tuple[str, ...]
    tier_results: Dict[str, bool] = field(default_factory=dict)
    tier_misses: Dict[str, List[str]] = field(default_factory=dict)
    parsed: Any = None

    @property
    def passed(self) -> bool:
        """Strict-AND across all requested tiers. Returns False if
        any requested tier failed; True only if every requested tier
        passed."""
        return all(self.tier_results.get(t, False) for t in self.requested_tiers)

    def tier_pass(self, tier: str) -> bool:
        return self.tier_results.get(tier, False)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "seed_id": self.seed_id,
            "fmt": self.fmt,
            "requested_tiers": list(self.requested_tiers),
            "tier_results": dict(self.tier_results),
            "tier_misses": {k: list(v) for k, v in self.tier_misses.items()},
        }


@dataclass
class RunReport:
    """Aggregated scores across a Bench run.

    Holds per-tier pass counts, per-format-class breakdowns, and
    raw scores for downstream paired comparisons (McNemar / Wilson-
    shifted diff CI). Consumers typically use ``RunReport.summary()``
    for the operator-facing table or ``RunReport.scores`` for
    programmatic access.
    """
    bench_name: str
    run_name: str
    n: int
    scores: List[Score]

    def tier_passes(self, tier: str) -> int:
        return sum(1 for s in self.scores if s.tier_pass(tier))

    def passed_count(self) -> int:
        return sum(1 for s in self.scores if s.passed)

    def by_format(self) -> Dict[str, "RunReport"]:
        """Return one sub-report per format value present in scores."""
        groups: Dict[str, List[Score]] = {}
        for s in self.scores:
            groups.setdefault(s.fmt, []).append(s)
        return {
            fmt: RunReport(
                bench_name=f"{self.bench_name}/{fmt}",
                run_name=self.run_name, n=len(grp), scores=grp,
            )
            for fmt, grp in groups.items()
        }

    def summary(self) -> Dict[str, Any]:
        """Headline numbers for the run as a plain dict."""
        all_tiers = sorted({
            t for s in self.scores for t in s.requested_tiers
        })
        per_tier = {
            t: {"passes": self.tier_passes(t), "n": self.n}
            for t in all_tiers
        }
        return {
            "bench": self.bench_name,
            "run": self.run_name,
            "n": self.n,
            "passed": self.passed_count(),
            "per_tier": per_tier,
        }


# ---------------------------------------------------------------------------
# Tier scorers
# ---------------------------------------------------------------------------


def get_path(obj: Any, path: str) -> Any:
    """Walk a dotted path into a parsed object. Empty segments and
    out-of-range list indices return None instead of raising."""
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


def _check_fields(parsed: Any, required: List[Dict[str, Any]]) -> List[str]:
    """Return the list of field-check misses; empty means all pass."""
    misses: List[str] = []
    if not required:
        return misses
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


def _check_substrings(artifact: str, required: List[str]) -> List[str]:
    """Return the list of substring misses; empty means all pass."""
    misses: List[str] = []
    if not required:
        return misses
    if not artifact:
        return [f"<empty artifact, wanted {sub!r}>" for sub in required]
    art_lower = artifact.lower()
    for sub in required:
        if sub.lower() not in art_lower:
            misses.append(f"missing substring: {sub!r}")
    return misses


# ---------------------------------------------------------------------------
# Top-level scorer
# ---------------------------------------------------------------------------


def score_record(eval_rec: Dict[str, Any], predicted: str,
                 parsers: Dict[str, Callable[[str], Any]],
                 behavioral_validators: Optional[Dict[str, Callable[[str], Any]]] = None,
                 ) -> Score:
    """Score one prediction against its eval record.

    Tiers requested are inferred from the eval record:
      - ``parse``        always requested if ``fmt`` has a parser in
                          ``parsers``; vacuously True if not.
      - ``fields``       requested if ``required_fields`` is non-empty.
      - ``substrings``   requested if ``required_substrings`` is non-empty.
      - ``behavioral``   requested if the eval record sets
                          ``behavioral: true`` AND a behavioural
                          validator is registered for the format.

    Tiers that aren't requested are NOT included in ``passed``'s
    AND, so a record with only ``required_substrings`` will have
    ``passed`` driven entirely by the substring check.

    Args:
        eval_rec: The eval JSONL record (with ``format``, ``prompt``,
                  ``required_fields``, ``required_substrings``,
                  optional ``behavioral`` flag).
        predicted: The model's prediction text.
        parsers: Map from ``format`` value to structural parser. The
                 parser returns a parsed object on success or None.
        behavioral_validators: Optional map from ``format`` value to
                                behavioural validator. Each validator
                                returns True (passed real-tool
                                validation), False (failed), or None
                                (not measurable). When None is
                                returned, the tier is treated as
                                not-measured and excluded from
                                ``passed``'s AND.

    Returns:
        A ``Score`` capturing per-tier results and miss reasons.
    """
    fmt = eval_rec.get("format", "")
    seed_id = eval_rec.get("seed_id") or eval_rec.get("prompt", "")[:60]
    required_fields = eval_rec.get("required_fields") or []
    required_subs = eval_rec.get("required_substrings") or []
    behavioral_requested = bool(eval_rec.get("behavioral"))

    parser = parsers.get(fmt)
    if parser is None:
        parsed = None
        parse_ok = True   # vacuously
        parse_requested = False
    else:
        parsed = parser(predicted)
        parse_ok = parsed is not None
        parse_requested = True

    field_misses = _check_fields(parsed, required_fields)
    sub_misses = _check_substrings(predicted, required_subs)

    requested: List[str] = []
    tier_results: Dict[str, bool] = {}
    tier_misses: Dict[str, List[str]] = {}

    if parse_requested:
        requested.append("parse")
        tier_results["parse"] = parse_ok
        tier_misses["parse"] = [] if parse_ok else ["<unparseable>"]
    if required_fields:
        requested.append("fields")
        tier_results["fields"] = parse_ok and not field_misses
        tier_misses["fields"] = field_misses
    if required_subs:
        requested.append("substrings")
        tier_results["substrings"] = not sub_misses
        tier_misses["substrings"] = sub_misses
    if behavioral_requested and behavioral_validators is not None:
        validator = behavioral_validators.get(fmt)
        if validator is not None:
            outcome = validator(predicted)
            # Outcome is True / False / None. None means
            # "not measurable"; treat as not-requested for the AND.
            if outcome is None:
                tier_misses["behavioral"] = ["<not measurable>"]
            else:
                requested.append("behavioral")
                tier_results["behavioral"] = bool(outcome)
                tier_misses["behavioral"] = (
                    [] if outcome else ["<failed behavioural validation>"]
                )

    return Score(
        seed_id=str(seed_id),
        fmt=fmt,
        requested_tiers=tuple(requested),
        tier_results=tier_results,
        tier_misses=tier_misses,
        parsed=parsed,
    )
