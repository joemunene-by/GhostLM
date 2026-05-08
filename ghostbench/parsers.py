"""Default parsers for GhostBench's nine bets.

Each parser returns the parsed object on success or None on
failure. ``None`` means "the prediction did not have valid
structure for this format"; the scorer interprets that as a
parse-tier failure.

For bets 7 (code-for-security) and 8 (binary-literacy) there is
no structural parser; the scorer treats parse as vacuously True
and runs only the substring / field tiers.
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _strip_code_fence(blob: str) -> str:
    """Strip a leading triple-backtick fence + an optional language
    tag, plus a trailing fence. Tolerant: returns the input unchanged
    if no fence is detected."""
    blob = blob.strip()
    if blob.startswith("```"):
        # Remove leading fence + language tag.
        m = re.match(r"^```(?:\w+)?\s*\n?", blob)
        if m:
            blob = blob[m.end():]
        if blob.endswith("```"):
            blob = blob[:-3]
    return blob.strip()


# ---------------------------------------------------------------------------
# STIX 2.1 indicator
# ---------------------------------------------------------------------------


_STIX_REQUIRED_KEYS = {"type", "spec_version", "id", "created", "modified"}


def parse_stix(blob: str) -> Optional[Dict]:
    """Validate that ``blob`` parses as a STIX 2.1 SDO. Returns the
    parsed object on success, None on any failure."""
    if not blob:
        return None
    blob = _strip_code_fence(blob)
    try:
        obj = json.loads(blob)
    except json.JSONDecodeError:
        return None
    if not isinstance(obj, dict):
        return None
    if not _STIX_REQUIRED_KEYS.issubset(obj.keys()):
        return None
    if obj.get("spec_version") != "2.1":
        return None
    return obj


# ---------------------------------------------------------------------------
# YARA rule
# ---------------------------------------------------------------------------


_YARA_RULE_RE = re.compile(r"^\s*rule\s+\w+\s*\{", re.MULTILINE)
_YARA_STRINGS_RE = re.compile(r"^\s*strings\s*:\s*", re.MULTILINE)
_YARA_CONDITION_RE = re.compile(r"^\s*condition\s*:\s*", re.MULTILINE)


def parse_yara(blob: str) -> Optional[str]:
    """Light-touch YARA validation: rule header + strings + condition
    + balanced braces. Returns the cleaned rule on success, None
    otherwise.

    A real YARA validator would invoke the ``yara`` CLI; that's a
    follow-up tier when the artifact is being exercised against a
    real ruleset."""
    if not blob:
        return None
    blob = _strip_code_fence(blob)
    if not _YARA_RULE_RE.search(blob):
        return None
    if not _YARA_STRINGS_RE.search(blob):
        return None
    if not _YARA_CONDITION_RE.search(blob):
        return None
    if blob.count("{") != blob.count("}"):
        return None
    return blob


# ---------------------------------------------------------------------------
# Sigma rule
# ---------------------------------------------------------------------------


_SIGMA_REQUIRED_FIELDS = ("title", "logsource", "detection")


def parse_sigma(blob: str) -> Optional[Dict]:
    """Sigma rule validation: parses as YAML, has the required top-
    level keys, and the detection block has a ``condition``.

    Falls back to a regex-based field probe when PyYAML isn't
    installed (the production loop normally has it; CI on a fresh
    M4 might not)."""
    if not blob:
        return None
    blob = _strip_code_fence(blob)
    try:
        import yaml  # type: ignore[import-not-found]
        try:
            obj = yaml.safe_load(blob)
        except yaml.YAMLError:
            # Free-form prose, hallucinated narrative, malformed YAML
            # all land here; treat as 'not a Sigma rule'.
            return None
        if not isinstance(obj, dict):
            return None
        if not all(k in obj for k in _SIGMA_REQUIRED_FIELDS):
            return None
        det = obj.get("detection")
        if not isinstance(det, dict) or "condition" not in det:
            return None
        return obj
    except ImportError:
        for k in _SIGMA_REQUIRED_FIELDS:
            if not re.search(rf"^\s*{k}\s*:", blob, re.MULTILINE):
                return None
        if not re.search(r"^\s*condition\s*:", blob, re.MULTILINE):
            return None
        return {"_unparsed": blob}


# ---------------------------------------------------------------------------
# MISP event
# ---------------------------------------------------------------------------


def parse_misp(blob: str) -> Optional[Dict]:
    """Validate MISP event structure: ``Event`` shell + populated
    ``Attribute`` array of typed IOCs."""
    if not blob:
        return None
    blob = _strip_code_fence(blob)
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
    if not all(
        isinstance(a, dict) and a.get("type") and a.get("value") for a in attrs
    ):
        return None
    return obj


# ---------------------------------------------------------------------------
# Provenance / cite tags
# ---------------------------------------------------------------------------


_CITE_TAG_RE = re.compile(r"<\|cite\|>([^<]+)<\|/cite\|>")


def parse_provenance(blob: str) -> Optional[List[str]]:
    """Return the list of well-formed cite tag bodies found in ``blob``.

    A cite is well-formed if it has the shape ``source_type:source_id``
    (contains a colon, both halves non-empty). Returns the list of
    valid bodies on success (>=1), or None if the prediction has no
    well-formed cites."""
    if not blob:
        return None
    matches = _CITE_TAG_RE.findall(blob)
    valid: List[str] = []
    for m in matches:
        m = m.strip()
        if not m or ":" not in m:
            continue
        head, _, tail = m.partition(":")
        if head and tail:
            valid.append(m)
    return valid if valid else None


# ---------------------------------------------------------------------------
# Default parser registry
# ---------------------------------------------------------------------------


DEFAULT_PARSERS = {
    "stix_indicator": parse_stix,
    "yara_rule": parse_yara,
    "sigma_rule": parse_sigma,
    "misp_event": parse_misp,
    "provenance": parse_provenance,
    # ``code_security`` and ``binary_literacy`` deliberately have no
    # parser registered; the scorer treats parse as vacuously True
    # and falls back to substring scoring for those bets.
}
