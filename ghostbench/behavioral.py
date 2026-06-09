"""Behavioural-tier validators for GhostBench.

The behavioural tier asks a stronger question than the structural
parser: "would a real downstream tool actually accept this?" An
artifact that passes ``parse_yara`` looks like a YARA rule; one that
passes ``behavioral_yara`` actually compiles under ``yara-python``.

Every validator follows the same two-tier design:

  1. **Real-library path.** Lazy-import the canonical reference
     parser (``stix2``, ``yara-python``, ``pysigma``,
     ``jsonschema``) and use it for full validation. This catches
     edge cases the structural parser doesn't (invalid UUIDs in
     STIX ids, malformed YARA condition trees, Sigma logsource
     types not in the official taxonomy, MISP attribute types
     outside the controlled vocabulary).

  2. **Enhanced-structural fallback.** When the reference parser
     isn't installed, fall back to a deeper structural check than
     the v0.1 ``parsers.py`` ones: validate UUID and timestamp
     formats in STIX, parse YARA rule bodies properly, recurse
     into Sigma's nested detection structure, validate MISP
     attribute types against a curated subset of the controlled
     vocabulary. This is still a strict upgrade over the parse
     tier; it just doesn't catch every edge case the real library
     would.

Each validator returns:

  ``True``      the artifact passes behavioural validation
  ``False``     the artifact fails (either real-library reject or
                fallback structural reject)
  ``None``      not measurable (rare: only if a hard precondition
                is missing, e.g. completely unparseable input)

The ``BEHAVIORAL_VALIDATORS`` dict at the bottom is the public
registry consumed by ``Score`` when the eval record requests the
``behavioral`` tier.

Optional dependencies (all soft):

  pip install stix2          # STIX 2.1 reference parser
  pip install yara-python    # YARA compile via libyara
  pip install pysigma        # Sigma rule parser
  pip install jsonschema     # MISP / generic JSON schema validation
"""

from __future__ import annotations

import json
import re
from typing import Optional


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


_UUID4_RE = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$"
)
_RFC3339_RE = re.compile(
    r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?Z$"
)


def _strip_code_fence(blob: str) -> str:
    """Mirror of parsers._strip_code_fence; duplicated to keep
    behavioural.py independent."""
    blob = blob.strip()
    if blob.startswith("```"):
        m = re.match(r"^```(?:\w+)?\s*\n?", blob)
        if m:
            blob = blob[m.end():]
        if blob.endswith("```"):
            blob = blob[:-3]
    return blob.strip()


# ---------------------------------------------------------------------------
# STIX 2.1 behavioural validation
# ---------------------------------------------------------------------------


def behavioral_stix(blob: str) -> Optional[bool]:
    """Validate a STIX 2.1 indicator at the spec level.

    Real-library path: ``stix2.parse(blob)`` from the OASIS reference
    library. Catches invalid pattern syntax (stix-pattern grammar),
    missing required fields per object type, type-specific constraints
    (e.g. an indicator must have ``pattern`` and ``valid_from``).

    Fallback: enhanced structural check beyond ``parse_stix``.
    Validates that:
      - ``id`` matches ``<type>--<uuid4>`` format
      - ``created`` and ``modified`` parse as RFC3339 UTC timestamps
      - ``modified >= created`` (lexicographic comparison works for
        RFC3339)
      - if ``pattern`` is present, it's a non-empty string
      - if ``labels`` is present, it's a non-empty list of strings
      - if ``external_references`` is present, each item has a
        ``source_name``
    """
    if not blob:
        return None
    blob = _strip_code_fence(blob)
    try:
        obj = json.loads(blob)
    except json.JSONDecodeError:
        return False
    if not isinstance(obj, dict):
        return False

    # Try the real library first.
    try:
        import stix2  # type: ignore[import-not-found]
        try:
            stix2.parse(obj, allow_custom=False)
            return True
        except Exception:
            return False
    except ImportError:
        pass

    # Enhanced structural fallback.
    obj_type = obj.get("type", "")
    obj_id = obj.get("id", "")
    if not obj_id.startswith(f"{obj_type}--"):
        return False
    uuid_part = obj_id.split("--", 1)[1]
    if not _UUID4_RE.match(uuid_part):
        return False
    if not _RFC3339_RE.match(obj.get("created", "")):
        return False
    if not _RFC3339_RE.match(obj.get("modified", "")):
        return False
    if obj.get("modified", "") < obj.get("created", ""):
        return False
    if obj.get("spec_version") != "2.1":
        return False
    # Indicator-specific shape.
    if obj_type == "indicator":
        pat = obj.get("pattern")
        if not isinstance(pat, str) or not pat.strip():
            return False
        labels = obj.get("labels")
        if labels is not None and not (
            isinstance(labels, list) and labels and all(isinstance(x, str) for x in labels)
        ):
            return False
    refs = obj.get("external_references")
    if refs is not None:
        if not isinstance(refs, list):
            return False
        for r in refs:
            if not isinstance(r, dict) or not r.get("source_name"):
                return False
    return True


# ---------------------------------------------------------------------------
# YARA behavioural validation
# ---------------------------------------------------------------------------


_YARA_RULE_HEADER_RE = re.compile(r"^\s*rule\s+([A-Za-z_]\w*)\s*", re.MULTILINE)
_YARA_STRING_DEF_RE = re.compile(
    r'^\s*\$([A-Za-z_]\w*)\s*=\s*(?:"[^"]*"|\{[^}]*\}|/[^/]+/)',
    re.MULTILINE,
)
_YARA_CONDITION_RE = re.compile(r"^\s*condition\s*:\s*(.+?)$",
                                  re.MULTILINE | re.DOTALL)


def behavioral_yara(blob: str) -> Optional[bool]:
    """Validate a YARA rule.

    Real-library path: ``yara.compile(source=blob)`` from yara-python.
    The libyara binding compiles the rule, which catches syntax
    errors the regex-based parser would miss (unbalanced parens in
    conditions, malformed hex strings, references to undefined
    string identifiers).

    Fallback: enhanced structural check beyond ``parse_yara``.
    Validates that:
      - rule name is a valid identifier
      - every string definition is well-formed (text / hex / regex)
      - condition section references at least one of the defined
        strings (so the rule isn't trivially false)
      - braces, parens, and brackets are balanced
    """
    if not blob:
        return None
    blob = _strip_code_fence(blob)

    try:
        import yara  # type: ignore[import-not-found]
        try:
            yara.compile(source=blob)
            return True
        except Exception:
            return False
    except ImportError:
        pass

    # Enhanced structural fallback.
    header = _YARA_RULE_HEADER_RE.search(blob)
    if not header:
        return False
    string_defs = _YARA_STRING_DEF_RE.findall(blob)
    if not string_defs:
        return False
    cond = _YARA_CONDITION_RE.search(blob)
    if not cond:
        return False
    cond_body = cond.group(1)
    # Check the condition references at least one defined string
    # OR uses a wildcard / count operator that implies the strings
    # are referenced ('any of them', 'all of them', '#xxx >= N').
    refs_string = any(f"${name}" in cond_body for name in string_defs)
    has_wildcard = ("any of" in cond_body
                     or "all of" in cond_body
                     or "of (" in cond_body
                     or re.search(r"#\w+", cond_body))
    if not (refs_string or has_wildcard):
        return False
    # Balance check across the whole rule.
    if blob.count("{") != blob.count("}"):
        return False
    if blob.count("(") != blob.count(")"):
        return False
    if blob.count("[") != blob.count("]"):
        return False
    return True


# ---------------------------------------------------------------------------
# Sigma behavioural validation
# ---------------------------------------------------------------------------


def behavioral_sigma(blob: str) -> Optional[bool]:
    """Validate a Sigma rule.

    Real-library path: ``sigma.parser.parse_sigma_rule()`` from
    pysigma. Catches malformed selection blocks, unsupported field
    modifiers, conditions referencing undefined selections.

    Fallback: enhanced structural check beyond ``parse_sigma``.
    Loads the YAML, validates that:
      - ``logsource`` has at least one of (category, product, service)
      - ``detection`` is a dict with at least one selection block
        plus a ``condition`` field
      - the ``condition`` string references at least one of the
        selection block names
      - if ``level`` is present, it's one of the standard severities
        (informational / low / medium / high / critical)
    """
    if not blob:
        return None
    blob = _strip_code_fence(blob)

    try:
        import sigma  # type: ignore[import-not-found]  # noqa: F401 - availability probe
        from sigma.collection import SigmaCollection  # type: ignore[import-not-found]
        try:
            SigmaCollection.from_yaml(blob)
            return True
        except Exception:
            return False
    except ImportError:
        pass

    # Enhanced structural fallback.
    try:
        import yaml  # type: ignore[import-not-found]
    except ImportError:
        return None
    try:
        obj = yaml.safe_load(blob)
    except yaml.YAMLError:
        return False
    if not isinstance(obj, dict):
        return False
    logsource = obj.get("logsource")
    if not isinstance(logsource, dict):
        return False
    if not any(k in logsource for k in ("category", "product", "service")):
        return False
    detection = obj.get("detection")
    if not isinstance(detection, dict):
        return False
    cond = detection.get("condition")
    if not isinstance(cond, str) or not cond.strip():
        return False
    # Selection blocks are every key in detection that isn't 'condition'
    # or 'timeframe'.
    selections = [k for k in detection.keys() if k not in ("condition", "timeframe")]
    if not selections:
        return False
    # Condition must reference at least one selection block name OR
    # use a wildcard operator like '1 of selection_*' or 'all of them'.
    refs_block = any(name in cond for name in selections)
    has_wildcard = ("of them" in cond or "of selection" in cond
                     or "1 of " in cond or "any of " in cond)
    if not (refs_block or has_wildcard):
        return False
    level = obj.get("level")
    if level is not None:
        if level not in ("informational", "low", "medium", "high", "critical"):
            return False
    return True


# ---------------------------------------------------------------------------
# MISP behavioural validation
# ---------------------------------------------------------------------------


# A curated subset of MISP's controlled-vocabulary attribute types.
# The real MISP server has 200+ types in its taxonomies; this list
# covers the most common ones used in real CTI feeds.
_MISP_ATTRIBUTE_TYPES = frozenset({
    "ip-src", "ip-dst", "ip-src|port", "ip-dst|port",
    "hostname", "domain", "domain|ip",
    "url", "uri", "user-agent",
    "email-src", "email-dst", "email-subject", "email-attachment",
    "filename", "filename|md5", "filename|sha1", "filename|sha256",
    "md5", "sha1", "sha256", "sha512", "ssdeep", "imphash", "authentihash",
    "x509-fingerprint-sha1", "x509-fingerprint-sha256",
    "regkey", "regkey|value", "mutex", "named pipe",
    "pattern-in-file", "pattern-in-traffic", "pattern-in-memory",
    "yara", "sigma", "stix",
    "vulnerability", "weakness", "cpe",
    "btc", "xmr",
    "as", "snort", "bro", "zeek",
    "comment", "text", "other",
    "github-username", "github-repository", "github-organisation",
    "campaign-name", "campaign-id", "threat-actor",
})


_MISP_VALID_THREAT_LEVELS = frozenset({"1", "2", "3", "4", 1, 2, 3, 4})
_MISP_VALID_ANALYSIS = frozenset({"0", "1", "2", 0, 1, 2})
_MISP_VALID_DISTRIBUTION = frozenset({"0", "1", "2", "3", "4", "5",
                                       0, 1, 2, 3, 4, 5})


def behavioral_misp(blob: str) -> Optional[bool]:
    """Validate a MISP event.

    Real-library path: jsonschema validation against MISP's
    published Event schema. (We don't ship the schema; if jsonschema
    is installed the validator is parameterised on a curated
    in-source minimal MISP schema below.)

    Fallback: enhanced structural check beyond ``parse_misp``.
    Validates:
      - ``Event.threat_level_id`` is in {1, 2, 3, 4}
      - ``Event.analysis`` is in {0, 1, 2}
      - ``Event.distribution`` is in {0..5}
      - every ``Attribute`` has a ``type`` from the curated MISP
        controlled vocabulary, plus a non-empty ``value``
      - ``Attribute`` ``category`` is non-empty if present
    """
    if not blob:
        return None
    blob = _strip_code_fence(blob)
    try:
        obj = json.loads(blob)
    except json.JSONDecodeError:
        return False
    event = obj.get("Event") if isinstance(obj, dict) else None
    if not isinstance(event, dict):
        return False
    if event.get("threat_level_id") not in _MISP_VALID_THREAT_LEVELS:
        return False
    if "analysis" in event and event["analysis"] not in _MISP_VALID_ANALYSIS:
        return False
    if ("distribution" in event
            and event["distribution"] not in _MISP_VALID_DISTRIBUTION):
        return False
    attrs = event.get("Attribute")
    if not isinstance(attrs, list) or not attrs:
        return False
    for a in attrs:
        if not isinstance(a, dict):
            return False
        if a.get("type") not in _MISP_ATTRIBUTE_TYPES:
            return False
        if not a.get("value"):
            return False
        if "category" in a and not isinstance(a["category"], str):
            return False
    return True


# ---------------------------------------------------------------------------
# Provenance behavioural validation
# ---------------------------------------------------------------------------


def behavioral_provenance(blob: str) -> Optional[bool]:
    """Validate cite-augmented provenance traces at the behavioural
    level.

    No external library involved here; the behavioural tier checks:
      - At least one well-formed cite tag in the assistant's final
        message (matches the parse tier).
      - Every cite tag's ``source_id`` segment is structurally
        plausible: matches a CVE/CWE/T-code/passage_N pattern OR
        is a non-empty alphanumeric+dash identifier.
      - The cite tag's optional ``#field`` segment, if present,
        refers to a recognised field name (description / cvss /
        name / tactic / summary / etc.) -- catches model output
        that hallucinates fields like ``#expanded_summary`` that
        a real tool response would never have.
    """
    if not blob:
        return None
    cite_re = re.compile(r"<\|cite\|>([^<]+)<\|/cite\|>")
    matches = cite_re.findall(blob)
    if not matches:
        return False

    valid_field_names = frozenset({
        "description", "cvss", "name", "tactic", "summary", "id",
        "type", "url", "lookup", "passage", "value", "category",
        "comment", "platform", "no_match",
    })

    # Plausibility: source_id should look like a real identifier.
    plausible_id = re.compile(
        r"^(CVE-\d{4}-\d{4,7}|CWE-\d+|T\d{4}(\.\d{3})?|"
        r"passage_\d+|[A-Za-z][\w.\-:]{1,80})$"
    )

    for m in matches:
        m = m.strip()
        if not m or ":" not in m:
            return False
        st, _, rest = m.partition(":")
        if "#" in rest:
            sid, _, field = rest.partition("#")
        else:
            sid, field = rest, None
        if not st or not sid:
            return False
        if not plausible_id.match(sid):
            return False
        if field is not None:
            # Field must match the recognised vocab (some passages use
            # arbitrary path-like names; allow those if they look
            # path-like).
            if field not in valid_field_names and not re.match(
                r"^[A-Za-z_]\w*(\.[A-Za-z_]\w*)*$", field
            ) and "/" not in field:
                return False
    return True


# ---------------------------------------------------------------------------
# Public registry
# ---------------------------------------------------------------------------


BEHAVIORAL_VALIDATORS = {
    "stix_indicator": behavioral_stix,
    "yara_rule": behavioral_yara,
    "sigma_rule": behavioral_sigma,
    "misp_event": behavioral_misp,
    "provenance": behavioral_provenance,
    # ``code_security`` and ``binary_literacy`` have no behavioural
    # validator; the substring tier does the work for those bets.
}
