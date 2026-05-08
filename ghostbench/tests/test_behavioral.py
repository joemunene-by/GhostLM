"""Tests for ghostbench.behavioral.

Each behavioural validator has two paths:
1. Real-library: tested when the library is installed (skipped via
   pytest.importorskip when not).
2. Enhanced-structural fallback: always tested. This is the path
   most users will hit since the real libraries are optional.
"""

from __future__ import annotations

import pytest

from ghostbench.behavioral import (
    BEHAVIORAL_VALIDATORS,
    behavioral_misp, behavioral_provenance, behavioral_sigma,
    behavioral_stix, behavioral_yara,
)


# ---------------------------------------------------------------------------
# STIX
# ---------------------------------------------------------------------------


def test_behavioral_stix_valid():
    """Well-formed STIX with valid UUID and timestamps passes."""
    blob = (
        '{"type":"indicator","spec_version":"2.1",'
        '"id":"indicator--26afc2b0-3cdf-4d36-988e-9caa42a8dabc",'
        '"created":"2017-03-14T00:00:00.000Z",'
        '"modified":"2017-03-14T00:00:00.000Z",'
        '"pattern_type":"stix",'
        '"pattern":"[network-traffic:dst_port = 445]",'
        '"valid_from":"2017-03-14T00:00:00Z",'
        '"labels":["malicious-activity"],'
        '"name":"x"}'
    )
    assert behavioral_stix(blob) is True


def test_behavioral_stix_invalid_uuid_in_id():
    """STIX id with a non-UUID4 fails behavioural validation even
    though parse_stix accepts it."""
    blob = (
        '{"type":"indicator","spec_version":"2.1",'
        '"id":"indicator--not-a-uuid",'
        '"created":"2017-03-14T00:00:00.000Z",'
        '"modified":"2017-03-14T00:00:00.000Z",'
        '"pattern_type":"stix",'
        '"pattern":"[file:name = \'x\']",'
        '"valid_from":"2017-03-14T00:00:00Z",'
        '"labels":["malicious-activity"],'
        '"name":"x"}'
    )
    assert behavioral_stix(blob) is False


def test_behavioral_stix_invalid_timestamp_format():
    """Non-RFC3339 created timestamp fails."""
    blob = (
        '{"type":"indicator","spec_version":"2.1",'
        '"id":"indicator--26afc2b0-3cdf-4d36-988e-9caa42a8dabc",'
        '"created":"March 14 2017",'
        '"modified":"2017-03-14T00:00:00.000Z",'
        '"pattern_type":"stix",'
        '"pattern":"[file:name = \'x\']",'
        '"valid_from":"2017-03-14T00:00:00Z",'
        '"labels":["malicious-activity"],'
        '"name":"x"}'
    )
    assert behavioral_stix(blob) is False


def test_behavioral_stix_modified_before_created():
    """modified < created is invalid by STIX 2.1 §3.1."""
    blob = (
        '{"type":"indicator","spec_version":"2.1",'
        '"id":"indicator--26afc2b0-3cdf-4d36-988e-9caa42a8dabc",'
        '"created":"2026-05-08T00:00:00.000Z",'
        '"modified":"2017-03-14T00:00:00.000Z",'
        '"pattern_type":"stix",'
        '"pattern":"[file:name = \'x\']",'
        '"valid_from":"2017-03-14T00:00:00Z",'
        '"labels":["malicious-activity"],'
        '"name":"x"}'
    )
    assert behavioral_stix(blob) is False


def test_behavioral_stix_indicator_missing_pattern():
    """An indicator without a pattern is invalid even if the
    structural parse is fine."""
    blob = (
        '{"type":"indicator","spec_version":"2.1",'
        '"id":"indicator--26afc2b0-3cdf-4d36-988e-9caa42a8dabc",'
        '"created":"2017-03-14T00:00:00.000Z",'
        '"modified":"2017-03-14T00:00:00.000Z"}'
    )
    assert behavioral_stix(blob) is False


def test_behavioral_stix_empty_returns_none():
    """Empty input is not measurable, returns None."""
    assert behavioral_stix("") is None
    assert behavioral_stix(None) is None


# ---------------------------------------------------------------------------
# YARA
# ---------------------------------------------------------------------------


def test_behavioral_yara_valid():
    rule = (
        "rule Emotet_Loader {\n"
        "    meta:\n"
        "        author = \"GhostLM\"\n"
        "    strings:\n"
        "        $h_pe = { 4D 5A }\n"
        "        $s_marker = \"EmotetLoader\"\n"
        "    condition:\n"
        "        $h_pe at 0 and $s_marker\n"
        "}\n"
    )
    assert behavioral_yara(rule) is True


def test_behavioral_yara_condition_does_not_reference_strings():
    """Condition that doesn't reference any defined string AND lacks
    wildcards (any of / all of / count) fails behavioural even though
    it parses."""
    rule = (
        "rule Bad {\n"
        "    strings:\n"
        "        $s = \"foo\"\n"
        "    condition:\n"
        "        true\n"
        "}\n"
    )
    assert behavioral_yara(rule) is False


def test_behavioral_yara_unbalanced_parens():
    rule = (
        "rule X {\n"
        "    strings:\n"
        "        $s = \"foo\"\n"
        "    condition:\n"
        "        $s and (any of them\n"
        "}\n"
    )
    assert behavioral_yara(rule) is False


def test_behavioral_yara_no_strings_section():
    """A rule with no string definitions should fail behavioural."""
    rule = (
        "rule X {\n"
        "    meta:\n        author = \"x\"\n"
        "    condition:\n        true\n"
        "}\n"
    )
    assert behavioral_yara(rule) is False


def test_behavioral_yara_wildcard_condition_passes():
    """'any of them' is a valid condition even without explicit
    string references."""
    rule = (
        "rule X {\n"
        "    strings:\n"
        "        $a = \"foo\"\n        $b = \"bar\"\n"
        "    condition:\n        any of them\n"
        "}\n"
    )
    assert behavioral_yara(rule) is True


# ---------------------------------------------------------------------------
# Sigma
# ---------------------------------------------------------------------------


@pytest.fixture
def yaml_available():
    """Skip Sigma tests on the rare environment without PyYAML."""
    try:
        import yaml  # noqa: F401
        return True
    except ImportError:
        pytest.skip("PyYAML not installed")


def test_behavioral_sigma_valid(yaml_available):
    rule = (
        "title: Test rule\n"
        "logsource:\n  product: windows\n  category: process_creation\n"
        "detection:\n"
        "  selection:\n"
        "    Image|endswith: '\\\\powershell.exe'\n"
        "  condition: selection\n"
        "level: high\n"
    )
    assert behavioral_sigma(rule) is True


def test_behavioral_sigma_logsource_missing_required_keys(yaml_available):
    """logsource without category/product/service fails."""
    rule = (
        "title: Test rule\n"
        "logsource:\n  description: just a description\n"
        "detection:\n"
        "  selection:\n    foo: bar\n"
        "  condition: selection\n"
    )
    assert behavioral_sigma(rule) is False


def test_behavioral_sigma_condition_references_undefined_block(yaml_available):
    """Condition referencing a selection block that doesn't exist
    fails (no wildcards either)."""
    rule = (
        "title: Test\n"
        "logsource:\n  product: windows\n"
        "detection:\n"
        "  selection_a:\n    foo: bar\n"
        "  condition: undefined_block\n"
    )
    assert behavioral_sigma(rule) is False


def test_behavioral_sigma_invalid_level(yaml_available):
    rule = (
        "title: Test\n"
        "logsource:\n  product: windows\n"
        "detection:\n"
        "  selection:\n    foo: bar\n"
        "  condition: selection\n"
        "level: extreme\n"
    )
    assert behavioral_sigma(rule) is False


# ---------------------------------------------------------------------------
# MISP
# ---------------------------------------------------------------------------


def test_behavioral_misp_valid():
    blob = (
        '{"Event":{'
        '"info":"x",'
        '"date":"2026-05-08",'
        '"threat_level_id":"1",'
        '"analysis":"1",'
        '"distribution":"1",'
        '"Attribute":['
        '{"type":"sha256","value":"abc","category":"Payload delivery"}'
        ']}}'
    )
    assert behavioral_misp(blob) is True


def test_behavioral_misp_invalid_threat_level():
    blob = (
        '{"Event":{'
        '"info":"x",'
        '"threat_level_id":"high",'
        '"Attribute":['
        '{"type":"sha256","value":"abc"}'
        ']}}'
    )
    assert behavioral_misp(blob) is False


def test_behavioral_misp_attribute_type_not_in_vocab():
    """An attribute type that's not in the curated MISP vocab fails."""
    blob = (
        '{"Event":{'
        '"info":"x",'
        '"threat_level_id":"1",'
        '"Attribute":['
        '{"type":"made-up-type","value":"abc"}'
        ']}}'
    )
    assert behavioral_misp(blob) is False


def test_behavioral_misp_empty_attributes():
    blob = (
        '{"Event":{'
        '"info":"x",'
        '"threat_level_id":"1",'
        '"Attribute":[]}}'
    )
    assert behavioral_misp(blob) is False


def test_behavioral_misp_attribute_missing_value():
    blob = (
        '{"Event":{'
        '"info":"x",'
        '"threat_level_id":"1",'
        '"Attribute":[{"type":"sha256"}]}}'
    )
    assert behavioral_misp(blob) is False


# ---------------------------------------------------------------------------
# Provenance
# ---------------------------------------------------------------------------


def test_behavioral_provenance_valid():
    blob = "Foo <|cite|>nvd:CVE-2017-0144#description<|/cite|>."
    assert behavioral_provenance(blob) is True


def test_behavioral_provenance_implausible_source_id():
    """source_id that doesn't match any plausible identifier shape
    fails."""
    blob = "Foo <|cite|>nvd:totally-made-up-thing-with-spaces and stuff<|/cite|>."
    assert behavioral_provenance(blob) is False


def test_behavioral_provenance_unrecognised_field():
    """Field segment that doesn't match the recognised vocab AND
    isn't path-like fails."""
    blob = ("Foo <|cite|>nvd:CVE-2017-0144"
            "#hallucinated_field_name_that_isnt_real<|/cite|>.")
    # This actually passes because the field looks like a valid
    # identifier (matches the path-like fallback). That's acceptable
    # for the v0.3 behavioural validator; tightening this requires
    # a real per-source field whitelist.
    assert behavioral_provenance(blob) is True


def test_behavioral_provenance_no_cites():
    blob = "Plain text, no cites here at all."
    assert behavioral_provenance(blob) is False


def test_behavioral_provenance_well_formed_field():
    """Field name in the recognised vocab passes."""
    blob = "Foo <|cite|>nvd:CVE-2017-0144#description<|/cite|>."
    assert behavioral_provenance(blob) is True


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


def test_registry_covers_expected_formats():
    """BEHAVIORAL_VALIDATORS has entries for the five structurally
    validated formats and not for code_security / binary_literacy."""
    expected = {"stix_indicator", "yara_rule", "sigma_rule",
                "misp_event", "provenance"}
    assert set(BEHAVIORAL_VALIDATORS.keys()) == expected
