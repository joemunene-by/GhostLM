"""Tests for ghostbench.parsers."""

import pytest

from ghostbench.parsers import (
    DEFAULT_PARSERS, parse_misp, parse_provenance, parse_sigma,
    parse_stix, parse_yara,
)


def test_parse_stix_valid():
    valid = (
        '{"type":"indicator","spec_version":"2.1","id":"x",'
        '"created":"x","modified":"x"}'
    )
    assert parse_stix(valid) is not None


def test_parse_stix_wrong_spec_version():
    bad = (
        '{"type":"indicator","spec_version":"2.0","id":"x",'
        '"created":"x","modified":"x"}'
    )
    assert parse_stix(bad) is None


def test_parse_stix_garbage():
    assert parse_stix("not json") is None
    assert parse_stix("") is None
    assert parse_stix(None) is None


def test_parse_stix_strips_code_fence():
    """STIX inside a triple-backtick fence still parses."""
    valid = (
        '```json\n'
        '{"type":"indicator","spec_version":"2.1","id":"x",'
        '"created":"x","modified":"x"}\n'
        '```'
    )
    assert parse_stix(valid) is not None


def test_parse_yara_valid():
    rule = (
        "rule X {\n"
        "  meta:\n    author = \"a\"\n"
        "  strings:\n    $s1 = \"foo\"\n"
        "  condition:\n    $s1\n"
        "}"
    )
    assert parse_yara(rule) is not None


def test_parse_yara_missing_condition():
    rule = "rule X { strings: $s = \"x\" foo: $s }"
    assert parse_yara(rule) is None


def test_parse_yara_unbalanced_braces():
    rule = (
        "rule X {\n"
        "  strings:\n    $s = \"x\"\n"
        "  condition:\n    $s\n"
    )  # missing closing brace
    assert parse_yara(rule) is None


def test_parse_sigma_valid():
    """Without yaml installed this exercises the regex fallback."""
    rule = (
        "title: Test\n"
        "logsource:\n  product: linux\n"
        "detection:\n  selection:\n    foo: bar\n  condition: selection\n"
    )
    assert parse_sigma(rule) is not None


def test_parse_sigma_missing_detection():
    assert parse_sigma("title: x\nlogsource: y\n") is None


def test_parse_misp_valid():
    valid = (
        '{"Event":{"info":"x","Attribute":'
        '[{"type":"sha256","value":"abc"}]}}'
    )
    assert parse_misp(valid) is not None


def test_parse_misp_empty_attributes():
    assert parse_misp('{"Event":{"info":"x","Attribute":[]}}') is None


def test_parse_misp_missing_event():
    assert parse_misp('{"info":"x"}') is None


def test_parse_provenance_picks_well_formed():
    """Two cite tags, one well-formed and one missing the colon."""
    blob = ("ok <|cite|>nvd:CVE-2017-0144#description<|/cite|> "
            "and <|cite|>broken<|/cite|>")
    out = parse_provenance(blob)
    assert out == ["nvd:CVE-2017-0144#description"]


def test_parse_provenance_no_cites():
    assert parse_provenance("plain text without any cite") is None


def test_parse_provenance_only_malformed():
    blob = "<|cite|>no_colon_here<|/cite|>"
    assert parse_provenance(blob) is None


def test_default_parsers_registered():
    """The four format-aware parsers + provenance are registered.
    code_security and binary_literacy are deliberately absent."""
    expected = {"stix_indicator", "yara_rule", "sigma_rule",
                "misp_event", "provenance"}
    assert set(DEFAULT_PARSERS.keys()) == expected
