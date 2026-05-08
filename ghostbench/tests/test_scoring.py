"""Tests for ghostbench.scoring."""

import pytest

from ghostbench.parsers import DEFAULT_PARSERS, parse_provenance, parse_stix
from ghostbench.scoring import Score, get_path, score_record


# ---------------------------------------------------------------------------
# get_path
# ---------------------------------------------------------------------------


def test_get_path_simple_dict():
    obj = {"a": {"b": "c"}}
    assert get_path(obj, "a.b") == "c"
    assert get_path(obj, "a.x") is None
    assert get_path(obj, "missing") is None


def test_get_path_list_index():
    obj = {"items": [{"x": 1}, {"x": 2}, {"x": 3}]}
    assert get_path(obj, "items.0.x") == 1
    assert get_path(obj, "items.2.x") == 3
    assert get_path(obj, "items.5.x") is None


def test_get_path_handles_none():
    """Walking through None returns None instead of raising."""
    assert get_path(None, "a.b") is None
    assert get_path({"a": None}, "a.b") is None


# ---------------------------------------------------------------------------
# Score (data class behaviour)
# ---------------------------------------------------------------------------


def test_score_passed_strict_and():
    """``passed`` is the strict-AND across requested tiers."""
    s = Score(seed_id="x", fmt="f",
              requested_tiers=("parse", "fields", "substrings"),
              tier_results={"parse": True, "fields": True, "substrings": True})
    assert s.passed is True
    s2 = Score(seed_id="x", fmt="f",
               requested_tiers=("parse", "fields", "substrings"),
               tier_results={"parse": True, "fields": False, "substrings": True})
    assert s2.passed is False


def test_score_unrequested_tier_does_not_drag():
    """If a tier was not requested, its absence from tier_results
    does not flip ``passed`` to False."""
    s = Score(seed_id="x", fmt="f", requested_tiers=("substrings",),
              tier_results={"substrings": True})
    assert s.passed is True


# ---------------------------------------------------------------------------
# score_record (end-to-end)
# ---------------------------------------------------------------------------


def test_score_record_no_parser_substring_pass():
    """Bet 7 / 8 shape: no parser registered, only substrings."""
    rec = {
        "format": "code_security",
        "prompt": "what CWE is X",
        "required_substrings": ["CWE-89", "parameterized"],
        "seed_id": "case1",
    }
    pred = "The answer is CWE-89, fix is to use parameterized queries."
    score = score_record(rec, pred, DEFAULT_PARSERS)
    assert score.passed
    assert "parse" not in score.requested_tiers
    assert score.tier_pass("substrings") is True


def test_score_record_no_parser_substring_fail():
    rec = {
        "format": "code_security",
        "prompt": "what CWE is X",
        "required_substrings": ["CWE-89"],
        "seed_id": "case1",
    }
    pred = "Different answer entirely."
    score = score_record(rec, pred, DEFAULT_PARSERS)
    assert not score.passed
    assert "missing substring: 'CWE-89'" in score.tier_misses["substrings"]


def test_score_record_stix_pass():
    rec = {
        "format": "stix_indicator",
        "prompt": "make a STIX indicator",
        "required_substrings": ["CVE-2017-0144"],
        "seed_id": "case_stix",
    }
    pred = (
        '{"type":"indicator","spec_version":"2.1",'
        '"id":"indicator--abc12345-1234-1234-1234-123456789abc",'
        '"created":"2017-03-14T00:00:00.000Z",'
        '"modified":"2017-03-14T00:00:00.000Z",'
        '"pattern_type":"stix",'
        '"pattern":"[network-traffic:dst_port = 445]",'
        '"valid_from":"2017-03-14T00:00:00Z",'
        '"labels":["malicious-activity"],'
        '"name":"CVE-2017-0144 EternalBlue SMBv1 RCE"}'
    )
    score = score_record(rec, pred, DEFAULT_PARSERS)
    assert score.tier_pass("parse") is True
    assert score.tier_pass("substrings") is True
    assert score.passed is True


def test_score_record_provenance_no_cite():
    """Provenance bet: predictions without cite tags fail parse."""
    rec = {
        "format": "provenance",
        "prompt": "tell me about CVE-X",
        "required_substrings": ["<|cite|>", "CVE-2017"],
        "seed_id": "case_prov",
    }
    pred = "CVE-2017-0144 is an SMB exploit. No cite here."
    score = score_record(rec, pred, DEFAULT_PARSERS)
    assert score.tier_pass("parse") is False


def test_score_record_provenance_with_cite():
    rec = {
        "format": "provenance",
        "prompt": "tell me about CVE-X",
        "required_substrings": ["<|cite|>", "CVE-2017"],
        "seed_id": "case_prov2",
    }
    pred = ("CVE-2017-0144 is an SMB exploit "
            "<|cite|>nvd:CVE-2017-0144#description<|/cite|>.")
    score = score_record(rec, pred, DEFAULT_PARSERS)
    assert score.tier_pass("parse") is True
    assert score.tier_pass("substrings") is True
    assert score.passed is True


def test_score_record_required_fields_path():
    """STIX field check resolves dotted paths into the parsed object."""
    rec = {
        "format": "stix_indicator",
        "prompt": "make a STIX indicator",
        "required_fields": [{"path": "type", "value": "indicator"}],
        "seed_id": "case_field",
    }
    pred = (
        '{"type":"indicator","spec_version":"2.1","id":"x",'
        '"created":"x","modified":"x"}'
    )
    score = score_record(rec, pred, DEFAULT_PARSERS)
    assert score.tier_pass("fields") is True


def test_score_record_required_fields_with_no_parser():
    """If parsing fails (or there's no parser) but required_fields
    is non-empty, fields tier is False with an unparseable miss."""
    rec = {
        "format": "code_security",
        "prompt": "x",
        "required_fields": [{"path": "type", "value": "indicator"}],
        "seed_id": "case_xfield",
    }
    score = score_record(rec, "anything", DEFAULT_PARSERS)
    assert score.tier_pass("fields") is False
