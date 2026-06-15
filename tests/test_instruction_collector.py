"""Tests for the general instruction-data collector's record formatting."""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from scripts.collect_instruction import format_record
from data.collect import domain_of


def test_format_with_context():
    out = format_record("Summarize this.", "Long context here.", "A summary.")
    assert out == "Summarize this.\n\nLong context here.\n\nA summary."


def test_format_without_context():
    out = format_record("What is 2+2?", "", "4.")
    assert out == "What is 2+2?\n\n4."


def test_format_strips_whitespace():
    out = format_record("  Q  ", "  ", "  A  ")
    assert out == "Q\n\nA"


def test_format_missing_parts_returns_empty():
    assert format_record("", "ctx", "resp") == ""
    assert format_record("instr", "ctx", "") == ""


def test_instruction_source_maps_to_instruction_domain():
    # The collector's source tag must route to the budgeted domain.
    assert domain_of("instruction") == "instruction"
