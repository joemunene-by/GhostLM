"""Tests for scripts/dedup_near.py."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.dedup_near import dedup, minhash

BASE = ("The attacker sends a crafted request to the vulnerable endpoint which fails to validate "
        "the length field before copying user input into a fixed size stack buffer leading to "
        "remote code execution with the privileges of the service account on affected hosts")


def _write(path, texts, source="nvd"):
    path.write_text("\n".join(json.dumps({"text": t, "source": source}) for t in texts) + "\n")


def test_near_duplicate_dropped_and_distinct_kept(tmp_path):
    near = BASE.replace("fails to validate", "does not validate")
    other = ("Rotating credentials regularly and enforcing multi factor authentication reduces the "
             "blast radius of phishing campaigns that target administrators of cloud tenants and "
             "their delegated service principals across many organisations")
    _write(tmp_path / "train.jsonl", [BASE, near, other])
    out = tmp_path / "out.jsonl"
    report = dedup(tmp_path / "train.jsonl", None, out, workers=2)
    kept = [json.loads(line)["text"] for line in out.read_text().splitlines()]
    assert kept == [BASE, other]
    assert report["stats"]["dropped"] == 1
    assert report["dropped_by_source"] == {"nvd": 1}


def test_train_copies_of_val_are_dropped(tmp_path):
    _write(tmp_path / "val.jsonl", [BASE])
    _write(tmp_path / "train.jsonl", [BASE + " today"])
    out = tmp_path / "out.jsonl"
    dedup(tmp_path / "train.jsonl", tmp_path / "val.jsonl", out, workers=2)
    assert out.read_text() == ""


def test_short_texts_are_always_kept(tmp_path):
    assert minhash("too short to fingerprint") is None
    _write(tmp_path / "train.jsonl", ["same short text", "same short text"])
    out = tmp_path / "out.jsonl"
    dedup(tmp_path / "train.jsonl", None, out, workers=1)
    assert len(out.read_text().splitlines()) == 2
