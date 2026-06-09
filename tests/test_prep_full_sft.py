"""Tests for prep_full_sft.py — the trace parser + chat-record converter."""

import importlib.util
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))


def _load():
    spec = importlib.util.spec_from_file_location(
        "prep_full_sft", str(REPO_ROOT / "scripts" / "prep_full_sft.py"),
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_parse_two_message_qa():
    m = _load()
    text = "USER: What does this code do?\nASSISTANT: It sorts a list."
    turns = m.parse_trace(text)
    assert turns is not None
    assert len(turns) == 2
    assert turns[0]["role"] == "user"
    assert turns[0]["content"] == "What does this code do?"
    assert turns[1]["role"] == "assistant"
    assert turns[1]["content"] == "It sorts a list."


def test_parse_four_message_tool_trace():
    m = _load()
    text = (
        "USER: What is CVE-2020-5179?\n"
        "ASSISTANT: <|tool_call|>{\"name\":\"search_cve_nvd\"}<|/tool_call|>\n"
        "TOOL: <|tool_response|>{\"cve\":\"CVE-2020-5179\"}<|/tool_response|>\n"
        "ASSISTANT: It is an OS command injection."
    )
    turns = m.parse_trace(text)
    assert turns is not None
    assert len(turns) == 4
    assert turns[0]["role"] == "user"
    assert turns[1]["role"] == "assistant"
    assert "tool_call" in turns[1]["content"]
    # TOOL gets folded into a user turn (ChatDataset has no tool role).
    assert turns[2]["role"] == "user"
    assert "tool_response" in turns[2]["content"]
    assert turns[3]["role"] == "assistant"
    assert turns[3]["content"] == "It is an OS command injection."


def test_parse_multiline_assistant():
    m = _load()
    text = (
        "USER: explain Rust ownership\n"
        "ASSISTANT: Ownership rules:\n"
        "1. each value has one owner\n"
        "2. when owner goes out of scope, value drops\n"
        "3. moves transfer ownership"
    )
    turns = m.parse_trace(text)
    assert turns is not None
    assert len(turns) == 2
    assert "1. each value" in turns[1]["content"]
    assert "3. moves" in turns[1]["content"]


def test_parse_rejects_no_user_first():
    m = _load()
    assert m.parse_trace("ASSISTANT: hi") is None
    assert m.parse_trace("") is None
    assert m.parse_trace(None) is None


def test_parse_rejects_no_final_assistant():
    m = _load()
    assert m.parse_trace("USER: hi") is None


def test_trace_to_chat_record_shape():
    m = _load()
    turns = [{"role": "user", "content": "hi"},
             {"role": "assistant", "content": "hello"}]
    rec = {"source": "synth_code_explain", "seed_source": "identify_lang",
           "seed_id": "ce_001"}
    out = m.trace_to_chat_record(turns, rec)
    assert out["turns"] == turns
    assert out["source"] == "synth_code_explain"
    assert out["seed_source"] == "identify_lang"
    assert out["seed_id"] == "ce_001"


def test_hash_for_split_stable():
    m = _load()
    rec = {"source": "synth_x", "seed_source": "y", "seed_id": "1"}
    a = m.hash_for_split(rec)
    b = m.hash_for_split(rec)
    c = m.hash_for_split({**rec, "seed_id": "2"})
    assert a == b
    assert a != c


def test_end_to_end_via_subprocess(tmp_path):
    """Smoke: build a tiny combined synth + base chat, run main(), check outputs."""
    import subprocess

    combined = tmp_path / "synth_v15_combined.jsonl"
    base_train = tmp_path / "base_train.jsonl"
    base_val = tmp_path / "base_val.jsonl"
    out_train = tmp_path / "out_train.jsonl"
    out_val = tmp_path / "out_val.jsonl"

    # Combined synth: 3 SFT records + 2 pretrain (skipped) + 1 unparseable.
    sft_records = [
        {"source": "synth_code_explain", "seed_source": "identify_lang",
         "seed_id": f"ce_{i}", "format_type": "sft",
         "text": f"USER: q{i}?\nASSISTANT: a{i}."}
        for i in range(3)
    ]
    pretrain_records = [
        {"source": "synth_code_explain", "seed_source": "pretrain_prose",
         "seed_id": f"pp_{i}", "format_type": "pretrain",
         "text": f"prose blob {i}"}
        for i in range(2)
    ]
    unparseable = [{"source": "x", "seed_source": "y", "seed_id": "z",
                    "format_type": "sft", "text": "no role prefixes here"}]
    with combined.open("w") as f:
        for r in sft_records + pretrain_records + unparseable:
            f.write(json.dumps(r) + "\n")

    base_train.write_text(json.dumps({
        "turns": [{"role": "user", "content": "hi"},
                   {"role": "assistant", "content": "hi back"}],
        "source": "small_talk",
    }) + "\n")
    base_val.write_text("")

    result = subprocess.run(
        [sys.executable, str(REPO_ROOT / "scripts" / "prep_full_sft.py"),
         "--in-combined", str(combined),
         "--base-train", str(base_train),
         "--base-val", str(base_val),
         "--out-train", str(out_train),
         "--out-val", str(out_val)],
        capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0, result.stderr
    train = [json.loads(line) for line in out_train.read_text().splitlines() if line]
    val = [json.loads(line) for line in out_val.read_text().splitlines() if line]
    # 1 base + 3 synth = 4 total split into train+val.
    assert len(train) + len(val) == 4
    # Base small_talk record should appear in train.
    sources = [r["source"] for r in train + val]
    assert "small_talk" in sources
    assert sources.count("synth_code_explain") == 3
