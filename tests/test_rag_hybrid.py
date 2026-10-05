"""Tests for the hybrid (exact-ID + dense) retriever and its agent wiring."""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ghostlm.rag import HybridRetriever, extract_ids

CHUNKS = [
    {"source": "nvd", "ref": "CVE-2021-44228", "text": "Apache Log4j2 JNDI lookup remote code execution."},
    {"source": "nvd", "ref": "CVE-2021-45046", "text": "Incomplete fix for CVE-2021-44228 in Log4j 2.15.0."},
    {"source": "mitre", "ref": "T1059.001", "text": "PowerShell abuse for execution."},
    {"source": "cwe", "ref": "CWE-79", "text": "Cross-site scripting: improper neutralization of input."},
]


def _retriever(query_vec_for=None):
    rng = np.random.default_rng(0)
    matrix = rng.normal(size=(len(CHUNKS), 8)).astype(np.float32)
    matrix /= np.linalg.norm(matrix, axis=1, keepdims=True)
    # Default embed points at chunk 3, so dense search alone would rank XSS first.
    embed = query_vec_for or (lambda q: matrix[3])
    return HybridRetriever(matrix, CHUNKS, embed)


def test_extract_ids_normalizes_and_dedupes():
    text = "see cve-2021-44228, CWE-79 and T1059.001; again CVE-2021-44228"
    assert extract_ids(text) == ["CVE-2021-44228", "CWE-79", "T1059.001"]


def test_exact_id_beats_dense_and_prefers_the_record_itself():
    hits = _retriever().search("What is CVE-2021-44228?", k=2)
    assert hits[0]["ref"] == "CVE-2021-44228" and hits[0]["match"] == "id"
    # The other chunk mentioning the ID comes next, ahead of any dense result.
    assert hits[1]["ref"] == "CVE-2021-45046" and hits[1]["match"] == "id"


def test_dense_fills_remaining_slots_without_duplicates():
    hits = _retriever().search("CWE-79 explained", k=3)
    refs = [h["ref"] for h in hits]
    assert refs[0] == "CWE-79"
    assert len(set(refs)) == 3
    assert [h["match"] for h in hits[1:]] == ["dense", "dense"]


def test_no_ids_is_pure_dense():
    hits = _retriever().search("how does cross-site scripting work", k=1)
    assert hits[0]["ref"] == "CWE-79" and hits[0]["match"] == "dense"


def test_load_and_agent_tool_use_real_index(tmp_path, monkeypatch):
    rag = tmp_path / "rag"
    rag.mkdir()
    matrix = np.eye(len(CHUNKS), 8, dtype=np.float32)
    np.save(rag / "index.npy", matrix)
    (rag / "chunks.jsonl").write_text("\n".join(json.dumps(c) for c in CHUNKS))

    from ghostlm.agent import tools
    monkeypatch.setenv("GHOSTLM_RAG_DIR", str(rag))
    monkeypatch.setattr(tools, "_RETRIEVER", None)
    loaded = tools._get_retriever()
    loaded._embed = lambda q: matrix[2]

    out = tools._backend_rag_retrieve({"query": "Log4Shell CVE-2021-44228", "k": 2})
    assert out["source"] == "rag_index"
    assert out["passages"][0]["id"] == "nvd:CVE-2021-44228"
    monkeypatch.setattr(tools, "_RETRIEVER", None)


def test_agent_tool_falls_back_without_index(tmp_path, monkeypatch):
    from ghostlm.agent import tools
    monkeypatch.setenv("GHOSTLM_RAG_DIR", str(tmp_path / "missing"))
    monkeypatch.setattr(tools, "_RETRIEVER", None)
    out = tools._backend_rag_retrieve({"query": "EternalBlue", "k": 2})
    assert out["source"] == "offline_cache"
