"""build_raft_data.augment_record uses the hybrid retriever's exact-ID hits."""

import random
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ghostlm.rag import HybridRetriever
from scripts.build_raft_data import augment_record

CHUNKS = [
    {"source": "nvd", "ref": "CVE-2017-0144", "text": "SMBv1 remote code execution (EternalBlue)."},
    {"source": "mitre", "ref": "T1021.002", "text": "SMB/Windows admin shares lateral movement."},
    {"source": "cwe", "ref": "CWE-79", "text": "Cross-site scripting."},
]


def _retriever():
    matrix = np.eye(3, 4, dtype=np.float32)
    return HybridRetriever(matrix, CHUNKS, embed=lambda q: matrix[2])


def _record(q):
    return {"source": "qa", "turns": [{"role": "user", "content": q}, {"role": "assistant", "content": "a"}]}


def test_oracle_mode_puts_exact_id_first():
    out = augment_record(_record("What is CVE-2017-0144?"), _retriever(), top_k=2,
                         mode="oracle", rng=random.Random(0))
    user = out["turns"][0]["content"]
    assert user.index("CVE-2017-0144") < user.index("CWE-79")
    assert out["raft_mode"] == "oracle" and out["turns"][1]["content"] == "a"


def test_distractor_mode_replaces_top_hit():
    out = augment_record(_record("What is CVE-2017-0144?"), _retriever(), top_k=1,
                         mode="distractor", rng=random.Random(0))
    assert "(nvd CVE-2017-0144)" not in out["turns"][0]["content"]
