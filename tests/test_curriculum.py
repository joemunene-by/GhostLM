"""Tests for the multi-stage domain curriculum and weighted sampler."""

import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from ghostlm.config import GhostLMConfig
from ghostlm.curriculum import (
    CurriculumStage,
    DomainCurriculum,
    DEFAULT_GENERALIST_CURRICULUM,
    parse_curriculum_spec,
)
from ghostlm.dataset import MultiDomainBinDataset


# ---------- DomainCurriculum ----------

def test_weights_normalized_and_sum_to_one():
    cur = DomainCurriculum([CurriculumStage(1.0, {"a": 3, "b": 1})])
    w = cur.weights_at(0.5)
    assert abs(sum(w.values()) - 1.0) < 1e-9
    assert abs(w["a"] - 0.75) < 1e-9 and abs(w["b"] - 0.25) < 1e-9


def test_interpolation_midpoint():
    cur = DomainCurriculum([
        CurriculumStage(0.0, {"a": 1.0, "b": 0.0}),
        CurriculumStage(1.0, {"a": 0.0, "b": 1.0}),
    ])
    # At progress 0.5 the two domains should be evenly mixed.
    w = cur.weights_at(0.5)
    assert abs(w["a"] - 0.5) < 1e-6 and abs(w["b"] - 0.5) < 1e-6
    # Endpoints honor the stage mixtures.
    assert cur.weights_at(0.0)["a"] == 1.0
    assert cur.weights_at(1.0)["b"] == 1.0


def test_step_mode_no_interpolation():
    cur = DomainCurriculum([
        CurriculumStage(0.5, {"a": 1.0}),
        CurriculumStage(1.0, {"b": 1.0}),
    ], interpolate=False)
    assert cur.weights_at(0.3)["a"] == 1.0
    assert cur.weights_at(0.8)["b"] == 1.0


def test_progress_clamped():
    cur = DomainCurriculum([CurriculumStage(1.0, {"a": 1, "b": 1})])
    assert cur.weights_at(-5.0)  # no exception, clamped to 0
    assert cur.weights_at(99.0)  # clamped to 1


def test_default_generalist_curriculum_shifts_toward_code_math():
    early = DEFAULT_GENERALIST_CURRICULUM.weights_at(0.05)
    late = DEFAULT_GENERALIST_CURRICULUM.weights_at(0.98)
    # General web dominates early; code/math are upweighted by the end.
    assert early["general_web"] > early["code"]
    assert late["code"] > early["code"]
    assert late["math"] > early["math"]
    assert late["general_web"] < early["general_web"]


def test_parse_curriculum_spec():
    cur = parse_curriculum_spec("0.5:general_web=5,cybersec=2;1.0:code=3,math=2")
    w0 = cur.weights_at(0.0)
    assert w0["general_web"] > w0["cybersec"]
    w1 = cur.weights_at(1.0)
    assert w1["code"] > w1["math"] > 0


# ---------- MultiDomainBinDataset ----------

def _make_bin(path, token_value, n=10000, dtype="uint16"):
    arr = np.full(n, token_value, dtype=np.dtype(dtype))
    arr.tofile(path)
    with path.with_suffix(".meta.json").open("w") as f:
        json.dump({"dtype": dtype}, f)


def _sample_domain_freq(ds, n_samples, token_to_domain):
    counts = Counter()
    it = iter(ds)
    for _ in range(n_samples):
        x, _ = next(it)
        counts[token_to_domain[int(x[0])]] += 1
    return counts


def test_sampler_matches_weights(tmp_path):
    cfg = GhostLMConfig(context_length=8, batch_size=1, seed=0)
    _make_bin(tmp_path / "a.bin", 1)
    _make_bin(tmp_path / "b.bin", 2)
    bins = {"a": str(tmp_path / "a.bin"), "b": str(tmp_path / "b.bin")}
    # 3:1 weighting toward domain 'a'.
    cur = DomainCurriculum([CurriculumStage(1.0, {"a": 3.0, "b": 1.0})])
    ds = MultiDomainBinDataset(bins, cfg, cur, progress_fn=lambda: 0.5, seed=0)
    freq = _sample_domain_freq(ds, 4000, {1: "a", 2: "b"})
    ratio = freq["a"] / (freq["a"] + freq["b"])
    assert 0.70 <= ratio <= 0.80, ratio  # ~0.75 expected


def test_sampler_follows_progress(tmp_path):
    cfg = GhostLMConfig(context_length=8, batch_size=1, seed=0)
    _make_bin(tmp_path / "a.bin", 1)
    _make_bin(tmp_path / "b.bin", 2)
    bins = {"a": str(tmp_path / "a.bin"), "b": str(tmp_path / "b.bin")}
    cur = DomainCurriculum([
        CurriculumStage(0.0, {"a": 1.0, "b": 0.0}),
        CurriculumStage(1.0, {"a": 0.0, "b": 1.0}),
    ])
    progress = {"v": 0.0}
    ds = MultiDomainBinDataset(bins, cfg, cur, progress_fn=lambda: progress["v"], seed=0)
    early = _sample_domain_freq(ds, 1000, {1: "a", 2: "b"})
    assert early["a"] > 0.9 * sum(early.values())  # nearly all 'a' early
    progress["v"] = 1.0
    late = _sample_domain_freq(ds, 1000, {1: "a", 2: "b"})
    assert late["b"] > 0.9 * sum(late.values())    # nearly all 'b' late


def test_sampler_block_shape(tmp_path):
    cfg = GhostLMConfig(context_length=16, batch_size=1, seed=0)
    _make_bin(tmp_path / "a.bin", 5)
    ds = MultiDomainBinDataset({"a": str(tmp_path / "a.bin")}, cfg,
                               DomainCurriculum([CurriculumStage(1.0, {"a": 1.0})]),
                               progress_fn=lambda: 0.0)
    x, y = next(iter(ds))
    assert x.shape == (16,) and y.shape == (16,)
    assert torch.equal(x, torch.full((16,), 5, dtype=torch.long))
