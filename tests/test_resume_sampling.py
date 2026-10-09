"""Resumed runs must continue the data stream, not replay its first samples."""

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ghostlm.config import GhostLMConfig  # noqa: E402
from ghostlm.curriculum import parse_curriculum_spec  # noqa: E402
from ghostlm.dataset import MultiDomainBinDataset, reseed_for_resume  # noqa: E402


def _bins(tmp_path):
    out = {}
    for d in ("a", "b"):
        arr = np.arange(5000, dtype=np.uint16) + (0 if d == "a" else 20000)
        p = tmp_path / f"train.{d}.bin"
        arr.tofile(p)
        out[d] = str(p)
    return out


def _take(ds, n):
    it = iter(ds)
    return [next(it)[0][0].item() for _ in range(n)]


def test_curriculum_stream_continues_after_resume(tmp_path):
    cfg = GhostLMConfig(context_length=16)
    cur = parse_curriculum_spec("1.0:a=1,b=1")
    full = _take(MultiDomainBinDataset(_bins(tmp_path), cfg, cur, lambda: 0.0, seed=1), 30)
    resumed = MultiDomainBinDataset(_bins(tmp_path), cfg, cur, lambda: 0.0, seed=1,
                                    start_sample_fn=lambda: 20)
    assert _take(resumed, 10) == full[20:30]


def test_resumes_at_different_steps_do_not_replay(tmp_path):
    cfg = GhostLMConfig(context_length=16)
    cur = parse_curriculum_spec("1.0:a=1,b=1")
    first = _take(MultiDomainBinDataset(_bins(tmp_path), cfg, cur, lambda: 0.0, start_sample_fn=lambda: 0), 20)
    later = _take(MultiDomainBinDataset(_bins(tmp_path), cfg, cur, lambda: 0.0, start_sample_fn=lambda: 640), 20)
    assert len(set(first) & set(later)) <= 2


def test_flat_loader_reshuffles_on_resume():
    data = list(range(100))
    gen = torch.Generator().manual_seed(42)
    loader = torch.utils.data.DataLoader(data, batch_size=10, shuffle=True, generator=gen)
    fresh = next(iter(loader)).tolist()
    reseed_for_resume(loader, 42, 0)
    gen.manual_seed(42)
    assert next(iter(loader)).tolist() == fresh  # step 0: unchanged order
    reseed_for_resume(loader, 42, 500)
    assert next(iter(loader)).tolist() != fresh
