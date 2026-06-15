"""Multi-stage domain curriculum for pretraining.

The modern small-LM recipe (SmolLM2, H2O-Danube3, MiniCPM) does not train
on a fixed data mixture. It trains in stages: a broad web-heavy mix early
to build general fluency, then progressively upweights the
higher-quality, higher-density domains (code, math, curated knowledge),
and finally a short "annealing" phase concentrated on the best data. This
schedule is worth several points on downstream benchmarks at fixed
compute versus a static mix.

``DomainCurriculum`` encodes that schedule as a function of training
*progress* (step / max_steps in [0, 1]) to a normalized set of per-domain
sampling weights. It is pure and deterministic so it can be unit-tested
without touching the data path; ``ghostlm.dataset.MultiDomainBinDataset``
consumes its output to actually sample documents.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List


@dataclass
class CurriculumStage:
    """One stage of the curriculum.

    ``until`` is the training-progress fraction (0..1] at which this
    stage's mixture is fully in effect. ``weights`` maps a training
    domain (see ``data.collect.SOURCE_DOMAINS``) to a relative sampling
    weight; absolute scale does not matter, the curriculum normalizes.
    """
    until: float
    weights: Dict[str, float]


class DomainCurriculum:
    """A progress-indexed schedule of per-domain sampling weights.

    Stages are sorted by ``until``. For a progress ``p``, the active
    weights are linearly interpolated between the surrounding stage
    boundaries (``interpolate=True``, the default) so the mixture drifts
    smoothly rather than jumping at stage edges; with ``interpolate=False``
    the first stage whose ``until >= p`` wins (step schedule).

    Weights are returned normalized to sum to 1. A domain absent from a
    stage is treated as weight 0 in that stage.
    """

    def __init__(self, stages: List[CurriculumStage], interpolate: bool = True):
        if not stages:
            raise ValueError("curriculum needs at least one stage")
        self.stages = sorted(stages, key=lambda s: s.until)
        if self.stages[-1].until < 1.0:
            # Extend the final mixture to the end of training.
            self.stages.append(CurriculumStage(1.0, dict(self.stages[-1].weights)))
        self.interpolate = interpolate
        self.domains = sorted({d for s in self.stages for d in s.weights})

    def _vec(self, stage: CurriculumStage) -> Dict[str, float]:
        return {d: float(stage.weights.get(d, 0.0)) for d in self.domains}

    @staticmethod
    def _normalize(w: Dict[str, float]) -> Dict[str, float]:
        total = sum(max(0.0, v) for v in w.values())
        if total <= 0:
            n = len(w)
            return {d: 1.0 / n for d in w}
        return {d: max(0.0, v) / total for d, v in w.items()}

    def weights_at(self, progress: float) -> Dict[str, float]:
        """Return normalized domain weights for a training-progress in [0, 1]."""
        p = min(1.0, max(0.0, progress))
        stages = self.stages

        if not self.interpolate:
            for s in stages:
                if p <= s.until:
                    return self._normalize(self._vec(s))
            return self._normalize(self._vec(stages[-1]))

        # Interpolate between the bracketing stages.
        prev = stages[0]
        if p <= prev.until:
            return self._normalize(self._vec(prev))
        for s in stages[1:]:
            if p <= s.until:
                span = s.until - prev.until
                t = 0.0 if span <= 0 else (p - prev.until) / span
                a, b = self._vec(prev), self._vec(s)
                return self._normalize({d: a[d] + t * (b[d] - a[d]) for d in self.domains})
            prev = s
        return self._normalize(self._vec(stages[-1]))


# Default generalist curriculum: broad web early, upweight code/math/
# knowledge through the middle, anneal on the densest domains at the end.
# Weights are relative; the curriculum normalizes them.
DEFAULT_GENERALIST_CURRICULUM = DomainCurriculum([
    # Stage 1 (-> 50%): fluency from broad web + a cybersec base.
    CurriculumStage(0.50, {
        "general_web": 5.0, "knowledge": 2.0, "cybersec": 2.0,
        "code": 1.0, "math": 1.0, "instruction": 0.5,
    }),
    # Stage 2 (-> 85%): balance up the high-density reasoning domains.
    CurriculumStage(0.85, {
        "general_web": 3.0, "knowledge": 2.0, "cybersec": 2.0,
        "code": 2.5, "math": 2.0, "instruction": 1.0,
    }),
    # Stage 3 (-> 100%): anneal on code / math / knowledge / instruction.
    CurriculumStage(1.00, {
        "general_web": 1.5, "knowledge": 2.0, "cybersec": 1.5,
        "code": 3.0, "math": 2.5, "instruction": 2.0,
    }),
])


def parse_curriculum_spec(spec: str) -> DomainCurriculum:
    """Parse a compact curriculum spec string into a ``DomainCurriculum``.

    Format: stages separated by ';', each ``until:dom=w,dom=w,...``. Example::

        "0.5:general_web=5,cybersec=2;1.0:code=3,math=2,general_web=1"

    Useful for passing a curriculum on the training CLI without code edits.
    """
    stages: List[CurriculumStage] = []
    for chunk in spec.split(";"):
        chunk = chunk.strip()
        if not chunk:
            continue
        until_s, _, weights_s = chunk.partition(":")
        weights: Dict[str, float] = {}
        for pair in weights_s.split(","):
            pair = pair.strip()
            if not pair:
                continue
            dom, _, w = pair.partition("=")
            weights[dom.strip()] = float(w)
        stages.append(CurriculumStage(float(until_s), weights))
    if not stages:
        raise ValueError(f"empty curriculum spec: {spec!r}")
    return DomainCurriculum(stages)
