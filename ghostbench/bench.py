"""Bench, Suite, EvalRecord, and Prediction abstractions.

A ``Bench`` is one bet's eval set plus its scoring config. A
``Suite`` is a collection of Benches that share a checkpoint.
The split mirrors how a real benchmark is run: pick a model
checkpoint, run every Bench in the Suite against it, get a
RunReport per Bench plus a SuiteReport across them all.

Eval records and predictions are the input data classes; both
are deserialised from JSONL files where each line carries the
fields documented in the dataclass.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional

from .scoring import Score, RunReport, score_record


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class EvalRecord:
    """One held-out prompt with its expected-content tags.

    JSONL shape:
        {
          "format": "<fmt>",
          "prompt": "<question>",
          "required_fields": [{"path": "...", "value": "..."}, ...],
          "required_substrings": ["substring", ...],
          "seed_id": "<optional, defaults to first 60 chars of prompt>"
        }
    """
    fmt: str
    prompt: str
    required_fields: List[Dict[str, Any]] = field(default_factory=list)
    required_substrings: List[str] = field(default_factory=list)
    seed_id: Optional[str] = None

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "EvalRecord":
        return cls(
            fmt=d.get("format", ""),
            prompt=d.get("prompt", ""),
            required_fields=list(d.get("required_fields", []) or []),
            required_substrings=list(d.get("required_substrings", []) or []),
            seed_id=d.get("seed_id"),
        )

    def to_score_dict(self) -> Dict[str, Any]:
        """Return the dict shape ``score_record`` expects."""
        return {
            "format": self.fmt,
            "prompt": self.prompt,
            "required_fields": self.required_fields,
            "required_substrings": self.required_substrings,
            "seed_id": self.seed_id or self.prompt[:60],
        }


@dataclass
class Prediction:
    """One model output paired with its eval record.

    JSONL shape (output of an inference run):
        {
          "format": "<fmt>",
          "prompt": "<question>",
          "predicted_artifact": "<model output>",
          "required_fields": [...],
          "required_substrings": [...]
        }

    The eval-record tags are propagated into the prediction file by
    the inference script so the scorer doesn't have to cross-reference
    eval and prediction files.
    """
    fmt: str
    prompt: str
    predicted_artifact: str
    required_fields: List[Dict[str, Any]] = field(default_factory=list)
    required_substrings: List[str] = field(default_factory=list)
    seed_id: Optional[str] = None

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Prediction":
        return cls(
            fmt=d.get("format", ""),
            prompt=d.get("prompt", ""),
            predicted_artifact=d.get("predicted_artifact", ""),
            required_fields=list(d.get("required_fields", []) or []),
            required_substrings=list(d.get("required_substrings", []) or []),
            seed_id=d.get("seed_id"),
        )

    def to_score_dict(self) -> Dict[str, Any]:
        return {
            "format": self.fmt,
            "prompt": self.prompt,
            "required_fields": self.required_fields,
            "required_substrings": self.required_substrings,
            "seed_id": self.seed_id or self.prompt[:60],
        }


# ---------------------------------------------------------------------------
# Bench
# ---------------------------------------------------------------------------


class Bench:
    """One bet's eval set plus its scoring config.

    A Bench holds:
      - ``name``           short stable identifier (e.g. "bet6_format_aware")
      - ``description``    human-readable purpose
      - ``records``        the list of EvalRecord
      - ``parsers``        the dict of format → parser used by score_record;
                            shared across Benches in the same Suite

    Calling ``Bench.score(predictions, run_name)`` produces a
    ``RunReport`` from a list of Predictions.
    """

    def __init__(self, name: str, description: str,
                 records: List[EvalRecord],
                 parsers: Dict[str, Callable[[str], Any]]):
        self.name = name
        self.description = description
        self.records = records
        self.parsers = parsers

    @classmethod
    def from_jsonl(cls, name: str, description: str, path: Path,
                   parsers: Dict[str, Callable[[str], Any]]) -> "Bench":
        """Load a Bench from an eval JSONL file."""
        records: List[EvalRecord] = []
        with Path(path).open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                records.append(EvalRecord.from_dict(rec))
        return cls(name=name, description=description,
                   records=records, parsers=parsers)

    def score(self, predictions: List[Prediction],
              run_name: str) -> RunReport:
        """Score ``predictions`` against this bench's records.

        Predictions are matched to eval records by matching ``prompt``
        text. Predictions without a matching eval record are
        scored against themselves (the prediction record carries
        the required_fields / required_substrings tags from the
        inference step). This keeps ``Bench.score`` decoupled from
        the eval JSONL when the prediction file is self-describing.
        """
        scores: List[Score] = []
        for pred in predictions:
            scores.append(
                score_record(pred.to_score_dict(),
                             pred.predicted_artifact, self.parsers)
            )
        return RunReport(
            bench_name=self.name,
            run_name=run_name,
            n=len(scores),
            scores=scores,
        )

    def __len__(self) -> int:
        return len(self.records)

    def __repr__(self) -> str:
        return f"Bench(name={self.name!r}, n={len(self.records)})"


# ---------------------------------------------------------------------------
# Suite
# ---------------------------------------------------------------------------


class Suite:
    """A collection of Benches that share a checkpoint.

    Typical usage:

        suite = Suite.from_dir("data/raw", parsers=BENCH_PARSERS)
        for bench in suite:
            preds = load_predictions(f"logs/{bench.name}.jsonl")
            report = bench.score(preds, run_name="ghost_base_v1")
            print(report.summary())
    """

    def __init__(self, benches: List[Bench]):
        self.benches = benches

    @classmethod
    def from_dir(cls, eval_dir: Path,
                 parsers: Dict[str, Callable[[str], Any]],
                 mapping: Optional[Dict[str, str]] = None) -> "Suite":
        """Discover Benches by scanning ``eval_dir`` for JSONL files
        matching the GhostBench naming convention.

        Default mapping (as of v0.9.5):
          format_aware_eval.jsonl       -> bet6_format_aware
          code_security_eval.jsonl      -> bet7_code_security
          binary_literacy_eval.jsonl    -> bet8_binary_literacy
          provenance_eval.jsonl         -> bet9_provenance

        Override via ``mapping={ "filename.jsonl": "bench_name", ... }``.
        """
        default_mapping = {
            "format_aware_eval.jsonl": "bet6_format_aware",
            "code_security_eval.jsonl": "bet7_code_security",
            "binary_literacy_eval.jsonl": "bet8_binary_literacy",
            "provenance_eval.jsonl": "bet9_provenance",
        }
        m = dict(default_mapping)
        if mapping:
            m.update(mapping)

        descriptions = {
            "bet6_format_aware": "Structural-format compliance "
                                  "(STIX 2.1 / YARA / Sigma / MISP).",
            "bet7_code_security": "Vulnerability-class identification "
                                   "and fix proposal on held-out CWEs.",
            "bet8_binary_literacy": "Hex / file-magic / disassembly "
                                     "recognition and explanation.",
            "bet9_provenance": "Inline cite-tag emission with valid "
                                "source_type:source_id format.",
        }

        benches: List[Bench] = []
        for fname, name in sorted(m.items()):
            path = Path(eval_dir) / fname
            if not path.exists():
                continue
            benches.append(Bench.from_jsonl(
                name=name,
                description=descriptions.get(name, ""),
                path=path, parsers=parsers,
            ))
        return cls(benches)

    def __iter__(self) -> Iterator[Bench]:
        return iter(self.benches)

    def __len__(self) -> int:
        return len(self.benches)

    def __getitem__(self, key) -> Bench:
        if isinstance(key, str):
            for b in self.benches:
                if b.name == key:
                    return b
            raise KeyError(key)
        return self.benches[key]

    def __repr__(self) -> str:
        return (f"Suite(benches=["
                + ", ".join(b.name for b in self.benches) + "])")
