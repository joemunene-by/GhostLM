#!/usr/bin/env python3
"""Publish ``data/raw/fact_recall_bench_v2.jsonl`` as a public HF
dataset at ``Ghostgim/cybersec-fact-recall``.

Why publish: free-form fact-recall is a meaningful eval surface for
small cybersec LMs that the existing public benchmarks (CTIBench,
SecQA) don't cover well. Both of those are multiple-choice, which
rewards register matching as much as actual recall. This bench
asks single-fact short-answer questions across CVE / MITRE / CWE /
OWASP / crypto / protocol / tool / misc topics, with a smart
grader that handles question-echoing and token-boundary edge
cases. Other small-LM-cybersec projects can reuse it; if anyone
else hits the same v0.9-class register-vs-fact wall, this gives
them a measurable ruler.

Pushes both the jsonl plus a dataset README with frontmatter
(license, task_categories, language, size) so it surfaces in HF
Datasets search.

Run on a host that has HF write auth (Joe's Mac):

    python3 scripts/publish_fact_recall_dataset.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

REPO_ID = "Ghostgim/cybersec-fact-recall"
REPO_TYPE = "dataset"

DATASET_README = """\
---
license: apache-2.0
task_categories:
  - text-generation
  - question-answering
language:
  - en
tags:
  - cybersecurity
  - benchmark
  - evaluation
  - fact-recall
  - cve
  - mitre-attack
  - cwe
  - owasp
size_categories:
  - n<1K
pretty_name: GhostLM Cybersec Fact-Recall Benchmark
---

# Cybersec Fact-Recall Benchmark (GhostLM v2)

Free-form short-answer benchmark for small cybersecurity language
models. Built and used by the [GhostLM](https://github.com/joemunene-by/GhostLM)
project as the truth metric for the ghost-base v1.0 acceptance gate.

## Why this exists

Multiple-choice cybersec benchmarks like CTIBench and SecQA reward
*register matching* (the model picks the option that "looks like" a
security answer) as much as actual factual recall. A small from-
scratch model can hit 28-30% on those without knowing a single CVE
identifier. Free-form generation strips that crutch away: if the
model knows EternalBlue is CVE-2017-0144, it surfaces the ID; if it
doesn't, it confabulates a plausible-looking but wrong CVE.

## Dataset structure

100-question seed (v2.0; growing toward 200 in subsequent releases):

| Topic | Count |
|---|---:|
| cve | 30 |
| mitre | 15 |
| cwe | 15 |
| protocol | 11 |
| owasp | 10 |
| crypto | 10 |
| tool | 6 |
| misc | 3 |

One JSON object per line with fields:

```json
{
  "id":             "fr2-001",
  "topic":          "cve",
  "prompt":         "What is the CVE identifier for the EternalBlue SMB vulnerability?",
  "answer":         "CVE-2017-0144",
  "alternates":     ["CVE-2017-0144"],
  "boundary_match": true,
  "disqualifiers":  ["EternalBlue"],
  "must_appear":    ["RFC 7519", "JWT"]
}
```

Every record has `id`, `topic`, `prompt`, `answer`. The rest is
optional; defaults are `alternates: []`, `boundary_match: true`,
`disqualifiers: []`, `must_appear: []`.

## Grading semantics

Three schema fields tighten the grader vs naive substring match:

- **`boundary_match`** (default `true`). Match only when the
  candidate appears with non-alphanumeric characters on both sides.
  `"10"` no longer matches inside `"100"`.
- **`disqualifiers`**. List of phrases that, if present in the
  completion, void any match credit. Used on echoing-risk questions
  where the model can surface the answer simply by quoting back the
  question prompt.
- **`must_appear`**. ALL listed substrings must appear (boundary-
  matched). Used for composite-fact questions like "what RFC defines
  JWT?" requiring both `"RFC 7519"` and `"JWT"` together.

Reference grader: [`scripts/eval_fact_recall_v2.py`](https://github.com/joemunene-by/GhostLM/blob/main/scripts/eval_fact_recall_v2.py)
in the GhostLM repo (Apache 2.0). 11 unit tests cover the boundary,
disqualifier, and must_appear edge cases.

## Baseline numbers

GhostLM ghost-small (45-81M parameters) chat checkpoints score 0%
across the board on this bench. That's expected: at this parameter
scale the line saturates as a register-matcher, not a fact-knower.
Posted as evidence the bench discriminates rather than as a leaderboard.

| Checkpoint | Params | Hits / 100 | Rate |
|---|---:|---:|---:|
| ghost-small v0.4 chat-v3 | 45M | 0 | 0.0% |
| ghost-small v0.7 chat | 81M | (TBD) | (TBD) |
| ghost-small v0.9 chat | 81M | (TBD) | (TBD) |

(v0.7 / v0.9 numbers land when their bench runs finish; check the
GhostLM RESULTS.md for the canonical table.)

## Acceptance criterion

The GhostLM project uses **>=30% on this bench** as one of three
ways the next architectural rung (ghost-base, ~360M parameters) can
clear its acceptance gate. If a 360M from-scratch model hits 30%+
free-form fact recall on cybersec questions, that validates the
parameter-scaling hypothesis. The other two gates are CTIBench full
>=40% and the in-repo CTF MCQ >=65%.

## License

Apache 2.0. Same license as the GhostLM source code.

## Citation

```bibtex
@misc{munene2026ghostlm,
  title         = {GhostLM: a from-scratch cybersecurity language model on a transparent scale ladder},
  author        = {Munene, Joe},
  year          = {2026},
  howpublished  = {\\url{https://github.com/joemunene-by/GhostLM}},
  note          = {Free-form fact-recall benchmark v2; 100-question seed, growing toward 200}
}
```
"""


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source-jsonl",
                   default="data/raw/fact_recall_bench_v2.jsonl")
    p.add_argument("--repo-id", default=REPO_ID)
    p.add_argument("--dry-run", action="store_true",
                   help="Print plan, don't actually push")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    src = Path(args.source_jsonl)
    if not src.exists():
        raise SystemExit(f"source jsonl missing: {src}")

    print(f"source jsonl: {src} ({src.stat().st_size / 1024:.1f} KB)")
    print(f"target repo:  datasets/{args.repo_id}")
    print(f"dataset card: {len(DATASET_README)} chars")
    if args.dry_run:
        print("\n(dry-run, exiting before push)")
        return 0

    from huggingface_hub import HfApi, CommitOperationAdd
    api = HfApi()
    me = api.whoami()
    print(f"\nauth: {me['name']}")

    # Create-or-noop the dataset repo. exist_ok = no-op if already there.
    api.create_repo(repo_id=args.repo_id, repo_type=REPO_TYPE, exist_ok=True,
                    private=False)

    ops = [
        CommitOperationAdd(path_in_repo="fact_recall_bench_v2.jsonl",
                           path_or_fileobj=str(src)),
        CommitOperationAdd(path_in_repo="README.md",
                           path_or_fileobj=DATASET_README.encode("utf-8")),
    ]
    info = api.create_commit(
        repo_id=args.repo_id, repo_type=REPO_TYPE,
        operations=ops,
        commit_message="feat: fact-recall v2 bench (n=100, smart grader)",
        commit_description=(
            "Free-form short-answer cybersec fact-recall bench from the\n"
            "GhostLM project. 100 questions across cve / mitre / cwe / owasp /\n"
            "crypto / protocol / tool / misc. Grader supports boundary-match,\n"
            "disqualifiers, and must_appear composite facts. Full schema +\n"
            "reference grader documented in the README."
        ),
    )
    print(f"\nfilled dataset commit: {info.oid}")
    print(f"dataset URL: https://huggingface.co/datasets/{args.repo_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
