# Free-form fact-recall benchmark v2

The truth metric for the ghost-base v1.0 acceptance gate. v0.9 chat
hit 1/50 on the v1 bench (`data/raw/fact_recall_bench.jsonl`); the
acceptance gate for ghost-base is **≥30% on this v2 bench**, per
[`docs/ghost_base_spec.md`](ghost_base_spec.md).

## What v2 fixes

The v1 bench had 50 questions and a plain substring grader. Two
known false-positive modes shipped in v0.9.2's reported numbers:

1. **Question echoing.** Some questions mention key terms that the
   answer also contains. The v1 grader couldn't tell whether the
   model actually answered or just rephrased the question. Concretely,
   v0.9's lone "hit" on the v1 bench was the alternate "256" matching
   inside "SHA-256" in a model completion that was effectively echoing
   the question. The hit was spurious.

2. **Token-boundary leakage.** The substring "10" matches inside
   "100" / "1000" / "T1059". On numeric or short-string answers, plain
   substring matching cannot distinguish "the model said 10" from "the
   model said 1000 which contains 10".

v2 fixes both with three schema additions (none mandatory; per-question
opt-in for the cases where they matter):

- **`boundary_match: true`** (default). Match only when the alternate
  appears delimited by non-alphanumeric characters on both sides (or
  string ends). "10" in "1000" no longer matches; "T1059" in "T1059.001"
  still matches because "." is a non-word char.
- **`disqualifiers: [...]`**. List of phrases that, if present in the
  completion, void any match credit for that question. Used on
  echoing-risk questions: a CVE-2017-0144 question can list
  `["EternalBlue"]` as a disqualifier if the question doesn't already
  mention the name, ensuring the model isn't getting credit for
  surfacing the question's own keywords.
- **`must_appear: [...]`**. List of phrases that ALL must be present
  for credit, with boundary matching honored. Used for composite-fact
  questions like "what RFC defines JWT?" where the answer requires
  both "RFC 7519" and "JWT" to appear together.

## Schema

One JSON object per line at `data/raw/fact_recall_bench_v2.jsonl`:

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
optional with sensible defaults (`alternates: []`, `boundary_match:
true`, `disqualifiers: []`, `must_appear: []`). `alternates` and
`must_appear` are mutually exclusive: a question is either testing one
synonymous fact (alternates, OR semantics) or a composite fact
(must_appear, AND semantics), never both.

## Topic distribution (v2 seed, n=100)

| Topic | Count | Examples |
|---|---:|---|
| cve | 30 | CVE-2017-0144 (EternalBlue), CVE-2021-44228 (Log4Shell), CVE-2024-6387 (regreSSHion) |
| mitre | 15 | T1059 (Command/Scripting), T1566 (Phishing), TA0001 (Initial Access) |
| cwe | 15 | CWE-89 (SQL injection), CWE-79 (XSS), CWE-787 (OOB write) |
| owasp | 10 | A01:2021 (Broken Access Control), API1 (BOLA), ASVS structure |
| crypto | 10 | AES round counts, hash digest sizes, KDF identities |
| protocol | 11 | TLS 1.3 RFC, JWT RFC, SSH/SMTPS/Kerberos default ports |
| tool | 6 | Nmap flags, Sysinternals tools, Windows event IDs |
| misc | 3 | CVSS bounds, PCI DSS version |
| **Total** | **100** | |

The benchmark grows toward n=200 as more questions land. The 100-item
seed is deliberately broad enough to detect the parameter-count gate:
ghost-small (81M, current) is near floor across every topic; if
ghost-base (~360M) lands at 30%+ overall or any topic at 50%+, the
parameter-scaling diagnosis is confirmed.

## Grading

[`scripts/eval_fact_recall_v2.py`](../scripts/eval_fact_recall_v2.py)
loads each checkpoint, runs greedy generation against each question's
prompt (default 120-token budget), and grades the completion via the
`grade_record` function. Per-row output goes to
`logs/fact_recall_v2/{label}.jsonl` for spot-checking.

Run all the chat checkpoints in one pass:

```bash
PYTHONPATH=. python3 scripts/eval_fact_recall_v2.py \
    --checkpoints \
        v0.4-chat-v3=checkpoints/phase5_chat_v3/best_model.pt \
        v0.7-chat=checkpoints/phase15_chat_v07/best_model.pt \
        v0.9-chat=checkpoints/phase19_chat_v09/best_model.pt
```

Expected baseline (per the v0.9.2 narrative): every ghost-small
checkpoint scores 0-2% across the v1 bench; v2 should sit at the same
floor with the false-positive cleanup. The bench has done its job
when v0.9 + RAG (or ghost-base) clears 30% on this set.

## Why this is the truth metric

CTIBench MCQ measures register matching: a model that emits the
right "shape" of cybersec-style answer scores well even with no
factual recall. Free-form generation strips that crutch away. If the
model knows EternalBlue is CVE-2017-0144, it surfaces it; if it
doesn't, it confabulates a plausible-looking but wrong CVE. The
grader catches both modes. The boundary-match + disqualifier scheme
catches the additional failure mode of "model echoed the question
prompt" that inflated v0.9's lone hit on the v1 bench to a spurious
hit.

The v1.0 acceptance gate at `docs/ghost_base_spec.md` accepts this
bench as the alternative gate (≥30% here is one of the three ways
ghost-base can clear). Free-form fact recall is the test where MCQ
register-matching tricks stop working.
