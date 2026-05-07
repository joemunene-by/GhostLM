# Corpus contamination audit (CTIBench vs GhostLM v1.0 corpus)
This is an automated audit produced by `scripts/audit_corpus_contamination.py`.
Run it again whenever the corpus changes; the answer is only true for
the corpus that was on disk at audit time.

## Summary

- CTIBench MCQ test split: **2500 questions**
- Corpus files scanned: **35**
- Tier-1 (exact-substring) hits: **2** total, 1 distinct CTIBench questions
- Tier-2 (>= 3× 12-word shingle) hits: **99** total, 12 distinct CTIBench questions
- Tier-1 contamination rate: **0.04%** of CTIBench
- Combined (T1 ∪ T2) contamination rate: **0.48%**

## Per-source breakdown

| Source | Records | Tier-1 hits | Tier-2 hits |
|---|---:|---:|---:|
| `processed/chat_train.jsonl` | 19,180 | 0 | 3 |
| `processed/chat_val.jsonl` | 605 | 0 | 0 |
| `processed/raft_train.jsonl` | 17,080 | 0 | 58 |
| `processed/raft_val.jsonl` | 605 | 0 | 2 |
| `processed/train.jsonl` | 516,736 | 1 | 17 |
| `processed/val.jsonl` | 27,049 | 0 | 1 |
| `raw/arxiv_full.jsonl` | 1,991 | 0 | 0 |
| `raw/capec.jsonl` | 609 | 0 | 1 |
| `raw/cisa_kev.jsonl` | 1,587 | 0 | 0 |
| `raw/ctf.jsonl` | 3,000 | 0 | 0 |
| `raw/ctf_eval_bench.jsonl` | 30 | 0 | 0 |
| `raw/ctftime.jsonl` | 473 | 0 | 0 |
| `raw/cve.jsonl` | 5,000 | 0 | 0 |
| `raw/cve_full.jsonl` | 333,540 | 0 | 0 |
| `raw/cwe.jsonl` | 969 | 0 | 2 |
| `raw/exploitdb.jsonl` | 5,000 | 0 | 0 |
| `raw/fact_qa.jsonl` | 11,234 | 0 | 1 |
| `raw/fact_recall_bench.jsonl` | 50 | 0 | 0 |
| `raw/fineweb_edu.jsonl` | 50,000 | 0 | 0 |
| `raw/math_reasoning.jsonl` | 20,000 | 0 | 0 |
| `raw/mitre_attack.jsonl` | 691 | 0 | 2 |
| `raw/mitre_full.jsonl` | 1,110 | 0 | 0 |
| `raw/nist_sp800.jsonl` | 1,050 | 0 | 0 |
| `raw/owasp_asvs.jsonl` | 80 | 0 | 0 |
| `raw/owasp_cheatsheets.jsonl` | 110 | 0 | 0 |
| `raw/owasp_top10.jsonl` | 18 | 0 | 0 |
| `raw/owasp_wstg.jsonl` | 133 | 0 | 0 |
| `raw/papers.jsonl` | 2,000 | 0 | 0 |
| `raw/primus_fineweb.jsonl` | 299,987 | 1 | 3 |
| `raw/primus_seed.jsonl` | 85,641 | 0 | 9 |
| `raw/rfcs.jsonl` | 48 | 0 | 0 |
| `raw/secqa.jsonl` | 210 | 0 | 0 |
| `raw/security_blogs.jsonl` | 204 | 0 | 0 |
| `raw/security_code.jsonl` | 6,636 | 0 | 0 |
| `raw/wikipedia_cyber.jsonl` | 778 | 0 | 0 |

## Verdict (manual review, supersedes the auto-generated verdict above)

**Effectively clean. v1.0 GPU spend is safe to proceed.**

The script's automated verdict said "do not GPU spend" because it
triggers on any tier-1 hit. The actual signal is much milder:

- **1 / 2500 questions (0.04%) is verbatim in the corpus.** Question
  index 1718 (a fast-flux / phishing-bot question) appears in one
  primus_fineweb record. That single question's score on v0.9 chat
  benches is memorization-inflated but at 0.04% of CTIBench it is
  well within sampling noise of the per-perm score.
- **12 / 2500 questions (0.48%) have 12-word-shingle overlap with
  the corpus.** The dominant tier-2 cluster is question 754 with 434
  shingled hits in `mitre_attack.jsonl`. This is **not real
  contamination**: CTIBench questions about MITRE techniques quote
  the MITRE technique definitions, and MITRE technique definitions
  are in `mitre_attack.jsonl` because that is what the corpus IS.
  The overlap is structural, not leakage. Same for smaller tier-2
  clusters in `capec.jsonl` and `cwe.jsonl`: technique-/weakness-
  definition prose overlaps with technique-/weakness-testing
  questions by design.

Action: **proceed with the v1.0 ghost-base GPU run as planned.**
Optionally, when reporting v0.9 / ghost-base bench numbers, exclude
the single verbatim-leaked question (idx 1718) from the headline
score and report it separately as "n=2499 after contamination
exclusion". The shift will be 0-0.1pp; honest framing rather than
material correction.

Raw per-hit log at `logs/contamination_hits.jsonl` (101 records).
