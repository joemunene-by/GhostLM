# Bet 3 followup: BPE corpus-mix ablation

## Why this exists

Bet 3 ([docs/differentiation.md](differentiation.md) §"Bet 3: cybersec-
native tokenizer") hypothesised a 25-35% compression win from a 32K
BPE retrained on the v1.0 corpus. The first measurement landed at
**+1.6%** averaged over a 99-record mixed-corpus sample. That's a
small win, but the sample mixed cyber-heavy and general-heavy text
in unknown proportions, so the headline number didn't tell us whether
the bet's underlying thesis (cybersec-text-compresses-better) was
sound or whether the corpus mix had drowned out the effect.

This doc captures the corpus-mix ablation that settles the question.
Train two BPEs:

  - `v1 mixed`  : 32K BPE trained on the full v1.0 train.jsonl
                  (516,736 records, includes fineweb_edu + math_reasoning)
  - `v1_cyber`  : 32K BPE trained on the cybersec-only subset
                  (450,235 records, drops fineweb_edu + math_reasoning)

Then score each on a fixed sample of cyber-only and general-only
text, both compared against GPT-2's 50K BPE as the baseline.

## Results

`scripts/score_tokenizer.py` against `data/processed/train.jsonl`,
`--max-records 500` per slice, 2026-05-08:

| Tokenizer | Cyber subset (n=496) | General subset (n=500) |
|---|---:|---:|
| GPT-2 BPE (50K) | baseline | baseline |
| v1 mixed BPE (32K) | **+4.0%** vs GPT-2 | **-2.5%** vs GPT-2 |
| v1_cyber BPE (32K) | **+4.3%** vs GPT-2 | **-7.6%** vs GPT-2 |

(Positive % = the v1 BPE is denser than GPT-2; negative = GPT-2 wins.)

## What it means

**Cyber-only training gives +0.3 percentage points on cyber text
and costs -5.1 percentage points on general text.** That is a clearly
bad trade. Switching ghost-base to v1_cyber would buy almost nothing
on cybersec inference and meaningfully hurt anything that touches
non-cybersec text.

**The mix-trained v1 BPE is the right default if a v1 BPE is to be
shipped at all.** It's +4.0% on cyber, only -2.5% on general, and
the numbers reflect the actual corpus distribution ghost-base will
see at training time.

**The bet 3 hypothesis ('cybersec text compresses 25-35% better with
custom BPE') is falsified at the magnitude claimed.** The real win
on cybersec text caps around +4% even with a corpus exclusively
trained on cyber. Possible reasons:

  - GPT-2's 50K BPE was trained on Common Crawl which already contained
    a non-trivial slice of cybersec terminology (CVE prose, security
    blogs, MITRE writeups). The marginal gain from re-prioritising
    those tokens at 32K is small.
  - The within-cyber sub-domains have different optimal vocabularies.
    `security_code` (raw shell / Python source) and `nist_sp800`
    (formal English specs) compress 12-21% better; `nvd` and
    `primus_fineweb` (the dominant volume sources) are already at
    +2.2% / +2.3% with mixed-trained v1 BPE so there's little left
    to gain there.

The +4% number is what bet 3 actually buys. That's small enough that
the recommendation in [docs/differentiation.md](differentiation.md)
stays unchanged: **default ghost-base to GPT-2 BPE; treat
`GhostTokenizerV1` (mixed) as opt-in for cyber-only inference paths
where the +4% buys ~5% more usable context. Don't ship `v1_cyber` at
all — the general-text regression isn't worth the marginal cybersec
lift.**

## Per-source detail

`v1 mixed` on cyber-only subset (where the bet 3 lift, if any, lives):

| source | n | win % |
|---|---:|---:|
| arxiv | 1 | +10.3% |
| arxiv_full | 2 | +0.4% |
| capec | 2 | +5.8% |
| cisa_kev | 3 | +9.1% |
| ctftime | 3 | +12.5% |
| cwe | 1 | +5.6% |
| exploitdb | 7 | +3.7% |
| fact_qa | 13 | +7.1% |
| nist_sp800 | 1 | +20.8% |
| nvd | 75 | +2.2% |
| primus_fineweb | 310 | +2.3% |
| primus_seed | 66 | +9.8% |
| security_blogs | 1 | +3.2% |
| security_code | 8 | +12.2% |
| synthetic | 2 | +4.7% |
| wikipedia_cyber | 1 | -4.8% |
| **overall** | **496** | **+4.0%** |

The biggest wins are on relatively rare formal sources
(`nist_sp800`, `ctftime`, `security_code`); the dominant bulk
sources (`primus_fineweb`, `nvd`) are at the floor +2-2.3%. Since
inference at deployment time will mostly look like
`primus_fineweb`-shaped cyber prose (technical English with CVE/
ATT&CK references), the realistic win is closer to +2.3% than +4%.

## Recommendation, restated

- Ghost-base: GPT-2 BPE default. (Established in
  [docs/differentiation.md](differentiation.md).)
- v1 mixed (`data/tokenizer/v1/`): keep on the shelf as opt-in
  for cyber-only inference paths.
- v1_cyber (`data/tokenizer/v1_cyber/`): do not ship; trained
  artifact preserved on the Mac for reproducibility but not
  added as a backend in `ghostlm/tokenizer.py`.

## Reproducing

```bash
# Filter the corpus
python3 -c "
import json
with open('data/processed/train.jsonl') as f, \
     open('data/processed/train_cyber.jsonl', 'w') as out:
    for line in f:
        rec = json.loads(line)
        if rec.get('source') in ('fineweb_edu', 'math_reasoning'):
            continue
        out.write(line)
"

# Train cyber-only BPE
PYTHONPATH=. python3 scripts/train_v1_bpe.py \
    --corpus data/processed/train_cyber.jsonl \
    --vocab-size 32000 \
    --out-dir data/tokenizer/v1_cyber

# Score both BPEs on both subsets
for tok in v1 v1_cyber; do
    for slice in "--drop-source fineweb_edu,math_reasoning" \
                 "--filter-source fineweb_edu,math_reasoning"; do
        PYTHONPATH=. python3 scripts/score_tokenizer.py \
            --tokenizer data/tokenizer/$tok/tokenizer.json \
            --corpus data/processed/train.jsonl \
            --max-records 500 $slice
    done
done
```
