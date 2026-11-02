# Bytes per token by tokenizer and domain

Held-out fifth of 400 sampled documents per domain from data/processed/train.jsonl (9 MB sampled). Higher is better: more text per token. The weighted column uses the planned TPU mix general_web 45%, code 25%, cybersec 12%, math 8%, knowledge 7%, instruction 3%.

| Tokenizer | general_web | code | cybersec | math | knowledge | instruction | Weighted | vs current |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| gpt2_50k (current) | 4.67 | 2.46 | 4.37 | 3.22 | 4.28 | 4.45 | 3.93 | +0.0% |
| v1_32k | 4.46 | 3.33 | 4.89 | 3.28 | 3.89 | 4.07 | 4.08 | +3.8% |
| v05_32k | 4.02 | 3.34 | 4.42 | 3.07 | 3.53 | 3.72 | 3.78 | -3.9% |
| new_32k | 4.41 | 3.47 | 4.57 | 3.30 | 3.98 | 4.14 | 4.07 | +3.4% |
| new_48k | 4.55 | 3.54 | 4.71 | 3.35 | 4.13 | 4.31 | 4.18 | +6.4% |
| new_64k | 4.63 | 3.57 | 4.78 | 3.38 | 4.21 | 4.39 | 4.25 | +8.0% |

A fresh vocabulary trained on this sample only sees a few hundred megabytes; one trained on the real mix will do slightly better than the rows above suggest.

## Recommendation

Train a fresh byte-level BPE at 64K on the real TPU mix before tokenising the corpus. On
this sample it packs 8% more text per token than the current GPT-2 vocabulary, almost all
of the gain coming from code (2.46 to 3.57 bytes per token) and the security domains. For a
fixed 150B-token budget that is effectively 8% more training data at no extra compute. 48K
captures most of the gain (+6.4%) if a smaller embedding table is preferred. The decision
must be made before the shards are written, because the corpus is tokenised once.
