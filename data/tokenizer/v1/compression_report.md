# v1 BPE compression report
Sample size: **99** corpus records

| Tokenizer | tokens/byte (avg) |
|---|---:|
| GhostLM v1 BPE (32K, this script) | 0.2190 |
| GPT-2 BPE (50K, current default) | 0.2225 |

**v1 BPE compresses corpus text by +1.6% more than GPT-2 BPE** (lower tokens/byte = denser tokens, more effective context per token).

Sample-level distribution (`bytes` is raw size, `v1` and `gpt2` are token counts on the same text):

| bytes | v1 | gpt2 | v1 tpb | gpt2 tpb |
|---:|---:|---:|---:|---:|
| 2,831 | 505 | 509 | 0.1784 | 0.1798 |
| 649 | 131 | 131 | 0.2018 | 0.2018 |
| 1,203 | 258 | 268 | 0.2145 | 0.2228 |
| 575 | 144 | 141 | 0.2504 | 0.2452 |
| 105 | 27 | 29 | 0.2571 | 0.2762 |
| 1,613 | 334 | 339 | 0.2071 | 0.2102 |
| 8,051 | 1,672 | 1,766 | 0.2077 | 0.2194 |
| 776 | 140 | 148 | 0.1804 | 0.1907 |
| 675 | 134 | 128 | 0.1985 | 0.1896 |
| 10,030 | 2,481 | 2,313 | 0.2474 | 0.2306 |
| 10,111 | 2,530 | 2,402 | 0.2502 | 0.2376 |
| 59,709 | 15,234 | 15,167 | 0.2551 | 0.2540 |
| 1,965 | 326 | 331 | 0.1659 | 0.1684 |
| 193 | 43 | 48 | 0.2228 | 0.2487 |
| 1,441 | 283 | 296 | 0.1964 | 0.2054 |
| 1,031 | 197 | 206 | 0.1911 | 0.1998 |
| 780 | 149 | 160 | 0.1910 | 0.2051 |
| 1,892 | 381 | 386 | 0.2014 | 0.2040 |
| 180 | 41 | 42 | 0.2278 | 0.2333 |
| 525 | 100 | 99 | 0.1905 | 0.1886 |

First 20 records shown; the full distribution and any outliers are visible by re-running this script and inspecting the rows list.
