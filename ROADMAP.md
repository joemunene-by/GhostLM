# GhostLM Roadmap

GhostLM is a multi-year, from-scratch effort. The released ghost-tiny is a 14.7M-param model on ~30M tokens — a learning artifact and a working pipeline, not a useful cyber-task model. The path to "useful" is the scale ladder below.

This roadmap is honest about what each rung needs (compute, corpus, time) and what each rung is expected to deliver. There are no shortcuts for "from scratch" at scale; the alternative path — fine-tuning a strong open base model — is acknowledged in the README and explicitly rejected for this project. Patience is a feature.

---

## Where we are: ghost-small line saturated at register-matching; v1.0 corpus built, ghost-base pending GPU

The v0.5.0 release reported chat-v3 at 36.9% on CTIBench MCQ. As of v0.6.0 we know that number was a positional-bias artifact: CTIBench's gold-letter distribution is 15/32/37/15 (A/B/C/D), the chat-v3 model collapsed to 98.6% C-emission during MCQ-format SFT, and a model that always emits "C" scores 37.1% on the v0.5.0 single-order metric. Real per-permutation accuracy under text scoring is **~30% across every chat-tune in the repo**. Full investigation in `docs/ctibench_bias_finding.md`.

Six independent attempts at the ghost-small (45-81M) parameter rung, all in a 4-point band on debiased text-scoring:

| Variant | Pretrain | Recipe | Debiased text-scoring |
|---|---|---|---:|
| ghost-small (v0.4) chat-v3 | 12.56M tokens, learned PE / GELU / LayerNorm | MCQ-tuned, 1.8K steps | 30.5% |
| ghost-small-v0.5 chat-v5 | v0.4.2 expanded, custom 32K BPE | hybrid raw×5 + CoT×2 | 29.7% |
| ghost-small-v0.5 chat-text | v0.4.2 expanded, text-loss SFT | answer-text gold | 30.1% |
| ghost-small-v0.6 chat | v0.5 arch + GPT-2 50K BPE | canonical chat-v3 recipe | 31.2% |
| **ghost-small-v0.7 chat (best)** | 81M wide (d_model 768), v0.5 arch | canonical chat-v3 recipe | **32.2%** |
| ghost-small-v0.8 chat | v0.7 arch + 11K Qwen-14B fact-QA in pretrain | canonical chat-v3 recipe | 31.2% |
| ghost-small-v0.9 chat | v0.7 arch + 273M-token corpus (PRIMUS + CWE + OWASP + RFC + fact-QA) | canonical chat-v3 recipe, n=2500 | 28.9% |

Three architectural axes ablated to within the 28-32% band: BPE size (32K vs 50K), positional encoding + FFN + normalization (learned PE + GELU + LayerNorm vs RoPE + SwiGLU + RMSNorm), and parameter count (45M vs 81M, 1.8×). A fourth axis (SFT objective: letter-loss vs text-loss) sits inside the same band. A fifth (corpus density via fact-QA distillation, then 273M-token open-cybersec expansion) added zero pp at v0.8 and slightly regressed at v0.9, likely because PRIMUS-FineWeb's TinyBERT-filtered crawl text dilutes the cyber-text register the model was scoring on.

**Apples-to-apples re-bench (v0.9.2) overturns both v0.9.0 and v0.9.1 diagnoses.** All earlier debiased CTIBench numbers in this repo were on a 500-record subset of the test split; only v0.9 was scored on the full 2500. Re-running every chat-tune on the full set produces a different ranking:

| Variant | CTIBench full (n=2500) | CTF eval (n=30) | SecQA (n=210, external) | Fact recall (n=50) |
|---|---:|---:|---:|---:|
| v0.4 chat-v3 (canonical from v0.5.0) | 27.6% | 50.0% | 35.0% | 0/50 |
| v0.6 chat | 28.2% | — | — | — |
| v0.7 chat (81M wide) | 27.2% | 50.0% | 37.6% | 1/50 |
| v0.7 chat-ctx1024 | 26.7% | 45.8% | — | — |
| v0.8 chat (fact-dense) | 27.4% | — | — | — |
| **v0.9 chat (273M corpus)** | **28.9%** | **59.2%** | **39.3%** | **1/50** |

v0.9 leads on every MCQ bench by 0.7-9.2 pp. SecQA confirms the cross-bench inversion against an independent third-party set; the v0.9 > v0.7 > v0.4 ranking is consistent across CTIBench full, CTF eval, and SecQA.

**But fact-recall is at floor.** A 50-question hand-written free-form fact-recall set (CVE / CWE / MITRE / OWASP / crypto / protocol / misc) graded by substring match gets 0-2% from every chat-tune in the line, and the two "hits" v0.7 and v0.9 each registered are spurious (v0.7's "Injection" surfaces in unrelated tangent prose; v0.9's "256" comes from echoing "SHA-256" in the question). **At the ghost-small parameter scale, the MCQ benches measure register matching and topic distinctness, not factual recall.** v0.9 is a better register-matcher than its predecessors but it can't tell you the CVE for EternalBlue.

This is the cleanest evidence we have that the ghost-small line saturates as a "cybersec parrot" and the next move is parameter count, not more corpus or recipe twiddling at this scale.

### v1.0 corpus is built

While ghost-small was being benched, the v1.0 corpus expansion landed (rebuild on 2026-05-06): **516,736 train / 27,049 val records / ~363M tokens / six domains**. Cybersec writeup-style content (PRIMUS, NVD, MITRE family, OWASP, RFCs, CTFtime, CISA, fact-QA, full-text arXiv) at 73%, plus three new domains the ghost-small line never saw: FineWeb-Edu general language (13%), open-web-math reasoning (6%), and security tool source code from 30 curated repos (2.4%), plus 26 NIST SP 800 publications and 11 RSS security research blogs as authoritative reference and writeup register. Per-source breakdown in [`CORPUS.md`](CORPUS.md). Leakage check returns 0.

### Ghost-base launcher shipped

`scripts/train_ghost_base.py` is the v1.0 pretrain entry point: 30L × 960d × 15h × 3200 d_ff architecture (~360M params, SmolLM2-360M shape, verified on M4 smoke), bf16, 30K-step recipe. Runs against the v1.0 `data/processed/train.jsonl`. Acceptance gate at [`docs/ghost_base_spec.md`](docs/ghost_base_spec.md): **≥40% on debiased CTIBench OR ≥65% on the CTF eval OR ≥30% on the 50-question fact-recall set**; passing any one validates the rung. The fact-recall bar is the truth metric (ghost-small fails on it; ghost-base needs to land there for a useful ship).

**v1.0 is gated on rented GPU compute** (~26h / ~$70 on a single spot H100 per the spec). Joe is sourcing GPU access; once available, the kick-off is one command. The longer-horizon hardware pathway (owned workstation, multi-year scale ladder through ghost-3B / ghost-7B, corpus-vs-hardware tradeoff, the wall at 100B+) is documented in [`docs/hardware_pathway.md`](docs/hardware_pathway.md).

```bash
PYTHONPATH=. python3 scripts/train_ghost_base.py \
  --batch-size 16 --grad-accum-steps 4 \
  --max-steps 30000 --warmup-steps 2000 \
  --learning-rate 2e-4 --dtype bfloat16
```

### GhostAgent runtime shipped (v0.9.9)

`ghostlm/agent/` is a production-shaped tool-using agent runtime that wraps any GhostLM checkpoint. It exercises bets 1 (`<|tool_call|>` traces) and 9 (`<|cite|>` tags) end-to-end: parser, tools registry (CVE / MITRE / CWE / RAG with offline caches), loop, JSON-serialisable trace. Works against any GhostLM checkpoint today (v0.9 chat will produce poor tool calls; the loop terminates safely via `answer_emitted` or the max-iterations cap). When ghost-base trains on `synth_v1.jsonl`, the runtime is already wrapped around it. CLI: `python -m ghostlm.agent --query "..."`. 31-case test suite at [`tests/test_agent.py`](tests/test_agent.py).

### Tool-use SFT pipeline shipped (v0.9.10)

`scripts/prep_tool_use_sft.py` converts the bet 1 + bet 9 synth traces (~850 records) into chat-format SFT records, with optional mixing into the existing chat data so v0.9's small-talk + identity SFT survives. `scripts/eval_agent.py` runs the agent loop against held-out provenance eval (n=15), scores on `required_substrings` with Wilson 95% CI, and supports a `--baseline` mode (max_iters=1) for paired comparison vs no-tool-use. The pipeline is M4-runnable in a few hours: synth -> prep -> SFT on v0.9 -> agent eval, with every step reproducible from one CLI line. Format compliance is the kind of narrow signal small models *can* learn even when fact recall floors at 81M params, so v0.9-chat-with-tools could plausibly produce a working agent demo before GPU compute lands.

### GhostBench agent runner shipped (v0.9.11)

`scripts/ghostbench_agent_run.py` composes GhostAgent with GhostBench: for every bet's held-out eval (bet 6 format-aware, bet 7 code-security, bet 8 binary-literacy, bet 9 provenance, bet 10 log-analysis, bet 11 IaC-security, bet 12 protocol-fields), runs the agent loop on every prompt and writes JSONL predictions that the existing `python -m ghostbench summary` and `python -m ghostbench compare` commands consume directly. A `--baseline` flag forces `max_iters=1` for the no-tools control, which produces the paired comparison via existing GhostBench machinery (Wilson CIs, McNemar p-values, Cohen's h). Until today, the agent runtime was unfalsifiable infrastructure; now every bet measures it with statistical rigor. When ghost-base lands, the same one-line invocation produces a publishable-shape per-bet table comparing ghost-base-with-tools vs ghost-base-baseline vs v0.9-chat-with-tools.

### Multi-vendor HTTP server shipped (v0.9.12)

`ghostlm/agent/server.py` exposes the agent loop over OpenAI Chat Completions (`/v1/chat/completions`), Anthropic Messages (`/v1/messages`), Google Gemini (`/v1beta/models/{model}:generateContent`), and Ollama (`/api/chat`, `/api/generate`, `/api/tags`). Tool calls happen server-side; final answers come back in the SDK's expected shape. Any team that already has an OpenAI / Anthropic / Gemini / Ollama integration in their stack can point it at GhostLM by changing one base_url. The server is a factory (`create_app(generator, config, model_name, tools)`) that takes any callable as the model abstraction, so tests inject stub generators while the CLI wires a real checkpoint. CLI: `python -m ghostlm.agent.server --checkpoint X --port 8000`. 22-case test suite at [`tests/test_agent_server.py`](tests/test_agent_server.py).

### Agent-trace distillation shipped (v0.9.13)

`ghostlm/agent/teacher.py` + `scripts/distill_agent_traces.py` close the data-quality loop. `OpenAICompatGenerator` wraps any OpenAI-compatible chat-completions endpoint (Ollama, vLLM, OpenAI, etc.) into a Generator the runtime can drive, so the teacher dispatches our tools, sees real responses, and produces fresh varied bet-1 + bet-9 traces. The distillation script reads a prompts JSONL, runs each through the teacher-backed agent, validates trace shape (USER / ASSISTANT-tool-call / TOOL / ASSISTANT-cited-answer) and optionally requires a parseable `<|cite|>` tag (bet 9 quality bar), and writes records in the format `scripts/prep_tool_use_sft.py` already consumes. Until today the SFT corpus was bounded by template variety; with this release, anyone running Ollama + Qwen-14B locally can generate thousands more high-quality traces overnight on M4 hardware. CLI: `python3 scripts/distill_agent_traces.py --teacher-base-url ... --teacher-model ... --prompts ... --out ...`. 13-case test suite at [`tests/test_agent_distill.py`](tests/test_agent_distill.py).

### MCP server retrofit shipped (v0.9.14)

`scripts/mcp_server.py` gains a `ghostlm_agent` tool that runs the full GhostAgent loop and returns the cite-tagged final answer. Claude Desktop / Cursor / any MCP-compatible client can now invoke the cybersec agent loop the same way they invoke any other tool. The retrofit reuses the model the MCP server already loaded (no second checkpoint load) via the new `make_generator_from_loaded(model, config, tokenizer, device, ...)` helper in `ghostlm/agent/runner.py`. `include_trace=True` prepends a JSON-serialised trace block before the final answer, which lets a Claude session inspect the loop's reasoning step-by-step. Combined with v0.9.12's HTTP server, GhostLM is now reachable from OpenAI / Anthropic / Gemini / Ollama / MCP clients with zero glue code; when ghost-base trains, every integration upgrades for free.

### Five new real-world cybersec tools shipped (v0.9.15)

The agent went from 4 demo-grade tools (CVE / MITRE / CWE / RAG) to 9 tools that correspond to a SOC analyst's actual investigative loop: `lookup_cisa_kev` (Known Exploited Vulnerabilities, public CISA feed, no API key), `lookup_greynoise` (IP scanner / benign / malicious classification, GREYNOISE_API_KEY), `lookup_virustotal_hash` (file-hash reputation, VIRUSTOTAL_API_KEY), `lookup_shodan` (IP service profile, SHODAN_API_KEY), `lookup_alienvault_otx` (IOC pulse search, OTX_API_KEY). Each follows the same try-real-then-cache pattern: live API when keys are set, offline cache fallback otherwise, structured `not_found` response when neither matches. Default system prompt updated to surface all 9 tools to the model. 15 new test cases bringing the suite to 210, all green.

### Static demo UI shipped (v0.9.16)

`ghostlm/agent/web_ui.py` exports a single-page HTML chat UI; `ghostlm/agent/server.py` serves it at `GET /`. No build step, no JS framework, no external dependencies, vanilla JS hitting the existing `/v1/agent/run` and `/healthz` endpoints. Tool calls render as inline panels with the tool name + args, tool responses appear as separate messages with the wrapping tags stripped, cite tags become coloured chips. Six canned example queries exercise different tools. Combined with the HTTP server (v0.9.12), the demo experience is now: clone the repo, run `python -m ghostlm.agent.server --offline`, open localhost in a browser. Three commands, zero configuration. 2 new test cases bringing the suite to 212, all green.

### Bet 7 expansion shipped (v0.9.17)

The original bet 7 bank (v0.9.5) had 12 code-security patterns, heavily Python-biased (10/12). v0.9.17 expands to 32 patterns covering Python, JavaScript, Java, Go, C, Ruby, and PHP, adding new CWE classes (1321 prototype pollution, 1333 ReDoS, 134 format string, 190 integer overflow, 285 missing authz, 326 weak crypto, 915 mass assignment, 98 LFI). The existing `scripts/synth_code_security.py` reuses unchanged; the SFT corpus grows from 48 to 128 records. Held-out eval extended from 20 to 32 prompts covering the new languages. 9 new test cases, total tests 221, all green. This is the first half of "is code at similar level as cybersec?": the SFT answer goes from 48 records to 128, comparable to bets 1/6/9 sizes.

### General-knowledge SFT bank shipped (v0.9.18)

`data/raw/chat/general_knowledge.jsonl` ships 98 hand-written 2-turn conversations across 15 topics (programming, math, science, geography, etymology, uncertainty/refusal, how-to, identity, comparison, definitions, reasoning, history, cross-domain, philosophy, conversation). `scripts/build_chat_dataset.py` gains `--general-knowledge` / `--general-knowledge-multiplier` / `--general-knowledge-val-frac` flags that mix the bank into chat training at ~5% of pairs. Combined with the existing 153-record `small_talk.jsonl` seed, the non-cybersec SFT floor is now 251 records (up from 153), enough to teach the model to recognize non-cybersec questions and admit uncertainty when appropriate. 11 new test cases, total tests 232, all green. This closes the second half of "code + general knowledge at similar level as cybersec?".

### Bet 7 phase 2 expansion shipped (v0.9.19)

`data/raw/code_security_patterns.jsonl` grows from 32 to 62 patterns across 11 languages: adds Rust, C#, Swift, Kotlin language coverage plus expanded Python / JavaScript / Java / Go / C / PHP / Ruby with new CWE classes (367 TOCTOU, 362 race condition, 90 LDAP injection, 1336 OGNL/EL injection, 113 HTTP header injection, 434 unrestricted upload, 122 heap overflow, 95 eval injection, 209 info leak via error, 1004 missing cookie flags). Held-out eval grows from 32 to 50 prompts covering the new languages and CWEs. SFT corpus produced by the existing `synth_code_security.py` reaches **243 records**, comparable to bets 1 (424), 6 (560), 9 (429). Tests still 232 passing (the test file assertions tightened to reflect the larger bank). The "is code SFT comparable to cybersec SFT?" question is now answered yes by both record count and language breadth (11 languages vs the cybersec bets' single-language focus).

### Programming Q&A SFT bank shipped (v0.9.20)

`data/raw/chat/programming_qa.jsonl` ships 66 hand-written 2-turn conversations covering broader-than-security programming chat: how-to, code-explain, debug-help, refactor, and language concepts across Python (20), JavaScript (5), Go (3), Rust (4), Java (1) plus 16 generic concepts (closures, big O, dependency injection, GIL, async, REST, GraphQL, etc.) and 5 tooling records (Docker, git, project structure). The chat builder gains `--programming-qa` flags mirroring the small-talk and general-knowledge wiring; default 5x multiplier brings the bank to ~5% of training pairs. 8 new test cases bringing the suite to 240, all green. Cross-domain SFT floor is now 317 records (~14% of unique SFT) with explicit coding-chat supervision — the model now has signal for everyday programming questions outside security.

### Math + reasoning SFT bank shipped (v0.9.21)

`data/raw/chat/math_reasoning.jsonl` ships 58 hand-written 2-turn conversations across 10 topics: arithmetic (11), word problems (12), algebra (7), logic (6), geometry (5), statistics (4), proofs (4), probability (3), combinatorics (3), concepts (3). Mix is intentionally weighted toward arithmetic + word problems (the patterns small models can actually learn) over deep proofs. Includes the classic Cognitive Reflection Test problems (bat-and-ball, lily-pad doubling, parallel-machine widget production) so the model learns to walk through computations rather than guess. The chat builder gains `--math-reasoning` flags; default 4x multiplier (slightly lower than the 5x for general-knowledge / programming-qa because the open-web-math pretrain already covers math at 6%). 9 new test cases bringing the suite to 249, all green. Cross-domain SFT total now 375 records (~16% of unique SFT) with explicit math + reasoning supervision; combined with v0.9.18-20 banks, the model has dedicated signal for cybersec, code, factual general knowledge, programming chat, and math/reasoning.

### Canonical models on disk
- **Density / generation:** `checkpoints/phase4_ghost_small/best_model.pt` (v0.4.0, val_loss 2.3535, val PPL 11.12). Unchanged since v0.5.0.
- **Chat (best ghost-small):** `checkpoints/phase19_chat_v09/best_model.pt` (v0.9, wins all three MCQ benches on apples-to-apples scoring).
- **Chat (single-order CTIBench winner, biased, historical):** `checkpoints/phase5_chat_v3/best_model.pt` (v0.5.0 canonical, 36.9% single-order, 27.6% debiased on full bench).

**Historical / preserved:**
- Phase 1 / 2: `checkpoints/best_model_phase{1,2}.pt`
- Phase 3 / 3.5 / 3.6: `checkpoints/phase{3_refresh,3.5_balanced,3.6_exploitdb}/best_model.pt`
- v0.5 base + chat: `checkpoints/phase{6_v05_pretrain,7_chat_v05_long,8_chat_v05_v5}/best_model.pt`
- v0.6 base + chat: `checkpoints/phase{9_v06_pretrain,10_chat_v06}/best_model.pt`
- v0.7 base + chat: `checkpoints/phase{14_v07_pretrain_v3,15_chat_v07}/best_model.pt`
- v0.8 base + chat: `checkpoints/phase{16_v08_pretrain,17_chat_v08}/best_model.pt`
- v0.9 base (training): `checkpoints/phase18_v09_pretrain/`

---

## Phase 3.5 — Corpus rebalance (complete, 2026-04-26)

Corpus is the long-term moat. Phase 3 brought volume (~30M tokens) but with NVD at ~90% token share, which made every other source statistically irrelevant. Phase 3.5 fixed the structural problem first; volume comes next.

**What landed:**
- **MITRE ATT&CK collector** — 691 enterprise techniques (Apache 2.0)
- **CAPEC collector** — 609 attack patterns (Apache 2.0)
- **CTFtime real-writeup collector** — 473 inline writeups across 28 curated 2020-2024 events; per-record attribution; off-site links deliberately not followed (per-page licensing not auditable)
- **GitHub-CTF-repos collector** — config-driven, JSON list of repos with explicit SPDX license per entry
- **NVD subsampling** — `rebuild_corpus.py --max-cve-tokens N` deterministically caps NVD's contribution by content-hash prefix. Without it, NVD owns ~90%; at `--max-cve-tokens 6000000` NVD owns 65.3% with diversity sources at ~35%.

**Current corpus state:**

| Source | Records | Tokens | Share |
|---|---|---|---|
| NVD CVE (subsampled) | 71,828 / 333,540 | 5.74M | 65.3% |
| Synthetic CTF (placeholder) | 3,000 | 1.51M | 17.2% |
| arXiv cs.CR abstracts | 2,000 | 0.74M | 8.4% |
| CTFtime real writeups | 467 | 0.47M | 5.3% |
| MITRE ATT&CK | 691 | 0.26M | 2.9% |
| CAPEC | 609 | 0.07M | 0.9% |
| **Total (post-dedup)** | **74,635** | **~8.8M** | |

Per-source license notes: see [CORPUS.md](CORPUS.md). Reproducible via `python3 scripts/rebuild_corpus.py --max-cve-tokens 6000000` (deterministic by content hash).

---

## Phase 3.6 — Corpus volume (attempted, capped by model capacity)

The structural rebalance worked; the volume add did not — at this rung. Phase 3.6 added Exploit-DB (~3.77M tokens, 30% of the new corpus mix) and re-trained ghost-tiny at the same 30K-step recipe. Eval suite regressed 14.4 pp (31.2% → 16.8%) and every existing per-source PPL got 28–42% worse. ghost-tiny at 14.7M params is at capacity — adding 43% more diverse text forces parameter reallocation away from the existing seven sources.

**What landed in v0.3.7 (corpus-side):**
- Exploit-DB collector hardened (persistent mirror, resume, Metasploit filter, structured metadata, date-desc CSV sort)
- 5,000 Exploit-DB records pulled (~3.77M tokens, GPL-2.0, mostly PHP webapps + Linux locals + Python PoCs from 2019–2025)
- arXiv full-text PDF collector built (uses pymupdf; not yet pulled at scale)
- CTFtime event-discovery script built (queries CTFtime API, filters by weight + participants; not yet run)
- v0.4.0 corpus-target tracker added to `scripts/data_audit.py`

**Status of the originally-planned Phase 3.6 sources:**

| Source | Phase 3.5 | v0.3.7 status |
|---|---|---|
| Exploit-DB | 0 | ~3.77M tokens pulled, 30% of corpus |
| arXiv cs.CR | 2,000 abstracts (~0.74M) | Full-text scaffolding shipped, not yet pulled |
| CTFtime real writeups | 473 | Discovery script shipped, not yet expanded |
| Security research blogs | 0 | Not yet built |
| Tool docs | 0 | Not yet built |
| **Total** | **~8.8M** | **~12.56M (Phase 3.6 corpus, sitting in `data/processed/`)** |

The Phase 3.6 corpus is reproducible: `python3 scripts/rebuild_corpus.py --max-cve-tokens 6000000`. It's the cleanest training target for the ghost-small run.

**The lesson:** more corpus at fixed model size has hit diminishing returns. The path forward is the model, not the data.

---

## Phase 4 — ghost-small (complete, 2026-04-30)

| Item | Value |
|---|---|
| Layers / d_model / heads | 6 / 512 / 8 |
| Params | ~45M actual (the 55M estimate was the config-side hand-waved figure; real count from `model.num_params()` is 45.17M) |
| Hardware | Mac Mini M4 (MPS), batch 8 × grad_accum 4, ~15h wall-clock |
| Training tokens | 12.56M (Phase 3.6 corpus, unchanged) |
| Steps | 30,000 |
| Final val_loss | **2.3535** |
| Best checkpoint | `checkpoints/phase4_ghost_small/best_model.pt` |

The first scale-up rung delivered cleanly. Loss curve was still descending at step 30k (train ~2.17, val 2.35) — no overfitting plateau visible at this step budget on this corpus, despite the corpus being well below Chinchilla-optimal for 45M params (~1.1B tokens would be the formal target).

**All four gates closed positively:**
1. ✓ Recipe-scales-with-data validated (Phase 2→3, +1.6 pp eval).
2. ✓ Source-mix structurally balanced (Phase 3→3.5, +11.2 pp eval, NVD 90% → 65%).
3. ✓ ghost-tiny capacity ceiling found (Phase 3.6, −14.4 pp + per-source PPL 28–42% worse on every source).
4. ✓ **Capacity-reallocation hypothesis confirmed.** ghost-small at 45M absorbed the same Phase 3.6 corpus that broke ghost-tiny, with **every existing source improving 59–78%** relative to the Phase 3.5 canonical. Overall val PPL dropped 66.05 → 11.12 (−83%).

**Eval methodology finding (worth flagging for v0.4.x):** the existing PMI security suite favored Phase 3.5 (31.2% vs Phase 4's 23.2%), but with conservative logp scoring Phase 4 wins (19.2% vs 17.6%). PMI subtracts unconditional candidate log-prob to break ties; a higher-capacity model with a tighter probability distribution gives PMI less separation. The 25-sample-per-task suite is small enough that this calibration asymmetry can flip headlines. Resolving cleanly probably requires both: (a) a bigger eval set, (b) a calibration-stable scoring rule.

---

## Phase 4.x — extension and corpus expansion (next on the critical path)

Phase 4 left two cheap, valuable follow-ups before the ghost-base jump:

- **v0.4.1 — extension run.** Train the existing ghost-small for another 30k–60k steps on the Phase 3.6 corpus and observe whether val_loss continues past 2.35 or plateaus into overfit. Cheap (one more overnight on M4). Tells us whether the 30k recipe was meaningfully undertrained at this corpus size, and gives a credible upper bound on what ghost-small can squeeze out of 12.56M tokens.
- **v0.4.2 — corpus expansion (Phase 4.5 corpus).** Run the arXiv full-text PDF collector at scale (collector landed in v0.3.6 but data not pulled), expand CTFtime past the curated 28-event seed via the discovery script, and ship the security-research-blogs collector. Target: ~50M-token corpus that ghost-small can train on without the per-source mode collapse Phase 3.6 forced on ghost-tiny. This is the corpus that makes Phase 5 (ghost-base) actually informative rather than overfit-prone.

Both are doable on local hardware over weeks. ghost-base is gated on at least the corpus expansion landing.

---

## Phase 5 — ghost-base (~350M params)

| Item | Value |
|---|---|
| Layers / d_model / heads | 12 / 768 / 12 |
| Params | ~350M |
| Hardware target | Rented GPU (A100 / H100 hours, ~hundreds of hours) or owned RTX 6000 Ada / 6000 Pro Blackwell. See [`docs/hardware_pathway.md`](docs/hardware_pathway.md). |
| Training tokens (Chinchilla-optimal) | ~7B |

The first rung that needs rented GPU compute. This is where domain-coherent generation should start to emerge — the model should be able to produce a few sentences of structurally correct cyber-text without falling apart. Still not factually reliable.

**Gating (after v0.4.0):**
1. ✓ ghost-small recipe validated (Phase 4 confirmed scaling works).
2. ⚠ Corpus volume — at 12.56M tokens, ~7B Chinchilla-optimal is 550× away. Phase 4.5 expansion needs to land first; otherwise ghost-base will overfit a tiny corpus before generalization kicks in.
3. ⚠ External GPU compute or owned 4090/5090-class hardware.

**Cost estimate:** at ~$2–3/H100-hour, a Chinchilla-optimal run is on the order of low-thousand-dollar compute. Doable as a focused-burst project; not casual.

---

## Phase 6 — ghost-1B (long-term goal)

| Item | Value |
|---|---|
| Layers / d_model / heads | 24 / 1024 / 16 |
| Params | ~1B |
| Hardware target | Rented H100 cluster, or owned RTX 6000 Pro Blackwell 96GB (single-card, fp8 native). 4090/5090 24GB hits the VRAM cliff at this rung. Pathway in [`docs/hardware_pathway.md`](docs/hardware_pathway.md). |
| Training tokens (Chinchilla-optimal) | ~20B |

The smallest scale at which a from-scratch cyber LM has a real shot at being **genuinely useful** for tasks like CVE-to-exploit explanation, CTF challenge reasoning, or structured log analysis. Note that "useful" does not mean "competitive with general-purpose 7B+ models" — those have ~20× the params and ~100× the training data. ghost-1B's value proposition is *narrow domain depth*, not breadth.

**Cost estimate:** Chinchilla-optimal training of a 1B model is in the ten-thousand-dollar range on rented compute. Or several months on a single owned 4090/5090. This is the rung where the project either gets serious external support, gets done over years on consumer hardware, or stalls.

---

## Realistic timeline

A useful from-scratch 1B cyber LM is **2–3 years of sustained evenings/weekends work** — not because the steps are hard individually, but because corpus curation is slow, compute access at scale is gated by money or patience, and each scale rung needs the previous rung's recipe to be validated first.

This is the actual shape of the work. There are no shortcuts for "from scratch."

What that timeline does *not* require:
- New architecture inventions (the recipe is stable)
- A team (single-maintainer is feasible at this pace)
- Continuous compute (corpus and eval work fills the gaps)

What it does require:
- Corpus curation as a first-class, ongoing track (see CONTRIBUTING.md)
- Eval harness built before scale-up so improvements are measurable
- Patience.

---

## Adjacent tracks (not on the critical path)

- **Eval harness expansion** — held-out CVE→description, vuln-type classification, exploit-vs-benign code, CTF-challenge classification. Build before scaling so we can detect real progress vs. memorization.
- **HuggingFace Hub publication** — once ghost-small has a checkpoint worth publishing, push safetensors weights + config sidecar.
- **Gradio web demo** — for ghost-small or above. Not worth doing on ghost-tiny.
- **Fine-tuning scripts** — once ghost-base or ghost-1B exists, expose adapters / LoRA pipelines so users can specialize the base model further.

These all become valuable at the upper rungs. Doing them on ghost-tiny would be premature.
