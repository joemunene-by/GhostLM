# Hardware pathway for the GhostLM scale ladder

This is the rolling document for "what does it take to keep going up the
ghost-tiny → ghost-small → ghost-base → ghost-1B → ghost-3B → ghost-7B
ladder, and beyond, from a solo-project home setup." It captures the
hardware decisions that have to land before each rung, plus the harder
truth that hardware stops being the binding constraint somewhere around
ghost-3B. Pair this with [`ghost_base_spec.md`](ghost_base_spec.md) (the
v1.0 design doc), [`distributed.md`](distributed.md) (multi-GPU bring-up
notes), and the corpus-expansion threads in [`dataset.md`](dataset.md).

## Where the project sits today

ghost-small (45-81M) saturated at ~28% on debiased CTIBench and 0-2% on
free-form fact recall. The diagnosis from
[`v05_postmortem.md`](v05_postmortem.md) and the v0.9.x release notes is
that 81M parameters cannot bind facts retrievably. The fix is parameter
count, not corpus polish, which is why ghost-base (~360M, SmolLM2-360M
shape) is the next rung. The launcher is shipped at
[`scripts/train_ghost_base.py`](../scripts/train_ghost_base.py); the run
itself is gated on GPU access. All ghost-small training was done on a
Mac M4 mini, which is the binding constraint that motivates this doc.

## What an M4 actually buys you

The M4's GPU sustains roughly 0.7-1.2 TFLOPS bf16 in PyTorch MPS, and
its unified-memory pool is shared with system RAM so 360M-scale training
will OOM at default batch sizes. The ghost_base launcher's smoke-test
flag (`--batch-size 1 --grad-accum-steps 32`) runs at ~5 s/step on an
M4. That is roughly **8% of an RTX 6000 Ada's throughput** on the same
shape. Practical M4 ceiling is 81M (the v0.9 line) and that is also
where the project hit its capability wall. This is consistent.

## The card to buy

**RTX 6000 Pro Blackwell 96GB.** New retail is around $11.5K; used from
a workstation reseller (ServerMonkey, B&H, Microcenter business) lands
in the $10K range. That is the buy if the goal is a multi-year
single-workstation path that does not bottleneck on hardware before
ghost-7B.

The case for Blackwell over the cheaper Ada 48GB:

1. **96GB VRAM**. Ghost-base, ghost-1B, ghost-3B are all native bf16
   training without offload tricks. Ghost-7B fits with fp8 + grad
   checkpointing, ghost-13B fits with fp8 + grad checkpointing + CPU
   optimizer offload (painful but viable).
2. **Hardware fp8/fp4 support**. Blackwell's tensor cores execute fp8
   GEMM natively, which is a ~1.8-2.2x speedup over bf16 on supported
   kernels (Transformer Engine, torchao). For 7B+ this is the
   difference between "trainable in weeks" and "trainable in months".
3. **Workstation form factor**. 300-600W TDP, blower cooling, ECC
   memory, single 12VHPWR. Fits a normal full-tower case. This matters
   because all of ghost-1B onward is multi-day or multi-week
   unattended training, and ECC catches the silent bit-flips that
   consumer cards (4090, 5090) have produced in long-haul bf16 runs in
   the wild.
4. **Headroom over the project lifetime**. Ada 48GB ages out at
   ghost-3B. Blackwell carries through ghost-7B. The $3.5K premium
   amortizes to about $700/year over a 5-year horizon, less than one
   spot-H100 weekend rental.

The Ada 48GB ($6.5K used) is the right card if the project wraps at
ghost-3B. The 4090 24GB ($1.8K) is the right card for nothing, given
this trajectory: it cannot do ghost-1B comfortably and you will resell
at a loss inside 6-12 months.

## Capability per rung on the Blackwell 96GB

| Rung | Params | Approach | Wall-clock at Chinchilla-optimal token count |
|---|---:|---|---:|
| ghost-base | 360M | bf16 native, batch 64 | ~6 hours |
| ghost-1B | 1B | bf16 native, batch 16-32 | ~2 days |
| ghost-3B | 3B | fp8 native, batch 8-16 | ~5 days |
| ghost-7B | 7B | fp8 + grad checkpointing | ~2-3 weeks |
| ghost-13B | 13B | fp8 + grad-ckpt + CPU offload | ~6-8 weeks (slow but possible) |
| ghost-30B+ | 30B+ | not viable on a single card, see below | n/a |

Wall-clocks assume 70-80% kernel efficiency, a modern PCIe 5.0 NVMe for
the dataloader, and 128GB system RAM for offload buffers. They get
worse fast if any of those are missing.

## The harder ceiling: corpus, not hardware

Chinchilla-optimal tokens scale linearly with parameters:

| Rung | Chinchilla tokens | Current GhostLM corpus is short by |
|---|---:|---:|
| ghost-base 360M | 7.2B | 25× |
| ghost-1B | 20B | 70× |
| ghost-3B | 60B | 200× |
| ghost-7B | 140B | 480× |
| ghost-13B | 260B | 900× |

The current corpus is 363M tokens spanning six domains
(see [`CORPUS.md`](../CORPUS.md)). A 7B model trained on 363M tokens
would be a beautifully expensive 7B that's worse than a properly-trained
1B. **Hardware investment past ghost-3B is wasted unless corpus
investment scales with it.**

Practical corpus expansion paths that need to land before each rung:

- **ghost-1B (target 20B tokens, 50× over current).** Full PRIMUS-FineWeb
  (current is a sampled shard), full arXiv cs.CR + cs.AI scrape with
  PDF-to-text, every Krebs / SANS / Schneier / Specops / DFIR.org blog
  pre-2025, full Cisco / Palo Alto / Fortinet / Crowdstrike whitepaper
  archive, full IETF RFC corpus (not just security-tagged), full PRIMUS
  TIE, full GreyNoise blog, full Mandiant report archive, paid threat-
  intel feeds (Recorded Future, Mandiant Advantage, MISP). Realistic
  cost: $5-15K of API + paid-feed budgets. Realistic time: 3-6 months
  if pursued seriously alongside training.
- **ghost-3B (target 60B tokens).** Distillation. At this scale, the
  remaining cybersec text on the public web is tapped out. The lever
  becomes synthetic data: Qwen-72B / Llama-3-70B / Claude-3.5 generating
  factual chains, multi-step CTF write-ups, threat-modeling exercises,
  deobfuscation walkthroughs, malware analysis prose. Distillation
  recipes from the SecQA / SecLM / TIE-LLM literature.
- **ghost-7B (target 140B tokens).** Now you need general-domain text
  alongside cybersec to keep the model's language modeling solid. RedPajama
  v2 or DCLM-baseline filtered for security adjacency. Books3-style
  long-context. Multi-turn conversation corpora. Code corpora (BigCode,
  StackOverflow). At 7B, 75% of the corpus is general; 25% is the
  cybersec specialization that defines GhostLM.
- **ghost-13B+.** Curated multi-trillion-token mixes. Realistically this
  means standing on top of a base model rather than training from
  scratch (LoRA, continued pretrain, distillation), which is a
  philosophical change to what "GhostLM" means.

## What sits above the Blackwell 96GB

Roughly in order of cost and complexity:

1. **2x Blackwell 96GB with FSDP** (~$20K). 192GB pooled, ghost-7B
   native, ghost-13B with offload. Adds NCCL + sharding + dual-PSU
   complexity that hurts solo-iteration speed; right answer for a
   2-3 person team, marginal for a solo project.
2. **Used H100 80GB** ($25-40K, dropping). Per-FLOP champion. But it's
   a datacenter card with passive cooling, needs a server chassis or
   serious DIY airflow. Wrong for home, right for a colocation
   half-rack.
3. **Used H100 80GB SXM 8x DGX node** ($150-250K). Datacenter pod,
   640GB pooled, NVLink fast interconnect. Realistic ceiling for a
   well-funded solo project; opens ghost-30B and ghost-65B from
   scratch, possibly ghost-100B with offload.
4. **Cloud rental for one-off runs** ($1-5K per Chinchilla run on H100,
   $50-200K per run on H100 cluster). Cheaper than ownership for any
   rung that runs less than ~10 times. The right way to do ghost-7B
   and above for a long time before any owned hardware purchase makes
   economic sense.

## Beyond ghost-7B: 100B+ params

100B-scale training is genuinely cluster territory. A 100B bf16 model
needs ~1.4 TB of param + grad + optimizer state alone before
activations. That's a minimum of 8 H100 80GB cards or 4 H200 141GB cards
under FSDP+ZeRO-3, and Chinchilla-optimal is 2 trillion tokens which is
CommonCrawl-scale dataset work, not curated corpus work. Realistic
trajectories from where GhostLM sits today:

- **Distill into a 100B base model rather than train from scratch.**
  Take Llama-3.1-70B / Mistral-Large / Qwen-2.5-72B as a starting
  point, continued-pretrain on the full GhostLM corpus, chat-tune with
  the GhostLM SFT recipe. The output is "GhostLM 70B (Llama-based)";
  the cybersec specialization and chat behavior is GhostLM, the
  language modeling foundation is borrowed. This is achievable on
  rented H100 cluster time within a year of consistent work, on the
  order of $10-30K per training run.
- **Skip 100B from scratch entirely.** The honest framing is that no
  single-person-funded project trains a 100B from-scratch transformer
  on a curated cybersec corpus this decade. The ones that do (Phi-3,
  SmolLM2, OLMo) are funded research labs. GhostLM's value
  proposition is a fully-from-scratch *small-to-mid* cybersec model
  with transparent training, honest evaluation, and a documented scale
  ladder. The "100B GhostLM" version of that proposition is a fine-tune
  on a borrowed foundation, not a full pretrain.
- **Wait for the hardware curve to bend.** A B200 successor in 2027-28
  with 384GB VRAM and fp4-native execution makes 100B-from-scratch on a
  4-card workstation thinkable for the first time. By then the corpus
  question is the real wall: even if hardware is free, where do 2T
  tokens of coherent cybersec-aligned text come from?

The honest direction setting: **plan to ghost-7B as the from-scratch
ceiling on owned hardware, and treat anything above that as a
fine-tune / continued-pretrain on a borrowed base.** The
"fully-from-scratch" identity holds for 7B and below, where it is
genuinely a from-scratch model rather than someone else's foundation
with a cybersec coat of paint.

## Investment allocation over a multi-year horizon

For a solo project that takes the trajectory seriously, the budget
shape that actually maximizes capability per dollar is roughly:

| Investment | One-time | Recurring | Why it matters |
|---|---:|---:|---|
| RTX 6000 Pro Blackwell 96GB workstation | $12K | $0 | Caps hardware spend until ghost-7B, then mostly retires |
| Corpus expansion (years 1-3) | $5-15K | $1-3K/yr | The actual binding constraint past ghost-3B |
| SFT data quality (annotation, distillation) | $2-5K | $1-2K/yr | A 50K-example chat SFT scored by Qwen-72B beats more pretrain at this scale |
| Cloud rentals for ghost-7B+ Chinchilla runs | $0 | $1-5K/run | Cheaper than owning a ghost-7B-capable rig |
| Eval infrastructure (private holdout sets) | $1-3K | $0.5K/yr | Cataclysmic if you skip; you cannot trust public benchmarks at this scale |
| **3-year total** | **~$22-35K** | **~$5-25K** | A model worth privatizing |

Note that less than half of the 3-year total is hardware. This is the
opposite shape from the natural intuition ("I need a bigger GPU") and
matches the empirical experience of every solo from-scratch LM project
that has shipped: corpus and eval are where the time and money go.

## When this doc gets revised

- After the ghost-base v1.0 run lands and the acceptance gate is
  evaluated. If 360M clears one of the acceptance criteria (≥40%
  CTIBench OR ≥65% CTF eval OR ≥30% fact recall), this doc is on the
  right track. If it doesn't, the diagnosis flips and this whole pathway
  may be premature; revisit
  [`docs/ghost_base_spec.md`](ghost_base_spec.md)'s diagnosis-flip
  priority list before more spend.
- After every concrete corpus expansion. Corpus tokens is the variable
  that shifts the per-rung wall-clocks; doubling the corpus changes the
  Chinchilla math.
- After every NVIDIA / AMD generation. Blackwell next-gen (rumored
  Vera-Rubin GR100 in 2026-27) and AMD MI400 may shift the per-dollar
  curve enough to retire the Blackwell 96GB recommendation.
