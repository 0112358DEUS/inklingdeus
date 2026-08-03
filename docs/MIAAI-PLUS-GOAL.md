# MiaAI-plus goal

User-expanded goal, 2026-08-03: make this two-Spark Inkling stack **significantly better than the
MiaAI-Lab version across every measurable serving dimension**, not merely equal to one screenshot.
This scorecard augments the original N1–N4 north stars. A MiaAI-plus victory requires every gate
below; hitting one original milestone alone is useful progress but no longer proves this expanded
goal.

## Flagship moonshot — Inkling SparkFlash-1M

The user further raised the bar from an incremental benchmark win to an engineering result that
would interest Thinking Machines Lab and NVIDIA. The flagship is an upstream-quality **SM121 FA4 +
native FP4 KV + speculative 1M-context reference stack** for Inkling, with lossless and quality
proof—not a collection of launcher tweaks.

The [official SGLang matrix at current upstream `3953788`](https://github.com/sgl-project/sglang/blob/39537885961c535656b2c93159b6ab89401ab450/docs/advanced_features/attention_backend.md)
documents FA4 MHA with page-128 FP4 KV on SM90/SM100. It does not claim SM121 support. A targeted
public search on 2026-08-03 found no end-to-end artifact combining SM121,
Inkling's hybrid full/SWA/sconv architecture, DSpark target and draft, native FP4 KV, exact T4, and
1M quality. This is a candidate novel contribution, but the project must repeat the prior-art audit
before publication and must not use “first” unless the evidence supports it.

### Required technical contribution

1. Native FA4 page-128 FP4 KV on SM121, including fused/on-the-fly quantized stores and fused
   attention reads with no full-pool BF16 materialization.
2. Correct coverage of all Inkling writers: target full attention, SWA, prefix-valid/radix moves,
   and DSpark hidden-state KV injection. Packed payload and scale rows must move together.
3. Target and speculative draft both execute FA4, with byte-exact T4 and 1M NIAH/GSM8K/tools green.
4. At least one speculation breakthrough beyond configuration tuning:
   - preferred: revive native width-1 MTP on FA4, removing the separate 0.9B draft; or
   - alternate: implement true compact/variable-width DSpark verification so calibrated draft
     widths reduce verify work instead of merely truncating acceptance.
5. Upstream-ready patches, isolated regression tests, a profiler breakdown, raw benchmark/power
   telemetry, a one-command reproducible launch, and a technical report explaining the failure
   walls and design.

### Wow gates

These stretch gates sit above the all-aspects 10% scorecard:

- open-ended real chat >=40 tok/s at n=32 with the lower 1-SE bound clearing 40;
- C8 aggregate >=100 tok/s with C8 no lower than C6;
- pooled open-ended accept length >=3.2, or native MTP at equal/higher throughput with no external
  draft allocation;
- usable KV capacity >=1,256,984 tokens and quality-proven 1M context;
- at least 1.5x the current champion's tokens per joule at matched T4/quality, measured with pinned
  power telemetry rather than estimated from TDP;
- zero errors across the replicated C1–C16 suite and post-suite T4;
- every runnable byte, image payload, dataset, and raw result content-addressed and independently
  reproducible on both controls.

If the performance stretch misses but the new SM121 kernel path is correct, quality-green, profiled,
and upstream-worthy, it remains a major engineering win but does not complete this moonshot.

## Frozen public reference

The comparison is pinned to MiaAI-Lab commit
[`ce6bfbe90d6e680e7b8172ebd1922fa148158e26`](https://github.com/MiaAI-Lab/Inkling-Small-NVFP4-Dual-DGX-Sparks/tree/ce6bfbe90d6e680e7b8172ebd1922fa148158e26),
rechecked as its current `main` on 2026-08-03. Its repository publishes one 512-token screenshot,
a launcher, and prose claims; it does not publish the benchmark source, prompts, raw records,
repetitions, error bars, T4 output, or quality results. The detailed comparability boundary remains
in [MIAAI-BENCHMARK-CROSSWALK.md](MIAAI-BENCHMARK-CROSSWALK.md).

“Significantly better” means the relevant confidence bound clears a 10% margin, not that one run
rounds higher. Throughput gates use our lower one-standard-error bound; latency gates use the upper
one-standard-error bound. Every performance curve must run twice in an unchanged session with raw
records and byte-exact T4 before, between, and after.

## Public-curve superiority gates

The first instrumentation iteration must add a streaming, 512-output-token compatibility harness
that reports the definitions needed for aggregate output rate, per-stream output rate, and TTFT.
Changing that harness is its own measurement-contract iteration and cannot be combined with a speed
claim. Until MiaAI-Lab publishes its benchmark source, passing this table is called **public-scoreboard
superiority**, not an exact reproduction.

| Concurrency | Mia aggregate | Aggregate gate | Mia stream | Stream gate | Mia TTFT | TTFT gate |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 33.9 | >=37.3 | 33.9 | >=37.3 | 3.37 s | <=3.03 s |
| 2 | 48.5 | >=53.4 | 24.9 | >=27.4 | 6.99 s | <=6.29 s |
| 3 | 56.6 | >=62.3 | 24.1 | >=26.6 | 9.72 s | <=8.74 s |
| 4 | 65.6 | >=72.2 | 18.8 | >=20.7 | 9.34 s | <=8.40 s |
| 6 | 78.7 | >=86.6 | 16.7 | >=18.4 | 8.49 s | <=7.64 s |
| 8 | 74.9 | >=82.4 | 11.9 | >=13.1 | 6.21 s | <=5.58 s |

The exact rule is at least `1.10 * Mia` for both throughput columns and at most `0.90 * Mia` for
TTFT; displayed thresholds are rounded inward so rounding cannot create a false pass. C8 aggregate
must also be no lower than C6, fixing the saturation downturn visible in MiaAI-Lab's screenshot.
Extend the transparent curve through C16 and report errors/OOMs even though the reference stops at
C8.

## Real-serving superiority gates

| Dimension | Required proof |
|---|---|
| Real single-user chat | open-ended, chat-templated n=32 >=37.3 tok/s, lower 1-SE bound; original N1 >=32 remains an intermediate milestone |
| Draft efficiency | pooled open-ended accept length >=2.8 through a finetuned draft, with the full positional histogram and GSM8K class unregressed |
| Correctness | byte-exact T4 before/between/after every measured arm; zero mismatches and no fluent-but-different exception |
| Long context capacity | usable KV pool >=1,256,984 tokens (10% above MiaAI-Lab's published 1,142,712) at declared 1M context |
| Long context quality | tokenizer-measured NIAH at 512K and 1M, depths 10/50/90%, all green |
| Reasoning quality | full 1,319-item GSM8K >=94.83%, with immutable dataset checksum and resumable raw responses |
| Tool behavior | four tools x four repetitions x two turns; valid structured arguments and zero parser-token leaks |
| Reliability | both replicated C1–C16 curves complete with zero request errors/OOMs, followed by exact T4 |
| Reproducibility | exact repo SHA, image IDs and payload fingerprints on both controls; committed raw JSON/logs and hosted CI |
| Startup | three unchanged warm boots, mean + 1 SE under four minutes, without persistent-cache correctness or permission failures |
| Operational safety | no champion/default/control configuration change until the full adoption gate passes; automatic cleanup leaves both ranks stopped after experiments |

Where MiaAI-Lab supplies no comparable evidence, the claim is that this project provides and passes
a stronger public gate—not that an undisclosed MiaAI result was numerically beaten.

## Highest-value execution order

1. **FA4 page-128 FP4 KV:** implement a native SM121 E2M1/block-scale read path rather than the
   stock full-pool BF16 dequant fallback; prove all target, SWA, and DSpark writers plus
   page-boundary attention numerics, then serve/T4. This is required for the capacity and
   1M-quality gates.
2. **FA4 DSpark block sweep:** re-sweep 5/6/7 in one unchanged FP4 session. FA4 removes the old
   Triton FP32-reduction factor, so the earlier block winner is not assumed current.
3. **Streaming compatibility harness:** implement TTFT/per-stream/aggregate definitions as a
   standalone measurement iteration; freeze it before comparing performance.
4. **Draft finetune:** train the 0.9B draft on 50–100M serving-generated target tokens from weak
   prompt classes; accept only at >=2.8 with GSM8K non-regression.
5. **Quality/capacity:** run NIAH 512K/1M, full GSM8K, and tool regression on the exact performance
   candidate.
6. **Replicated speed/latency curves:** same-session champion-vs-candidate n=32 plus the public
   C1–C8 scorecard and C16 extension, all T4-bracketed.
7. **Boot/reliability/upstream:** clear the warm-boot gate, file the four upstream submissions,
   rebase on a current upstream image, and rerun the complete scorecard.

No single run, benchmark-contract substitution, weakened quality threshold, or aggregate-only win
can satisfy this expanded goal.
