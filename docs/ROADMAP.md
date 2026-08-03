# Roadmap

## Active north-star loop — control1/control2

The north-star loop resumed from current `main` on 2026-08-03. Completed rows are immutable:
E1 retained the single-HCA transport, E2 was killed as not applicable, and E3 promoted DSpark
block 5. E4 re-baselined that champion in the same session at **26.57 +/- 0.33 tok/s** and
**2.142 +/- 0.026 accept**, with byte-exact T4 before and after, then rejected native width-1 MTP
at its predeclared boot-dead gate. E8 retained NCCL autotuning after three forced-protocol arms
produced no adoptable gain, then accepted continuous decode steps 2 at +0.645 tok/s with acceptance
and latency improvements. The exact-default all-task adoption gate passed. E8 then retained 8 KV
splits after both alternatives lost. E5's first session localized manifest wall #25; its clean-root
restart cleared that wall but rejected cache adoption as too small. E6 is next.

North-star distance on the adopted E8 steps-2 champion:

- **N1:** 5.77 tok/s below 32 tok/s on the 26.225 +/- 0.323 all-task-gate open-ended n=32 result.
- **N2:** 0.662 accept below 2.8 on the 2.138 +/- 0.026 open-ended result; no finetuned draft
  exists yet. The pooled/task-class values are not substitutes for the N2 gate.
- **N3:** no FA4 serving lane on sm_121; E7 remains engineering work.
- **N4:** NIAH@1M, full GSM8K, tool regression, four upstream submissions, and current-image
  rebase are all still open.

| Rank | Status | Artifact / next proof |
|---|---|---|
| E1 dual RoCE twins | **COMPLETE — INCONCLUSIVE** | Both twins 111.62 Gb/s; dual 24.636 +/- 0.291 vs single 24.928 +/- 0.295 tok/s. Retain single `rocep1s0f1`. |
| E2 dense FP4 GEMM | **COMPLETE — NOT APPLICABLE** | `flashinfer_trtllm` unsupported on capability 121; checkpoint has no dense NVFP4 layer controlled by this flag. No serving A/B. |
| E3 block 5/6/7 | **COMPLETE — ACCEPT block 5** | **26.007 +/- 0.334** vs block 7 at 24.747 +/- 0.208 tok/s; delta +1.260, combined SE 0.394, T4 ×6. |
| E4 width-1 native MTP | **COMPLETE — REJECTED, NEW WALL #24** | Baseline 26.57 +/- 0.33 tok/s, accept 2.142 +/- 0.026, T4 x2. Candidate loaded target + MTP and allocated a 1,160,700-token pool, then the fp4 Triton extend kernel failed to parse during width-2 graph capture. No candidate serving claim. |
| E8 decode-latency sweep | **COMPLETE — ADOPT CDS=2** | Steps 2: +0.645 tok/s vs same-session baseline, combined SE 0.425; accept +0.046; latency -0.153 s. Default contract/all-task/T4 adoption passed. Protocol forced arms were null. KV splits 4 was -0.323/inconclusive; splits 16 -0.478 and rejected on acceptance. Retain protocol autotuning and splits 8. |
| E5 persistent JIT caches | **COMPLETE — INCONCLUSIVE** | Clean restart: no-mount warm T4 395.7 +/- 3.3 s vs cache warm 372.0 +/- 9.1 s; only 23.7 s / 6% saved and still >240 s. Serving +0.137 tok/s, neutral. T4 x11. Keep cache mounts off. |
| E6 mem-fraction under C8-C16 | **OPEN — NEXT** | `docs/EXPERIMENT-E6-MEMFRAC.md`; active-earlyoom preflight, scoped 12-GiB guard, T4, chat n=32 at C8/C16. |
| E7 FA4 paged-KV port | **SCOPED — ENGINEERING REQUIRED** | `docs/EXPERIMENT-E7-FA4-PORT.md`; pinned donor, four incompatible seams, staged GPU numerics/T4/quality gates; no runnable port yet |
| Q1 chat-templated harness | **LIVE-PROVEN** | Used for E1 and E3 exact n=32 serving measurements with plan-identity checks and T4. |
| Q2 depth quality | **READY — NOT RUN** | `docs/QUALITY-GATES-Q2-Q3.md`; token-measured NIAH 512K/1M at 3 depths plus full 1,319-item GSM8K ≥94.83% |
| Q3 tool-call regression | **READY — NOT RUN** | `docs/QUALITY-GATES-Q2-Q3.md`; 4 tools ×4 reps ×2 turns, structured args and zero parser-token leaks |
| Q4 C1→C16 curve | **READY — NOT RUN** | `benchmarks/concurrency_bench.py`; chat-templated exact n=32 at C1/2/4/8/16. |
| Q5 no-GPU CI | **COMPLETE** | Python compile/tests, local Markdown links, launch dry-run, and shell syntax pass. |

Exhaustion counter: **1 / 4**. E5's initial wall reset the sequence; its clean-root restart had no
adoption, new wall, or profiling win and begins a new null sequence. E4 likewise does not count
because it localized a new boot-dead wall.

## Historical campaigns

Committed follow-on campaigns (in order):

1. **STS + SPS calibration for cap-accept scheduling** (tooling shipped in `benchmarks/`): the
   confidence head is trained and present, but unusable until per-position temperatures are fitted
   and a cost table is profiled. The only remaining pure-config lever on accept.

2. ~~**mxfp8 KV cache**~~ — ✅ DONE (1.94× pool, superseded by fp4).

3. ~~**NVFP4 KV cache on the triton backend**~~ — ✅ **DONE: 3.12× pool (1.1M tokens), shipped in
   `patches/kv-quant/`.** See the README headline section. Remaining follow-ons: long-context needle
   tests at 500K–1M, concurrency benchmarks under fp4, and a TTFT check (the draft KV write falls back
   to a per-layer python loop under quantized dtypes — correctness-neutral, possible prefill cost).

   *(historical scoping note, kept for context:)* the capacity unlock toward true 1M in-flight
   tokens. The pool/storage side exists (fa4 uses it); the gap was `q/k/v_descale` handling + fp4
   block-scale dequant in the triton extend/decode/verify kernels. Donor code identified:
   upstream PR #32333 (DSV4 fp4 triton dequant) + a fleet-internal MLA nvfp4-KV triton mod.
   Full kernel-level plan in `KV-QUANT-TRITON-PLAN.md` (pre-implementation document; the as-built
   description is `KV-QUANT-IMPLEMENTATION-NOTES.md`).

4. ~~**A4Q native-fp4 attention**~~ — ❌ **NOT APPLICABLE to Inkling-Small on SGLang.** Evaluated and
   rejected on four independent grounds:
   - **KV width 1024** (8 kv-heads × 128 head_dim). A4Q's gain scales with KV width and needs
     **≥4096** to amortize its quantization overhead — at 1024 it is expected to be a net loss.
   - **Implementation is a FlashInfer FA2 kernel**, but Inkling asserts `attention_backend in
     ("fa4","triton")` and fa4 is sm_100-only ⇒ triton is the only legal lane on GB10 (wall 1).
     There is no seam to attach it to.
   - **It is a vLLM integration**, not SGLang.
   - **Inkling is sliding-window (512) + short-conv**, so its attention prefill is closer to linear
     than quadratic; A4Q's advantage comes precisely from accelerating quadratic prefill.

   A4Q remains excellent for *dense-GQA, wide-KV* models on vLLM (measured elsewhere on this fleet:
   Nemotron-3-Omni TTFT −22% @60K scaling to −39% @256K). It is simply the wrong tool for this model.

5. ~~**DSpark cap-accept calibration**~~ — ✅ **RESOLVED: it works, and it still loses.** Both
   artifacts were produced (SPS table with `match_fraction=1.00`; STS from 19,871 samples, ECE
   0.03651→0.03453 with a joint coordinate-descent fitter that beats the shipped greedy one).
   Calibrated cap-accept reaches accept 3.39 ± 0.18 — statistically level with static — but at
   23.4 ± 1.3 tok/s vs 34.7 ± 1.5. Full mechanism in
   [DSPARK-CALIBRATION-FINDINGS.md](DSPARK-CALIBRATION-FINDINGS.md). **Keep static scheduling.**
   The E3 campaign later promoted static block 5; the figures in this historical campaign used
   block 7 and should not be reinterpreted as block-5 measurements.
   *(superseded note, kept for context:)* The confidence (STS) recorder only runs
   inside the cap-accept planner, but the planner degenerates to verify-all until an SPS cost table
   exists (`sps_table=uninitialized ... zero scheduling gain`), and the SPS recorder writes through an
   info-dumper with no retrievable output path exposed. So cap-accept cannot be calibrated here and
   measures worse than static block-7. Tooling for both fits is in `benchmarks/` if a future build
   exposes the dump path.

6. **Draft finetune** — the only remaining lever that raises accept fundamentally (a 0.9B draft
   predicting a 276B target caps around accept 3.5). Everything else is scheduling.

7. **Helion native autotune** (`HELION_AOT_AUTOTUNE=create`) to replace the seeded sm_100 configs.
