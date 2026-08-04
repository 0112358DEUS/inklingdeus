# Roadmap

## Expanded user goal — significantly exceed MiaAI-Lab

The user expanded completion on 2026-08-03 from hitting any one original north star to proving
all-aspects superiority over MiaAI-Lab's pinned public result. The authoritative scorecard is
[`docs/MIAAI-PLUS-GOAL.md`](MIAAI-PLUS-GOAL.md): every published C1–C8 throughput/TTFT cell must
clear a 10% confidence-bound margin, C8 must not regress from C6, and the same candidate must also
pass real-chat >=37.3 tok/s, accept >=2.8, T4, >=1,256,984 usable KV tokens, NIAH@1M, full GSM8K,
tools, reliability, reproducibility, and warm boot. Original N1–N4 remain intermediate milestones;
none alone completes the expanded goal. The flagship **Inkling SparkFlash-1M** moonshot adds an
upstream-quality SM121 FA4+native-FP4-KV contribution, a native-MTP or true compact-verify
breakthrough, real chat >=40 tok/s, C8 >=100 tok/s, acceptance >=3.2 (or draft-free native MTP),
and measured >=1.5x champion tokens/joule.

## Queued successor campaign — SparkFlash-X

**State: QUEUED, not yet active.** SparkFlash-X begins only after SparkFlash-1M v1.0 has a tagged,
released champion. Its baseline tuple — tok/s, acceptance, usable pool, J/token, and C8 — must be
read from that release and reproduced under the same harness; roadmap prose and remembered numbers
are never baseline authority. The loop remains plan -> implement -> measure -> adopt/reject ->
commit -> PR -> merge until VICTORY or EXHAUSTION.

### Phase 0 — inheritance audit (mandatory, one iteration)

1. Start from a fresh clone and reproduce every v1.0 headline number on the then-current hardware
   and firmware. A result outside one combined standard error becomes the campaign's first bug.
2. If v1.0 terminates in EXHAUSTION, import its ranked "next experiment" list. Close or explicitly
   re-park every item before opening frontier work.
3. Record the release tag, exact SHAs, image payloads, firmware, harness plan identities, raw samples,
   and derived baseline tuple in this file before measuring any frontier candidate.

Inherited invariants remain unchanged: I1 numerical integrity (byte-exact within one numerics config;
logprob parity plus GSM8K across kernel changes), usable pool never below v1.0, open-ended n=32
one-variable same-session A/B, no gate edits alongside performance claims, explicit PENDING-HW
status, and known-fatal settings remaining dead.

### Tier 1 — frontier baseline (all required)

- **F1 single stream:** >=25% open-ended n=32 tok/s over the reproduced v1.0 champion.
- **F2 prefill:** publish TTFT p50/p99 at 1K, 32K, 256K, and 1M; reduce p99 TTFT@32K by >=30%.
- **F3 concurrency:** improve C8 aggregate by >=30% over v1.0's curve, keep C16 stable, and publish
  p99 inter-token latency.
- **F4 quality:** every adoption retains NIAH@1M, GSM8K >=94.83%, and the tool-call wall.
- **F5 regression wall:** expose all inherited v1.0 gates as one documented per-adoption command;
  no frontier change may land without its complete result.

### Tier 2 — frontier stretch (any two)

- **X1 context 2M:** audit RoPE and draft caps, prove pool accounting, and pass NIAH@2M at three
  depths. The claim requires a declared 2M recipe on desktop hardware, not projected capacity.
- **X2 KV beyond FP4:** exploit Inkling's 35/42 SWA-512 layers by compressing or evicting only the
  seven global-layer caches (H2O/SnapKV-class scoring, or 2–3 bit with outlier channels). Target
  >=1.5x effective pool or 2M context while retaining NIAH@1M and GSM8K within 0.5 point.
- **X3 speculation architecture:** price and test tree/top-k>1 verification, cheap STS-driven dynamic
  gamma without the rejected cap-accept layout, or weekly-arm online draft LoRA distillation. Target
  +0.5 acceptance over v1.0 on open-ended traffic.
- **X4 fused-step engineering:** use the v1.0 cost map to fuse its largest eligible item — native FP4
  KV dequant in the attention load, draft forward in the decode graph, sconv commit with verify, or
  asynchronous TP all-reduce with MoE compute. One end-to-end proven fusion is a win.
- **X5 scale-out 4x:** with four Sparks, test DP2 routing across two TP2 pairs or prefill/decode
  disaggregation. Require >=3.4x single-pair aggregate at C16 and no worse per-request latency than
  one pair at C8. Otherwise record **PENDING-HW** without proxy claims.

### Tier 3 — moonshot and publication

Compound X1+X2 into a verified 2M-context desktop stack sustaining >=100 tok/s aggregate at C8, then
release SparkFlash-X v2.0 with a reproducibility paper: methods, all null results, energy curve,
v1.0 comparison, and upstream-official comparison cells. The paper and raw reproduction package are
part of the deliverable, not follow-up work.

### Frontier operating rules

- **N1 price first:** profile every lever and record an Amdahl ceiling before implementation. Skip
  any lever whose predicted end-to-end ceiling is below 3%.
- **N2 reference before serve:** every custom kernel gets bitwise/tolerance tests against a reference
  implementation before its first server launch, following `tests_verify_nvfp4.py`.
- **N3 gate quality-changing numerics:** eviction, lossy compression, or stochastic/tree verification
  must add a dedicated quality check to F5 in the same PR.
- **N4 upstream first:** file every adopted generally applicable kernel or fix upstream within one
  iteration; enumerate any private patch as technical debt.

SparkFlash-X reaches **EXHAUSTION** after six consecutive iterations with no adoption, no newly
localized wall, and no new ceiling-pricing insight. Its terminal report must quantify the remaining
distance to every target and rank the handoff backlog.

## Active north-star loop — control1/control2

The north-star loop resumed from current `main` on 2026-08-03. Completed rows are immutable:
E1 retained the single-HCA transport, E2 was killed as not applicable, and E3 promoted DSpark
block 5. E4 re-baselined that champion in the same session at **26.57 +/- 0.33 tok/s** and
**2.142 +/- 0.026 accept**, with byte-exact T4 before and after, then rejected native width-1 MTP
at its predeclared boot-dead gate. E8 retained NCCL autotuning after three forced-protocol arms
produced no adoptable gain, then accepted continuous decode steps 2 at +0.645 tok/s with acceptance
and latency improvements. The exact-default all-task adoption gate passed. E8 then retained 8 KV
splits after both alternatives lost. E5's first session localized manifest wall #25; its clean-root
restart cleared that wall but rejected cache adoption as too small. E6 is on HOLD because its
mandatory read-only preflight found `earlyoom` absent on both nodes. E7 passed the pinned-donor and
relative-bias numerical probes on both SM121 controls, localized wall #26 at the SM100 sheared-bias
seam, cleared it with Inkling's guarded score-mod path, then enabled DSpark block 5 as the sole
factor. Both target and draft ran FA4, all verify graph tiers captured, and T4 passed twice.

North-star distance on the adopted E8 steps-2 champion:

- **N1:** 5.77 tok/s below 32 tok/s on the 26.225 +/- 0.323 all-task-gate open-ended n=32 result.
- **N2:** 0.662 accept below 2.8 on the 2.138 +/- 0.026 open-ended result; no finetuned draft
  exists yet. The pooled/task-class values are not substitutes for the N2 gate.
- **N3:** FA4 BF16/page-128 with DSpark block 5 now passes full target/draft graph setup and T4 on
  sm_121; depth/GSM8K/tool quality and same-session performance gates are still open.
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
| E6 mem-fraction under C8-C16 | **HOLD — EARLYOOM ABSENT ON BOTH NODES** | Mandatory read-only preflight failed before any arm. No host service was installed or changed. Resume only after separately authorized setup. |
| E7 FA4 paged-KV port | **IN PROGRESS — DSPARK T4 PASS** | Both controls: base + relative-bias probes 14/14. Target and DSpark draft use FA4; block 5/gamma 5, all 12 target verify graph tiers, draft graph fold, runtime contract, and T4 x2 pass. Separate BF16 dev lane only; quality/performance next. |
| Q1 chat-templated harness | **LIVE-PROVEN** | Used for E1 and E3 exact n=32 serving measurements with plan-identity checks and T4. |
| Q2 depth quality | **READY — NOT RUN** | `docs/QUALITY-GATES-Q2-Q3.md`; token-measured NIAH 512K/1M at 3 depths plus full 1,319-item GSM8K ≥94.83% |
| Q3 tool-call regression | **READY — NOT RUN** | `docs/QUALITY-GATES-Q2-Q3.md`; 4 tools ×4 reps ×2 turns, structured args and zero parser-token leaks |
| Q4 concurrency curves | **READY — NOT RUN** | Authoritative chat-templated n=32 at C1/2/4/8/16. The pinned Mia-shaped 512-token diagnostic adds C1/2/3/4/6/8 at n=48 per level, replicated and T4-bracketed; targets C1 >=33.9 and C8 aggregate >=74.9 without changing N1. See `docs/MIAAI-BENCHMARK-CROSSWALK.md`. |
| Q5 no-GPU CI | **COMPLETE** | Python compile/tests, local Markdown links, launch dry-run, and shell syntax pass. |

Exhaustion counter: **0 / 4**. E7's DSpark-on-FA4 graph and T4 proof is a new engineering win and
resets the counter. E4 likewise did not count because it localized a new boot-dead wall.

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
