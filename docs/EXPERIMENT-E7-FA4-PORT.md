# E7 — SM120/121 FA4 paged-KV port

Status: **IN PROGRESS — DSPARK-ON-FA4 T4 PASS**. The pinned donor and dedicated relative-bias
probes pass 14/14 on both controls. Inkling's guarded FA4 score-mod path clears wall #26, and the
next one-factor stage loads DSpark block 5 with both target and draft on FA4, captures all target
verify graph tiers, and passes T4 twice. Quality and performance gates remain. No champion image
tag or default was changed.

## Pinned donor and provenance

- Integration donor: `eugr/spark-vllm-docker` commit
  [`552b618c6e0b092d45a7290916547a7ccc1a078d`](https://github.com/eugr/spark-vllm-docker/tree/552b618c6e0b092d45a7290916547a7ccc1a078d/mods/inkling-sm12-paged-kv).
- Vendored kernel provenance recorded by that donor:
  `SecondNatureComputing/flash-attn-4-sm120` commit
  `60117041e10fcc6f19882afd274318c755a5ef6e`, hosted on
  [Hugging Face](https://huggingface.co/SecondNatureComputing/flash-attn-4-sm120), plus the two
  CUTLASS DSL 4.6 mechanical migrations from `vllm-project/tml-fa4@b206834`.
- The donor retains BSD-3-Clause license and authors files. Any SGLang port must retain the same
  provenance and a byte-hash manifest for every vendored file.

The donor is a vLLM Inkling-only adapter, not an SGLang backend. It clamps `num_splits=1`, carries
vLLM's preallocated `out=` contract, and patches only compute-capability-major 12 dispatch.

## Completed preflight

- The champion image already contains the required CUDA bindings, CUTLASS DSL, Einops, Quack, and
  TVM-FFI dependencies. Importing the unchanged donor inside that image passed on SM121 with CUDA
  13.0; see `artifacts/e7-fa4-preflight-20260803/dependency-import.txt`.
- `benchmarks/fa4_sm121_paged_probe.py` uses Inkling's real TP2 shape (`Hq=16`, `Hkv=4`, `D=128`)
  with BF16 page size 128. Full attention and SWA-512 passed at lengths
  1/127/128/129/511/512/513. All outputs were finite and the worst max absolute error was
  0.001813 against the torch reference, below the predeclared 0.05 limit; see
  `artifacts/e7-fa4-preflight-20260803/paged-bf16-numerics.jsonl`.
- `scripts/verify-fa4-vendor.py` fail-closes on any file-set or byte drift. The baker refuses the
  champion tag and creates `local/sglang-inkling:fa4-sm121-dev` instead.

This proof is deliberately limited: the donor covers BF16 paged KV and retains the score-mod seam,
but it does not carry SGLang's current MXFP8 helpers or Inkling relative-bias extensions. It is not
a drop-in replacement for the FP4 champion. The next gate is a two-node BF16, page-128, spec-OFF T4
run from the separate development image; DSpark and FP4 remain later gates.

## Stage 2 pre-registration — spec-off serving seam

- **Hypothesis:** the pinned SM120 donor can replace only the FA4 CuTe package in the current
  SGLang image and provide a two-node SM121 Inkling serving lane with BF16 page-128 KV while
  preserving full-attention and SWA-512 semantics.
- **Expected effect:** correctness only; no throughput claim. The server should reach health and
  pass two byte-exact T4 probes. The image overlay and GPU probes should take under two minutes;
  the two-node boot should take 6–10 minutes.
- **One coherent factor:** relative to the BF16 fallback, attention backend and its required page
  layout move together from triton/page-1 to FA4/page-128. Speculation stays off, graphs stay on,
  context stays 64K, and MoE/dense-FP4/network/default decode settings stay fixed.
- **Kill criterion:** any donor/hash/image mismatch, either 14-case GPU probe failing, boot death,
  mixed runtime flags, or one-byte T4 mismatch kills this stage before DSpark or performance work.
  A boot-dead result becomes a new wall; no alternate graph, memory, or kernel flag may be slipped
  into the same iteration.
- **Reproducible command:** run scripts/run-e7-fa4-specoff.sh on control1 with its existing
  passwordless private-link SSH to control2 and the site values supplied as environment knobs.

## Stage 2 result — rejected, new wall #26

The exact runner commit was `164d84bb7116ac26f726d0815b13c3fd029aadf4`. Repository payload,
champion payload, and full FA4 payload matched between controls. Both committed GPU probes passed
14/14 again. The target then loaded on both ranks, allocated BF16 page-128 pools
(`full=429184`, `swa=42880` tokens per rank), and began full decode-graph capture.

Capture died before health with:

```text
TypeError: flash_attn_varlen_func() got an unexpected keyword argument 'rel_bias'
```

The full SGLang wrapper always forwards its Inkling relative-bias extension, but the pinned donor
interface lacks that keyword. The standalone kernel probe exercised full attention and SWA windows
but did not traverse this wrapper seam. The predeclared kill criterion therefore rejects the stage;
both containers were stopped and no T4 was attempted. This is the first dead E7 stage after the
standalone numerical win, not two consecutive dead stages. The next E7 iteration must implement and
numerically gate a guarded relative-bias adapter before another serve; disabling graphs or dropping
the bias would be a different factor and is not a retry of this result.

## Stage 3 pre-registration — guarded score-mod relative bias

- **Hypothesis:** Inkling's existing FA4-only score-mod path is the SM121 adapter for wall #26. It
  adds the same relative logits through the donor's supported `score_mod` and `aux_tensors`
  interface, while bypassing only the dedicated SM100 shearing optimization.
- **Expected effect:** correctness only; no throughput claim. A real-shape relative-bias probe must
  pass 14/14 full/SWA page-boundary cases on both controls before the server is started. The
  unchanged development image should then reach health and pass two byte-exact T4 probes.
- **One variable:** set `SGLANG_OPT_USE_INKLING_SHEARED_BIAS=0` inside the separate E7 launch.
  Image payload, donor, FA4/page-128/BF16, graphs, spec-off, context, network, MoE, and dense-FP4
  settings stay identical to stage 2. The champion launch leaves this environment variable unset.
- **Reference gate:** `benchmarks/fa4_sm121_rel_bias_probe.py` uses the real per-rank
  `Hq=16, Hkv=4, D=128` shape, relative extent 1024, page/SWA boundaries, and the exact Inkling
  score-mod callable. Require finite output and max absolute error at most 0.05 versus torch.
- **Kill criterion:** any reference failure, payload/contract drift, boot death, or one-byte T4
  mismatch rejects this stage. Because that would be the second dead E7 serving stage in a row,
  E7 must then be parked under the goal's two-dead-stage rule.
- **Cost:** about one minute for four GPU probes plus one 6–10 minute two-node boot.

## Stage 3 result — pass, wall #26 cleared for spec-off

The exact runner commit was `a9d4857ccf7ed586429f68bf0563a05e06a2a120`. Both controls matched
repository, champion-image, and FA4-image payloads. The base paged-KV probe passed 14/14 on each,
and the new relative-bias probe passed 14/14 on each with worst max absolute error
0.0023084282875061035 against the torch reference (limit 0.05).

With only `SGLANG_OPT_USE_INKLING_SHEARED_BIAS=0` added to the separate development launch, both
ranks loaded, allocated BF16 page-128 pools, and captured all 12 full decode graph sizes. Docker
inspection proved FA4/page-128/BF16/spec-off/score-mod on both ranks. Two consecutive T4 probes then
matched byte-for-byte. The runner stopped both containers afterward.

This is a serving-correctness stage, not a throughput adoption. Wall #26 is resolved without
dropping relative bias or changing the donor: the existing Inkling score-mod callable carries the
same relative logits through the donor's supported auxiliary-tensor interface. Next is a separately
pre-registered DSpark-on-FA4 T4 gate; the champion remains triton/page-1/FP4 KV.

## Stage 4 pre-registration — DSpark on FA4

- **Hypothesis:** the already adopted DSpark block-5 draft path can write/inject BF16 page-128 KV
  while the target uses FA4 score-mod attention, without changing T4 output.
- **Single factor:** enable DSpark block 5. The successful stage-3 image, FA4 target, BF16 KV,
  page 128, score-mod relative bias, marlin MoE, `flashinfer_trtllm` FP4 GEMM, graph tiers,
  max requests, continuous decode steps 2, transport, model paths, and site knobs stay fixed.
- **Preflight:** both controls must again match repo, FA4-image, and champion-image payloads; base
  paged-KV and score-mod relative-bias probes must each pass 14/14 on both controls.
- **Runtime proof:** Docker inspection must show the locked common contract plus exactly one DSpark
  block-5 flag set on both ranks. The head log must prove gamma 5 initialized and its greedy
  proposal folded into the draft CUDA graph.
- **Pass gate:** the full two-node server reaches health, the runtime proof passes, and two
  consecutive byte-exact T4 requests match the frozen expected output.
- **Kill criterion:** any payload/probe drift, boot death, missing DSpark graph proof, or one-byte
  T4 mismatch rejects the stage. Do not change the draft backend, block, graphs, page size, KV
  dtype, memory fraction, or bias path inside this run.
- **No claim yet:** a pass advances E7 to quality/performance measurement; it does not establish N3
  or permit a champion/default change.

## Stage 4 result — pass, DSpark target and draft on FA4

The exact runner commit was `14b8c31457f7f43ca3b9d57c5e26c6779c579b99`. Both controls matched
the repository, champion-image, and FA4-image payloads. Base paged-KV and score-mod relative-bias
probes again passed 14/14 on each control before serving started.

The target and DSpark draft loaded with BF16 page-128 KV. The head log proves gamma 5 initialization
with `attention_backend=fa4`, followed by SGLang's explicit `Overriding draft attention backend to
fa4`. The target full/SWA pools held 359,936/35,968 tokens and the draft allocated its separate
359,936-token BF16 pool. All 12 target verify graph tiers captured with six tokens per request, and
the DSpark greedy proposal folded into the draft CUDA graph.

Docker inspection on both ranks proved FA4/page-128/BF16/DSpark-block-5/score-mod with continuous
decode steps 2. Two consecutive T4 requests matched the frozen output byte-for-byte. The runner
then stopped both containers; the worker's final connection-reset traceback is the expected tail
after the intentional rank-0 shutdown, not a serving failure.

This clears E7's DSpark serving-correctness checkpoint. It is not N3 yet: the FA4 lane still needs
the locked depth/GSM8K/tool quality suite and a same-session, replicated triton-vs-FA4 performance
gate. The champion remains triton/page-1/FP4 KV.

## Stage 5 pre-registration — native page-128 FP4 KV

### Hardware and format decision

SM121 has two distinct Blackwell execution families that must not be conflated. It lacks the
SM100 `tcgen05`/TMEM attention path, but PTX 8.8 exposes warp-level block-scaled
`mma.sync.aligned` FP4 on the SM120 family. NVIDIA's current
[PTX feature table](https://docs.nvidia.com/cuda/archive/12.9.2/parallel-thread-execution/index.html)
lists the `.e2m1`, `.kind`, `.block_scale`, and `.scale_vec_size` warp-MMA features for
`sm_120f`. The installed CUTLASS DSL 4.6.0 on both development images contains
`warp.MmaMXF4Op` and `warp.MmaMXF4NVF4Op` for SM121a.

That does **not** make the current cache bytes a legal native-MMA operand. The frozen
`fp4_mx_block16` cache is packed E2M1 plus one UE8M0 scale per 16 elements. NVIDIA's
[warp-MMA programming guide](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/mma_docs/wmma_programming.html#block-scaled-mma)
defines:

- `MmaMXF4Op`: E2M1, UE8M0, scale-vector 32;
- `MmaMXF4NVF4Op`: E2M1, UE4M3, scale-vector 16.

Therefore a direct MMA over the current block-16/UE8M0 storage would apply the wrong scale
contract. Stage 5 preserves the proven cache format and separates capacity/correctness from a
later cache-format experiment.

### Stage 5A — fused exact block-16 reader

- **Hypothesis:** FA4 can consume the existing packed block-16 cache by decoding only the K/V
  tile selected by the page table into the existing BF16 shared-memory tiles. This removes stock
  SGLang's whole-pool BF16 materialization while leaving the already proven BF16 MMA, score-mod,
  softmax, and P×V numerics unchanged.
- **One coherent factor:** the FA4 development lane moves from BF16 page-128 storage to the
  existing `fp4_mx_block16` payload/scale contract plus its required fused paged reader. Model,
  MoE/dense-FP4 backends, DSpark block, score-mod, graph tiers, context, network, and champion
  defaults remain fixed. A new image tag must be used; `fa4-sm121-dev` and the champion tag are
  immutable inputs.
- **Implementation contract:** the SGLang pool returns raw packed K/V and separate scale buffers;
  payload and scale rows move together for ordinary writes, prefix-valid commits, radix moves,
  SWA translation, and DSpark injection. The FA4 interface accepts explicit `sfk`/`sfv` plus a
  fail-closed FP4-format marker. On SM121 paged attention, each selected page row is decoded
  on-the-fly into the kernel's K/V shared-memory tile. No per-layer or per-request full-pool BF16
  tensor may be allocated.
- **Expected effect:** storage cost falls from 2 bytes to 0.5625 bytes per KV element, a
  theoretical 3.5556× pool-capacity multiplier. From the stage-4 359,936-token full pool this is
  sufficient in principle for at least 1,279,772 tokens, above the 1,256,984-token moonshot gate.
  This first rung is a correctness/capacity gate; it makes no throughput claim until the reader is
  vectorized and profiled.
- **Primitive gate:** on both controls, independently seeded tensors must prove packed payload and
  scale identity against SGLang's reference quantizer, exact E2M1/UE8M0 element reconstruction,
  finite outputs, deterministic repeatability, and page/SWA boundary attention at
  1/127/128/129/511/512/513 tokens. Attention max absolute error must remain at most 0.05 versus a
  torch dequantized reference.
- **Writer gate:** ordinary target, target prefix-valid/radix move, SWA, and DSpark hidden-state
  injection fixtures must each prove that payload and scale bytes reach identical destination
  rows. Any payload-only move is a hard failure.
- **Allocation gate:** source inspection plus CUDA memory snapshots must prove that one attention
  call allocates no BF16 object proportional to total pool capacity. Only fixed-size per-CTA
  shared-memory tiles and existing output/split buffers are allowed.
- **Serving gate:** spec-off first, then unchanged DSpark block 5. Each must reach health and pass
  two consecutive byte-exact T4 probes. Pool logs must report at least 1,256,984 usable full tokens
  without OOM. A mismatch or whole-pool materialization rejects the stage immediately.
- **Cost bound:** one isolated primitive/image build and at most two serving launches per rung.
  Compile or numerical failure is documented before another implementation factor is introduced.

### Stage 5B — hardware-native block-scaled QK

Only after 5A passes may a separate image change the cache scale vector to block-32 UE8M0 and
quantize Q to the same legal `MmaMXF4Op` contract for Q×K. P×V stays on the exact fused V-dequant
path because the attention probabilities are not an E2M1 operand. This is a new numerical format,
not an optimization flag: it requires fresh payload/scale tests, page-boundary attention, T4,
NIAH, GSM8K, tool, and same-session performance gates. It is adopted only if it preserves every
quality gate and improves the lower 1-SE throughput bound; otherwise 5A remains the native
FP4-storage reference path.

### Stage 5A primitive result — pass on both controls

The correctness-first fused reader compiled and ran at exact commit
`c607b8465cb169ef55fa06ddd2b29980f4313e9d`. Independently baked images on the two controls had
different Docker layer IDs, as expected, but the same FA4 payload fingerprint
`9a7c390cf13732244c068e69241c0291021318c3a4d299523ed34e5a234048e4`.

Both controls passed all 14 full/SWA cases at lengths 1/127/128/129/511/512/513. Their complete
logs are byte-identical. The worst max absolute error against the torch reference over the
reference-dequantized payload was `0.0019738078117370605`, versus the predeclared `0.05` limit;
length 1 was exact. Each case also re-quantized K/V twice and required payload and scale bytes to
match before the FA4 call. Raw logs and image identities are in
`artifacts/e7-fa4-fp4-20260803/`.

This clears only the isolated reader primitive. Live SGLang backend plumbing, the four-writer
gate, allocation proof, usable-capacity log, serving health, and byte-exact T4 remain pending; no
champion or control default changed.

### Stage 5A serving attempts 1-2 — capacity clears, dispatch seams reject

The first isolated serving attempt failed before model load because SGLang's stock FP4/FA4
compatibility table rejected `fa4` as a decode backend (wall #27). The next exact image narrowed
that exception to `fp4_mx_block16` on SM120-family hardware with page size 128.

The second attempt cleared that route, loaded both ranks, and allocated
`full_layer_tokens=1445504` plus `swa_layer_tokens=144512`. The usable full-attention capacity is
188,520 tokens (14.998%) above the 1,256,984-token gate. Per-rank logs reported 2.71 GB for each
full K/V payload, 1.36 GB for each SWA K/V payload, 8.14 GB for the combined SWA pool, and
16.93 GB still available when target decode-graph capture began.

The first batch-16 graph then failed before health with
`flash_attn_with_kvcache() got an unexpected keyword argument 'kv_fp4'`. Runtime inspection tied
the callable to `sglang.kernels.ops.attention.flash_attention`, whose generic signature lacked the
marker even though its existing `ver == 4` branch calls the already extended FA4 wrapper. This is
wall #28: the next image adds a fail-closed, version-4-only generic dispatch seam and changes no
kernel, cache format, model, graph, network, or champion flag. The two failed launches exhaust the
original Stage 5A serving-attempt bound; this documented dispatch correction starts the next
bounded routing rung. Neither attempt reached health or T4, and both containers were stopped.

### Stage 5A routing rung result — spec-off serving pass

Exact commit `9f62de49ca6318a8008d03750e06eb8d24683751` added only the documented
version-4 dispatcher seam and an executable capacity assertion. Independently baked Control 1 and
Control 2 images had different Docker layer IDs but the same complete payload fingerprint
`8b88229301b6b5817cec177c7570d8b1ee24a88640c24bb2d75aeb09f4b11a7f`. Both FP4 paged
numerical probes passed 14/14 before serving.

The two-node target allocated `full_layer_tokens=1537152` and `swa_layer_tokens=153600`. The
full pool exceeded the 1,256,984-token gate by 280,168 tokens (22.29%). It captured all 12 locked
decode graph tiers in 179.88 seconds, reached health with 13.96 GB available, and the runtime
inspection proved FA4/page-128/`fp4_mx_block16`/score-mod/spec-off on both ranks. Two consecutive
T4 responses matched the frozen expected bytes and SHA-256 exactly. The runner then intentionally
stopped both containers.

Raw identities, independent numerical logs, container inspections, server logs, capacity/runtime
contracts, and both T4 records are in `artifacts/e7-fa4-fp4-specoff-9f62de4/`. This clears native
FP4 storage for spec-off serving correctness and the target-capacity moonshot gate. It is not yet
a DSpark, 1M NIAH, quality, throughput, or adoption result; the next gate reuses this exact image
and changes only speculation from off to DSpark block 5.

### Stage 5A DSpark result — serving correctness pass

The next runner reused exact commit `9f62de49ca6318a8008d03750e06eb8d24683751` and the identical
`8b88229301b6b5817cec177c7570d8b1ee24a88640c24bb2d75aeb09f4b11a7f` image payload.
The only serving change was enabling DSpark block 5. Both controls again passed the 14-case FP4
paged numerical probe before launch.

Target and draft loaded with native FP4 KV. The target retained `full_layer_tokens=1280896` and
`swa_layer_tokens=128000`, clearing the 1,256,984-token gate by 23,912 tokens while the draft also
allocated its separate 1,280,896-token FP4 pool. Runtime logs prove gamma 5, FA4 for target and
draft, all 12 six-token target verify graph tiers, all 12 five-token draft tiers, and the greedy
proposal folded into the draft CUDA graph. The locked container inspection passed on both ranks,
the server reached health, and two consecutive T4 responses matched byte-for-byte. The runner
then stopped both containers.

Raw evidence is in `artifacts/e7-fa4-fp4-dspark-b5-9f62de4/`. This clears the Stage 5A spec-off and
DSpark serving-correctness gates plus the usable-capacity threshold. Dedicated payload-plus-scale
writer fixtures and the no-pool-sized-BF16 allocation audit remain before quality/performance
promotion; no champion or default changed.

## Why this is not a config experiment

Four independent seams must be implemented before a launch is meaningful:

1. SGLang's current backend matrix says FA4 MHA requires page size 128 and does **not** support
   sliding-window attention, while Inkling mixes full attention, SWA-512, and sconv layers.
2. The champion uses page size 1 with a custom `fp4_mx_block16` pool. FA4's page-128 layout and
   scale contract cannot be assumed compatible with the existing E2M1 + UE8M0 block-16 storage.
3. Inkling has three KV writers, including DSpark's direct hidden-state injector. A new page-128
   pool must cover all writers; adapting only the attention backend reproduces wall 19.
4. The pinned image rejects FA4 on SM121. Dispatch, CuTe DSL dependencies, architecture guards,
   and CUDA-graph behavior all need a separate baked image. vLLM's PIECEWISE graph recommendation
   cannot be copied: piecewise and prefill graphs are known-fatal in this SGLang stack.

The upstream constraint source is SGLang's
[attention-backend matrix](https://github.com/sgl-project/sglang/blob/main/docs/advanced_features/attention_backend.md).

## Required implementation sequence

1. Vendor the pinned BSD source with license/authors and a SHA256 manifest; make import/compile
   preflight fail before changing SGLang dispatch.
2. Add an Inkling-specific SGLang FA4 adapter with explicit SM120/121 and `num_splits=1` guards.
3. Implement page-128 FP4 KV allocation, quantize/dequantize, page-table planning, and DSpark KV
   injection without changing the champion triton path.
4. Preserve per-layer semantics: full attention and SWA-512 must each have a reference-backed path;
   sconv layers must remain untouched.
5. Bake a separate image tag and extend `scripts/image-fingerprint.sh`; never overwrite the
   champion image tag during development.

## Gates before any throughput claim

- **Static/build:** pinned source hashes and license pass; SM121 import and CuTe compilation pass;
  no dispatch on other models/architectures; the existing triton image payload remains byte-exact.
- **GPU numerics:** for Inkling's real per-rank shapes, page-boundary lengths 1/127/128/129,
  SWA boundary 511/512/513, and long decode, compare FA4 with a torch/triton reference; require
  repeatability, finite outputs, packed FP4 data/scale identity, and declared tolerances before
  serving. Exercise all three KV writers.
- **Serving correctness:** spec OFF T4, DSpark ON T4, and T4 before/after every measured arm. Any
  fluent-but-different T4 output kills the port immediately.
- **Depth/tool quality:** rerun Q2 NIAH 512K/1M, full GSM8K, and Q3 post-tool regressions on the FA4
  image before it can replace the champion.
- **Performance:** same-session chat-templated n=32 triton-vs-FA4 A/B, one coherent backend factor,
  accepted only at +0.5 tok/s or better with non-overlapping 1-SE bars and no quality regression.

Until the serving, quality, and performance gates pass, E7 remains a development lane—not a flag
to add to the champion `EXTRA_ARGS` and not a reason to change either control's configuration.
