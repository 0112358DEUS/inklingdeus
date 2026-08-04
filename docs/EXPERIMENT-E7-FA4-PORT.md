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

### Stage 5A pool integrity result — dual-control pass

Exact test commit `71c164532b71aa37ad9a2cf5550d46b6bc53b31d` ran against the unchanged
SparkFlash image payload on both controls. The GPU fixture compared every destination row against
SGLang's reference FP4 quantizer and passed ordinary target writes, prefix-valid DSpark injection,
SWA full/local routing, direct radix moves, and hybrid-SWA radix moves. In every case K/V payload
and K/V scale bytes moved together; uncommitted prefix-valid destinations remained zero.

The allocation fixture then created a 262,144-token pool, whose logical BF16 K/V size is
537,133,056 bytes. Its packed payload plus scales occupied 151,068,672 bytes, exactly 0.28125 of
BF16 (3.5556× capacity). After a warm call, an FA4 attention call over the large raw pool added
only 4,096 peak allocated bytes on each control, versus the predeclared 67,108,864-byte ceiling.
The raw accessors returned uint8 storage and source inspection confirmed that they contain no
whole-pool `batched_dequantize` call.

Raw dual-control logs, identity, machine-checked contract, and decision are in
`artifacts/e7-fa4-fp4-pool-gate-71c1645/`. Stage 5A has now cleared its primitive, writer,
allocation, capacity, spec-off serving, DSpark serving, and T4 gates. Quality, 1M-context behavior,
performance, energy, soak, and upstream readiness remain; the champion is still unchanged.

### Stage 5A DSpark block sweep — block 5 selected

Exact runner commit `f79ed99f6c32b4c0c6705124df01095d8a7fe9c3` reused the unchanged
SparkFlash image and ran blocks 7, 5, and 6 in one contiguous measurement session. Every arm
proved the locked FA4/page-128/FP4/score-mod runtime on both ranks, retained at least 1,256,984
full tokens, initialized and graph-folded its declared DSpark width, completed exactly 32
chat-templated open-ended samples, and passed T4 before and after measurement.

| Block | Full tokens | Open-ended tok/s | Accept length | T4 pre/post |
|---:|---:|---:|---:|---:|
| 7 | 1,349,760 | 24.258 +/- 0.286 | 2.183 +/- 0.023 | PASS/PASS |
| 5 | 1,371,264 | 25.267 +/- 0.320 | 2.115 +/- 0.025 | PASS/PASS |
| 6 | 1,276,800 | 24.824 +/- 0.287 | 2.163 +/- 0.023 | PASS/PASS |

Block 5 improved throughput over block 7 by 1.009 tok/s, or 2.35 combined standard errors; its
lower 1-SE bound (24.946) also cleared block 7's upper bound (24.544). Block 6 gained 0.566 tok/s
but its error bars overlapped, so it was rejected by the frozen selector. Block 5 is the measured
Stage 5A candidate. Its 25.267 tok/s result is a candidate-selection result, not a moonshot speed
pass: it remains below the 40 tok/s lower-bound goal and below the separately measured champion
until a same-session A/B proves otherwise.

Raw chat samples, positional acceptance histograms, all six T4 records, capacity/runtime contracts,
container inspections, rank logs, and the machine decision are in
`artifacts/e7-fa4-fp4-block-sweep-f79ed99/`.

### Stage 5A.1 pre-registration — block-16 scale-hoisted reader

- **Profiled mechanism:** the correctness-first reader assigns one packed byte to a thread at a
  time. Because eight consecutive bytes share one UE8M0 scale, it reloads that byte and evaluates
  `exp2(scale - 127)` eight times per 16-element block. At a 128x128 K or V tile this is 8,192
  scale loads/exp2 evaluations instead of the format-minimum 1,024.
- **One implementation factor:** change loop ownership from one packed byte to one complete
  block-16 group. Each participating thread loads one scale, expands it once, then decodes the
  group's eight packed bytes with the identical nibble-to-E2M1 algebra into the same BF16 shared
  tile. Cache bytes, scale format, page lookup, output layout, MMA, softmax, score-mod, model,
  DSpark block 5, graph sizes, and all launch flags stay unchanged.
- **Expected effect:** remove seven eighths of scale loads and special-function `exp2` work from
  both K and V tile loads. This should improve decode throughput without changing storage,
  capacity, or numerical error. No magnitude is claimed before measurement.
- **Primitive gate:** both controls must again pass all 14 full/SWA page-boundary cases, with
  byte-identical payload/scale inputs and max absolute error no worse than the 0.05 contract.
- **Serving gate:** the separately tagged image must preserve the >=1,256,984-token pool, locked
  runtime contract, graph capture, and two exact T4 responses under block 5.
- **Adoption gate:** a same-session scalar-versus-scale-hoisted open-ended n=32 A/B, T4-bracketed,
  must show the optimized lower 1-SE throughput bound above the scalar upper bound. Any numerical
  mismatch, boot failure, T4 mismatch, or overlapping/worse throughput rejects the change.
- **Cost bound:** one image/primitive build, one serving correctness launch, and one two-arm A/B.

### Stage 5A.1 result — rejected, scalar reader retained

Exact kernel commit `b58400609cd3d5e62ffb4b107be5e33629456e43` passed 14/14 primitive cases
on both controls with byte-identical outputs and unchanged worst max absolute error
`0.0019738078117370605`. Its independently baked images shared payload fingerprint
`53ff159c7ca2ea9a9dbf434335a84d599ecfa8ca7f6a461d5e6308ed9d4f9ca9`. The block-5
serving gate retained 1,327,616 full tokens, captured target and draft graphs, reached health, and
passed T4 twice.

The frozen same-session A/B at runner commit `1231e49652c3a1a8fc504f079231f43cc14315bb`
then measured:

| Reader | Open-ended tok/s | Accept length | Mean request latency |
|---|---:|---:|---:|
| scalar | 25.652 +/- 0.331 | 2.134 +/- 0.026 | 6.270 +/- 0.083 s |
| scale-hoisted | 25.973 +/- 0.305 | 2.094 +/- 0.024 | 6.187 +/- 0.073 s |

The `+0.321 tok/s` gain missed the 0.5 floor and the 1-SE bars overlapped. Acceptance fell by
0.040 versus a 0.036 combined SE, tripping the frozen no-regression guard; latency improved only
inside overlapping error bars. Both arms retained capacity and passed T4 before and after. The
optimization is therefore rejected, not quoted as a speedup, and the source tree restores the
byte-identical scalar reader from `dc43db1`/`9f62de4`.

Raw correctness evidence is in `artifacts/e7-fa4-fp4-scale-hoist-correctness-b584006/`; raw A/B
samples, histograms, contracts, inspections, logs, and the fail-closed decision are in
`artifacts/e7-scale-hoist-ab-1231e49/`.

### Stage 5A.2 pre-registration — native width-1 MTP on FA4

- **Hypothesis:** E4's native-MTP wall #24 was specific to Triton's FP4 target-verify parser path.
  The now-proven FA4 FP4 target and draft route should compile the two-token EAGLE verify graph,
  allowing Inkling's own MTP head to replace the separate 0.9B DSpark model.
- **One coherent factor:** relative to the measured Stage 5A block-5 candidate, change speculation
  from external DSpark block 5 to native EAGLE width 1 (`steps=1`, `topk=1`, two draft tokens,
  multi-layer EAGLE, rejection sampling). Keep the scalar FA4 reader, FP4 format, page 128,
  score-mod bias, graphs, model, MoE/dense backends, memory fraction, transport, and 64K context
  unchanged.
- **Preflight:** `mtp.safetensors` must exist and hash identically on both controls; repo and image
  payloads must match; the runtime command must contain exactly the EAGLE path and no DSpark model
  or flags.
- **Correctness gate:** the target and native draft load, allocate at least 1,256,984 usable full
  tokens, capture all target and draft graph tiers with FA4, reach health, and pass two exact T4
  requests. The logs must prove no external draft model allocation.
- **Kill criterion:** any MTP hash mismatch, memory shortfall, Triton fallback, mixed speculative
  path, graph failure, health timeout, or T4 mismatch rejects the rung without changing memory,
  graph, page, or context settings.
- **Promotion gate:** only after correctness passes, run a same-session block-5 DSpark versus MTP
  n=32 A/B with T4 brackets, acceptance and latency no-regression guards, and power telemetry. MTP
  must be equal or faster while eliminating the external draft allocation; otherwise DSpark stays.
- **Cost bound:** one candidate-only correctness launch, then at most one two-arm adoption A/B.

### Stage 5A.2 result — FA4 clears wall #24, capacity gate rejects rung

Exact runner commit `18ff162a74fd745fbc912f4e98f490f8eb58cd62` and the scalar SparkFlash
image passed repo/image reproducibility. `mtp.safetensors` matched across controls at SHA-256
`d286dd21cb982a0052d24ee0077ec6fc38f5a766dc6953e6c4b85c6473bfa7b3`. The target loaded,
then the native `InklingForConditionalGenerationMTP` loaded 10 shards using 1.77 GB on rank 0;
no `DSparkDraftModel` was loaded and the runtime contract proved EAGLE width 1 on FA4/page-128/FP4.

Unlike E4's Triton run, FA4 compiled all 12 two-token target-verify graph tiers in 182.05 seconds
and the server reached health with 14.36 GB available after capture. This is direct evidence that
FA4 removes wall #24's Triton parser failure.

The same immutable run allocated only `full_layer_tokens=1253248`, 3,736 tokens (0.297%) below the
expanded 1,256,984-token gate. The fail-closed runner therefore rejected the rung immediately
after health and before T4 or performance measurement. No MTP serving-correctness or speed claim
is made. Raw identities, weight hash, runtime inspections, both rank logs, and the failed capacity
contract are in `artifacts/e7-fa4-fp4-mtp-width1-18ff162/`.

### Stage 5A.3 pre-registration — minimal MTP capacity rescue

- **Hypothesis:** the native-MTP rung missed the capacity gate by only 3,736 tokens, approximately
  23-30 MB across its target/SWA/draft FP4 pool geometry. Raising the isolated candidate's static
  memory fraction from 0.850 to 0.851 adds roughly 128 MB of budget and should clear the target
  without reducing graph coverage or request concurrency.
- **Only changed factor:** `--mem-fraction-static 0.851`. Native EAGLE width 1, scalar FA4 reader,
  FP4 format, page 128, score-mod, all 12 graph tiers through batch 16, max requests 16, context
  64K, transport, and model backends remain byte-for-byte identical to Stage 5A.2.
- **Safety/correctness gate:** usable full tokens >=1,256,984; all graphs capture; post-capture
  available GPU memory remains at least 13 GB on rank 0; runtime proves no external draft; health
  and two exact T4 requests pass. Both containers are stopped afterward.
- **Kill criterion:** any capacity miss, post-capture headroom below 13 GB, graph/health/T4 failure,
  or runtime drift rejects the rescue. Do not try 0.852 or remove graph tiers inside this rung.
- **No adoption claim:** a pass permits one power-instrumented, same-session DSpark-versus-MTP A/B;
  it does not change the champion's 0.85 default.

### Stage 5A.3 result — rejected, capacity is not monotonic at this margin

Exact runner commit `94d1ef6893f067b5caecbc91d4b4dc573c6f14f2` changed only the MTP
candidate's static fraction to 0.851 and machine-checked that value on both ranks. Repo, image, and
MTP-weight identities matched; native MTP again loaded, captured the two-token FA4 target-verify
graph, reached health, and proved no external draft allocation.

The pool nevertheless fell to 1,163,904 full tokens, 93,080 below the gate and 89,344 below the
prior 0.850 run. Model/MTP available-memory snapshots were comparable, so the small fraction
increase was dominated by runtime profiling/allocation variability rather than producing a
monotonic 3,736-token rescue. The runner rejected before T4 as required. This invalidates the
minimal-memory hypothesis; no higher fraction is inferred or attempted from this result.

Raw evidence is in `artifacts/e7-fa4-fp4-mtp-rescue-94d1ef6/`. Native MTP remains a valuable FA4
graph breakthrough but is not the Stage 5A candidate; DSpark block 5 remains selected for quality
and further performance work.

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

### Stage 5A.4 quality attempt 1 — runtime passes, tokenizer API seam blocks NIAH

Exact runner commit `e95e839` and scalar SparkFlash image payload
`8b88229301b6b5817cec177c7570d8b1ee24a88640c24bb2d75aeb09f4b11a7f` matched across both
controls. The DSpark block-5 server allocated 1,280,768 full-layer tokens, 23,784 above the gate;
captured every target and draft graph tier through batch 16; reached health at 1,048,576 context;
and passed the pre-quality byte-exact T4 gate.

The first NIAH calibration call then stopped before generation because `/v1/tokenize` returned
HTTP 500. The traceback ended in ORJSON with `Integer exceeds 64-bit range`. Source and model
inspection identified the sole out-of-range response field: the tokenizer's conventional
no-intrinsic-limit sentinel `1000000000000000019884624838656` was exposed as `max_model_len`.
This is an API serialization failure, not a NIAH answer or model-quality failure. Both containers
were stopped and the incomplete run was preserved in
`artifacts/e7-fa4-fp4-quality-e95e839/`.

The bounded retry changes only that serving metadata seam: `/v1/tokenize` reports SGLang's
resolved `model_config.context_len`. A new fail-closed preflight requires a consistent token list
and count plus `max_model_len=1048576` before T4 and NIAH. Kernel, FP4 representation, DSpark,
memory, graph, model, benchmark prompts, and quality thresholds remain unchanged.

The first retry preparation at `74b8da1` also stopped before launch because the repository
fingerprint included each worktree's site-specific `.git` pointer file. Both worktrees were clean
and their 152 runnable entries matched; only the embedded absolute metadata path differed. The
fingerprinter now excludes `.git` in both directory and file form, with a checkout-versus-worktree
regression test. This is an evidence-harness correction only; no serving process started.

At the next exact run, repository/image identity, 1,315,584-token capacity, all graph tiers, the
new tokenize contract, and T4 passed. The first 512K/10%-depth NIAH request measured 511,984
tokens and returned the exact secret after 1,886.8 seconds. That result exposed a second harness
weakness: NIAH retained all six results only in memory until the suite ended, so a later failure or
transport interruption could erase hours of evidence. The run was deliberately terminated during
case two and both containers were stopped. Because case one was not durably checkpointed, it is
reported as diagnostic evidence only and must be rerun.

The next exact retry adds atomic per-case checkpointing and explicit resume. Resume is accepted
only when schema, model, full plan, and completed case-order prefix match exactly. Prompts, token
calibration, depths, context targets, generation settings, correctness rule, model image, and all
serving flags are unchanged.

### Stage 5A.4 result — 512K/1M NIAH pass, duplicate tool calls stop quality

Exact commit `f8e5540ad1fd68fad30a011f86e3929e1e96aa8b`, runnable payload
`88ea26dc61016b5be7bb0059658d69b0d6a9a2cc813186274c6c54520d058639`, and full image payload
`3660ab042e4cbfb33f026d06c7b37a7657f4ea89f7411d52ea2d716f5930a2de` matched across controls.
The server allocated 1,325,312 full-layer tokens, 68,328 above the gate; captured all target/draft
tiers through C16; passed the live tokenizer contract; and passed T4 before quality.

NIAH then passed all six durable cases. The measured 512K prompts were 511,984/512,026/511,973
tokens at 10/50/90% depth and completed in 1,882.6/1,875.6/1,876.4 seconds. The measured 1M
prompts were 999,987/1,000,020/999,952 tokens and completed in 6,649.0/6,655.1/6,646.0 seconds.
Every answer contained its exact unique secret; the atomic checkpoint is `complete=true`, 6/6,
and `all_passed=true`. Post-NIAH T4 also passed.

The next gate failed 0/16 tool flows. Every response selected the correct forced tool and emitted
valid required arguments, but it emitted the same call twice with distinct generated call IDs.
The harness correctly rejected those as two external actions; it did not execute any tool. GSM8K
did not start, and both containers stopped. Raw evidence is in
`artifacts/e7-fa4-fp4-quality-f8e5540/`.

### Stage 5A.5 pre-registration — continuous-decode tool-stop isolation

- **Hypothesis:** E8's accepted `--num-continuous-decode-steps 2` text-speed optimization crosses
  Inkling's structured `END_MESSAGE` boundary and permits a second call. E8 tested T4 and text
  workloads, not structured tools.
- **Only changed factor:** continuous decode steps 2 to 1. Image, FA4 reader, native FP4 KV,
  DSpark block 5, page 128, 1M context, memory fraction, graph tiers, max requests, MoE/GEMM,
  transport, tool prompts, forced-choice objects, and four repetitions remain unchanged.
- **Gate:** exact identities, capacity >=1,256,984, all graphs, pre/post T4, and 16/16 complete tool
  flows including post-tool turns. Full raw OpenAI responses are retained in the candidate artifact.
- **Kill/adoption rule:** any duplicate, wrong call, malformed arguments, parser-token leak, empty
  post-tool answer, extra post-tool call, or T4 failure rejects CDS1. A pass identifies the stop
  boundary but does not silently change the champion; its known text throughput cost must be
  measured or the compiled scheduler stop logic fixed before final adoption.
