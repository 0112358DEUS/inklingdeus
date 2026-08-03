# E7 — SM120/121 FA4 paged-KV port

Status: **IN PROGRESS — SPEC-OFF SERVING REJECTED AT NEW WALL #26**. The pinned donor imports and
passes all 14 real-shape BF16 page-boundary cases on both controls, but the full two-node server dies
during decode-graph capture because SGLang forwards an unsupported `rel_bias` keyword. No T4 or
throughput claim exists. No champion image tag, launcher default, or host configuration was changed.

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
