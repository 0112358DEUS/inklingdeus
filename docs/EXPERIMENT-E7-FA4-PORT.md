# E7 — SM120/121 FA4 paged-KV port

Status: **SCOPED — ENGINEERING REQUIRED, NOT RUNNABLE**. No FA4 code has been added to the image,
and no command has been run on `control1` or `control2`. This item must not be labeled
Unlike E1–E6, there is not yet an implementation to measure; E7 remains engineering scope rather
than a runnable experiment.

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

Until steps 1–5 exist and the GPU numerical gate passes, E7 is an engineering project—not a flag
to add to `EXTRA_ARGS` and not a reason to touch the controls.
