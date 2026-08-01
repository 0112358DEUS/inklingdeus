# E2 — dense FP4 GEMM backend

Status: **COMPLETE — NOT APPLICABLE; killed at the numerical gate** on 2026-08-02.

## Live result

The standalone dense-GEMM gate rejected the baseline on GB10 before either serving arm launched:

```text
BackendSupportedError: mm_fp4 does not support backend 'trtllm' with capability 121
```

The marlin path independently executed bitwise-repeatably with finite outputs at M=1, 2, 8, and
32. A checkpoint audit then showed why the full Inkling serve can still pass T4 with the
`flashinfer_trtllm` flag present: `hf_quant_config.json` excludes all 42 attention modules and all
40 shared-expert modules from NVFP4, while the checkpoint's 156 scale tensors belong only to routed
experts. Routed experts use `--moe-runner-backend marlin`; this checkpoint has no dense NVFP4 layer
for `--fp4-gemm-backend` to control.

The synthetic dense layer is therefore useful as an architecture capability test but not as an A/B
factor for this checkpoint. Per the predeclared kill criterion, no serving A/B was run and no
performance claim is made. The champion flags remain unchanged. The complete per-shape gate record
is `artifacts/e2-fp4-gemm/dense-gemm-numerics.txt`, SHA-256
`8ac08434810c18613857663bf11d844e9a2738e1b8c0f93db78c033a88343769`.

## Hypothesis and gates

- **Hypothesis:** `--fp4-gemm-backend marlin`, recommended by the official Spark recipe, is faster
  or more numerically robust on GB10/sm_121a than the current `flashinfer_trtllm` dense path.
- **Expected effect:** +0.5 to +2 tok/s if dense projections are a meaningful decode bottleneck.
- **Only changed serving variable:** `FP4GEMM=flashinfer_trtllm` versus `FP4GEMM=marlin`.
- **Kill criteria:** the standalone dense-GEMM numerical gate fails; either arm fails byte-exact T4
  before or after measurement; candidate throughput loses at least one combined SE; or the n=32
  plans differ.
- **Acceptance:** numerical gate and all four T4 probes pass, the candidate gains at least
  0.5 tok/s, and the arms' 1-SE throughput bars do not overlap.

## Ready-to-run sequence

Run from the head Spark after the same commit, image, and model paths are present on both nodes:

```bash
export WORKER_SSH=control2@<control2-host>
export WORKER_REPO=/home/control2/code/inklingdeus
export MASTER_IP=<control1-primary-roce-ip>
export IF=<control1-primary-roce-netdev>
export HCA=<the-same-single-or-dual-HCA-list-for-both-arms>
export MODELS=<same-absolute-model-path-on-both-nodes>

./scripts/run-e2-fp4-gemm-ab.sh
```

`benchmarks/tests_compare_fp4_gemm.py` first constructs identical serialized NVFP4 dense layers,
runs the backend-specific repacks, checks that each backend is bitwise repeatable, and bounds the
cross-backend numerical delta over decode and small-batch shapes. Only then does the runner launch
the target model for the same-session chat-templated n=32 A/B.
