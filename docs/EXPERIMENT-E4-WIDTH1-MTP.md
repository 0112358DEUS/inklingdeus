# E4 — DSpark block 5 vs native width-1 MTP

Status: **COMPLETE — REJECTED, boot-dead at the declared kill gate** on 2026-08-03.

## Live result

The same-session block-5 baseline completed exact chat-templated open-ended `n=32` measurement at
**26.57 +/- 0.33 tok/s** and **2.142 +/- 0.026 accept**. Its runtime-command contract passed and
T4 was byte-exact before and after measurement.

The width-1 native-MTP arm loaded the target and all ten MTP shards successfully. The MTP load used
1.84 GB on rank 0 and 2.55 GB on rank 1, leaving 24.47 GB and 23.74 GB available; fp4 KV allocation
then produced a 1,160,700-token full pool, still above the 1,048,576 declared context. At target
verify graph capture, with 18.00/18.01 GB reported available and `num_tokens_per_req=2`, Triton's
fp4 extend kernel failed in
`kv_quant_attention.py:_fwd_kernel_kv_quant` with `RuntimeError: error encountered during parsing`.
The scheduler terminated before HTTP readiness.

This is not the older full-width memory deficit: width 1 fit. It is a distinct native-MTP +
fp4-KV Triton compile wall (wall #24). Per the predeclared kill criterion, the candidate was
rejected without changing mem-fraction, graph coverage, KV dtype, or any champion default. No
candidate T4 or throughput claim exists.

Evidence is in `artifacts/e4-width1-mtp-20260803/`, including the exact baseline JSON, pre/post T4,
per-rank runtime records, candidate logs, MTP hash, and fail-closed boot record.

- baseline JSON: `41bdea2273d8872dbec2c3310fe661d1285d9b5431cb1f8f76c84a48ccd35d73`
- each baseline T4 record: `aac69468d03ab55a8da2d9f15e7939103b479a8e93d9b7839e12953299119de7`
- candidate boot-failure record: `2fb72cb28671bf1dc4e1abcef86116c05d865670d5ca9c484bd6b89725c33038`
- candidate head/worker logs: `8a632f26807dacb8345fd903c2d1f411f47bbc020416e2b28a32d17648da9846` /
  `eaf336f0ad2f2c325c44bc57c797b96751429583f9bdbff8dda4b1ee60c06b69`

## Hypothesis and gates

- **Hypothesis:** the target's native MTP head can avoid the external DSpark draft's weight and
  communication costs at width 1, while staying inside the two-Spark memory envelope.
- **One changed serving factor:** speculative implementation. The baseline is the accepted DSpark
  block from E3 (`CHAMPION_BLOCK`, default 5); the candidate is one coherent native-MTP
  configuration. HCA, target weights, image payload, dense
  backend, KV dtype, context, memory fraction, graphs, prompts, and tokens remain fixed.
- **Candidate flags:** EAGLE, one draft step, top-k 1, two draft tokens, multi-layer EAGLE, and
  rejection sampling. SGLang's official Inkling recipe is 8-1-9 and explicitly requires
  `--enable-multi-layer-eagle`; width 1 therefore keeps the same `steps + 1` verify-window relation.
- **Preflight:** exact repo SHA, runnable worktree payload, and patched image payload match;
  `mtp.safetensors` exists and hashes
  identically on both nodes; the live container commands prove that both ranks use exactly one
  speculative path; per-rank memory snapshots are retained.
- **Kill criteria:** missing/mismatched MTP weights, mixed DSpark/EAGLE flags, boot timeout or OOM,
  any T4 mismatch before or after an arm, an arm other than exact n=32, or a throughput loss of at
  least one combined SE. A boot failure is a rejection, not permission to change `MEMFRAC` in E4.
- **Acceptance:** native MTP gains at least 0.5 tok/s and the 1-SE throughput bars do not overlap.

The flag basis is the current
[official SGLang Inkling-Small cookbook](https://github.com/sgl-project/sglang/blob/main/docs_new/cookbook/autoregressive/ThinkingMachines/Inkling-Small.mdx)
and its
[deployment configuration](https://github.com/sgl-project/sglang/blob/main/docs_new/src/snippets/configs/thinkingmachines/inkling-small.jsx).
This width-1 variant is deliberately **unverified** until the runner completes on the target pair.

## Ready-to-run sequence

```bash
export WORKER_SSH=control2@<control2-host>
export WORKER_REPO=/home/control2/code/inklingdeus
export MASTER_IP=<control1-primary-roce-ip>
export IF=<control1-primary-roce-netdev>
export HCA=<unchanged-HCA-list-for-both-arms>
export MODELS=<same-absolute-model-path-on-both-nodes>
export CHAMPION_BLOCK=5

./scripts/run-e4-width1-mtp-ab.sh
```

The runner measures the inherited DSpark champion first and native width-1 MTP second in the same
session. Both arms receive T4 before and after exact n=32 chat-templated measurements. No result is
accepted from a different command, a partial run, or a post-hoc memory adjustment.
