# E4 — DSpark block 5 vs native width-1 MTP

Status: **NOT RUN — optimization loop stopped after E3 success**. This remains a prepared
follow-on experiment against the promoted block-5 champion.

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
