# E5 — persistent compiler/JIT caches

Status: **COMPLETE — INCONCLUSIVE; persistent cache mounts remain off by default**.
The new cache mounts are opt-in, so the measured champion's default launch behavior is unchanged.

## Clean-root restart — live result

The complete eight-boot restart ran on 2026-08-03 from exact SHA
`7e44a67c06f535650a55902761d3709eb859ab78`, matched repo payload
`3782ed02d8be385f424f13c5e10c495a73d50a8cc377f3ef665136fc769a30a6`, and matched image bytes.
Every boot passed the two-node champion/cache-mount contract and T4; the prime and both compared
serving arms also passed post-measurement T4.

| timing arm | T4 seconds |
|---|---:|
| no-mount cold | 390 |
| cache prime | 389 |
| no-mount warm 1 / 2 / 3 | 399 / 389 / 399 |
| cache warm 1 / 2 / 3 | 389 / 358 / 369 |

The three-run no-mount warm mean was **395.7 +/- 3.3 seconds**; persisted-cache warm was
**372.0 +/- 9.1 seconds**. The 23.7-second saving is only 6.0% (`warm_ratio=0.940`), far short of
the predeclared >=60-second, >=20%, and <240-second gates. Serving did not regress: baseline
open-ended `n=32` was 26.052 +/- 0.292 tok/s versus cache-warm 26.189 +/- 0.314, delta +0.137 with
combined SE 0.428. The formal decision is **INCONCLUSIVE — do not adopt**.

Read-only manifests proved both nodes populated 1,077 files at prime and 1,515 files by the final
sample; cache sizes grew from about 370 MiB to 397 MiB. Wall #25 is cleared, but the cache itself
does not deliver an adoptable boot improvement on this image. Evidence is in
`artifacts/e5-jit-cache-restart-20260803/`; `decision.txt` hashes to
`4445d3d510d2bb5a74cec1320e79c25b583a467b9103894a5950816bc6ee20a8`, and all 11 T4 records hash
to `aac69468d03ab55a8da2d9f15e7939103b479a8e93d9b7839e12953299119de7`.

## First live session — invalidated after prime

The first session ran on 2026-08-03 from exact SHA
`8c19e4312addca1956be57ec2efd90d8dac41c27`, matched repo payload
`879fcd3eab274cbd2c6843173bfbe64314855e048e5105fbe1564788fcb68d6f`, and matched image bytes.
The no-mount cold boot reached T4 in 399 seconds. The persisted-cache prime reached T4 in 384
seconds, passed its two-node champion/mount contract, completed exact open-ended `n=32` at
25.857 +/- 0.290 tok/s and 2.123 +/- 0.023 accept, and remained T4-exact before and after.

The required host-side cache manifest then failed because root-owned CUDA cache hash directories
were not traversable by the host user. The runner stopped before any of the six balanced comparison
boots, so there is no warm-cache result or decision. This is harness wall #25, not serving evidence.
Partial artifacts are retained in `artifacts/e5-jit-cache-20260803/`. The restart must use a new
empty cache root and collect manifests through a read-only helper container; changing cache
permissions would mutate the factor under test and is not allowed.

The restart runner mounts the cache root read-only into a short-lived helper container for
`find` and `du`. This gives the evidence collector root traversal without changing any cache byte,
owner, mode, serving mount, or champion flag.

## Hypothesis and gates

- **Hypothesis:** persisting generated Triton, FlashInfer, SGLang, Torch extension/Inductor, and
  CUDA compute artifacts makes repeated boots pay compilation once and cuts warm time-to-T4 from
  roughly eight minutes to less than four.
- **Only changed serving factor:** six compiler-cache directories are bind-mounted from an explicit
  host root. The image, repo SHA/worktree payload, HCA, target/speculator weights, inherited
  block-5/CDS2 speculator profile, KV dtype, context, memory
  fraction, graphs, prompts, and tokens remain fixed.
- **Isolation:** the runner refuses an existing cache root. It captures a no-mount cold boot, primes
  the new mounts with exact n=32 traffic, then alternates six comparison boots in B-A-A-B-B-A order.
  Three no-mount and three persisted-cache warm samples prevent a warm OS model-page cache or simple
  time drift from being mistaken for a JIT-cache win.
- **Evidence:** T4 runs on every boot and again after both compared n=32 serving arms; live Docker
  inspection proves the six intended mounts are either all present or all absent and verifies the
  complete champion command on both nodes; both nodes must produce non-empty file manifests and
  size records.
- **Kill criteria:** repo/image drift, pre-existing or empty cache roots, a wrong live mount, any boot
  timeout, any T4 mismatch, an arm other than exact n=32, or warm serving throughput losing at least
  one combined SE.
- **Acceptance:** three-run persisted-cache warm mean time-to-T4 is under 240 seconds, saves at least
  60 seconds and 20% versus the three-run no-mount warm mean, the 1-SE boot bars do not overlap, and
  serving throughput does not regress by one combined SE.

The mounted locations follow the projects' current documented defaults: Triton's `.triton` cache,
[FlashInfer's `~/.cache/flashinfer`](https://github.com/flashinfer-ai/flashinfer/blob/main/CLAUDE.md),
[SGLang's `~/.cache/sglang`](https://github.com/sgl-project/sglang/blob/main/docs/references/environment_variables.md),
and
[TorchInductor's `/tmp/torchinductor_root`](https://github.com/sgl-project/sglang/blob/main/docs/advanced_features/server_arguments.md).
Torch extension and CUDA compute caches are mounted alongside them. Cache contents are executable
compiler artifacts: use a dedicated trusted path and do not share it with untrusted containers.

## Ready-to-run sequence

Choose a path that does not yet exist on either node; the runner will create it but never delete it.

```bash
export WORKER_SSH=control2@<control2-host>
export WORKER_REPO=/home/control2/code/inklingdeus
export MASTER_IP=<control1-primary-roce-ip>
export IF=<control1-primary-roce-netdev>
export HCA=<unchanged-HCA-list-for-all-boots>
export MODELS=<same-absolute-model-path-on-both-nodes>
export E5_CACHE_ROOT=/var/cache/inkling-e5-<exact-image-or-run-id>
export CHAMPION_BLOCK=5
export CHAMPION_SPECULATOR=<dspark-or-mtp-width1>

./scripts/run-e5-jit-cache-ab.sh
```

The experiment is intentionally long: eight boots are the cost of separating compiler-cache reuse
from model-page-cache warmth. A lone fast restart is not acceptable evidence.
