# E6 — memory fraction under C8/C16 load

Status: **HOLD — mandatory read-only preflight failed on both Sparks on 2026-08-03**.
Nothing in this branch installs, enables, or reconfigures `earlyoom` on either Spark.

Both control1 and control2 returned `earlyoom is not installed; do not start E6`. Per the declared
box-protection and kill criteria, no serving arm was launched and no memory-fraction inference is
valid. The exact preflight outputs are committed in `artifacts/e6-preflight-20260803/`. E6 can
resume only after separately authorized host setup; this experiment does not grant that authority.

## Hypothesis and gates

- **Hypothesis:** the current `MEMFRAC=0.85` either has proven host-level headroom through C8/C16,
  or it approaches unified-memory exhaustion and should fall back to the official DSpark-style
  `0.68` reservation.
- **Only changed serving variable:** `MEMFRAC=0.85` versus `0.68`. HCA, weights, repo SHA/worktree
  payload, image
  payload, target/draft backends, block size, KV dtype, context, graphs, prompts, and tokens stay
  fixed.
- **Traffic:** chat-templated open-ended work at C8 and C16, exact n=32 per concurrency and arm.
  Aggregate tokens/s is summarized across fixed-size waves; the old raw `/generate` concurrency
  script has been replaced and cannot support a serving claim.
- **Box protection:** an already-active system `earlyoom` service is mandatory. In addition, the
  experiment starts a scoped guard on each node that samples `MemAvailable` and memory PSI every
  second and removes only `inkling-sglang` below 12 GiB (configurable by `MIN_AVAILABLE_KIB`).
- **Evidence:** per-node guard logs/minima, trip files, kernel OOM journal, `earlyoom` journal,
  Docker stats, T4 before/after each completed arm, and machine-readable stability status.
- **Kill criteria:** absent/inactive `earlyoom`, unreadable journals, repo/image drift, any T4
  mismatch, malformed/non-n=32 traffic, or an unexplained server failure. An unexplained baseline
  failure invalidates the experiment; it is not evidence for lowering the fraction.
- **Decision:** accept `0.68` only when `0.85` triggers concrete host-memory/OOM protection and
  `0.68` completes C8/C16 losslessly. If `0.85` is safe, retain it; a smaller KV pool has no
  compensating win. Any candidate instability rejects `0.68`.

## Earlyoom prerequisite — not executed by this branch

On DGX OS/Ubuntu, an administrator can install and enable the package separately:

```bash
sudo apt-get update
sudo apt-get install earlyoom
sudo systemctl enable --now earlyoom
systemctl status earlyoom --no-pager
```

Those commands change host state and are **not** run by the experiment. After an authorized setup,
the read-only `scripts/earlyoom-preflight.sh` must pass on both nodes before load starts.

## Ready-to-run sequence

Use a new result directory each time; the runner refuses to overwrite prior evidence.

```bash
export WORKER_SSH=control2@<control2-host>
export WORKER_REPO=/home/control2/code/inklingdeus
export MASTER_IP=<control1-primary-roce-ip>
export IF=<control1-primary-roce-netdev>
export HCA=<unchanged-HCA-list-for-both-arms>
export MODELS=<same-absolute-model-path-on-both-nodes>
export RESULT_DIR=artifacts/e6-memfrac-<run-id>
export CHAMPION_BLOCK=5
export CHAMPION_SPECULATOR=<dspark-or-mtp-width1>

./scripts/run-e6-memfrac-ab.sh
```

The baseline runs first. A memory-guard trip is expected to terminate the container; the runner
continues to `0.68` only when it has explicit trip, kernel-OOM, or `earlyoom` evidence.
