# E8 — decode-latency micro-tuning: NCCL protocol, continuous decode steps, KV splits

Status: **NOT RUN — prepared follow-on against the promoted block-5 champion**. No arm has been
executed; nothing in this branch changes the measured champion's default launch behavior (the new
`NCCL_ALGO`/`NCCL_PROTO` launcher knobs inject only when explicitly set, and
`locked-experiment-launch.sh` pins them empty for every other experiment).

## Hypothesis and gates

Single-stream decode on this stack pays three per-step overheads that no experiment has swept:

1. **The TP2 all-reduce runs on every decode step** and is small-message latency-bound, not
   bandwidth-bound — which is consistent with E1's result (doubling link bandwidth changed
   nothing). NCCL's protocol choice (`LL` / `LL128` / `Simple`) governs exactly this small-message
   latency path, and the champion currently accepts whatever NCCL autotunes. E1 varied the HCA
   count; protocol has never been varied.
2. **Scheduler overhead per decode step.** `--num-continuous-decode-steps` batches N decode
   iterations per scheduler tick, amortizing that overhead at a bounded cost to streaming
   granularity (`--stream-interval 32` already exceeds it).
3. **The fp32 split-KV reduction** behind `--triton-attention-reduce-in-fp32` (the +11%-accept
   flag) scales with `--triton-attention-num-kv-splits` (build default 8). Fewer splits mean less
   reduction work on short-KV requests; more splits help long-KV. The benchmark's short prompts
   sit at the short-KV end, so 4 may beat 8 — and 16 bounds the other direction.

Every arm changes **exactly one factor** vs the same-session baseline and must pass:

- byte-exact T4 lossless gate before and after its `n=32` measurement;
- the runtime contract check (the intended factor — and no other — visible in `docker inspect`
  on both nodes);
- acceptance: `+0.5` tok/s minimum gain vs baseline per `benchmarks/compare_ab.py` defaults,
  with accept length not degraded beyond combined standard error.

## Factors and arms

| arm | factor changed | value |
|---|---|---|
| `e8-baseline` | none (block-5 champion) | — |
| `e8-proto-ll` | `NCCL_PROTO` | `LL` |
| `e8-proto-ll128` | `NCCL_PROTO` | `LL128` |
| `e8-proto-simple` | `NCCL_PROTO` | `Simple` |
| `e8-cds-2` | `--num-continuous-decode-steps` | `2` |
| `e8-cds-4` | `--num-continuous-decode-steps` | `4` |
| `e8-ksplit-4` | `--triton-attention-num-kv-splits` | `4` |
| `e8-ksplit-16` | `--triton-attention-num-kv-splits` | `16` |

`FACTORS="proto cds ksplit"` (default) selects which groups run; a partial sweep is valid because
every arm compares only against its own session's baseline.

## Kill criteria

- A candidate arm that fails to boot is **REJECTED — factor unavailable in this build** (recorded
  with log tails in `<label>-boot-failure.txt`); the sweep continues. This covers the possibility
  that `--num-continuous-decode-steps` or `--triton-attention-num-kv-splits` does not exist in the
  pinned image's ServerArgs, and that NCCL rejects a protocol value on this platform.
- A baseline boot failure aborts the whole session — no comparison may be made.
- Any T4 mismatch on any arm invalidates that arm regardless of throughput. `ksplit` arms change
  reduction order, so temp-0 outputs may legitimately differ **only if** the T4 probe itself
  differs — per the repo's constraint that lossless references are comparable only within one
  numerics configuration, a `ksplit` arm that fails byte-exactness is REJECTED, not re-baselined.
- `cds` arms additionally must not regress `chat_bench` per-request latency by more than one
  combined standard error even if throughput holds (streaming responsiveness guard).

## Ready-to-run sequence

```bash
# On the idle head Spark, from the repo root, with the worker idle:
WORKER_SSH=<user@worker> WORKER_REPO=<abs repo path on worker> \
MASTER_IP=<head link IP> IF=<link netdev> HCA=<champion HCA list> \
MODELS=<models dir> ./scripts/run-e8-decode-latency-ab.sh
```

Artifacts land in `artifacts/e8-decode-latency/`: per-arm `chat_bench` JSON with accept
histograms, pre/post lossless transcripts, `docker inspect` contract records, per-arm
`decision-*.txt` from `compare_ab.py`, and `rejected-arms.txt` for boot-failed factors.

Adoption of any winning arm is a separate, explicit champion change: update the launcher default,
re-run T4 + `chat_bench --task all`, and record the new champion numbers in the README — the same
procedure E3 followed for block 5.
