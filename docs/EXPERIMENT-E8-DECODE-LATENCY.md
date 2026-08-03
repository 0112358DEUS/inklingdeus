# E8 — decode-latency micro-tuning: NCCL protocol, continuous decode steps, KV splits

Status: **COMPLETE — continuous decode steps 2 adopted; NCCL protocol and KV-split defaults retained**.

The fail-closed runner was hardened before measurement: it rejects acceptance loss at one combined
SE for every arm, rejects per-request latency regression at one combined SE for `cds` arms, records
all planned decisions even when an earlier arm is rejected/inconclusive, and terminates a boot arm
as soon as its serving container disappears.

## NCCL protocol subgroup — live result

The protocol subgroup ran on control1/control2 on 2026-08-03 from exact repo SHA
`d104f8bdcaa0be893a1c62996d64b5b87f749824`. Both clean checkouts had repo payload
`45af9635faf581a57c908762d158bb762fd214be528ee963a96dec4b75b627a3`; both patched images had
payload `b2272bfef54e3dd37eea30b67a1fa8c3ae54f5f1b4c877e1a6661903a4a8ac61`. Each arm changed only
`NCCL_PROTO`, passed its two-node runtime contract, and produced byte-identical T4 output before
and after exact chat-templated open-ended `n=32` measurement.

| arm | tok/s | delta vs baseline | accept | decision |
|---|---:|---:|---:|---|
| baseline/autotuned | 26.189 +/- 0.321 | — | 2.132 +/- 0.025 | reference |
| `LL` | 25.808 +/- 0.385 | -0.381 | 2.148 +/- 0.031 | INCONCLUSIVE — no adoption |
| `LL128` | 25.852 +/- 0.265 | -0.337 | 2.136 +/- 0.021 | INCONCLUSIVE — no adoption |
| `Simple` | 26.249 +/- 0.318 | +0.059 | 2.173 +/- 0.026 | INCONCLUSIVE — no adoption |

The `Simple` delta was only 0.059 tok/s against a 0.452 combined SE and missed the predeclared
+0.5 tok/s threshold. The other forced protocols were slower. Acceptance did not regress in any
arm, but no throughput bar cleared the adoption rule. Keep NCCL protocol autotuning; do not quote
any forced protocol as a serving improvement.

Evidence is in `artifacts/e8-proto-20260803/`. The four benchmark JSON hashes are
`9985b289866b3feb87539b8ca465069258e97bdea3f205be3bc10428242f7325`,
`01dd3cabacc2bd22e463c78fa8b4c4efef1c948f533444653f993de5b8f7afa3`,
`e8f8d6078c6275f3379809b0dd518c5d0201a40d09d7a4852e24612239af3967`, and
`46f8995500f3e86da9d1d72d361a73056a060c464a283134b74f29755458881e` in table order. All eight
pre/post T4 records hash to
`aac69468d03ab55a8da2d9f15e7939103b479a8e93d9b7839e12953299119de7`.

## Continuous-decode-step subgroup — live result

The continuous-decode-step subgroup ran on the same pair on 2026-08-03 from exact repo SHA
`d5e1a015799e0da8bbf44a08a91ca33503209e45`, with matched repo payload
`03cf58ee92c2123e392d54ffa32cd2bf7beb17079ebb9e38d54411443a7c74d5` and the same matched image
payload. Each candidate changed only `--num-continuous-decode-steps`, passed its two-node runtime
contract, and produced byte-identical T4 output before and after exact open-ended `n=32`.

| arm | tok/s | delta vs baseline | accept | mean request latency | decision |
|---|---:|---:|---:|---:|---|
| baseline / steps 1 | 25.888 +/- 0.302 | — | 2.112 +/- 0.025 | 6.208 +/- 0.075 s | reference |
| steps 2 | 26.533 +/- 0.299 | +0.645 | 2.158 +/- 0.023 | 6.055 +/- 0.071 s | **ACCEPT** |
| steps 4 | 26.006 +/- 0.307 | +0.118 | 2.109 +/- 0.024 | 6.179 +/- 0.073 s | INCONCLUSIVE |

Steps 2 cleared the +0.5 tok/s threshold and the one-combined-SE throughput rule (combined SE
0.425), while acceptance improved by 0.046 and mean request latency improved by 0.153 s. Steps 4
did not clear the throughput rule. The accepted steps-2 candidate now proceeds through the
separate champion-adoption procedure: promote the launcher default, re-run T4 and
`chat_bench --task all`, and only then publish new champion numbers. No default changes in this
measurement commit.

### Champion adoption gate — complete

The default promotion was validated on 2026-08-03 from exact SHA
`63c65c572f42d04756a389e140dc49a497596af0`, matched repo payload
`51f45e78a61f0f7da91a37c8b15a3691f7ca3765ac384bb915a277c03ba3501c`, and matched image payload
`b2272bfef54e3dd37eea30b67a1fa8c3ae54f5f1b4c877e1a6661903a4a8ac61`. Both node commands
contained exactly one `--num-continuous-decode-steps 2` and the full champion contract; neither had
an NCCL or ragged override. T4 was byte-exact before and after `chat_bench --task all`.

| adoption task | n | tok/s | accept |
|---|---:|---:|---:|
| GSM8K-style | 32 | 48.738 +/- 0.585 | 4.188 +/- 0.053 |
| code | 32 | 33.226 +/- 0.478 | 2.834 +/- 0.039 |
| chat | 32 | 26.330 +/- 0.371 | 2.149 +/- 0.033 |
| open-ended | 32 | 26.225 +/- 0.323 | 2.138 +/- 0.026 |
| pooled | 128 | 33.630 +/- 0.844 | 2.827 +/- 0.077 |

Evidence is in `artifacts/e8-cds2-adoption-20260803/`. The all-task JSON hashes to
`685b804f69bb236e646cffc805bd2878212c254eb9de5e72fcf509239e730298`; both T4 records hash to
`aac69468d03ab55a8da2d9f15e7939103b479a8e93d9b7839e12953299119de7`. The champion default is
therefore promoted to continuous decode steps 2. The E8 runner inherited that value for every
subsequent KV-split arm.

## KV-split subgroup — live result

The final subgroup ran on 2026-08-03 from exact adopted-champion SHA
`1bb973cc76c6fd903c134b9f23d78ae04d29e118`, with matched repo payload
`e21e8e0d5acbcfb3ef5c4fdf37424c6a6f422fe5a3e2202e3ef271974202ac7c` and the same matched image
payload. Each arm retained continuous decode steps 2 and changed only
`--triton-attention-num-kv-splits`. Both runtime contracts and all six pre/post T4 gates passed.

| arm | tok/s | delta vs baseline | accept | decision |
|---|---:|---:|---:|---|
| baseline / splits 8 | 26.313 +/- 0.340 | — | 2.146 +/- 0.025 | reference |
| splits 4 | 25.990 +/- 0.351 | -0.323 | 2.129 +/- 0.028 | INCONCLUSIVE — no adoption |
| splits 16 | 25.835 +/- 0.337 | -0.478 | 2.108 +/- 0.026 | REJECT — acceptance loss |

Splits 4 was slower and did not clear the throughput rule. Splits 16 was slower and its acceptance
loss of 0.038 was at least the 0.037 combined SE, so the acceptance guard rejected it. Retain the
SGLang default of 8 KV splits. Evidence is in `artifacts/e8-ksplit-20260803/`; the baseline,
splits-4, and splits-16 JSON hashes are
`ad3434e31c757bfcb7e1ec5682ec266e63e7d1ac1fe104670ad76b479bda9c57`,
`ec29e8328cc175d8e2aab304c94e3cb858c5099ef24eae74e3076e72441ce0b9`, and
`2f2353fd1fc07bdee7feaa77113cb8b7bf88c4dba1588f47fe761b528d393400`. All six T4 records hash to
`aac69468d03ab55a8da2d9f15e7939103b479a8e93d9b7839e12953299119de7`.

E8's adopted result is therefore exactly one default change: continuous decode steps 1 to 2.
NCCL protocol remains autotuned and KV splits remains 8.

Evidence is in `artifacts/e8-cds-20260803/`. The baseline, steps-2, and steps-4 benchmark JSON
hashes are `134421549d7f2417e67897bd505a6c37208ca30205033fa411160f9e27d3d3d3`,
`8446b547d134c29cba2cd2a59849876dd07c48889343fa219d922704e27b4acd`, and
`fcd41a4c2f7cba43f37f7bb273bea73f64029fdf64555472d1343d4ab1330010`. All six pre/post T4
records hash to `aac69468d03ab55a8da2d9f15e7939103b479a8e93d9b7839e12953299119de7`.

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
