# E1 — dual RoCE twins

Status: **COMPLETE — INCONCLUSIVE; retain single HCA** on 2026-08-02.

## Live result

Both physical twins passed the raw-link gate at **111.62 Gb/s** independently. The serving A/B
then changed only `NCCL_IB_HCA` and ran in one session with byte-exact T4 before and after each
chat-templated open-ended `n=32` arm:

| arm | throughput | accept length |
|---|---:|---:|
| single `rocep1s0f1` | 24.928 +/- 0.295 tok/s | 2.177 +/- 0.023 |
| dual `rocep1s0f1,roceP2p1s0f1` | 24.636 +/- 0.291 tok/s | 2.146 +/- 0.022 |

The dual-HCA delta was **-0.292 tok/s** with combined SE **0.414**. It did not meet the +0.5 tok/s
acceptance threshold, its error bars overlapped, and the loss did not reach one combined SE. The
fail-closed comparator therefore returned `INCONCLUSIVE`; single HCA remains the champion setting.

Evidence is in `artifacts/e1-dual-roce/`. The immutable result digests are:

- single JSON: `9977b99343bac8d45f4d2acb91002b9dbac374ad23dc1e8410d2ff4257283c93`
- dual JSON: `d2a302d654258ab2d078541bd2fccad4b679f2f2e761eafd3ed28c3f44dc3c4f`
- each of the four T4 records: `aac69468d03ab55a8da2d9f15e7939103b479a8e93d9b7839e12953299119de7`

## Hypothesis and gates

- **Hypothesis:** exposing both QSFP twins through `NCCL_IB_HCA` removes a PCIe x4 bottleneck and
  increases chat-templated open-ended decode throughput.
- **Expected effect:** each twin sustains about 111.7 Gb/s raw; the serving gain should exceed
  +0.5 tok/s if inter-node traffic is limiting decode.
- **Kill criteria:** either twin is inactive, lacks its own IPv4 RoCEv2 GID, or measures below
  100 Gb/s; either T4 output differs by one byte; dual-HCA loses at least one combined SE; or the
  two arms differ in any setting other than `NCCL_IB_HCA`.
- **Acceptance:** both T4 probes pass, both arms are same-session chat-templated n=32, dual-HCA
  gains at least 0.5 tok/s, and the arms' 1-SE throughput bars do not overlap.

The second twin needs a separate point-to-point subnet on both nodes. Configuration is deliberately
not automated because it requires site-specific privileged network changes. Confirm with
`show_gids` that both named HCAs have IPv4 RoCEv2 entries (normally GID index 3) before running E1.

## Ready-to-run sequence

Run from the head Spark after the same commit, image, and model paths are present on both nodes:

```bash
export WORKER_SSH=control2@<control2-host>
export WORKER_REPO=/home/control2/code/inklingdeus
export MASTER_IP=<control1-primary-roce-ip>
export IF=<control1-primary-roce-netdev>
export MODELS=<same-absolute-model-path-on-both-nodes>
export HEAD_IPS_CSV=<control1-twin-a-ip>,<control1-twin-b-ip>
export WORKER_IPS_CSV=<control2-twin-a-ip>,<control2-twin-b-ip>
export HCAS_CSV=rocep1s0f1,roceP2p1s0f1

./scripts/run-e1-dual-roce-ab.sh
```

The script first requires identical repo SHAs, byte-level runnable worktree fingerprints (including
uncommitted pre-T4 files), and SHA-256 fingerprints for the patched image
payload across the pair (independently baked Docker image IDs may legitimately differ). It then runs
raw `ib_write_bw` on both twins, boots the single-HCA arm, runs T4 before and after the open-ended
n=32 chat benchmark, then repeats with the comma-separated dual-HCA list. It finally
invokes `benchmarks/compare_ab.py`, which exits 0 only for an accepted result, 2 for a regression,
and 3 for an inconclusive comparison.
