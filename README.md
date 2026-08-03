# keys-1M-context · Inkling-Small-NVFP4 + DSpark + NVFP4 KV Cache · SGLang · sm_121a · Two DGX Sparks

> **Fork notice**: this is `0112358DEUS/inklingdeus`, a maintained fork of
> [drowzeys/keys-1M-CTX-…-Two-DGX-Sparks](https://github.com/drowzeys/keys-1M-CTX-Inkling-Small-NVFP4-Dspark-NVFP4-KV-Cache-SGlang-SM121-optimized-on-Two-DGX-Sparks)
> with review fixes (see [NOTICE.md](NOTICE.md) for provenance and licensing).

**A full 1M-token context on two desktop DGX Sparks, first implemented NVFP4 KV cache on SGlang — for
Inkling-Small NVFP4 + DSpark.**

Serve [thinkingmachines/Inkling-Small-NVFP4](https://huggingface.co/thinkingmachines/Inkling-Small-NVFP4)
(276B total / 12B active MoE) with the [RadixArk DSpark speculator](https://huggingface.co/RadixArk/Inkling-Small-DSpark-Preview)
across **two desktop DGX Sparks** (GB10 / sm_121a), at a **full 1,048,576-token context**.

This is a field port. SGLang's triton backend — the only attention lane this model can use on
consumer Blackwell — shipped with **no KV quantization at all**, and none of this had been run on
GB10 before. Everything needed is here: patched files, bake script, launcher, benchmarks, and every
wall we hit with its fix.

---

## Quick start — the champion stack

**Prereqs**: 2× DGX Spark / GB10 (128 GB unified each, DGX OS, CUDA 13, docker + nvidia runtime); a
direct 200G CX7↔CX7 link with IPs on both ends; `ls /dev/infiniband` non-empty on both nodes; ~165 GB
of storage for weights, reachable at the **same path** on both nodes (NFS or local copies).

```bash
# 0) on the HEAD node (rank 0), with SSH access to the worker
git clone https://github.com/0112358DEUS/inklingdeus.git
cd inklingdeus

# 1) weights — once, wherever the shared storage lives
python3 -m venv ~/hfdl-venv && ~/hfdl-venv/bin/pip install -q huggingface_hub hf_transfer
HF_HUB_ENABLE_HF_TRANSFER=1 ~/hfdl-venv/bin/hf download \
  thinkingmachines/Inkling-Small-NVFP4  --local-dir <STORE>/inkling/inkling-small-nvfp4
HF_HUB_ENABLE_HF_TRANSFER=1 ~/hfdl-venv/bin/hf download \
  RadixArk/Inkling-Small-DSpark-Preview --local-dir <STORE>/inkling/dspark-draft

# 2) get the image — ON EACH NODE.  Either pull the prebuilt one (fastest):
docker pull ghcr.io/drowzeys/inkling-sglang-gb10:kvquant
docker tag  ghcr.io/drowzeys/inkling-sglang-gb10:kvquant local/sglang-inkling:gb10-kvquant
#    ...or build it yourself from the digest-pinned upstream + patches in this repo:
# KVQUANT=1 ./scripts/bake-image.sh

# 3) launch — WORKER FIRST, then head. Defaults are the champion config.
#    worker:
MASTER_IP=<rank0-link-ip> IF=<link-nic> HCA=<rdma-dev> MODELS=<mount>/inkling ./scripts/nvfp4-kv-boot.sh 1
#    head:
MASTER_IP=<rank0-link-ip> IF=<link-nic> HCA=<rdma-dev> MODELS=<mount>/inkling ./scripts/nvfp4-kv-boot.sh 0
```

**On the image size**: it reports ~43 GB, but **only ~473 MB of that is ours** — 42.2 GB is
upstream's `lmsysorg/sglang:dev-cu13-inkling-dspark` dev image (CUDA 13 toolkit + PyTorch +
FlashInfer AOT cubins + SGLang kernels for sm_80/90/100/110/120). Docker layers are
content-addressed, so **if you already have that upstream image, pulling ours downloads only the
differing layers (~473 MB)** — of which 471 MB is the mandatory NCCL 2.30 upgrade; every code patch
is under half a megabyte. To make the pull cheap deliberately:

```bash
docker pull lmsysorg/sglang@sha256:fbea1a4e25b26660dbc2384a27ead8817e9b7670f257b5c3143e0450d14524d7
docker pull ghcr.io/drowzeys/inkling-sglang-gb10:kvquant   # now only ~473 MB of new layers
```

(We don't ship a slimmed base: the "unused" architecture cubins aren't safely removable — `sgl_kernel`
loads its **sm100** variant on GB10, which is one of the quirks documented in the walls table.)

The prebuilt image is exactly what `KVQUANT=1 ./scripts/bake-image.sh` produces — same patches, same
digest-pinned base — published so you don't have to rebuild. Verify it if you like:
`docker run --rm --entrypoint bash ghcr.io/drowzeys/inkling-sglang-gb10:kvquant -c 'ls /sgl-workspace/sglang/python/sglang/srt/mem_cache/kv_quant_pools.py'`

`IF` is the NIC carrying the inter-node link, `HCA` its RDMA device (`ibv_devices`). Boot takes
~8 min (156 GB weight load + first-run JIT); follow it with `docker logs -f inkling-sglang`.
OpenAI-compatible endpoint on port **30000**.

### Verify (30 seconds)

```bash
curl -s localhost:30000/v1/completions -H 'Content-Type: application/json' \
  -d '{"model":"inkling-small","prompt":"The capital of France is","max_tokens":12,"temperature":0}'
```

Expect byte-for-byte: ` Paris. The capital of Germany is Berlin. The capital of`

That is the bf16 reference output — matching it proves both the quantized-KV path and the speculator
are numerically clean. Then confirm the pool exceeds your context:

```
grep -aoE 'context_len=[0-9]+|max_total_num_tokens=[0-9]+' ~/inkling-serve.log | tail -2
→ context_len=1048576    max_total_num_tokens=1331001
```

---

## The exact recipe (what the launcher actually runs)

If you prefer to run it by hand, or need to adapt it, this is the champion invocation verbatim.
`scripts/nvfp4-kv-boot.sh` is exactly this with the site values as env vars.

```bash
docker run --name inkling-sglang --rm --gpus all --network host --ipc host \
  --shm-size 16g --device /dev/infiniband --cap-add IPC_LOCK \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  -v <STORE>/inkling:/models:ro \
  -e SGLANG_ENABLE_UNIFIED_RADIX_TREE=1 \
  -e INKLING_TORCH_CONV_COMMIT=1 -e INKLING_COMMIT_STEP_BIAS=1 \
  -e NCCL_IB_HCA=<rdma-dev> -e NCCL_IB_GID_INDEX=3 \
  -e NCCL_SOCKET_IFNAME=<link-nic> -e GLOO_SOCKET_IFNAME=<link-nic> -e TP_SOCKET_IFNAME=<link-nic> \
  -e NCCL_NET=IB -e NCCL_IB_DISABLE=0 -e NCCL_NET_PLUGIN=none \
  -e NCCL_CUMEM_ENABLE=0 -e NCCL_NVLS_ENABLE=0 -e NCCL_CROSS_NIC=0 \
  -e NCCL_IGNORE_CPU_AFFINITY=1 -e NCCL_DEBUG=WARN \
  -e TORCH_CUDA_ARCH_LIST=12.1a -e FLASHINFER_CUDA_ARCH_LIST=12.1a \
  -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  --entrypoint python3 local/sglang-inkling:gb10-kvquant \
  -m sglang.launch_server \
    --model-path /models/inkling-small-nvfp4 --trust-remote-code \
    --served-model-name inkling-small \
    --host 0.0.0.0 --port 30000 \
    --tp-size 2 --nnodes 2 --node-rank <0|1> --dist-init-addr <rank0-link-ip>:25000 \
    --context-length 1048576 \
    --quantization modelopt_fp4 \
    --kv-cache-dtype fp4_mx_block16 \
    --attention-backend triton --triton-attention-reduce-in-fp32 \
    --page-size 1 \
    --fp4-gemm-backend flashinfer_trtllm \
    --moe-runner-backend marlin \
    --mamba-radix-cache-strategy extra_buffer \
    --mem-fraction-static 0.85 \
    --swa-full-tokens-ratio 0.1 --mamba-full-memory-ratio 0.1 \
    --max-running-requests 16 \
    --chunked-prefill-size 8192 \
    --reasoning-parser inkling --tool-call-parser inkling \
    --skip-server-warmup --disable-flashinfer-autotune \
    --stream-interval 32 \
    --num-continuous-decode-steps 2 \
    --speculative-algorithm DSPARK \
    --speculative-draft-model-path /models/dspark-draft \
    --speculative-draft-model-quantization unquant \
    --speculative-dspark-block-size 5 \
    --cuda-graph-bs 1 2 3 4 5 6 7 8 10 12 14 16 \
    --disable-piecewise-cuda-graph --disable-prefill-cuda-graph
```

**Every non-obvious flag, and why it is not optional:**

| Flag | Why |
|---|---|
| `--kv-cache-dtype fp4_mx_block16` | the triton-compatible fp4 recipe. `nvfp4` selects the flashinfer/trtllm packing this lane cannot read; `fp8_e4m3` produces garbage (wall 13) |
| `--attention-backend triton` | Inkling asserts `fa4\|triton`, and fa4 is sm_100-only ⇒ triton is the only legal lane on GB10 |
| `--triton-attention-reduce-in-fp32` | bf16 accumulation across KV splits perturbs logits → fewer draft matches. ~+11% accept |
| `--moe-runner-backend marlin` | the only numerically-correct NVFP4 MoE runner on sm_121: cutlass **silently miscomputes**, trtllm hard-fails on sm_100-only cubins |
| `--page-size 1` | page-128 (the fa4 layout) corrupts the triton verify path |
| `--disable-prefill-cuda-graph` | the triton backend cannot replay `EXTEND` mode |
| `--disable-piecewise-cuda-graph` | the sm_121 piecewise compiler hard-fails |
| `--cuda-graph-bs 1 2 … 16` | an explicit list; `--cuda-graph-max-bs` does **not** filter and the default list OOMs the pool |
| `--speculative-dspark-block-size 5` | E3 winner: 26.007 ± 0.334 vs block 7 at 24.747 ± 0.208 tok/s (`n=32`, T4 before/after every arm) |
| `--num-continuous-decode-steps 2` | E8 winner: +0.645 tok/s vs steps 1 (combined SE 0.425), accept +0.046, mean request latency -0.153 s; all adoption tasks and T4 passed |
| `INKLING_TORCH_CONV_COMMIT=1` + `INKLING_COMMIT_STEP_BIAS=1` | the conv-state commit fix. Without them output degenerates into prompt-replay whenever accept > 1 |
| `--device /dev/infiniband --cap-add IPC_LOCK` | without RDMA passthrough NCCL fails with a bare `invalid usage` |
| `--mem-fraction-static 0.85` | 0.87 boots fine but buys nothing measurable |

**Order matters**: start `--node-rank 1` (worker) first, then `--node-rank 0` (head). The head is the
rendezvous point at `--dist-init-addr`. If you script this across nodes, put the environment in a
**file on each node** rather than passing it through nested SSH — multi-flag `EXTRA_ARGS` gets split
by the second shell and the worker silently never launches (you'll see `1/2 clients joined`).

---

## What you get

**Context is nearly free with fp4 KV.** Historical 64K/1M measurements moved the KV pool by only
~2%, because the SWA/mamba reserves that scale with context are
small next to a quantized pool. That is *not* true on bf16, where the same change costs most of the
pool — which is why `scripts/inkling-sglang-launch.sh` (the bf16 path) still defaults to 64K while
the champion launcher defaults to the full 1M.

| | **1M profile** (default) | 64K profile |
|---|---|---|
| launch | `./scripts/nvfp4-kv-boot.sh <rank>` | `CTX=65536 ./scripts/nvfp4-kv-boot.sh <rank>` |
| context | **1,048,576** | 65,536 |
| KV pool | **1,331,001 tokens in the E8 adoption run** | boot-dependent; must exceed declared context |
| decode | **26.225 ± 0.323 tok/s open-ended** | remeasure after profile change |
| accept (of 6) | **2.138 ± 0.026 open-ended** | remeasure after profile change |

### Throughput depends heavily on workload — quote a task class, always

Measured on this stack, stock draft, temp 0. Serving claims use the chat-templated
[`benchmarks/chat_bench.py`](benchmarks/chat_bench.py); the legacy raw-continuation probe is
retained separately for historical comparison. E8's adoption gate measured every task class on the
current block-5, continuous-decode-steps-2 champion in one session. `GSM8K-style` here is the
serving-harness prompt class, not the separate full 1,319-item accuracy gate:

| workload | accept (of 6) | tok/s |
|---|---|---|
| GSM8K-style (current champion, n=32) | **4.188 ± 0.053** | **48.738 ± 0.585** |
| code explanation (current champion, n=32) | 2.834 ± 0.039 | 33.226 ± 0.478 |
| chat (current champion, n=32) | 2.149 ± 0.033 | 26.330 ± 0.371 |
| **open-ended prose (current champion, n=32)** | **2.138 ± 0.026** | **26.225 ± 0.323** |
| pooled all-task evidence (current champion, n=128) | 2.827 ± 0.077 | 33.630 ± 0.844 |

**Plan against ~26 tok/s for open-ended work on the current champion.** The task-class spread is
not noise — it's the same gradient RadixArk's card shows across its nine datasets (GSM8K 4.79 →
Arena-Hard 2.70): predictable, templated text drafts well; novel open-ended prose does not. The
older block-7 GSM8K-style result (4.81) reproduced their card; the current block-5/CDS2 all-task
gate is a different serving configuration and is reported independently above.

> **Correction (2026-08-01):** earlier revisions of this README published **3.44 accept / 34.3 tok/s**
> from a raw-continuation probe (untemplated text through `/generate`). That probe turns out to
> inflate acceptance by ~60%, because this model *echoes* untemplated input — and repetitive text is
> trivially draftable. The number was real but unrepresentative. It is kept below as a legacy
> reference so older figures remain comparable, not as an expectation.
>
> | legacy raw-continuation probe | 3.44 ± 0.17 | 34.3 ± 1.7 |
> |---|---|---|

- **Speculation is lossless** — DSpark output is byte-exact vs non-speculative decoding at temp 0.
- **fp4 KV is quality-neutral** — byte-exact vs bf16 KV on the reference probe; needle retrieval
  verified at **21K, 64K and 113K** token depths.
- **Without quantized KV the pool caps near 354K tokens** — fp4 is what makes 1M reachable at all.
- Reference point: no speculation at all is ~13 tok/s.

### Config detail worth matching

The chat template defaults to **thinking effort 0.9**; RadixArk's card measured at **0.99**. Setting
0.99 is worth +0.06 accept (~1.4σ) and costs nothing — but note it explains very little: sweeping the
full 0.0 → 0.99 range moves accept only +0.11. Workload composition dominates everything else.

---

## Why patches are needed

`scripts/bake-image.sh` bakes them all. Full symptom → cause → fix table for **25 walls** lives in
[`docs/BUGS-AND-FIXES.md`](docs/BUGS-AND-FIXES.md). The load-bearing ones:

| Area | Fix |
|---|---|
| **KV quantization** | The triton backend had none. `patches/kv-quant/` adds it: quantize **inside the pool** (Inkling has *three* KV writers — DSpark's hidden-state injector writes KV directly), an fp4 branch for the hybrid-SWA pool upstream never wrote, e2m1 nibble decode + block-16 scales in cloned kernels, correct fp4 byte accounting. Upstream's `decode_attention.py`/`extend_attention.py` stay **byte-untouched**. |
| **DSpark draft OOB** | [sglang#30555](https://github.com/sgl-project/sglang/issues/30555) fixed *correctly*: one `-1` on the draft worker's width in `triton_backend.py`. (The issue's own suggested ServerArgs pin double-corrects on current builds — don't use it.) Fixes OOB draft-KV reads **and** unblocks decode CUDA graphs. |
| **Conv-state commit off-by-one** *(novel)* | DSpark's `commit_lens` excludes the bonus token, but the sconv commit used it as a last-step index — state regressed and output degenerated into prompt-replay whenever accept > 1. Hits every non-symm-mem deployment, i.e. everything that isn't a B200-class single node. |
| **GB10 kernel limits** | MoE grouped-GEMM `num_stages` 3/4→2 (99 KB smem vs B200's 228 KB); Helion sm_121 configs seeded; `emit_packed_topk=False`; **marlin is the only numerically-correct NVFP4 MoE runner** here (cutlass silently miscomputes, trtllm hard-fails). |
| **Long-context speed** | The draft's context is pinned to its 64K adaptation, so declaring a huge context no longer craters acceptance. |

---

## Repo map

| Path | What |
|---|---|
| `scripts/nvfp4-kv-boot.sh` | **the champion launcher** (1M context, fp4 KV) |
| `scripts/dual-roce-preflight.sh` | E1 per-twin RDMA/GID/≥100-Gb/s fail-closed preflight |
| `scripts/read-only-spark-preflight.sh` | no-mutation two-node access/runtime/repo/model/RDMA readiness audit |
| `scripts/run-e1-dual-roce-ab.sh` | same-session, one-variable single-vs-dual HCA experiment |
| `scripts/run-e2-fp4-gemm-ab.sh` | same-session dense FP4 backend experiment with GPU numerical gate |
| `scripts/run-e3-block-sweep.sh` | block 5/6/7 sweep with native accept-by-position evidence |
| `scripts/run-e4-width1-mtp-ab.sh` | DSpark block 5 vs native width-1 MTP; E4 rejected at wall #24 |
| `scripts/run-e5-jit-cache-ab.sh` | completed eight-boot cache experiment; 6% warm saving was inconclusive, mounts remain off |
| `scripts/run-e6-memfrac-ab.sh` | MEMFRAC 0.85/0.68 C8/C16 experiment with host-memory evidence |
| `scripts/run-e8-decode-latency-ab.sh` | completed one-factor E8 sweep; adopted CDS=2, retained NCCL autotuning and KV splits 8 |
| `scripts/run-e8-cds2-adoption.sh` | exact-SHA champion runtime contract, T4, and all-task adoption gate |
| `scripts/earlyoom-preflight.sh` | read-only check for active earlyoom and readable OOM journals |
| `scripts/host-memory-guard.sh` | experiment-scoped 12-GiB guard that removes only the serving container |
| `scripts/locked-experiment-launch.sh` | fixes every non-factor serving knob for reproducible A/B arms |
| `scripts/champion-profile.sh` | carries an accepted E3 block/E4 speculator into every later experiment |
| `scripts/repo_fingerprint.py` | hashes tracked and untracked runnable bytes/modes across both nodes |
| `scripts/run-quality-gates.sh` | locked Q2/Q3 NIAH, full GSM8K, and post-tool regression suite |
| `scripts/fetch-gsm8k.sh` | immutable official GSM8K test-split fetch and checksum gate |
| `scripts/check_launch_render.py` | no-GPU dry-run validation of the champion command contract |
| `scripts/image-fingerprint.sh` | stable patched-payload identity check across independent bakes |
| `scripts/bake-image.sh` | builds `local/sglang-inkling:gb10[-kvquant]` from a digest-pinned upstream |
| `scripts/inkling-sglang-launch.sh` | underlying launcher; every knob is an env var |
| `patches/kv-quant/` | the KV-quantization implementation (6 files) |
| `patches/files/` + `patches/all-patches.diff` | base GB10 patches, byte-exact and as a reviewable diff |
| `docs/MEASUREMENT-PROTOCOL.md` | **read before benchmarking anything** |
| `docs/BUGS-AND-FIXES.md` | 25 walls: symptom → root cause → fix |
| `docs/KV-QUANT-IMPLEMENTATION-NOTES.md` | how the fp4 KV path works internally |
| `docs/DRAFT-FINETUNE-PLAN.md` | the remaining accept lever (+ A4Q applicability appendix) |
| `docs/EXPERIMENT-E7-FA4-PORT.md` | pinned SM120 donor and the pre-implementation SGLang port gates |
| `docs/MIAAI-BENCHMARK-CROSSWALK.md` | pinned MiaAI C1/C8 diagnostic targets and comparability boundary |
| `docs/ROADMAP.md` | done / blocked / why |
| `benchmarks/chat_bench.py` | chat-templated 4-seed × 8-rep harness for serving comparisons |
| `benchmarks/concurrency_bench.py` | chat-templated exact-n C1→C16 aggregate-throughput harness |
| `benchmarks/niah_eval.py` | tokenizer-measured 512K/1M needle checks at 10/50/90% depth |
| `benchmarks/gsm8k_eval.py` | resumable full 1,319-item GSM8K scorer against the pinned reference |
| `benchmarks/tool_call_regression.py` | structured call + post-tool parser-token leak regression suite |
| `benchmarks/accept_probe.py` | legacy raw `/generate` harness; not valid for serving claims |
| `benchmarks/tests_verify_nvfp4.py` | 20 bitwise-exactness tests for the fp4 kernels |
| `specs/001-oneshot-install/` | the same install as a **gated** task list for agents |
| [`Journal log/`](Journal%20log/) | long-form build write-ups — the story behind the fixes |

**Agents**: start at [`specs/001-oneshot-install/tasks.md`](specs/001-oneshot-install/tasks.md).

---

## Knobs

All are env vars on the launchers: `CTX` · `MEMFRAC` (0.85 default; 0.87 works, buys nothing
measurable) · `MAXREQ` · `BLOCK` (**5 is the accepted E3 champion**) · `KVD` (`fp4_mx_block16` default —
**not** `nvfp4`, which selects the flashinfer/trtllm recipe the triton lane cannot consume; read by
`nvfp4-kv-boot.sh`) · `CONTINUOUS_DECODE_STEPS` (**2 is the accepted E8 champion**) · `GRAPH_BS` ·
`IMAGE` · `EXTRA_ARGS` · `LOG` (server log path, default
`~/inkling-serve.log` — the verify grep reads this) · `PERSIST_JIT_CACHE=1` plus an absolute
`JIT_CACHE_ROOT` (E5 opt-in; default off after an inconclusive 6% warm-boot saving) · `INKLING_DRAFT_CTX_CAP` (default 65536; pins
the draft to its 64K adaptation so a huge declared context doesn't crater acceptance — baked patch #6).

Ranked runners after E3 also accept `CHAMPION_BLOCK` (default `5`) and
`CHAMPION_SPECULATOR=dspark|mtp-width1` (default `dspark`) so an accepted result becomes the next
experiment's baseline instead of silently reverting to the original recipe.

`SGLANG_RAGGED_VERIFY_MODE` stays **unset** unless you set `RAGGED=...` (the launcher only injects
it on request): `compact` crashes Inkling's sconv JIT, `static` costs accept, and `cap-accept` is
calibration-only and measures slower even calibrated (walls 14, 15, 17).

## Provenance

Upstream image `lmsysorg/sglang@sha256:fbea1a4e25b26660dbc2384a27ead8817e9b7670f257b5c3143e0450d14524d7`
(`dev-cu13-inkling-dspark`, 2026-07-30); all patches are against files inside it. Not affiliated with
LMSYS, Thinking Machines, RadixArk, or NVIDIA — an independent field port.
