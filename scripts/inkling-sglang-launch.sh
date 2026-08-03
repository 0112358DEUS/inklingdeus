#!/bin/bash
# Inkling-Small-NVFP4 + RadixArk DSpark draft — TP=2 across two DGX Sparks (GB10, sm_121a)
#
# Usage:  ./inkling-sglang-launch.sh <rank 0|1>     (start rank 1 on the worker FIRST, then rank 0 on the head)
#
# DEFAULTS = the measured champion (see README; numbers are mean +/- se over 32 samples,
# NOT single runs): marlin MoE, triton attention + fp32 reduction, page-size 1, DSpark block 5,
# 2 continuous decode steps, decode CUDA graphs, mem-fraction 0.85, 64K ctx, conv-commit fix ON,
# draft-context cap ON.
#
# Measure ANY change with benchmarks/chat_bench.py — this stack is nondeterministic at temp 0
# and single-run comparisons are worthless (docs/MEASUREMENT-PROTOCOL.md).
#
# Site knobs (env): MASTER_IP IF HCA GID MODELS IMAGE SGLANG_PORT
# Tuning knobs (env): ATTN MOE FP4GEMM MEMFRAC CTX SPEC GRAPHS GRAPH_BS RAGGED BLOCK MAXREQ PAGE
# CONTINUOUS_DECODE_STEPS EXTRA_ARGS INKLING_SHEARED_BIAS
# Boot-cache knobs (env): PERSIST_JIT_CACHE JIT_CACHE_ROOT
# Decode-latency knobs (env, E8 experiment only): NCCL_ALGO NCCL_PROTO — unset preserves NCCL's
# autotuned defaults (the measured champion); set only inside a same-session A/B.
# Validation knob: DRY_RUN=1 renders the exact docker command without requiring Docker or weights.
set -euo pipefail
RANK=${1:?rank 0|1}
case "$RANK" in 0|1) ;; *) echo "rank must be 0 or 1, got: $RANK" >&2; exit 2 ;; esac

# ---- site config (edit or override) ----
MASTER_IP=${MASTER_IP:-10.100.20.1}     # rank0's IP on the 200G link
IF=${IF:-enp1s0f0np0}                   # NIC carrying the inter-node link (both nodes)
HCA=${HCA:-rocep1s0f0}                  # RDMA device for that NIC (ibv_devices)
GID=${GID:-3}                           # RoCEv2 IPv4 GID index (show_gids)
MODELS=${MODELS:-/mnt/models-7552/inkling}  # dir with inkling-small-nvfp4/ + dspark-draft/
IMAGE=${IMAGE:-local/sglang-inkling:gb10}   # baked by scripts/bake-image.sh
LOG=${LOG:-$HOME/inkling-serve.log}         # server log (the README verify step greps this)

# The defaults above are one site's values. Say so loudly when they are in use, and fail
# early on the mistakes that otherwise surface as a silent rendezvous hang.
for v in MASTER_IP IF HCA MODELS; do
  if ! env | grep -q "^$v="; then
    echo "WARN: $v not set — using built-in default '${!v}' (another site's value)" >&2
  fi
done
if [ "${DRY_RUN:-0}" != 1 ]; then
  [ -d "$MODELS" ] || { echo "ERROR: MODELS dir '$MODELS' does not exist on this node" >&2; exit 2; }
  [ -d "$MODELS/inkling-small-nvfp4" ] || echo "WARN: '$MODELS/inkling-small-nvfp4' not found — weights missing?" >&2
  docker image inspect "$IMAGE" >/dev/null 2>&1 || {
    echo "ERROR: image '$IMAGE' not found on this node — build it with: ./scripts/bake-image.sh (or KVQUANT=1 for the fp4-KV image), or docker pull the prebuilt one (README)" >&2
    exit 2
  }
fi

# ---- champion defaults ----
PORT=${SGLANG_PORT:-30000}
ATTN=${ATTN:-triton}                    # Inkling asserts fa4|triton; fa4 is sm_100-only -> triton
MOE=${MOE:-marlin}                      # ONLY numerically-correct NVFP4 MoE runner on sm_121
FP4GEMM=${FP4GEMM:-flashinfer_trtllm}   # dense FP4 GEMMs are fine on sm_121
MEMFRAC=${MEMFRAC:-0.85}      # 0.87 works but buys nothing measurable; 0.85 is the fleet-safe ceiling
CTX=${CTX:-65536}          # bf16-KV default. With bf16 the pool shrinks sharply as ctx grows
                           # (SWA/mamba reserves scale with ctx), so 64K is the sane bf16 setting.
                           # For 1M use scripts/nvfp4-kv-boot.sh — with fp4 KV the pool barely moves
                           # (1,104,683 @64K vs 1,082,627 @1M, ~2%), so 1M is essentially free there.
SPEC=${SPEC:-1}
GRAPHS=${GRAPHS:-1}
PAGE=${PAGE:-1}                         # page 128 corrupts the triton verify path
CONTINUOUS_DECODE_STEPS=${CONTINUOUS_DECODE_STEPS:-2}  # E8 winner: +0.645 tok/s, latency -0.153 s
case "$CONTINUOUS_DECODE_STEPS" in
  *[!0-9]*|0|'') echo "ERROR: CONTINUOUS_DECODE_STEPS must be a positive integer" >&2; exit 2 ;;
esac
export INKLING_TORCH_CONV_COMMIT=${INKLING_TORCH_CONV_COMMIT:-1}   # conv-commit fix (see docs/BUGS-AND-FIXES.md)
export INKLING_COMMIT_STEP_BIAS=${INKLING_COMMIT_STEP_BIAS:-1}

# SGLANG_RAGGED_VERIFY_MODE must stay UNSET unless explicitly requested: `compact` crashes
# Inkling's sconv JIT at first request (wall 14), `static`/`cap-accept` are calibration-only
# modes that cost accept or speed (walls 15, 17). Only inject the env when RAGGED is non-empty.
RAGGED_ENV=()
if [ -n "${RAGGED:-}" ]; then
  RAGGED_ENV+=(-e SGLANG_RAGGED_VERIFY_MODE="$RAGGED")
fi

# E7 opt-in: choose Inkling's FA4 relative-bias implementation. Unset preserves the image default
# and therefore the champion contract. Zero selects the model-guarded score_mod + aux-tensor path;
# one selects the dedicated SM100 sheared-bias path.
INKLING_BIAS_ENV=()
case "${INKLING_SHEARED_BIAS:-}" in
  '') ;;
  0|1)
    INKLING_BIAS_ENV+=(
      -e SGLANG_OPT_USE_INKLING_SHEARED_BIAS="$INKLING_SHEARED_BIAS"
    )
    ;;
  *) echo "ERROR: INKLING_SHEARED_BIAS must be unset, 0, or 1" >&2; exit 2 ;;
esac

# E8 opt-in: NCCL collective tuning for the per-decode-step TP2 all-reduce. Injected only when
# set, so the default launch keeps NCCL's own protocol/algorithm selection (the measured champion).
NCCL_TUNE_ENV=()
if [ -n "${NCCL_ALGO:-}" ]; then
  NCCL_TUNE_ENV+=(-e NCCL_ALGO="$NCCL_ALGO")
fi
if [ -n "${NCCL_PROTO:-}" ]; then
  NCCL_TUNE_ENV+=(-e NCCL_PROTO="$NCCL_PROTO")
fi

EXTRA=()
if [ "$SPEC" = 1 ]; then
  EXTRA+=(--speculative-algorithm DSPARK
          --speculative-draft-model-path /models/dspark-draft
          --speculative-draft-model-quantization unquant
          --speculative-dspark-block-size "${BLOCK:-5}")
fi
if [ "$GRAPHS" = 1 ]; then
  read -r -a GRAPH_BS_VALUES <<<"${GRAPH_BS:-1 2 3 4 5 6 7 8 10 12 14 16}"
  EXTRA+=(--cuda-graph-bs "${GRAPH_BS_VALUES[@]}" --disable-piecewise-cuda-graph --disable-prefill-cuda-graph)
else
  EXTRA+=(--disable-cuda-graph --disable-prefill-cuda-graph)
fi
read -r -a USER_EXTRA <<<"${EXTRA_ARGS:-}"

# E5 opt-in: persist only compiler/JIT artifacts. Keeping the default off preserves the measured
# champion until the cold/prime/warm hardware experiment clears T4 and the boot-time threshold.
CACHE_MOUNTS=()
case "${PERSIST_JIT_CACHE:-0}" in
  0) ;;
  1)
    JIT_CACHE_ROOT=${JIT_CACHE_ROOT:-$HOME/.cache/inkling-sglang-jit}
    case "$JIT_CACHE_ROOT" in
      /*) ;;
      *) echo "ERROR: JIT_CACHE_ROOT must be an absolute host path" >&2; exit 2 ;;
    esac
    CACHE_DIRS=(triton flashinfer sglang torch_extensions torchinductor cuda)
    if [ "${DRY_RUN:-0}" != 1 ]; then
      for cache_dir in "${CACHE_DIRS[@]}"; do
        mkdir -p "$JIT_CACHE_ROOT/$cache_dir"
      done
    fi
    CACHE_MOUNTS+=(
      -v "$JIT_CACHE_ROOT/triton:/root/.triton"
      -v "$JIT_CACHE_ROOT/flashinfer:/root/.cache/flashinfer"
      -v "$JIT_CACHE_ROOT/sglang:/root/.cache/sglang"
      -v "$JIT_CACHE_ROOT/torch_extensions:/root/.cache/torch_extensions"
      -v "$JIT_CACHE_ROOT/torchinductor:/tmp/torchinductor_root"
      -v "$JIT_CACHE_ROOT/cuda:/root/.nv/ComputeCache"
    )
    ;;
  *) echo "ERROR: PERSIST_JIT_CACHE must be 0 or 1" >&2; exit 2 ;;
esac

DOCKER_CMD=(
  docker run --name inkling-sglang --rm --gpus all --network host --ipc host
  --shm-size 16g --device /dev/infiniband --cap-add IPC_LOCK
  --ulimit memlock=-1 --ulimit stack=67108864
  -v "$MODELS:/models:ro"
  ${CACHE_MOUNTS[@]+"${CACHE_MOUNTS[@]}"}
  -e SGLANG_ENABLE_UNIFIED_RADIX_TREE=1
  ${RAGGED_ENV[@]+"${RAGGED_ENV[@]}"}
  ${INKLING_BIAS_ENV[@]+"${INKLING_BIAS_ENV[@]}"}
  -e NCCL_IB_HCA="$HCA" -e NCCL_IB_GID_INDEX="$GID"
  -e NCCL_SOCKET_IFNAME="$IF" -e GLOO_SOCKET_IFNAME="$IF" -e TP_SOCKET_IFNAME="$IF"
  -e NCCL_NET=IB -e NCCL_IB_DISABLE=0 -e NCCL_NET_PLUGIN=none
  -e NCCL_CUMEM_ENABLE=0 -e NCCL_NVLS_ENABLE=0 -e NCCL_CROSS_NIC=0
  -e NCCL_IGNORE_CPU_AFFINITY=1 -e NCCL_DEBUG=WARN
  ${NCCL_TUNE_ENV[@]+"${NCCL_TUNE_ENV[@]}"}
  -e TORCH_CUDA_ARCH_LIST=12.1a -e FLASHINFER_CUDA_ARCH_LIST=12.1a
  -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1
  -e INKLING_TORCH_CONV_COMMIT -e INKLING_COMMIT_STEP_BIAS
  -e INKLING_NOOP_CONV_COMMIT="${INKLING_NOOP_CONV_COMMIT:-0}"
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
  --entrypoint python3 "$IMAGE"
  -m sglang.launch_server
  --model-path /models/inkling-small-nvfp4 --trust-remote-code
  --served-model-name inkling-small
  --host 0.0.0.0 --port "$PORT"
  --tp-size 2 --nnodes 2 --node-rank "$RANK" --dist-init-addr "$MASTER_IP:25000"
  --context-length "$CTX"
  --quantization modelopt_fp4
  --attention-backend "$ATTN" --page-size "$PAGE"
  --triton-attention-reduce-in-fp32
  --fp4-gemm-backend "$FP4GEMM" --moe-runner-backend "$MOE"
  --mamba-radix-cache-strategy extra_buffer
  --mem-fraction-static "$MEMFRAC" --swa-full-tokens-ratio 0.1 --mamba-full-memory-ratio 0.1
  --max-running-requests "${MAXREQ:-16}"
  --chunked-prefill-size 8192
  --reasoning-parser inkling --tool-call-parser inkling
  --skip-server-warmup --disable-flashinfer-autotune
  --stream-interval 32
  --num-continuous-decode-steps "$CONTINUOUS_DECODE_STEPS"
  "${EXTRA[@]}" ${USER_EXTRA[@]+"${USER_EXTRA[@]}"}
)

if [ "${DRY_RUN:-0}" = 1 ]; then
  printf '%q ' "${DOCKER_CMD[@]}"
  printf '\n'
  exit 0
fi

docker rm -f inkling-sglang 2>/dev/null || true
# Foreground run, teed to $LOG so the README's pool-verification grep works and the log
# survives the container (--rm). Use `docker logs -f inkling-sglang` from another shell.
"${DOCKER_CMD[@]}" 2>&1 | tee "$LOG"
