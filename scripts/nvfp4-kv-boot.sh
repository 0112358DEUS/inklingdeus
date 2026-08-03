#!/bin/bash
# CHAMPION LAUNCHER: Inkling-Small + DSpark + NVFP4 KV @ 1,048,576-token context.
#   pool 1,331,001 tokens in the E8 adoption run (exceeds the context) | lossless | task-dependent:
#   26.23 +/- 0.32 tok/s / accept 2.138 +/- 0.026 on open-ended work (n=32)
#   (see README "Throughput depends heavily on workload"). CTX=65536 for the short-context
#   profile (same speed, 1.1M pool).
#
# Requires the KV-quant image: KVQUANT=1 ./scripts/bake-image.sh  (see docs/KV-QUANT-*.md)
# Usage: ./nvfp4-kv-boot.sh <rank 0|1>   — rank 1 (worker) FIRST, then rank 0.
#
# NOTE the dtype: `--kv-cache-dtype nvfp4` selects the flashinfer/trtllm recipe (e4m3 block
# scales + checkpoint global scales) which the triton lane cannot consume. The triton-compatible
# fp4 recipe is `fp4_mx_block16` (packed e2m1 + uint8 biased-exponent scale per 16 elements).
# Capacity is identical: 0.5625 bytes/element.
set -euo pipefail
RANK=${1:?rank 0|1}
export IMAGE="${IMAGE:-local/sglang-inkling:gb10-kvquant}"
export INKLING_TORCH_CONV_COMMIT=1 INKLING_COMMIT_STEP_BIAS=1
export MOE="${MOE:-marlin}" GRAPHS="${GRAPHS:-1}" MEMFRAC="${MEMFRAC:-0.85}" CTX="${CTX:-1048576}"
export MAXREQ="${MAXREQ:-16}" BLOCK="${BLOCK:-5}"
export CONTINUOUS_DECODE_STEPS="${CONTINUOUS_DECODE_STEPS:-2}"
# (--triton-attention-reduce-in-fp32 is already always passed by the underlying launcher.)
export EXTRA_ARGS="--kv-cache-dtype ${KVD:-fp4_mx_block16} ${EXTRA_ARGS:-}"
exec "$(dirname "$0")/inkling-sglang-launch.sh" "$RANK"
