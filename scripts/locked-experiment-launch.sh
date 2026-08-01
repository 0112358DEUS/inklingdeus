#!/bin/bash
# Render/run an A/B arm with every serving knob locked except explicit positional factors.
set -euo pipefail

[ "$#" -eq 8 ] || {
  echo "usage: $0 RANK FP4_BACKEND BLOCK SPEC MEMFRAC PERSIST_JIT EXTRA_ARGS JIT_CACHE_ROOT" >&2
  exit 2
}
RANK=$1
FP4_BACKEND=$2
EXPERIMENT_BLOCK=$3
EXPERIMENT_SPEC=$4
EXPERIMENT_MEMFRAC=$5
EXPERIMENT_PERSIST_JIT=$6
EXPERIMENT_EXTRA_ARGS=$7
EXPERIMENT_JIT_CACHE_ROOT=$8

case "$RANK" in 0|1) ;; *) echo "RANK must be 0 or 1" >&2; exit 2 ;; esac
case "$EXPERIMENT_SPEC" in 0|1) ;; *) echo "SPEC must be 0 or 1" >&2; exit 2 ;; esac
case "$EXPERIMENT_PERSIST_JIT" in 0|1) ;; *) echo "PERSIST_JIT must be 0 or 1" >&2; exit 2 ;; esac
case "$EXPERIMENT_BLOCK" in *[!0-9]*|0|'') echo "BLOCK must be a positive integer" >&2; exit 2 ;; esac
case "$FP4_BACKEND" in
  flashinfer_trtllm|marlin) ;;
  *) echo "FP4_BACKEND must be flashinfer_trtllm or marlin" >&2; exit 2 ;;
esac
awk -v value="$EXPERIMENT_MEMFRAC" \
  'BEGIN {exit !(value ~ /^[0-9]+([.][0-9]+)?$/ && value > 0 && value < 1)}' || {
  echo "MEMFRAC must be a number strictly between 0 and 1" >&2
  exit 2
}
if [ "$EXPERIMENT_PERSIST_JIT" = 1 ]; then
  case "$EXPERIMENT_JIT_CACHE_ROOT" in
    /*) ;;
    *) echo "persisted JIT experiments require an absolute JIT cache root" >&2; exit 2 ;;
  esac
fi

export ATTN=triton
export MOE=marlin
export FP4GEMM="$FP4_BACKEND"
export MEMFRAC="$EXPERIMENT_MEMFRAC"
export CTX=1048576
export SPEC="$EXPERIMENT_SPEC"
export BLOCK="$EXPERIMENT_BLOCK"
export GRAPHS=1
export GRAPH_BS="1 2 3 4 5 6 7 8 10 12 14 16"
export RAGGED=
export MAXREQ=16
export PAGE=1
export KVD=fp4_mx_block16
export EXTRA_ARGS="$EXPERIMENT_EXTRA_ARGS"
export PERSIST_JIT_CACHE="$EXPERIMENT_PERSIST_JIT"
export JIT_CACHE_ROOT="$EXPERIMENT_JIT_CACHE_ROOT"
export INKLING_NOOP_CONV_COMMIT=0

exec "$(dirname "$0")/nvfp4-kv-boot.sh" "$RANK"
