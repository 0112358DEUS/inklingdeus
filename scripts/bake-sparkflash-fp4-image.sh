#!/bin/bash
# Bake the separate Stage 5A FA4 + page-128 fp4_mx_block16 development image.
set -euo pipefail

SOURCE_IMAGE=${SOURCE_IMAGE:-local/sglang-inkling:gb10-kvquant}
TAG=${TAG:-local/sglang-inkling:sparkflash-fp4-dev}
REPO_DIR=$(cd "$(dirname "$0")/.." && pwd)
VENDOR="$REPO_DIR/third_party/inkling_sm120_fa4"
OVERLAY_SCRIPT="$REPO_DIR/scripts/apply-sparkflash-fp4-overlay.py"
SGL=/sgl-workspace/sglang/python/sglang
TARGET=$SGL/kernels/ops/attention/flash_attn/cute
CONTAINER=inkling-sparkflash-fp4-bake

case "$TAG" in
  local/sglang-inkling:gb10-kvquant|local/sglang-inkling:fa4-sm121-dev)
    echo "refusing to overwrite an immutable input image tag" >&2
    exit 2
    ;;
esac

python3 "$REPO_DIR/scripts/verify-fa4-vendor.py"
docker image inspect "$SOURCE_IMAGE" >/dev/null
docker rm -f "$CONTAINER" >/dev/null 2>&1 || true
docker create --name "$CONTAINER" --entrypoint bash "$SOURCE_IMAGE" -lc "
  set -euo pipefail
  python3 /tmp/apply-sparkflash-fp4-overlay.py $SGL
  python3 -m py_compile \
    $SGL/srt/layers/attention/flashattention_backend.py \
    $SGL/kernels/ops/attention/flash_attention.py \
    $SGL/kernels/ops/attention/flash_attention_v4.py \
    $SGL/srt/server_args.py \
    $SGL/srt/mem_cache/kv_quant_pools.py \
    $SGL/srt/mem_cache/kv_cache_configurator.py
" >/dev/null

docker cp "$VENDOR/." "$CONTAINER:$TARGET"
docker cp "$OVERLAY_SCRIPT" "$CONTAINER:/tmp/apply-sparkflash-fp4-overlay.py"
docker cp "$REPO_DIR/patches/kv-quant/srt/mem_cache/kv_quant_pools.py" \
  "$CONTAINER:$SGL/srt/mem_cache/kv_quant_pools.py"
docker cp "$REPO_DIR/patches/kv-quant/srt/mem_cache/kv_cache_configurator.py" \
  "$CONTAINER:$SGL/srt/mem_cache/kv_cache_configurator.py"

docker start -a "$CONTAINER"
docker commit "$CONTAINER" "$TAG" >/dev/null
docker rm "$CONTAINER" >/dev/null

docker run --rm --entrypoint python3 "$TAG" -c \
  'from sglang.kernels.ops.attention.flash_attn.cute import flash_attn_varlen_func; from sglang.srt.mem_cache.kv_quant_pools import MHATokenToKVPoolFP4Native; assert callable(flash_attn_varlen_func); print("SPARKFLASH FP4 IMAGE IMPORT PASS")'
echo "SPARKFLASH FP4 DEV IMAGE BAKED: $TAG"
