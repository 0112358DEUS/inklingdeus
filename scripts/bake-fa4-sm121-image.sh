#!/bin/bash
# Build a separate BF16-KV E7 development image; never overwrite the champion tag.
set -euo pipefail

SOURCE_IMAGE=${SOURCE_IMAGE:-local/sglang-inkling:gb10-kvquant}
TAG=${TAG:-local/sglang-inkling:fa4-sm121-dev}
REPO_DIR=$(cd "$(dirname "$0")/.." && pwd)
VENDOR="$REPO_DIR/third_party/inkling_sm120_fa4"
TARGET=/sgl-workspace/sglang/python/sglang/kernels/ops/attention/flash_attn/cute
CONTAINER=inkling-fa4-sm121-bake

case "$TAG" in
  local/sglang-inkling:gb10-kvquant)
    echo "refusing to overwrite the champion image tag" >&2
    exit 2
    ;;
esac

python3 "$REPO_DIR/scripts/verify-fa4-vendor.py"
docker image inspect "$SOURCE_IMAGE" >/dev/null
docker rm -f "$CONTAINER" >/dev/null 2>&1 || true
docker create --name "$CONTAINER" --entrypoint true "$SOURCE_IMAGE" >/dev/null
docker cp "$VENDOR/." "$CONTAINER:$TARGET"
docker commit "$CONTAINER" "$TAG" >/dev/null
docker rm "$CONTAINER" >/dev/null

docker run --rm --entrypoint python3 "$TAG" -c \
  'from sglang.kernels.ops.attention.flash_attn.cute import __version__, flash_attn_varlen_func; assert callable(flash_attn_varlen_func); print(f"FA4 image import PASS version={__version__}")'
echo "FA4 DEV IMAGE BAKED: $TAG"
