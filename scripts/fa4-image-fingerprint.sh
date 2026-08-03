#!/bin/bash
# Stable payload fingerprint for the separate E7 FA4 development image.
set -euo pipefail

IMAGE=${IMAGE:-local/sglang-inkling:fa4-sm121-dev}

docker image inspect "$IMAGE" >/dev/null
docker run --rm --entrypoint sh "$IMAGE" -c '
set -eu
sha256sum \
  /sgl-workspace/sglang/python/sglang/srt/layers/attention/flashattention_backend.py \
  /sgl-workspace/sglang/python/sglang/kernels/ops/attention/flash_attention_v4.py
find /sgl-workspace/sglang/python/sglang/kernels/ops/attention/flash_attn/cute \
  -type f -print0 | sort -z | xargs -0 sha256sum
' | sha256sum | awk '{print $1}'
