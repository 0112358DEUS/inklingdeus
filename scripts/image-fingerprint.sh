#!/bin/bash
# Stable payload fingerprint for independently baked Inkling images.
set -euo pipefail

IMAGE=${IMAGE:-local/sglang-inkling:gb10-kvquant}
FILES=(
  /sgl-workspace/sglang/python/sglang/srt/models/inkling.py
  /sgl-workspace/sglang/python/sglang/srt/models/inkling_common/moe.py
  /sgl-workspace/sglang/python/sglang/srt/models/inkling_common/sconv.py
  /sgl-workspace/sglang/python/sglang/srt/layers/attention/triton_backend.py
  /sgl-workspace/sglang/python/sglang/kernels/ops/moe/inkling_moe.py
  /sgl-workspace/sglang/python/sglang/srt/speculative/draft_worker_common.py
  /sgl-workspace/sglang/python/sglang/srt/mem_cache/kv_quant_pools.py
  /sgl-workspace/sglang/python/sglang/kernels/ops/attention/kv_quant_attention.py
)

docker run --rm --entrypoint sha256sum "$IMAGE" "${FILES[@]}" | sha256sum | awk '{print $1}'
