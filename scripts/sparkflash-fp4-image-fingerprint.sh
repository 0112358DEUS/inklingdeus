#!/bin/bash
# Stable payload fingerprint for a separately baked SparkFlash FP4 image.
set -euo pipefail

IMAGE=${IMAGE:-local/sglang-inkling:sparkflash-fp4-dev}

docker image inspect "$IMAGE" >/dev/null
docker run --rm --entrypoint sh "$IMAGE" -c '
set -eu
sha256sum \
  /sgl-workspace/sglang/python/sglang/srt/layers/attention/flashattention_backend.py \
  /sgl-workspace/sglang/python/sglang/kernels/ops/attention/flash_attention.py \
  /sgl-workspace/sglang/python/sglang/kernels/ops/attention/flash_attention_v4.py \
  /sgl-workspace/sglang/python/sglang/srt/server_args.py \
  /sgl-workspace/sglang/python/sglang/srt/entrypoints/openai/serving_tokenize.py \
  /sgl-workspace/sglang/python/sglang/srt/function_call/function_call_parser.py \
  /sgl-workspace/sglang/python/sglang/srt/layers/quantization/kvfp4_tensor.py \
  /sgl-workspace/sglang/python/sglang/srt/mem_cache/kv_quant_pools.py \
  /sgl-workspace/sglang/python/sglang/srt/mem_cache/kv_cache_configurator.py
find /sgl-workspace/sglang/python/sglang/kernels/ops/attention/flash_attn/cute \
  -type f -print0 | sort -z | xargs -0 sha256sum
' | sha256sum | awk '{print $1}'
