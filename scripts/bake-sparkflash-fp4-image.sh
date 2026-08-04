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
    $SGL/srt/entrypoints/openai/serving_tokenize.py \
    $SGL/srt/function_call/function_call_parser.py \
    $SGL/srt/layers/quantization/kvfp4_tensor.py \
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

docker run --rm --gpus all --entrypoint python3 "$TAG" -c \
  'import torch; from sglang.kernels.ops.attention.flash_attn.cute import flash_attn_varlen_func; from sglang.srt.mem_cache.kv_quant_pools import MHATokenToKVPoolFP4Native; from sglang.srt.layers.quantization.kvfp4_tensor import FP4MXBlock16KVQuantizeUtil; from sglang.srt.entrypoints.openai.protocol import Function, Tool, ToolChoice, ToolChoiceFuncName; from sglang.srt.function_call.function_call_parser import FunctionCallParser; tool = Tool(type="function", function=Function(name="probe", parameters={"type":"object"})); choice = ToolChoice(type="function", function=ToolChoiceFuncName(name="probe")); kind, schema = FunctionCallParser([tool], "inkling").get_structure_constraint(choice, parallel_tool_calls=False); assert kind == "json_schema" and schema["minItems"] == schema["maxItems"] == 1; assert hasattr(FP4MXBlock16KVQuantizeUtil.batched_quantize, "_torchdynamo_orig_callable"); q, s = FP4MXBlock16KVQuantizeUtil.batched_quantize(torch.ones((2, 4, 128), dtype=torch.bfloat16, device="cuda")); torch.cuda.synchronize(); assert q.shape == (2, 4, 64) and s.shape == (2, 32); assert callable(flash_attn_varlen_func); print("SPARKFLASH FP4 IMAGE IMPORT PASS")'
echo "SPARKFLASH FP4 DEV IMAGE BAKED: $TAG"
