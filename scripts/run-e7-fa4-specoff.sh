#!/bin/bash
# E7 stage 2: two-node SM121 FA4 serving correctness with BF16 KV and speculation disabled.
set -euo pipefail

WORKER_SSH=${WORKER_SSH:?set passwordless worker SSH}
WORKER_REPO=${WORKER_REPO:?set the absolute inklingdeus repo path on the worker}
MASTER_IP=${MASTER_IP:?set the head IP on the primary RoCE link}
IF=${IF:?set the primary link netdev}
HCA=${HCA:?set the unchanged NCCL HCA for both nodes}
MODELS=${MODELS:?set the identical model directory path present on both nodes}
GID=${GID:-3}
IMAGE=${IMAGE:-local/sglang-inkling:fa4-sm121-dev}
CHAMPION_IMAGE=${CHAMPION_IMAGE:-local/sglang-inkling:gb10-kvquant}
RESULT_DIR=${RESULT_DIR:-artifacts/e7-fa4-specoff}
READY_TIMEOUT=${READY_TIMEOUT:-900}
REL_BIAS_MODE=${REL_BIAS_MODE:-sheared}
REPO_DIR=$(cd "$(dirname "$0")/.." && pwd)
LABEL=e7-fa4-bf16-specoff

case "$REL_BIAS_MODE" in
  sheared) INKLING_SHEARED_BIAS= ;;
  scoremod) INKLING_SHEARED_BIAS=0 ;;
  *) echo "REL_BIAS_MODE must be sheared or scoremod" >&2; exit 2 ;;
esac

mkdir -p "$RESULT_DIR"

stop_arm() {
  docker rm -f inkling-sglang >/dev/null 2>&1 || true
  ssh -o BatchMode=yes "$WORKER_SSH" docker rm -f inkling-sglang >/dev/null 2>&1 || true
}
trap stop_arm EXIT

wait_stopped() {
  local deadline=$((SECONDS + 60))
  while [ "$SECONDS" -lt "$deadline" ]; do
    if ! docker inspect inkling-sglang >/dev/null 2>&1 \
      && ! ssh -o BatchMode=yes "$WORKER_SSH" docker inspect inkling-sglang >/dev/null 2>&1; then
      return 0
    fi
    sleep 2
  done
  echo "existing serving containers did not stop within 60 seconds" >&2
  return 1
}

verify_reproducibility() {
  local local_sha worker_sha local_payload worker_payload
  local champion_payload worker_champion_payload local_fa4 worker_fa4
  local_sha=$(git -C "$REPO_DIR" rev-parse HEAD)
  worker_sha=$(ssh -o BatchMode=yes "$WORKER_SSH" \
    "git -C $(printf '%q' "$WORKER_REPO") rev-parse HEAD")
  [ "$local_sha" = "$worker_sha" ] || {
    echo "repo SHA mismatch: head=$local_sha worker=$worker_sha" >&2
    exit 2
  }
  local_payload=$(python3 "$REPO_DIR/scripts/repo_fingerprint.py" --digest-only)
  worker_payload=$(ssh -o BatchMode=yes "$WORKER_SSH" \
    "python3 $(printf '%q' "$WORKER_REPO/scripts/repo_fingerprint.py") --root $(printf '%q' "$WORKER_REPO") --digest-only")
  [ "$local_payload" = "$worker_payload" ] || {
    echo "repo payload mismatch: head=$local_payload worker=$worker_payload" >&2
    exit 2
  }
  "$REPO_DIR/scripts/verify-fa4-vendor.py" >/dev/null
  ssh -o BatchMode=yes "$WORKER_SSH" \
    "$(printf '%q' "$WORKER_REPO/scripts/verify-fa4-vendor.py")" >/dev/null
  local_fa4=$(IMAGE="$IMAGE" "$REPO_DIR/scripts/fa4-image-fingerprint.sh")
  worker_fa4=$(ssh -o BatchMode=yes "$WORKER_SSH" \
    "IMAGE=$(printf '%q' "$IMAGE") $(printf '%q' "$WORKER_REPO/scripts/fa4-image-fingerprint.sh")")
  [ "$local_fa4" = "$worker_fa4" ] || {
    echo "FA4 image payload mismatch: head=$local_fa4 worker=$worker_fa4" >&2
    exit 2
  }
  champion_payload=$(IMAGE="$CHAMPION_IMAGE" "$REPO_DIR/scripts/image-fingerprint.sh")
  worker_champion_payload=$(ssh -o BatchMode=yes "$WORKER_SSH" \
    "IMAGE=$(printf '%q' "$CHAMPION_IMAGE") $(printf '%q' "$WORKER_REPO/scripts/image-fingerprint.sh")")
  [ "$champion_payload" = "$worker_champion_payload" ] || {
    echo "champion payload mismatch between nodes" >&2
    exit 2
  }
  {
    printf 'repo_sha=%s\nrepo_payload=%s\n' "$local_sha" "$local_payload"
    printf 'fa4_payload=%s\nchampion_payload=%s\n' "$local_fa4" "$champion_payload"
    printf 'rel_bias_mode=%s\n' "$REL_BIAS_MODE"
    printf 'head_champion_id='
    docker image inspect "$CHAMPION_IMAGE" --format '{{.Id}}'
    printf 'worker_champion_id='
    ssh -o BatchMode=yes "$WORKER_SSH" \
      docker image inspect "$CHAMPION_IMAGE" --format '{{.Id}}'
    printf 'head_fa4_id='
    docker image inspect "$IMAGE" --format '{{.Id}}'
    printf 'worker_fa4_id='
    ssh -o BatchMode=yes "$WORKER_SSH" \
      docker image inspect "$IMAGE" --format '{{.Id}}'
    echo "reproducibility PASS"
  } | tee "$RESULT_DIR/identity.txt"
}

run_numerics() {
  docker run --rm --gpus all -v "$REPO_DIR:/repo:ro" \
    --entrypoint python3 "$IMAGE" /repo/benchmarks/fa4_sm121_paged_probe.py \
    >"$RESULT_DIR/control1-paged-bf16-numerics.log" 2>&1
  ssh -o BatchMode=yes "$WORKER_SSH" \
    "docker run --rm --gpus all -v $(printf '%q' "$WORKER_REPO:/repo:ro") --entrypoint python3 $(printf '%q' "$IMAGE") /repo/benchmarks/fa4_sm121_paged_probe.py" \
    >"$RESULT_DIR/control2-paged-bf16-numerics.log" 2>&1
  grep -q 'FA4 PAGED BF16 NUMERICS PASS cases=14' \
    "$RESULT_DIR/control1-paged-bf16-numerics.log"
  grep -q 'FA4 PAGED BF16 NUMERICS PASS cases=14' \
    "$RESULT_DIR/control2-paged-bf16-numerics.log"
  if [ "$REL_BIAS_MODE" = scoremod ]; then
    docker run --rm --gpus all -v "$REPO_DIR:/repo:ro" \
      --entrypoint python3 "$IMAGE" /repo/benchmarks/fa4_sm121_rel_bias_probe.py \
      >"$RESULT_DIR/control1-rel-bias-numerics.log" 2>&1
    ssh -o BatchMode=yes "$WORKER_SSH" \
      "docker run --rm --gpus all -v $(printf '%q' "$WORKER_REPO:/repo:ro") --entrypoint python3 $(printf '%q' "$IMAGE") /repo/benchmarks/fa4_sm121_rel_bias_probe.py" \
      >"$RESULT_DIR/control2-rel-bias-numerics.log" 2>&1
    grep -q 'FA4 REL BIAS NUMERICS PASS cases=14' \
      "$RESULT_DIR/control1-rel-bias-numerics.log"
    grep -q 'FA4 REL BIAS NUMERICS PASS cases=14' \
      "$RESULT_DIR/control2-rel-bias-numerics.log"
  fi
}

wait_ready() {
  local deadline=$((SECONDS + READY_TIMEOUT))
  local seen_container=0
  while [ "$SECONDS" -lt "$deadline" ]; do
    if curl -fsS http://127.0.0.1:30000/health >/dev/null 2>&1; then
      return 0
    fi
    if docker inspect inkling-sglang >/dev/null 2>&1; then
      seen_container=1
      if [ "$(docker inspect -f '{{.State.Running}}' inkling-sglang 2>/dev/null)" != true ]; then
        echo "head server container exited before readiness" >&2
        return 1
      fi
    elif [ "$seen_container" = 1 ]; then
      echo "head server container disappeared before readiness" >&2
      return 1
    fi
    sleep 5
  done
  echo "server did not become healthy within $READY_TIMEOUT seconds" >&2
  return 1
}

record_boot_failure() {
  {
    echo "FAIL: FA4 BF16 spec-off server did not become healthy; no serving claim"
    echo "head log tail:"
    tail -n 160 "$RESULT_DIR/$LABEL-head.log" 2>/dev/null || true
    echo "worker log tail:"
    ssh -o BatchMode=yes "$WORKER_SSH" \
      "tail -n 160 $(printf '%q' "$WORKER_REPO/$RESULT_DIR/$LABEL-worker.log")" \
      2>/dev/null || true
  } | tee "$RESULT_DIR/boot-failure.txt" >&2
}

capture_runtime_contract() {
  docker inspect inkling-sglang >"$RESULT_DIR/$LABEL-head-inspect.json"
  ssh -o BatchMode=yes "$WORKER_SSH" docker inspect inkling-sglang \
    >"$RESULT_DIR/$LABEL-worker-inspect.json"
  python3 - "$IMAGE" "$REL_BIAS_MODE" "$RESULT_DIR/$LABEL-head-inspect.json" \
    "$RESULT_DIR/$LABEL-worker-inspect.json" <<'PY'
import json
import sys
from pathlib import Path

expected_image = sys.argv[1]
rel_bias_mode = sys.argv[2]
required_pairs = {
    "--attention-backend": "fa4",
    "--page-size": "128",
    "--context-length": "65536",
    "--quantization": "modelopt_fp4",
    "--moe-runner-backend": "marlin",
    "--fp4-gemm-backend": "flashinfer_trtllm",
    "--mem-fraction-static": "0.85",
    "--num-continuous-decode-steps": "2",
}
required_flags = {
    "--disable-piecewise-cuda-graph",
    "--disable-prefill-cuda-graph",
}
for path_text in sys.argv[3:]:
    path = Path(path_text)
    payload = json.loads(path.read_text(encoding="utf-8"))[0]
    command = payload["Config"]["Cmd"]
    if payload["Config"]["Image"] != expected_image:
        raise SystemExit(
            f"{path}: image={payload['Config']['Image']!r}, expected={expected_image!r}"
        )
    for flag, expected in required_pairs.items():
        if command.count(flag) != 1:
            raise SystemExit(f"{path}: expected exactly one {flag}")
        actual = command[command.index(flag) + 1]
        if actual != expected:
            raise SystemExit(f"{path}: {flag}={actual!r}, expected={expected!r}")
    missing = required_flags.difference(command)
    if missing:
        raise SystemExit(f"{path}: missing {sorted(missing)}")
    forbidden = {
        "--kv-cache-dtype",
        "--speculative-algorithm",
        "--speculative-draft-model-path",
        "--speculative-draft-model-quantization",
        "--speculative-dspark-block-size",
    }
    present = forbidden.intersection(command)
    if present:
        raise SystemExit(f"{path}: forbidden spec-off flags: {sorted(present)}")
    env = set(payload["Config"]["Env"])
    bias_env = {
        value for value in env if value.startswith("SGLANG_OPT_USE_INKLING_SHEARED_BIAS=")
    }
    expected_bias_env = (
        {"SGLANG_OPT_USE_INKLING_SHEARED_BIAS=0"}
        if rel_bias_mode == "scoremod"
        else set()
    )
    if bias_env != expected_bias_env:
        raise SystemExit(
            f"{path}: relative-bias env={sorted(bias_env)}, expected={sorted(expected_bias_env)}"
        )
print(
    "E7 runtime contract PASS attention=fa4 page=128 kv=bf16 "
    f"spec=off rel_bias={rel_bias_mode}"
)
PY
}

lossless_gate() {
  local output_path=$1
  python3 - "$output_path" <<'PY'
import hashlib
import json
import sys
import urllib.request
from pathlib import Path

expected = " Paris. The capital of Germany is Berlin. The capital of"
body = {
    "model": "inkling-small",
    "prompt": "The capital of France is",
    "max_tokens": 12,
    "temperature": 0,
}
request = urllib.request.Request(
    "http://127.0.0.1:30000/v1/completions",
    data=json.dumps(body).encode(),
    headers={"Content-Type": "application/json"},
)
with urllib.request.urlopen(request, timeout=300) as response:
    actual = json.load(response)["choices"][0]["text"]
record = {
    "actual": actual,
    "actual_sha256": hashlib.sha256(actual.encode()).hexdigest(),
    "expected": expected,
    "expected_sha256": hashlib.sha256(expected.encode()).hexdigest(),
    "match": actual == expected,
}
Path(sys.argv[1]).write_text(json.dumps(record, sort_keys=True) + "\n", encoding="utf-8")
if actual != expected:
    raise SystemExit(f"T4 LOSSLESS FAIL\nexpected={expected!r}\nactual={actual!r}")
print("T4 LOSSLESS PASS")
PY
}

start_server() {
  stop_arm
  wait_stopped
  ssh -f -o BatchMode=yes "$WORKER_SSH" \
    "mkdir -p $(printf '%q' "$WORKER_REPO/$RESULT_DIR") && cd $(printf '%q' "$WORKER_REPO") && exec env MASTER_IP=$(printf '%q' "$MASTER_IP") IF=$(printf '%q' "$IF") HCA=$(printf '%q' "$HCA") GID=$(printf '%q' "$GID") MODELS=$(printf '%q' "$MODELS") IMAGE=$(printf '%q' "$IMAGE") LOG=$(printf '%q' "$WORKER_REPO/$RESULT_DIR/$LABEL-worker.log") ATTN=fa4 MOE=marlin FP4GEMM=flashinfer_trtllm MEMFRAC=0.85 CTX=65536 SPEC=0 GRAPHS=1 GRAPH_BS=$(printf '%q' '1 2 3 4 5 6 7 8 10 12 14 16') RAGGED= INKLING_SHEARED_BIAS=$(printf '%q' "$INKLING_SHEARED_BIAS") MAXREQ=16 PAGE=128 CONTINUOUS_DECODE_STEPS=2 EXTRA_ARGS= ./scripts/inkling-sglang-launch.sh 1 </dev/null >/dev/null 2>&1"
  sleep 3
  nohup env MASTER_IP="$MASTER_IP" IF="$IF" HCA="$HCA" GID="$GID" \
    MODELS="$MODELS" IMAGE="$IMAGE" LOG="$REPO_DIR/$RESULT_DIR/$LABEL-head.log" \
    ATTN=fa4 MOE=marlin FP4GEMM=flashinfer_trtllm MEMFRAC=0.85 CTX=65536 \
    SPEC=0 GRAPHS=1 GRAPH_BS="1 2 3 4 5 6 7 8 10 12 14 16" RAGGED= \
    INKLING_SHEARED_BIAS="$INKLING_SHEARED_BIAS" MAXREQ=16 PAGE=128 \
    CONTINUOUS_DECODE_STEPS=2 EXTRA_ARGS= \
    "$REPO_DIR/scripts/inkling-sglang-launch.sh" 0 </dev/null >/dev/null 2>&1 &
}

verify_reproducibility
run_numerics
start_server
if ! wait_ready; then
  record_boot_failure
  exit 4
fi
capture_runtime_contract | tee "$RESULT_DIR/runtime-contract.txt"
lossless_gate "$RESULT_DIR/lossless-1.json" | tee "$RESULT_DIR/lossless-1.txt"
lossless_gate "$RESULT_DIR/lossless-2.json" | tee "$RESULT_DIR/lossless-2.txt"
echo "PASS: FA4 BF16 page-128 spec-off serving reached health and two exact T4 gates (rel_bias=$REL_BIAS_MODE)" \
  | tee "$RESULT_DIR/decision.txt"
