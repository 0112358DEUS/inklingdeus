#!/bin/bash
# E4: same-session, one-variable accepted DSpark champion vs native width-1 MTP experiment.
# Run on the idle head Spark with passwordless SSH to the idle worker.
set -euo pipefail

WORKER_SSH=${WORKER_SSH:?set passwordless worker SSH}
WORKER_REPO=${WORKER_REPO:?set the absolute inklingdeus repo path on the worker}
MASTER_IP=${MASTER_IP:?set the head IP on the primary RoCE link}
IF=${IF:?set the primary link netdev}
HCA=${HCA:?set the unchanged NCCL HCA list for both arms}
MODELS=${MODELS:?set the identical model directory path present on both nodes}
GID=${GID:-3}
IMAGE=${IMAGE:-local/sglang-inkling:gb10-kvquant}
FP4GEMM=${FP4GEMM:-flashinfer_trtllm}
CHAMPION_BLOCK=${CHAMPION_BLOCK:-5}
RESULT_DIR=${RESULT_DIR:-artifacts/e4-width1-mtp}
READY_TIMEOUT=${READY_TIMEOUT:-900}
REPS=${REPS:-8}
TOKENS=${TOKENS:-160}
REPO_DIR=$(cd "$(dirname "$0")/.." && pwd)
MTP_ARGS="--speculative-algorithm EAGLE --speculative-num-steps 1 --speculative-eagle-topk 1 --speculative-num-draft-tokens 2 --enable-multi-layer-eagle --speculative-use-rejection-sampling"

mkdir -p "$RESULT_DIR"

stop_arm() {
  docker rm -f inkling-sglang >/dev/null 2>&1 || true
  ssh -o BatchMode=yes "$WORKER_SSH" docker rm -f inkling-sglang >/dev/null 2>&1 || true
}
trap stop_arm EXIT

verify_reproducibility() {
  local local_sha worker_sha local_payload worker_payload local_image worker_image
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
  local_image=$(IMAGE="$IMAGE" "$REPO_DIR/scripts/image-fingerprint.sh")
  worker_image=$(ssh -o BatchMode=yes "$WORKER_SSH" \
    "IMAGE=$(printf '%q' "$IMAGE") $(printf '%q' "$WORKER_REPO/scripts/image-fingerprint.sh")")
  [ "$local_image" = "$worker_image" ] || {
    echo "patched image payload mismatch" >&2
    exit 2
  }
  printf 'reproducibility PASS sha=%s repo_payload=%s image_payload=matched\n' \
    "$local_sha" "$local_payload"
}

verify_mtp_weights() {
  local mtp_path head_hash worker_hash
  mtp_path="$MODELS/inkling-small-nvfp4/mtp.safetensors"
  [ -f "$mtp_path" ] || {
    echo "native MTP weight missing on head: $mtp_path" >&2
    exit 2
  }
  ssh -o BatchMode=yes "$WORKER_SSH" "test -f $(printf '%q' "$mtp_path")" || {
    echo "native MTP weight missing on worker: $mtp_path" >&2
    exit 2
  }
  head_hash=$(sha256sum "$mtp_path" | awk '{print $1}')
  worker_hash=$(ssh -o BatchMode=yes "$WORKER_SSH" \
    "sha256sum $(printf '%q' "$mtp_path") | awk '{print \$1}'")
  [ "$head_hash" = "$worker_hash" ] || {
    echo "mtp.safetensors mismatch between nodes" >&2
    exit 2
  }
  printf 'MTP weight preflight PASS sha256=%s\n' "$head_hash" \
    | tee "$RESULT_DIR/mtp-weight-preflight.txt"
}

wait_ready() {
  local deadline=$((SECONDS + READY_TIMEOUT))
  while [ "$SECONDS" -lt "$deadline" ]; do
    if curl -fsS http://127.0.0.1:30000/health >/dev/null 2>&1; then
      return 0
    fi
    sleep 5
  done
  echo "server did not become healthy within $READY_TIMEOUT seconds" >&2
  return 1
}

lossless_gate() {
  python3 - <<'PY'
import json
import urllib.request

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
if actual != expected:
    raise SystemExit(f"T4 LOSSLESS FAIL\nexpected={expected!r}\nactual={actual!r}")
print("T4 LOSSLESS PASS")
PY
}

capture_and_verify_runtime() {
  local label=$1 mode=$2
  docker inspect inkling-sglang >"$RESULT_DIR/$label-head-inspect.json"
  ssh -o BatchMode=yes "$WORKER_SSH" docker inspect inkling-sglang \
    >"$RESULT_DIR/$label-worker-inspect.json"
  docker stats --no-stream inkling-sglang >"$RESULT_DIR/$label-head-memory.txt"
  ssh -o BatchMode=yes "$WORKER_SSH" docker stats --no-stream inkling-sglang \
    >"$RESULT_DIR/$label-worker-memory.txt"
  python3 - "$mode" "$CHAMPION_BLOCK" "$RESULT_DIR/$label-head-inspect.json" \
    "$RESULT_DIR/$label-worker-inspect.json" <<'PY'
import json
import sys
from pathlib import Path

mode = sys.argv[1]
champion_block = sys.argv[2]
required_by_mode = {
    "dspark": {
        "--speculative-algorithm": "DSPARK",
        "--speculative-draft-model-path": "/models/dspark-draft",
        "--speculative-draft-model-quantization": "unquant",
        "--speculative-dspark-block-size": champion_block,
    },
    "mtp": {
        "--speculative-algorithm": "EAGLE",
        "--speculative-num-steps": "1",
        "--speculative-eagle-topk": "1",
        "--speculative-num-draft-tokens": "2",
    },
}
required_flags = {"mtp": {"--enable-multi-layer-eagle", "--speculative-use-rejection-sampling"}}
for path_text in sys.argv[3:]:
    path = Path(path_text)
    payload = json.loads(path.read_text(encoding="utf-8"))
    command = payload[0]["Config"]["Cmd"]
    for flag, expected in required_by_mode[mode].items():
        if command.count(flag) != 1:
            raise SystemExit(f"{path}: expected exactly one {flag}")
        actual = command[command.index(flag) + 1]
        if actual != expected:
            raise SystemExit(f"{path}: {flag}={actual!r}, expected {expected!r}")
    missing = required_flags.get(mode, set()).difference(command)
    if missing:
        raise SystemExit(f"{path}: missing {sorted(missing)}")
    if mode == "mtp":
        forbidden = {
            "DSPARK",
            "--speculative-draft-model-path",
            "--speculative-draft-model-quantization",
            "--speculative-dspark-block-size",
        }
    else:
        forbidden = {"EAGLE", "--enable-multi-layer-eagle"}
    present = forbidden.intersection(command)
    if present:
        raise SystemExit(f"{path}: mixed speculative paths: {sorted(present)}")
print(f"runtime command contract PASS mode={mode}")
PY
}

record_boot_failure() {
  local label=$1
  {
    echo "REJECT: $label failed to become healthy; no serving claim is allowed"
    echo "head log tail:"
    tail -n 120 "$REPO_DIR/$RESULT_DIR/$label-head.log" 2>/dev/null || true
    echo "worker log tail:"
    ssh -o BatchMode=yes "$WORKER_SSH" \
      "tail -n 120 $(printf '%q' "$WORKER_REPO/$RESULT_DIR/$label-worker.log")" \
      2>/dev/null || true
  } | tee "$RESULT_DIR/$label-boot-failure.txt" >&2
}

start_arm() {
  local label=$1 mode=$2 spec=$3 extra_args=$4
  stop_arm
  ssh -f -o BatchMode=yes "$WORKER_SSH" \
    "mkdir -p $(printf '%q' "$WORKER_REPO/$RESULT_DIR") && cd $(printf '%q' "$WORKER_REPO") && exec env MASTER_IP=$(printf '%q' "$MASTER_IP") IF=$(printf '%q' "$IF") HCA=$(printf '%q' "$HCA") GID=$(printf '%q' "$GID") MODELS=$(printf '%q' "$MODELS") IMAGE=$(printf '%q' "$IMAGE") LOG=$(printf '%q' "$WORKER_REPO/$RESULT_DIR/$label-worker.log") ./scripts/locked-experiment-launch.sh 1 $(printf '%q' "$FP4GEMM") $(printf '%q' "$CHAMPION_BLOCK") $(printf '%q' "$spec") 0.85 0 $(printf '%q' "$extra_args") '' </dev/null >/dev/null 2>&1"
  sleep 3
  nohup env MASTER_IP="$MASTER_IP" IF="$IF" HCA="$HCA" GID="$GID" \
    MODELS="$MODELS" IMAGE="$IMAGE" LOG="$REPO_DIR/$RESULT_DIR/$label-head.log" \
    "$REPO_DIR/scripts/locked-experiment-launch.sh" \
    0 "$FP4GEMM" "$CHAMPION_BLOCK" "$spec" 0.85 0 "$extra_args" "" </dev/null >/dev/null 2>&1 &
  if ! wait_ready; then
    record_boot_failure "$label"
    return 4
  fi
  capture_and_verify_runtime "$label" "$mode"
  lossless_gate | tee "$RESULT_DIR/$label-lossless-pre.txt"
  INKLING_URL=http://127.0.0.1:30000 python3 "$REPO_DIR/benchmarks/chat_bench.py" \
    "$label" --task open-ended --reps "$REPS" --tokens "$TOKENS" \
    --output "$RESULT_DIR/$label.json"
  lossless_gate | tee "$RESULT_DIR/$label-lossless-post.txt"
}

verify_reproducibility
verify_mtp_weights
start_arm e4-dspark-baseline dspark 1 ""
start_arm e4-native-mtp-width-1 mtp 0 "$MTP_ARGS"
stop_arm
python3 "$REPO_DIR/benchmarks/compare_ab.py" \
  "$RESULT_DIR/e4-dspark-baseline.json" "$RESULT_DIR/e4-native-mtp-width-1.json" \
  --task open-ended | tee "$RESULT_DIR/decision.txt"
