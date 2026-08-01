#!/bin/bash
# E2: same-session, one-variable dense FP4 GEMM backend experiment.
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
BASELINE_BACKEND=${BASELINE_BACKEND:-flashinfer_trtllm}
CANDIDATE_BACKEND=${CANDIDATE_BACKEND:-marlin}
RESULT_DIR=${RESULT_DIR:-artifacts/e2-fp4-gemm}
READY_TIMEOUT=${READY_TIMEOUT:-900}
REPS=${REPS:-8}
TOKENS=${TOKENS:-160}
REPO_DIR=$(cd "$(dirname "$0")/.." && pwd)

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

run_dense_numerical_gate() {
  docker run --rm --gpus all \
    -v "$REPO_DIR:/inklingdeus:ro" \
    --entrypoint python3 "$IMAGE" \
    /inklingdeus/benchmarks/tests_compare_fp4_gemm.py \
    | tee "$RESULT_DIR/dense-gemm-numerics.txt"
}

start_arm() {
  local label=$1 backend=$2
  stop_arm
  ssh -f -o BatchMode=yes "$WORKER_SSH" \
    "mkdir -p $(printf '%q' "$WORKER_REPO/$RESULT_DIR") && cd $(printf '%q' "$WORKER_REPO") && exec env MASTER_IP=$(printf '%q' "$MASTER_IP") IF=$(printf '%q' "$IF") HCA=$(printf '%q' "$HCA") GID=$(printf '%q' "$GID") MODELS=$(printf '%q' "$MODELS") IMAGE=$(printf '%q' "$IMAGE") LOG=$(printf '%q' "$WORKER_REPO/$RESULT_DIR/$label-worker.log") ./scripts/locked-experiment-launch.sh 1 $(printf '%q' "$backend") 7 1 0.85 0 '' '' </dev/null >/dev/null 2>&1"
  sleep 3
  nohup env MASTER_IP="$MASTER_IP" IF="$IF" HCA="$HCA" GID="$GID" \
    MODELS="$MODELS" IMAGE="$IMAGE" LOG="$REPO_DIR/$RESULT_DIR/$label-head.log" \
    "$REPO_DIR/scripts/locked-experiment-launch.sh" \
    0 "$backend" 7 1 0.85 0 "" "" </dev/null >/dev/null 2>&1 &
  wait_ready
  lossless_gate | tee "$RESULT_DIR/$label-lossless-pre.txt"
  INKLING_URL=http://127.0.0.1:30000 python3 "$REPO_DIR/benchmarks/chat_bench.py" \
    "$label" --task open-ended --reps "$REPS" --tokens "$TOKENS" \
    --output "$RESULT_DIR/$label.json"
  lossless_gate | tee "$RESULT_DIR/$label-lossless-post.txt"
}

verify_reproducibility
run_dense_numerical_gate
start_arm e2-flashinfer-trtllm "$BASELINE_BACKEND"
start_arm e2-marlin "$CANDIDATE_BACKEND"
stop_arm
python3 "$REPO_DIR/benchmarks/compare_ab.py" \
  "$RESULT_DIR/e2-flashinfer-trtllm.json" "$RESULT_DIR/e2-marlin.json" \
  --task open-ended
