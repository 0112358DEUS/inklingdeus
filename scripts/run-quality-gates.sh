#!/bin/bash
# Q2/Q3: launch the locked champion once, then run depth, GSM8K, and tool-call gates.
set -euo pipefail

WORKER_SSH=${WORKER_SSH:?set passwordless worker SSH}
WORKER_REPO=${WORKER_REPO:?set the absolute inklingdeus repo path on the worker}
MASTER_IP=${MASTER_IP:?set the head IP on the primary RoCE link}
IF=${IF:?set the primary link netdev}
HCA=${HCA:?set the unchanged NCCL HCA list}
MODELS=${MODELS:?set the identical model directory path present on both nodes}
GSM8K_DATA=${GSM8K_DATA:?set the official checksum-verified GSM8K test JSONL path}
GID=${GID:-3}
IMAGE=${IMAGE:-local/sglang-inkling:gb10-kvquant}
FP4GEMM=${FP4GEMM:-flashinfer_trtllm}
CHAMPION_SPECULATOR=${CHAMPION_SPECULATOR:-dspark}
CHAMPION_BLOCK=${CHAMPION_BLOCK:-5}
RESULT_DIR=${RESULT_DIR:-artifacts/quality-gates}
READY_TIMEOUT=${READY_TIMEOUT:-900}
GSM_CONCURRENCY=${GSM_CONCURRENCY:-8}
REPO_DIR=$(cd "$(dirname "$0")/.." && pwd)
# shellcheck source=scripts/champion-profile.sh
source "$REPO_DIR/scripts/champion-profile.sh"
resolve_champion_profile "$CHAMPION_SPECULATOR" "$CHAMPION_BLOCK"
HEAD_RESULT_DIR="$REPO_DIR/$RESULT_DIR"
WORKER_RESULT_DIR="$WORKER_REPO/$RESULT_DIR"

case "$RESULT_DIR" in
  /*|*..*) echo "RESULT_DIR must be a safe relative path" >&2; exit 2 ;;
esac
mkdir -p "$HEAD_RESULT_DIR"
ssh -o BatchMode=yes "$WORKER_SSH" "mkdir -p $(printf '%q' "$WORKER_RESULT_DIR")"

stop_arm() {
  docker rm -f inkling-sglang >/dev/null 2>&1 || true
  ssh -o BatchMode=yes "$WORKER_SSH" docker rm -f inkling-sglang >/dev/null 2>&1 || true
}
trap stop_arm EXIT

verify_reproducibility() {
  local local_sha worker_sha local_payload worker_payload local_image worker_image identity
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
  identity="repo_sha=$local_sha repo_payload=$local_payload image_payload=$local_image"
  if [ -f "$HEAD_RESULT_DIR/run-identity.txt" ]; then
    [ "$(cat "$HEAD_RESULT_DIR/run-identity.txt")" = "$identity" ] || {
      echo "result directory belongs to a different repo/image identity" >&2
      exit 2
    }
  else
    printf '%s\n' "$identity" >"$HEAD_RESULT_DIR/run-identity.txt"
  fi
  printf 'reproducibility PASS %s\n' "$identity"
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

start_champion() {
  stop_arm
  ssh -o BatchMode=yes "$WORKER_SSH" \
    "cd $(printf '%q' "$WORKER_REPO") && nohup env MASTER_IP=$(printf '%q' "$MASTER_IP") IF=$(printf '%q' "$IF") HCA=$(printf '%q' "$HCA") GID=$(printf '%q' "$GID") MODELS=$(printf '%q' "$MODELS") IMAGE=$(printf '%q' "$IMAGE") LOG=$(printf '%q' "$WORKER_RESULT_DIR/quality-worker.log") ./scripts/locked-experiment-launch.sh 1 $(printf '%q' "$FP4GEMM") $(printf '%q' "$PROFILE_BLOCK") $(printf '%q' "$PROFILE_SPEC") 0.85 0 $(printf '%q' "$PROFILE_EXTRA_ARGS") '' >/dev/null 2>&1 &"
  sleep 3
  nohup env MASTER_IP="$MASTER_IP" IF="$IF" HCA="$HCA" GID="$GID" \
    MODELS="$MODELS" IMAGE="$IMAGE" LOG="$HEAD_RESULT_DIR/quality-head.log" \
    "$REPO_DIR/scripts/locked-experiment-launch.sh" \
    0 "$FP4GEMM" "$PROFILE_BLOCK" "$PROFILE_SPEC" 0.85 0 \
    "$PROFILE_EXTRA_ARGS" "" >/dev/null 2>&1 &
  wait_ready
  docker inspect inkling-sglang >"$HEAD_RESULT_DIR/quality-head-inspect.json"
  ssh -o BatchMode=yes "$WORKER_SSH" docker inspect inkling-sglang \
    >"$HEAD_RESULT_DIR/quality-worker-inspect.json"
}

verify_reproducibility
python3 "$REPO_DIR/benchmarks/gsm8k_eval.py" "$GSM8K_DATA" \
  --responses "$HEAD_RESULT_DIR/gsm8k-responses.jsonl" \
  --summary "$HEAD_RESULT_DIR/gsm8k-summary.json" --dry-run \
  >"$HEAD_RESULT_DIR/gsm8k-dataset-preflight.json"
start_champion
lossless_gate | tee "$HEAD_RESULT_DIR/lossless-before-quality.txt"

niah_rc=0
if INKLING_URL=http://127.0.0.1:30000 \
  python3 "$REPO_DIR/benchmarks/niah_eval.py" \
  --output "$HEAD_RESULT_DIR/niah.json" --resume; then
  :
else
  niah_rc=$?
fi
lossless_gate | tee "$HEAD_RESULT_DIR/lossless-after-niah.txt"

gsm_rc=0
if INKLING_URL=http://127.0.0.1:30000 \
  python3 "$REPO_DIR/benchmarks/gsm8k_eval.py" "$GSM8K_DATA" \
  --concurrency "$GSM_CONCURRENCY" \
  --responses "$HEAD_RESULT_DIR/gsm8k-responses.jsonl" \
  --summary "$HEAD_RESULT_DIR/gsm8k-summary.json"; then
  :
else
  gsm_rc=$?
fi
lossless_gate | tee "$HEAD_RESULT_DIR/lossless-after-gsm8k.txt"

tool_rc=0
if INKLING_URL=http://127.0.0.1:30000 \
  python3 "$REPO_DIR/benchmarks/tool_call_regression.py" \
  --reps 4 --output "$HEAD_RESULT_DIR/tool-call-regression.json"; then
  :
else
  tool_rc=$?
fi
lossless_gate | tee "$HEAD_RESULT_DIR/lossless-after-tools.txt"

printf '{"schema_version":1,"niah_exit":%s,"gsm8k_exit":%s,"tool_call_exit":%s,"q2_passed":%s,"q3_passed":%s}\n' \
  "$niah_rc" "$gsm_rc" "$tool_rc" \
  "$([ "$niah_rc" -eq 0 ] && [ "$gsm_rc" -eq 0 ] && echo true || echo false)" \
  "$([ "$tool_rc" -eq 0 ] && echo true || echo false)" \
  >"$HEAD_RESULT_DIR/quality-decision.json"

if [ "$niah_rc" -ne 0 ] || [ "$gsm_rc" -ne 0 ] || [ "$tool_rc" -ne 0 ]; then
  exit 2
fi
printf 'Q2/Q3 quality gates PASS\n'
