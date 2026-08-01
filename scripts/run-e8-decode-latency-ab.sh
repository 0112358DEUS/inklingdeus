#!/bin/bash
# E8: same-session decode-latency sweep — NCCL protocol, continuous decode steps, KV splits.
# Each arm changes EXACTLY ONE factor vs the accepted block-5 champion baseline.
# Run on the idle head Spark with passwordless SSH to the idle worker.
set -euo pipefail

WORKER_SSH=${WORKER_SSH:?set passwordless worker SSH}
WORKER_REPO=${WORKER_REPO:?set the absolute inklingdeus repo path on the worker}
MASTER_IP=${MASTER_IP:?set the head IP on the primary RoCE link}
IF=${IF:?set the primary link netdev}
HCA=${HCA:?set the unchanged NCCL HCA list for every arm}
MODELS=${MODELS:?set the identical model directory path present on both nodes}
GID=${GID:-3}
IMAGE=${IMAGE:-local/sglang-inkling:gb10-kvquant}
FP4GEMM=${FP4GEMM:-flashinfer_trtllm}
CHAMPION_BLOCK=${CHAMPION_BLOCK:-5}
RESULT_DIR=${RESULT_DIR:-artifacts/e8-decode-latency}
READY_TIMEOUT=${READY_TIMEOUT:-900}
REPS=${REPS:-8}
TOKENS=${TOKENS:-160}
# Which factor groups to sweep this session. Every arm is independent, so a partial
# sweep is still valid — each arm compares only against this session's baseline.
FACTORS=${FACTORS:-proto cds ksplit}
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

# Verify from docker inspect that the arm's single intended factor (and nothing else) is live.
verify_runtime_contract() {
  local label=$1 nccl_proto=$2 extra_args=$3
  docker inspect inkling-sglang >"$RESULT_DIR/$label-head-inspect.json"
  ssh -o BatchMode=yes "$WORKER_SSH" docker inspect inkling-sglang \
    >"$RESULT_DIR/$label-worker-inspect.json"
  python3 - "$nccl_proto" "$extra_args" "$RESULT_DIR/$label-head-inspect.json" \
    "$RESULT_DIR/$label-worker-inspect.json" <<'PY'
import json
import sys
from pathlib import Path

nccl_proto = sys.argv[1]
extra_args = sys.argv[2].split()
for path_text in sys.argv[3:]:
    path = Path(path_text)
    payload = json.loads(path.read_text(encoding="utf-8"))
    env = payload[0]["Config"]["Env"]
    command = payload[0]["Config"]["Cmd"]
    proto_entries = [entry for entry in env if entry.startswith("NCCL_PROTO=")]
    if nccl_proto:
        if proto_entries != [f"NCCL_PROTO={nccl_proto}"]:
            raise SystemExit(f"{path}: NCCL_PROTO entries {proto_entries!r}, expected {nccl_proto!r}")
    elif proto_entries:
        raise SystemExit(f"{path}: unexpected NCCL_PROTO in a non-proto arm: {proto_entries!r}")
    if any(entry.startswith("NCCL_ALGO=") for entry in env):
        raise SystemExit(f"{path}: NCCL_ALGO must not be set in this experiment")
    if extra_args:
        flag, value = extra_args[0], extra_args[1]
        if command.count(flag) != 1:
            raise SystemExit(f"{path}: expected exactly one {flag}")
        actual = command[command.index(flag) + 1]
        if actual != value:
            raise SystemExit(f"{path}: {flag}={actual!r}, expected {value!r}")
    else:
        for forbidden in ("--num-continuous-decode-steps", "--triton-attention-num-kv-splits"):
            if forbidden in command:
                raise SystemExit(f"{path}: unexpected {forbidden} in this arm")
print("runtime contract PASS")
PY
}

record_boot_failure() {
  local label=$1
  {
    echo "REJECT: $label failed to become healthy; factor unavailable or broken in this build"
    echo "head log tail:"
    tail -n 120 "$REPO_DIR/$RESULT_DIR/$label-head.log" 2>/dev/null || true
    echo "worker log tail:"
    ssh -o BatchMode=yes "$WORKER_SSH" \
      "tail -n 120 $(printf '%q' "$WORKER_REPO/$RESULT_DIR/$label-worker.log")" \
      2>/dev/null || true
  } | tee "$RESULT_DIR/$label-boot-failure.txt" >&2
}

# start_arm LABEL NCCL_PROTO_VALUE EXTRA_ARGS — empty proto/extra = champion baseline.
start_arm() {
  local label=$1 nccl_proto=$2 extra_args=$3
  stop_arm
  ssh -f -o BatchMode=yes "$WORKER_SSH" \
    "mkdir -p $(printf '%q' "$WORKER_REPO/$RESULT_DIR") && cd $(printf '%q' "$WORKER_REPO") && exec env MASTER_IP=$(printf '%q' "$MASTER_IP") IF=$(printf '%q' "$IF") HCA=$(printf '%q' "$HCA") GID=$(printf '%q' "$GID") MODELS=$(printf '%q' "$MODELS") IMAGE=$(printf '%q' "$IMAGE") EXPERIMENT_NCCL_PROTO=$(printf '%q' "$nccl_proto") LOG=$(printf '%q' "$WORKER_REPO/$RESULT_DIR/$label-worker.log") ./scripts/locked-experiment-launch.sh 1 $(printf '%q' "$FP4GEMM") $(printf '%q' "$CHAMPION_BLOCK") 1 0.85 0 $(printf '%q' "$extra_args") '' </dev/null >/dev/null 2>&1"
  sleep 3
  nohup env MASTER_IP="$MASTER_IP" IF="$IF" HCA="$HCA" GID="$GID" \
    MODELS="$MODELS" IMAGE="$IMAGE" EXPERIMENT_NCCL_PROTO="$nccl_proto" \
    LOG="$REPO_DIR/$RESULT_DIR/$label-head.log" \
    "$REPO_DIR/scripts/locked-experiment-launch.sh" \
    0 "$FP4GEMM" "$CHAMPION_BLOCK" 1 0.85 0 "$extra_args" "" </dev/null >/dev/null 2>&1 &
  if ! wait_ready; then
    record_boot_failure "$label"
    return 4
  fi
  verify_runtime_contract "$label" "$nccl_proto" "$extra_args"
  lossless_gate | tee "$RESULT_DIR/$label-lossless-pre.txt"
  INKLING_URL=http://127.0.0.1:30000 python3 "$REPO_DIR/benchmarks/chat_bench.py" \
    "$label" --task open-ended --reps "$REPS" --tokens "$TOKENS" \
    --block-size "$CHAMPION_BLOCK" --require-histogram \
    --output "$RESULT_DIR/$label.json"
  lossless_gate | tee "$RESULT_DIR/$label-lossless-post.txt"
}

# Arms: label, NCCL_PROTO, EXTRA_ARGS. Baseline always runs first, in the same session.
ARM_SPECS=("e8-baseline||")
read -r -a FACTOR_LIST <<<"$FACTORS"
for factor in "${FACTOR_LIST[@]}"; do
  case "$factor" in
    proto)
      ARM_SPECS+=("e8-proto-ll|LL|" "e8-proto-ll128|LL128|" "e8-proto-simple|Simple|")
      ;;
    cds)
      ARM_SPECS+=("e8-cds-2||--num-continuous-decode-steps 2"
                  "e8-cds-4||--num-continuous-decode-steps 4")
      ;;
    ksplit)
      ARM_SPECS+=("e8-ksplit-4||--triton-attention-num-kv-splits 4"
                  "e8-ksplit-16||--triton-attention-num-kv-splits 16")
      ;;
    *) echo "unknown factor '$factor' (valid: proto cds ksplit)" >&2; exit 2 ;;
  esac
done

verify_reproducibility
FAILED_ARMS=()
for spec in "${ARM_SPECS[@]}"; do
  IFS='|' read -r label nccl_proto extra_args <<<"$spec"
  if ! start_arm "$label" "$nccl_proto" "$extra_args"; then
    [ "$label" = "e8-baseline" ] && { echo "baseline arm failed — no comparison is possible" >&2; exit 4; }
    FAILED_ARMS+=("$label")
  fi
done
stop_arm

for spec in "${ARM_SPECS[@]:1}"; do
  IFS='|' read -r label _ _ <<<"$spec"
  case " ${FAILED_ARMS[*]-} " in *" $label "*) continue ;; esac
  python3 "$REPO_DIR/benchmarks/compare_ab.py" \
    "$RESULT_DIR/e8-baseline.json" "$RESULT_DIR/$label.json" \
    --task open-ended | tee "$RESULT_DIR/decision-$label.txt"
done
if [ "${#FAILED_ARMS[@]}" -gt 0 ]; then
  printf 'REJECTED (boot failure, factor unavailable): %s\n' "${FAILED_ARMS[@]}" \
    | tee "$RESULT_DIR/rejected-arms.txt"
fi
