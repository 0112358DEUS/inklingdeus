#!/bin/bash
# Promote the accepted E8 CDS=2 candidate through the champion T4 + all-task gate.
set -euo pipefail

WORKER_SSH=${WORKER_SSH:?set passwordless worker SSH}
WORKER_REPO=${WORKER_REPO:?set the absolute inklingdeus repo path on the worker}
MASTER_IP=${MASTER_IP:?set the head IP on the primary RoCE link}
IF=${IF:?set the primary link netdev}
HCA=${HCA:?set the accepted champion HCA}
MODELS=${MODELS:?set the identical model directory path present on both nodes}
GID=${GID:-3}
IMAGE=${IMAGE:-local/sglang-inkling:gb10-kvquant}
RESULT_DIR=${RESULT_DIR:-artifacts/e8-cds2-adoption}
READY_TIMEOUT=${READY_TIMEOUT:-900}
REPS=${REPS:-8}
TOKENS=${TOKENS:-160}
REPO_DIR=$(cd "$(dirname "$0")/.." && pwd)

mkdir -p "$RESULT_DIR"

stop_champion() {
  docker rm -f inkling-sglang >/dev/null 2>&1 || true
  ssh -o BatchMode=yes "$WORKER_SSH" docker rm -f inkling-sglang >/dev/null 2>&1 || true
}
trap stop_champion EXIT

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
  printf 'repo_sha=%s\nrepo_payload=%s\nimage_payload=%s\n' \
    "$local_sha" "$local_payload" "$local_image"
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
        echo "champion container exited before readiness" >&2
        return 1
      fi
    elif [ "$seen_container" = 1 ]; then
      echo "champion container disappeared before readiness" >&2
      return 1
    fi
    sleep 5
  done
  echo "champion did not become healthy within $READY_TIMEOUT seconds" >&2
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

verify_runtime_contract() {
  docker inspect inkling-sglang >"$RESULT_DIR/champion-head-inspect.json"
  ssh -o BatchMode=yes "$WORKER_SSH" docker inspect inkling-sglang \
    >"$RESULT_DIR/champion-worker-inspect.json"
  python3 - "$RESULT_DIR/champion-head-inspect.json" \
    "$RESULT_DIR/champion-worker-inspect.json" <<'PY'
import json
import sys
from pathlib import Path

expected_pairs = {
    "--num-continuous-decode-steps": "2",
    "--speculative-dspark-block-size": "5",
    "--context-length": "1048576",
    "--kv-cache-dtype": "fp4_mx_block16",
    "--attention-backend": "triton",
    "--moe-runner-backend": "marlin",
    "--page-size": "1",
}
for path_text in sys.argv[1:]:
    path = Path(path_text)
    payload = json.loads(path.read_text(encoding="utf-8"))
    command = payload[0]["Config"]["Cmd"]
    env = payload[0]["Config"]["Env"]
    for flag, expected in expected_pairs.items():
        if command.count(flag) != 1:
            raise SystemExit(f"{path}: expected exactly one {flag}")
        actual = command[command.index(flag) + 1]
        if actual != expected:
            raise SystemExit(f"{path}: {flag}={actual!r}, expected {expected!r}")
    for required in (
        "--triton-attention-reduce-in-fp32",
        "--disable-piecewise-cuda-graph",
        "--disable-prefill-cuda-graph",
    ):
        if command.count(required) != 1:
            raise SystemExit(f"{path}: expected exactly one {required}")
    forbidden_env = ("NCCL_ALGO=", "NCCL_PROTO=", "SGLANG_RAGGED_VERIFY_MODE=")
    for prefix in forbidden_env:
        if any(entry.startswith(prefix) for entry in env):
            raise SystemExit(f"{path}: unexpected environment entry {prefix}")
print("champion runtime contract PASS")
PY
}

stop_champion
verify_reproducibility | tee "$RESULT_DIR/identity.txt"

ssh -f -o BatchMode=yes "$WORKER_SSH" \
  "mkdir -p $(printf '%q' "$WORKER_REPO/$RESULT_DIR") && cd $(printf '%q' "$WORKER_REPO") && exec env MASTER_IP=$(printf '%q' "$MASTER_IP") IF=$(printf '%q' "$IF") HCA=$(printf '%q' "$HCA") GID=$(printf '%q' "$GID") MODELS=$(printf '%q' "$MODELS") IMAGE=$(printf '%q' "$IMAGE") LOG=$(printf '%q' "$WORKER_REPO/$RESULT_DIR/champion-worker.log") ./scripts/nvfp4-kv-boot.sh 1 </dev/null >/dev/null 2>&1"
sleep 3
nohup env MASTER_IP="$MASTER_IP" IF="$IF" HCA="$HCA" GID="$GID" \
  MODELS="$MODELS" IMAGE="$IMAGE" LOG="$REPO_DIR/$RESULT_DIR/champion-head.log" \
  "$REPO_DIR/scripts/nvfp4-kv-boot.sh" 0 </dev/null >/dev/null 2>&1 &

if ! wait_ready; then
  {
    echo "champion adoption boot failed"
    echo "head log tail:"
    tail -n 160 "$RESULT_DIR/champion-head.log" 2>/dev/null || true
    echo "worker log tail:"
    ssh -o BatchMode=yes "$WORKER_SSH" \
      "tail -n 160 $(printf '%q' "$WORKER_REPO/$RESULT_DIR/champion-worker.log")" \
      2>/dev/null || true
  } | tee "$RESULT_DIR/boot-failure.txt" >&2
  exit 4
fi

verify_runtime_contract | tee "$RESULT_DIR/runtime-contract.txt"
lossless_gate | tee "$RESULT_DIR/lossless-pre.txt"
INKLING_URL=http://127.0.0.1:30000 python3 "$REPO_DIR/benchmarks/chat_bench.py" \
  e8-cds2-champion --task all --reps "$REPS" --tokens "$TOKENS" \
  --block-size 5 --require-histogram --output "$RESULT_DIR/champion-all.json"
lossless_gate | tee "$RESULT_DIR/lossless-post.txt"
