#!/bin/bash
# Isolate whether continuous-decode-steps=2 duplicates Inkling tool calls.
set -euo pipefail

WORKER_SSH=${WORKER_SSH:?set passwordless worker SSH}
WORKER_REPO=${WORKER_REPO:?set the absolute repository path on the worker}
MASTER_IP=${MASTER_IP:?set the head IP on the primary RoCE link}
IF=${IF:?set the primary link netdev}
HCA=${HCA:?set the unchanged NCCL HCA}
MODELS=${MODELS:?set the identical model directory on both nodes}
IMAGE=${IMAGE:-local/sglang-inkling:sparkflash-fp4-dev}
GID=${GID:-3}
RESULT_DIR=${RESULT_DIR:-artifacts/e7-tool-cds1}
READY_TIMEOUT=${READY_TIMEOUT:-900}
MIN_FULL_TOKENS=${MIN_FULL_TOKENS:-1256984}
REPO_DIR=$(cd "$(dirname "$0")/.." && pwd)

case "$RESULT_DIR" in
  /*|*..*) echo "RESULT_DIR must be a safe relative path" >&2; exit 2 ;;
esac
mkdir -p "$RESULT_DIR"

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
  local_image=$(IMAGE="$IMAGE" "$REPO_DIR/scripts/sparkflash-fp4-image-fingerprint.sh")
  worker_image=$(ssh -o BatchMode=yes "$WORKER_SSH" \
    "IMAGE=$(printf '%q' "$IMAGE") $(printf '%q' "$WORKER_REPO/scripts/sparkflash-fp4-image-fingerprint.sh")")
  [ "$local_image" = "$worker_image" ] || {
    echo "SparkFlash image payload mismatch" >&2
    exit 2
  }
  identity="repo_sha=$local_sha repo_payload=$local_payload image_payload=$local_image"
  printf '%s\n' "$identity" >"$RESULT_DIR/run-identity.txt"
  {
    printf '%s\n' "$identity"
    printf 'head_image_id='
    docker image inspect "$IMAGE" --format '{{.Id}}'
    printf 'worker_image_id='
    ssh -o BatchMode=yes "$WORKER_SSH" \
      docker image inspect "$IMAGE" --format '{{.Id}}'
    echo 'reproducibility PASS'
  } | tee "$RESULT_DIR/identity.txt"
}

wait_ready() {
  local deadline=$((SECONDS + READY_TIMEOUT)) seen=0
  while [ "$SECONDS" -lt "$deadline" ]; do
    if curl -fsS http://127.0.0.1:30000/health >/dev/null 2>&1; then
      return 0
    fi
    if docker inspect inkling-sglang >/dev/null 2>&1; then
      seen=1
      if [ "$(docker inspect -f '{{.State.Running}}' inkling-sglang)" != true ]; then
        echo "head server exited before readiness" >&2
        return 1
      fi
    elif [ "$seen" = 1 ]; then
      echo "head server disappeared before readiness" >&2
      return 1
    fi
    sleep 5
  done
  echo "server did not become healthy within $READY_TIMEOUT seconds" >&2
  return 1
}

lossless_gate() {
  local output=$1
  python3 - "$output" <<'PY'
import hashlib
import json
import sys
import urllib.request
from pathlib import Path

expected = " Paris. The capital of Germany is Berlin. The capital of"
request = urllib.request.Request(
    "http://127.0.0.1:30000/v1/completions",
    data=json.dumps({
        "model": "inkling-small",
        "prompt": "The capital of France is",
        "max_tokens": 12,
        "temperature": 0,
    }).encode(),
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
    raise SystemExit(f"T4 LOSSLESS FAIL expected={expected!r} actual={actual!r}")
print("T4 LOSSLESS PASS")
PY
}

start_server() {
  stop_arm
  ssh -f -o BatchMode=yes "$WORKER_SSH" \
    "mkdir -p $(printf '%q' "$WORKER_REPO/$RESULT_DIR") && cd $(printf '%q' "$WORKER_REPO") && exec env MASTER_IP=$(printf '%q' "$MASTER_IP") IF=$(printf '%q' "$IF") HCA=$(printf '%q' "$HCA") GID=$(printf '%q' "$GID") MODELS=$(printf '%q' "$MODELS") IMAGE=$(printf '%q' "$IMAGE") LOG=$(printf '%q' "$WORKER_REPO/$RESULT_DIR/tool-cds1-worker.log") ATTN=fa4 MOE=marlin FP4GEMM=flashinfer_trtllm MEMFRAC=0.85 CTX=1048576 SPEC=1 BLOCK=5 GRAPHS=1 GRAPH_BS=$(printf '%q' '1 2 3 4 5 6 7 8 10 12 14 16') RAGGED= INKLING_SHEARED_BIAS=0 MAXREQ=16 PAGE=128 CONTINUOUS_DECODE_STEPS=1 EXTRA_ARGS=$(printf '%q' '--kv-cache-dtype fp4_mx_block16') ./scripts/inkling-sglang-launch.sh 1 </dev/null >/dev/null 2>&1"
  sleep 3
  nohup env MASTER_IP="$MASTER_IP" IF="$IF" HCA="$HCA" GID="$GID" \
    MODELS="$MODELS" IMAGE="$IMAGE" LOG="$REPO_DIR/$RESULT_DIR/tool-cds1-head.log" \
    ATTN=fa4 MOE=marlin FP4GEMM=flashinfer_trtllm MEMFRAC=0.85 CTX=1048576 \
    SPEC=1 BLOCK=5 GRAPHS=1 GRAPH_BS="1 2 3 4 5 6 7 8 10 12 14 16" RAGGED= \
    INKLING_SHEARED_BIAS=0 MAXREQ=16 PAGE=128 CONTINUOUS_DECODE_STEPS=1 \
    EXTRA_ARGS="--kv-cache-dtype fp4_mx_block16" \
    "$REPO_DIR/scripts/inkling-sglang-launch.sh" 0 </dev/null >/dev/null 2>&1 &
}

verify_runtime() {
  docker inspect inkling-sglang >"$RESULT_DIR/tool-cds1-head-inspect.json"
  ssh -o BatchMode=yes "$WORKER_SSH" docker inspect inkling-sglang \
    >"$RESULT_DIR/tool-cds1-worker-inspect.json"
  python3 - "$IMAGE" "$MIN_FULL_TOKENS" "$RESULT_DIR/tool-cds1-head.log" \
    "$RESULT_DIR/tool-cds1-head-inspect.json" "$RESULT_DIR/tool-cds1-worker-inspect.json" <<'PY' \
    | tee "$RESULT_DIR/runtime-contract.txt"
import json
import re
import sys
from pathlib import Path

image = sys.argv[1]
minimum = int(sys.argv[2])
log_path = Path(sys.argv[3])
text = log_path.read_text(encoding="utf-8", errors="replace")
match = re.search(
    r"Use sliding window memory pool\. full_layer_tokens=(\d+), swa_layer_tokens=(\d+)",
    text,
)
if match is None:
    raise SystemExit("missing capacity record")
full, swa = map(int, match.groups())
if full < minimum:
    raise SystemExit(f"capacity fail full={full} minimum={minimum}")
required = {
    "--attention-backend": "fa4",
    "--page-size": "128",
    "--context-length": "1048576",
    "--kv-cache-dtype": "fp4_mx_block16",
    "--speculative-algorithm": "DSPARK",
    "--speculative-dspark-block-size": "5",
    "--num-continuous-decode-steps": "1",
    "--max-running-requests": "16",
}
for path_text in sys.argv[4:]:
    payload = json.loads(Path(path_text).read_text(encoding="utf-8"))[0]
    if payload["Config"]["Image"] != image:
        raise SystemExit(f"{path_text}: wrong image")
    command = payload["Config"]["Cmd"]
    for flag, expected in required.items():
        if command.count(flag) != 1 or command[command.index(flag) + 1] != expected:
            raise SystemExit(f"{path_text}: {flag} contract failed")
for proof in (
    "Initialized DSpark draft runner. attention_backend=fa4",
    "Capture target verify CUDA graph end",
    "Capture draft verify CUDA graph end",
):
    if proof not in text:
        raise SystemExit(f"missing runtime proof {proof!r}")
print(
    f"CDS1 RUNTIME PASS full_layer_tokens={full} "
    f"headroom={full - minimum} swa_layer_tokens={swa}"
)
PY
}

verify_reproducibility
start_server
if ! wait_ready; then
  tail -n 180 "$RESULT_DIR/tool-cds1-head.log" >&2 || true
  exit 4
fi
verify_runtime
lossless_gate "$RESULT_DIR/lossless-before-tools.json" \
  | tee "$RESULT_DIR/lossless-before-tools.txt"

tool_rc=0
if INKLING_URL=http://127.0.0.1:30000 \
  python3 "$REPO_DIR/benchmarks/tool_call_regression.py" \
  --reps 4 --output "$RESULT_DIR/tool-call-regression.json"; then
  :
else
  tool_rc=$?
fi
lossless_gate "$RESULT_DIR/lossless-after-tools.json" \
  | tee "$RESULT_DIR/lossless-after-tools.txt"

if [ "$tool_rc" -ne 0 ]; then
  echo "REJECT: CDS1 did not clear the 16-flow tool gate" \
    | tee "$RESULT_DIR/decision.txt"
  exit "$tool_rc"
fi
echo "PASS: CDS1 cleared all 16 tool flows and bracketed T4" \
  | tee "$RESULT_DIR/decision.txt"
