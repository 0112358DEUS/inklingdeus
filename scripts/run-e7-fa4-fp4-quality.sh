#!/bin/bash
# 1M NIAH + tools + full GSM8K on the selected SM121 FA4 FP4 DSpark stack.
set -euo pipefail

WORKER_SSH=${WORKER_SSH:?set passwordless worker SSH}
WORKER_REPO=${WORKER_REPO:?set the absolute repository path on the worker}
MASTER_IP=${MASTER_IP:?set the head IP on the primary RoCE link}
IF=${IF:?set the primary link netdev}
HCA=${HCA:?set the unchanged NCCL HCA}
MODELS=${MODELS:?set the identical model directory on both nodes}
GSM8K_DATA=${GSM8K_DATA:?set the checksum-pinned GSM8K test JSONL path}
IMAGE=${IMAGE:-local/sglang-inkling:sparkflash-fp4-dev}
GID=${GID:-3}
RESULT_DIR=${RESULT_DIR:-artifacts/e7-fa4-fp4-quality}
READY_TIMEOUT=${READY_TIMEOUT:-900}
MIN_FULL_TOKENS=${MIN_FULL_TOKENS:-1256984}
GSM_CONCURRENCY=${GSM_CONCURRENCY:-8}
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
  if [ -f "$RESULT_DIR/run-identity.txt" ]; then
    [ "$(cat "$RESULT_DIR/run-identity.txt")" = "$identity" ] || {
      echo "result directory belongs to a different repo/image identity" >&2
      exit 2
    }
  else
    printf '%s\n' "$identity" >"$RESULT_DIR/run-identity.txt"
  fi
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

record_boot_failure() {
  {
    echo "FAIL: FA4 FP4 1M quality server did not become healthy"
    echo "head log tail:"
    tail -n 180 "$RESULT_DIR/quality-head.log" 2>/dev/null || true
    echo "worker log tail:"
    ssh -o BatchMode=yes "$WORKER_SSH" \
      "tail -n 180 $(printf '%q' "$WORKER_REPO/$RESULT_DIR/quality-worker.log")" \
      2>/dev/null || true
  } | tee "$RESULT_DIR/boot-failure.txt" >&2
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
    "mkdir -p $(printf '%q' "$WORKER_REPO/$RESULT_DIR") && cd $(printf '%q' "$WORKER_REPO") && exec env MASTER_IP=$(printf '%q' "$MASTER_IP") IF=$(printf '%q' "$IF") HCA=$(printf '%q' "$HCA") GID=$(printf '%q' "$GID") MODELS=$(printf '%q' "$MODELS") IMAGE=$(printf '%q' "$IMAGE") LOG=$(printf '%q' "$WORKER_REPO/$RESULT_DIR/quality-worker.log") CUDA_LAUNCH_BLOCKING=$(printf '%q' "${CUDA_LAUNCH_BLOCKING:-}") ATTN=fa4 MOE=marlin FP4GEMM=flashinfer_trtllm MEMFRAC=0.85 CTX=1048576 SPEC=1 BLOCK=5 GRAPHS=1 GRAPH_BS=$(printf '%q' '1 2 3 4 5 6 7 8 10 12 14 16') RAGGED= INKLING_SHEARED_BIAS=0 MAXREQ=16 PAGE=128 CONTINUOUS_DECODE_STEPS=2 EXTRA_ARGS=$(printf '%q' '--kv-cache-dtype fp4_mx_block16') ./scripts/inkling-sglang-launch.sh 1 </dev/null >/dev/null 2>&1"
  sleep 3
  nohup env MASTER_IP="$MASTER_IP" IF="$IF" HCA="$HCA" GID="$GID" \
    MODELS="$MODELS" IMAGE="$IMAGE" LOG="$REPO_DIR/$RESULT_DIR/quality-head.log" \
    ATTN=fa4 MOE=marlin FP4GEMM=flashinfer_trtllm MEMFRAC=0.85 CTX=1048576 \
    SPEC=1 BLOCK=5 GRAPHS=1 GRAPH_BS="1 2 3 4 5 6 7 8 10 12 14 16" RAGGED= \
    INKLING_SHEARED_BIAS=0 MAXREQ=16 PAGE=128 CONTINUOUS_DECODE_STEPS=2 \
    EXTRA_ARGS="--kv-cache-dtype fp4_mx_block16" \
    "$REPO_DIR/scripts/inkling-sglang-launch.sh" 0 </dev/null >/dev/null 2>&1 &
}

verify_runtime_and_capacity() {
  docker inspect inkling-sglang >"$RESULT_DIR/quality-head-inspect.json"
  ssh -o BatchMode=yes "$WORKER_SSH" docker inspect inkling-sglang \
    >"$RESULT_DIR/quality-worker-inspect.json"
  python3 - "$IMAGE" "$MIN_FULL_TOKENS" "$RESULT_DIR/quality-head.log" \
    "$RESULT_DIR/quality-head-inspect.json" "$RESULT_DIR/quality-worker-inspect.json" <<'PY' \
    | tee "$RESULT_DIR/runtime-capacity-contract.txt"
import json
import re
import sys
from pathlib import Path

image = sys.argv[1]
minimum = int(sys.argv[2])
log_path = Path(sys.argv[3])
text = log_path.read_text(encoding="utf-8", errors="replace")
match = re.search(
    r"Use sliding window memory pool\. full_layer_tokens=(\d+), "
    r"swa_layer_tokens=(\d+)",
    text,
)
if match is None:
    raise SystemExit(f"{log_path}: missing target capacity")
full, swa = map(int, match.groups())
if full < minimum:
    raise SystemExit(f"1M CAPACITY FAIL full={full} minimum={minimum}")
required = {
    "--attention-backend": "fa4",
    "--page-size": "128",
    "--context-length": "1048576",
    "--kv-cache-dtype": "fp4_mx_block16",
    "--speculative-algorithm": "DSPARK",
    "--speculative-dspark-block-size": "5",
    "--max-running-requests": "16",
    "--moe-runner-backend": "marlin",
    "--fp4-gemm-backend": "flashinfer_trtllm",
}
for path_text in sys.argv[4:]:
    path = Path(path_text)
    payload = json.loads(path.read_text(encoding="utf-8"))[0]
    if payload["Config"]["Image"] != image:
        raise SystemExit(f"{path}: wrong image")
    command = payload["Config"]["Cmd"]
    for flag, expected in required.items():
        if command.count(flag) != 1 or command[command.index(flag) + 1] != expected:
            raise SystemExit(f"{path}: {flag} contract failed")
    if "SGLANG_OPT_USE_INKLING_SHEARED_BIAS=0" not in set(payload["Config"]["Env"]):
        raise SystemExit(f"{path}: score-mod bias contract failed")
for proof in (
    "Initialized DSpark draft runner. attention_backend=fa4",
    "DSpark draft greedy proposal folded into the draft cuda graph",
    "Capture target verify CUDA graph end",
    "Capture draft verify CUDA graph end",
):
    if proof not in text:
        raise SystemExit(f"{log_path}: missing {proof!r}")
print(
    f"FA4 FP4 1M RUNTIME PASS full_layer_tokens={full} "
    f"headroom={full - minimum} swa_layer_tokens={swa} block=5 graphs=C16"
)
PY
}

verify_tokenize_contract() {
  python3 - "$RESULT_DIR/tokenize-contract.json" <<'PY'
import json
import sys
import urllib.request
from pathlib import Path

request = urllib.request.Request(
    "http://127.0.0.1:30000/v1/tokenize",
    data=json.dumps({
        "model": "inkling-small",
        "messages": [{"role": "user", "content": "tokenize contract probe"}],
        "reasoning_effort": "none",
    }).encode(),
    headers={"Content-Type": "application/json"},
)
with urllib.request.urlopen(request, timeout=300) as response:
    payload = json.load(response)
tokens = payload.get("tokens")
count = payload.get("count")
if not isinstance(tokens, list) or not all(isinstance(token, int) for token in tokens):
    raise SystemExit("TOKENIZE CONTRACT FAIL: tokens is not List[int]")
if not isinstance(count, int) or count != len(tokens):
    raise SystemExit("TOKENIZE CONTRACT FAIL: inconsistent count/token list")
if payload.get("max_model_len") != 1048576:
    raise SystemExit(
        "TOKENIZE CONTRACT FAIL: effective max_model_len is not runtime context"
    )
Path(sys.argv[1]).write_text(
    json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8"
)
print(f"TOKENIZE CONTRACT PASS count={count} max_model_len=1048576")
PY
}

verify_reproducibility
python3 "$REPO_DIR/benchmarks/gsm8k_eval.py" "$GSM8K_DATA" \
  --responses "$RESULT_DIR/gsm8k-responses.jsonl" \
  --summary "$RESULT_DIR/gsm8k-summary.json" --dry-run \
  >"$RESULT_DIR/gsm8k-dataset-preflight.json"
start_server
if ! wait_ready; then
  record_boot_failure
  exit 4
fi
verify_runtime_and_capacity
verify_tokenize_contract | tee "$RESULT_DIR/tokenize-contract.txt"
lossless_gate "$RESULT_DIR/lossless-before-quality.json" \
  | tee "$RESULT_DIR/lossless-before-quality.txt"

INKLING_URL=http://127.0.0.1:30000 \
  python3 "$REPO_DIR/benchmarks/niah_eval.py" \
  --output "$RESULT_DIR/niah.json" --resume
lossless_gate "$RESULT_DIR/lossless-after-niah.json" \
  | tee "$RESULT_DIR/lossless-after-niah.txt"

INKLING_URL=http://127.0.0.1:30000 \
  python3 "$REPO_DIR/benchmarks/tool_call_regression.py" \
  --reps 4 --output "$RESULT_DIR/tool-call-regression.json"
lossless_gate "$RESULT_DIR/lossless-after-tools.json" \
  | tee "$RESULT_DIR/lossless-after-tools.txt"

INKLING_URL=http://127.0.0.1:30000 \
  python3 "$REPO_DIR/benchmarks/gsm8k_eval.py" "$GSM8K_DATA" \
  --concurrency "$GSM_CONCURRENCY" \
  --responses "$RESULT_DIR/gsm8k-responses.jsonl" \
  --summary "$RESULT_DIR/gsm8k-summary.json"
lossless_gate "$RESULT_DIR/lossless-after-gsm8k.json" \
  | tee "$RESULT_DIR/lossless-after-gsm8k.txt"

printf '{"schema_version":1,"niah_passed":true,"tool_call_passed":true,"gsm8k_passed":true,"t4_brackets_passed":true}\n' \
  >"$RESULT_DIR/quality-decision.json"
echo "PASS: FA4 FP4 DSpark quality gates at 1M context" \
  | tee "$RESULT_DIR/decision.txt"
