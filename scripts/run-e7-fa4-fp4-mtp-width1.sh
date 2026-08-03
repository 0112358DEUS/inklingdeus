#!/bin/bash
# Candidate-only native width-1 MTP correctness gate on SM121 FA4 + FP4 KV.
set -euo pipefail

WORKER_SSH=${WORKER_SSH:?set passwordless worker SSH}
WORKER_REPO=${WORKER_REPO:?set the absolute repository path on the worker}
MASTER_IP=${MASTER_IP:?set the head IP on the primary RoCE link}
IF=${IF:?set the primary link netdev}
HCA=${HCA:?set the unchanged NCCL HCA}
MODELS=${MODELS:?set the identical model directory on both nodes}
IMAGE=${IMAGE:-local/sglang-inkling:sparkflash-fp4-dev}
GID=${GID:-3}
RESULT_DIR=${RESULT_DIR:-artifacts/e7-fa4-fp4-mtp-width1}
READY_TIMEOUT=${READY_TIMEOUT:-900}
MIN_FULL_TOKENS=${MIN_FULL_TOKENS:-1256984}
MIN_POST_GRAPH_GB=${MIN_POST_GRAPH_GB:-13}
MEMFRAC=${MEMFRAC:-0.85}
REPO_DIR=$(cd "$(dirname "$0")/.." && pwd)
MTP_ARGS="--kv-cache-dtype fp4_mx_block16 --speculative-algorithm EAGLE --speculative-num-steps 1 --speculative-eagle-topk 1 --speculative-num-draft-tokens 2 --enable-multi-layer-eagle --speculative-use-rejection-sampling"

mkdir -p "$RESULT_DIR"
awk -v value="$MEMFRAC" \
  'BEGIN {exit !(value ~ /^[0-9]+([.][0-9]+)?$/ && value > 0 && value < 1)}' || {
  echo "MEMFRAC must be a number strictly between 0 and 1" >&2
  exit 2
}

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
  local_image=$(IMAGE="$IMAGE" "$REPO_DIR/scripts/sparkflash-fp4-image-fingerprint.sh")
  worker_image=$(ssh -o BatchMode=yes "$WORKER_SSH" \
    "IMAGE=$(printf '%q' "$IMAGE") $(printf '%q' "$WORKER_REPO/scripts/sparkflash-fp4-image-fingerprint.sh")")
  [ "$local_image" = "$worker_image" ] || {
    echo "SparkFlash image payload mismatch" >&2
    exit 2
  }
  {
    printf 'repo_sha=%s\nrepo_payload=%s\nimage_payload=%s\n' \
      "$local_sha" "$local_payload" "$local_image"
    printf 'head_image_id='
    docker image inspect "$IMAGE" --format '{{.Id}}'
    printf 'worker_image_id='
    ssh -o BatchMode=yes "$WORKER_SSH" \
      docker image inspect "$IMAGE" --format '{{.Id}}'
    echo 'reproducibility PASS'
  } | tee "$RESULT_DIR/identity.txt"
}

verify_mtp_weights() {
  local path head_hash worker_hash
  path="$MODELS/inkling-small-nvfp4/mtp.safetensors"
  [ -f "$path" ] || { echo "missing head MTP weights: $path" >&2; exit 2; }
  ssh -o BatchMode=yes "$WORKER_SSH" "test -f $(printf '%q' "$path")" || {
    echo "missing worker MTP weights: $path" >&2
    exit 2
  }
  head_hash=$(sha256sum "$path" | awk '{print $1}')
  worker_hash=$(ssh -o BatchMode=yes "$WORKER_SSH" \
    "sha256sum $(printf '%q' "$path") | awk '{print \$1}'")
  [ "$head_hash" = "$worker_hash" ] || {
    echo "MTP weight mismatch: head=$head_hash worker=$worker_hash" >&2
    exit 2
  }
  printf 'MTP weight preflight PASS sha256=%s\n' "$head_hash" \
    | tee "$RESULT_DIR/mtp-weight-preflight.txt"
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
    echo "REJECT: native MTP failed before health; no serving claim"
    echo "head log tail:"
    tail -n 180 "$RESULT_DIR/mtp-head.log" 2>/dev/null || true
    echo "worker log tail:"
    ssh -o BatchMode=yes "$WORKER_SSH" \
      "tail -n 180 $(printf '%q' "$WORKER_REPO/$RESULT_DIR/mtp-worker.log")" \
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

verify_runtime() {
  docker inspect inkling-sglang >"$RESULT_DIR/mtp-head-inspect.json"
  ssh -o BatchMode=yes "$WORKER_SSH" docker inspect inkling-sglang \
    >"$RESULT_DIR/mtp-worker-inspect.json"
  python3 - "$IMAGE" "$MEMFRAC" "$RESULT_DIR/mtp-head-inspect.json" \
    "$RESULT_DIR/mtp-worker-inspect.json" <<'PY' \
    | tee "$RESULT_DIR/runtime-contract.txt"
import json
import sys
from pathlib import Path

image, memfrac = sys.argv[1:3]
required = {
    "--attention-backend": "fa4",
    "--page-size": "128",
    "--context-length": "65536",
    "--kv-cache-dtype": "fp4_mx_block16",
    "--speculative-algorithm": "EAGLE",
    "--speculative-num-steps": "1",
    "--speculative-eagle-topk": "1",
    "--speculative-num-draft-tokens": "2",
    "--moe-runner-backend": "marlin",
    "--fp4-gemm-backend": "flashinfer_trtllm",
    "--mem-fraction-static": memfrac,
}
required_flags = {"--enable-multi-layer-eagle", "--speculative-use-rejection-sampling"}
for path_text in sys.argv[3:]:
    path = Path(path_text)
    payload = json.loads(path.read_text(encoding="utf-8"))[0]
    if payload["Config"]["Image"] != image:
        raise SystemExit(f"{path}: wrong image")
    command = payload["Config"]["Cmd"]
    for flag, expected in required.items():
        if command.count(flag) != 1 or command[command.index(flag) + 1] != expected:
            raise SystemExit(f"{path}: {flag} contract failed")
    missing = required_flags.difference(command)
    if missing:
        raise SystemExit(f"{path}: missing {sorted(missing)}")
    forbidden = {
        "DSPARK",
        "--speculative-draft-model-path",
        "--speculative-draft-model-quantization",
        "--speculative-dspark-block-size",
    }
    present = forbidden.intersection(command)
    if present:
        raise SystemExit(f"{path}: mixed speculative paths: {sorted(present)}")
    if "SGLANG_OPT_USE_INKLING_SHEARED_BIAS=0" not in set(payload["Config"]["Env"]):
        raise SystemExit(f"{path}: score-mod bias contract failed")
print(
    f"MTP runtime contract PASS attention=fa4 page=128 kv=fp4 width=1 "
    f"memfrac={memfrac} external_draft=none"
)
PY
}

verify_capacity_and_logs() {
  python3 - "$RESULT_DIR/mtp-head.log" "$MIN_FULL_TOKENS" \
    "$MIN_POST_GRAPH_GB" <<'PY' \
    | tee "$RESULT_DIR/capacity-contract.txt"
import re
import sys
from pathlib import Path

path = Path(sys.argv[1])
minimum = int(sys.argv[2])
minimum_post_graph_gb = float(sys.argv[3])
text = path.read_text(encoding="utf-8", errors="replace")
matches = re.findall(
    r"Use sliding window memory pool\. full_layer_tokens=(\d+), "
    r"swa_layer_tokens=(\d+)",
    text,
)
if not matches:
    raise SystemExit(f"{path}: missing MTP target capacity")
full, swa = map(int, matches[0])
if full < minimum:
    raise SystemExit(f"MTP CAPACITY FAIL full={full} minimum={minimum}")
required = (
    "type=InklingForConditionalGenerationMTP",
    "Capture target verify CUDA graph end",
)
missing = [item for item in required if item not in text]
if missing:
    raise SystemExit(f"{path}: missing MTP log proofs: {missing}")
if "type=DSparkDraftModel" in text:
    raise SystemExit(f"{path}: external DSpark draft was loaded")
headroom_matches = re.findall(
    r"Capture target verify CUDA graph end\..*?avail mem=([0-9.]+) GB", text
)
if not headroom_matches:
    raise SystemExit(f"{path}: missing post-capture memory headroom")
post_graph_gb = float(headroom_matches[-1])
if post_graph_gb < minimum_post_graph_gb:
    raise SystemExit(
        f"MTP HEADROOM FAIL post_graph_gb={post_graph_gb} "
        f"minimum={minimum_post_graph_gb}"
    )
print(
    f"MTP CAPACITY PASS full_layer_tokens={full} minimum={minimum} "
    f"headroom={full - minimum} swa_layer_tokens={swa} "
    f"post_graph_gb={post_graph_gb} external_draft=none"
)
PY
}

start_server() {
  stop_arm
  ssh -f -o BatchMode=yes "$WORKER_SSH" \
    "mkdir -p $(printf '%q' "$WORKER_REPO/$RESULT_DIR") && cd $(printf '%q' "$WORKER_REPO") && exec env MASTER_IP=$(printf '%q' "$MASTER_IP") IF=$(printf '%q' "$IF") HCA=$(printf '%q' "$HCA") GID=$(printf '%q' "$GID") MODELS=$(printf '%q' "$MODELS") IMAGE=$(printf '%q' "$IMAGE") LOG=$(printf '%q' "$WORKER_REPO/$RESULT_DIR/mtp-worker.log") ATTN=fa4 MOE=marlin FP4GEMM=flashinfer_trtllm MEMFRAC=$(printf '%q' "$MEMFRAC") CTX=65536 SPEC=0 BLOCK=5 GRAPHS=1 GRAPH_BS=$(printf '%q' '1 2 3 4 5 6 7 8 10 12 14 16') RAGGED= INKLING_SHEARED_BIAS=0 MAXREQ=16 PAGE=128 CONTINUOUS_DECODE_STEPS=2 EXTRA_ARGS=$(printf '%q' "$MTP_ARGS") ./scripts/inkling-sglang-launch.sh 1 </dev/null >/dev/null 2>&1"
  sleep 3
  nohup env MASTER_IP="$MASTER_IP" IF="$IF" HCA="$HCA" GID="$GID" \
    MODELS="$MODELS" IMAGE="$IMAGE" LOG="$REPO_DIR/$RESULT_DIR/mtp-head.log" \
    ATTN=fa4 MOE=marlin FP4GEMM=flashinfer_trtllm MEMFRAC="$MEMFRAC" CTX=65536 \
    SPEC=0 BLOCK=5 GRAPHS=1 GRAPH_BS="1 2 3 4 5 6 7 8 10 12 14 16" RAGGED= \
    INKLING_SHEARED_BIAS=0 MAXREQ=16 PAGE=128 CONTINUOUS_DECODE_STEPS=2 \
    EXTRA_ARGS="$MTP_ARGS" \
    "$REPO_DIR/scripts/inkling-sglang-launch.sh" 0 </dev/null >/dev/null 2>&1 &
}

verify_reproducibility
verify_mtp_weights
start_server
if ! wait_ready; then
  record_boot_failure
  exit 4
fi
verify_runtime
verify_capacity_and_logs
lossless_gate "$RESULT_DIR/lossless-1.json" | tee "$RESULT_DIR/lossless-1.txt"
lossless_gate "$RESULT_DIR/lossless-2.json" | tee "$RESULT_DIR/lossless-2.txt"
echo "PASS: native width-1 MTP serves on FA4 FP4 with capacity and exact T4" \
  | tee "$RESULT_DIR/decision.txt"
