#!/bin/bash
# E5: cold/prime/warm persistent compiler-cache experiment with OS-cache-balanced boot samples.
# Run on the idle head Spark with passwordless SSH to the idle worker.
set -euo pipefail

WORKER_SSH=${WORKER_SSH:?set passwordless worker SSH}
WORKER_REPO=${WORKER_REPO:?set the absolute inklingdeus repo path on the worker}
MASTER_IP=${MASTER_IP:?set the head IP on the primary RoCE link}
IF=${IF:?set the primary link netdev}
HCA=${HCA:?set the unchanged NCCL HCA list for every boot}
MODELS=${MODELS:?set the identical model directory path present on both nodes}
E5_CACHE_ROOT=${E5_CACHE_ROOT:?set a new, nonexistent absolute cache path on both nodes}
GID=${GID:-3}
IMAGE=${IMAGE:-local/sglang-inkling:gb10-kvquant}
FP4GEMM=${FP4GEMM:-flashinfer_trtllm}
CHAMPION_SPECULATOR=${CHAMPION_SPECULATOR:-dspark}
CHAMPION_BLOCK=${CHAMPION_BLOCK:-5}
RESULT_DIR=${RESULT_DIR:-artifacts/e5-jit-cache}
READY_TIMEOUT=${READY_TIMEOUT:-900}
REPS=${REPS:-8}
TOKENS=${TOKENS:-160}
REPO_DIR=$(cd "$(dirname "$0")/.." && pwd)
# shellcheck source=scripts/champion-profile.sh
source "$REPO_DIR/scripts/champion-profile.sh"
resolve_champion_profile "$CHAMPION_SPECULATOR" "$CHAMPION_BLOCK"

case "$E5_CACHE_ROOT" in
  /*) ;;
  *) echo "E5_CACHE_ROOT must be absolute" >&2; exit 2 ;;
esac
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

prepare_new_cache_roots() {
  [ ! -e "$E5_CACHE_ROOT" ] || {
    echo "refusing pre-existing head cache root: $E5_CACHE_ROOT" >&2
    exit 2
  }
  ssh -o BatchMode=yes "$WORKER_SSH" "test ! -e $(printf '%q' "$E5_CACHE_ROOT")" || {
    echo "refusing pre-existing worker cache root: $E5_CACHE_ROOT" >&2
    exit 2
  }
  mkdir -p "$E5_CACHE_ROOT"
  ssh -o BatchMode=yes "$WORKER_SSH" "mkdir -p $(printf '%q' "$E5_CACHE_ROOT")"
}

wait_stopped() {
  local deadline=$((SECONDS + 30))
  while curl -fsS http://127.0.0.1:30000/health >/dev/null 2>&1; do
    if [ "$SECONDS" -ge "$deadline" ]; then
      echo "stale server still answers after container stop" >&2
      return 1
    fi
    sleep 1
  done
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
        echo "server container exited before readiness" >&2
        return 1
      fi
    elif [ "$seen_container" = 1 ]; then
      echo "server container disappeared before readiness" >&2
      return 1
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

verify_runtime_mounts() {
  local label=$1 persist=$2
  docker inspect inkling-sglang >"$RESULT_DIR/$label-head-inspect.json"
  ssh -o BatchMode=yes "$WORKER_SSH" docker inspect inkling-sglang \
    >"$RESULT_DIR/$label-worker-inspect.json"
  python3 - "$persist" "$E5_CACHE_ROOT" "$PROFILE_BLOCK" "$PROFILE_SPEC" \
    "$RESULT_DIR/$label-head-inspect.json" "$RESULT_DIR/$label-worker-inspect.json" <<'PY'
import json
import sys
from pathlib import Path

persist = sys.argv[1] == "1"
root = Path(sys.argv[2])
profile_block = sys.argv[3]
profile_spec = sys.argv[4] == "1"
expected = {
    "/root/.triton": root / "triton",
    "/root/.cache/flashinfer": root / "flashinfer",
    "/root/.cache/sglang": root / "sglang",
    "/root/.cache/torch_extensions": root / "torch_extensions",
    "/tmp/torchinductor_root": root / "torchinductor",
    "/root/.nv/ComputeCache": root / "cuda",
}
expected_pairs = {
    "--num-continuous-decode-steps": "2",
    "--context-length": "1048576",
    "--kv-cache-dtype": "fp4_mx_block16",
    "--attention-backend": "triton",
    "--moe-runner-backend": "marlin",
    "--page-size": "1",
}
if profile_spec:
    expected_pairs["--speculative-dspark-block-size"] = profile_block
for path_text in sys.argv[5:]:
    path = Path(path_text)
    payload = json.loads(path.read_text(encoding="utf-8"))
    mounts = {item["Destination"]: Path(item["Source"]) for item in payload[0]["Mounts"]}
    command = payload[0]["Config"]["Cmd"]
    env = payload[0]["Config"]["Env"]
    for flag, wanted in expected_pairs.items():
        if command.count(flag) != 1:
            raise SystemExit(f"{path}: expected exactly one {flag}")
        actual = command[command.index(flag) + 1]
        if actual != wanted:
            raise SystemExit(f"{path}: {flag}={actual!r}, expected {wanted!r}")
    if "--triton-attention-num-kv-splits" in command:
        raise SystemExit(f"{path}: unexpected KV-split override")
    for prefix in ("NCCL_ALGO=", "NCCL_PROTO=", "SGLANG_RAGGED_VERIFY_MODE="):
        if any(entry.startswith(prefix) for entry in env):
            raise SystemExit(f"{path}: unexpected environment entry {prefix}")
    if persist:
        for destination, source in expected.items():
            if mounts.get(destination) != source:
                raise SystemExit(
                    f"{path}: {destination} maps to {mounts.get(destination)}, expected {source}"
                )
    else:
        unexpected = set(expected).intersection(mounts)
        if unexpected:
            raise SystemExit(f"{path}: baseline unexpectedly persists {sorted(unexpected)}")
print(f"runtime champion/cache-mount contract PASS persist={int(persist)}")
PY
}

record_boot_failure() {
  local label=$1
  {
    echo "REJECT: $label failed to become healthy"
    echo "head log tail:"
    tail -n 120 "$REPO_DIR/$RESULT_DIR/$label-head.log" 2>/dev/null || true
    echo "worker log tail:"
    ssh -o BatchMode=yes "$WORKER_SSH" \
      "tail -n 120 $(printf '%q' "$WORKER_REPO/$RESULT_DIR/$label-worker.log")" \
      2>/dev/null || true
  } | tee "$RESULT_DIR/$label-boot-failure.txt" >&2
}

run_arm() {
  local label=$1 persist=$2 run_benchmark=$3
  local started healthy t4
  stop_arm
  wait_stopped
  started=$(date +%s)
  ssh -f -o BatchMode=yes "$WORKER_SSH" \
    "mkdir -p $(printf '%q' "$WORKER_REPO/$RESULT_DIR") && cd $(printf '%q' "$WORKER_REPO") && exec env MASTER_IP=$(printf '%q' "$MASTER_IP") IF=$(printf '%q' "$IF") HCA=$(printf '%q' "$HCA") GID=$(printf '%q' "$GID") MODELS=$(printf '%q' "$MODELS") IMAGE=$(printf '%q' "$IMAGE") LOG=$(printf '%q' "$WORKER_REPO/$RESULT_DIR/$label-worker.log") ./scripts/locked-experiment-launch.sh 1 $(printf '%q' "$FP4GEMM") $(printf '%q' "$PROFILE_BLOCK") $(printf '%q' "$PROFILE_SPEC") 0.85 $(printf '%q' "$persist") $(printf '%q' "$PROFILE_EXTRA_ARGS") $(printf '%q' "$E5_CACHE_ROOT") </dev/null >/dev/null 2>&1"
  sleep 3
  nohup env MASTER_IP="$MASTER_IP" IF="$IF" HCA="$HCA" GID="$GID" \
    MODELS="$MODELS" IMAGE="$IMAGE" LOG="$REPO_DIR/$RESULT_DIR/$label-head.log" \
    "$REPO_DIR/scripts/locked-experiment-launch.sh" \
    0 "$FP4GEMM" "$PROFILE_BLOCK" "$PROFILE_SPEC" 0.85 "$persist" \
    "$PROFILE_EXTRA_ARGS" "$E5_CACHE_ROOT" </dev/null >/dev/null 2>&1 &
  if ! wait_ready; then
    record_boot_failure "$label"
    return 4
  fi
  healthy=$(date +%s)
  verify_runtime_mounts "$label" "$persist"
  lossless_gate | tee "$RESULT_DIR/$label-lossless-pre.txt"
  t4=$(date +%s)
  printf '{"schema_version":1,"label":"%s","persist_jit_cache":%s,"time_to_health_seconds":%s,"time_to_t4_seconds":%s}\n' \
    "$label" "$persist" "$((healthy - started))" "$((t4 - started))" \
    >"$RESULT_DIR/$label-timing.json"
  if [ "$run_benchmark" = 1 ]; then
    INKLING_URL=http://127.0.0.1:30000 python3 "$REPO_DIR/benchmarks/chat_bench.py" \
      "$label" --task open-ended --reps "$REPS" --tokens "$TOKENS" \
      --output "$RESULT_DIR/$label.json"
    lossless_gate | tee "$RESULT_DIR/$label-lossless-post.txt"
  fi
}

capture_cache_state() {
  local label=$1
  # Cache artifacts are written by the serving container as root. Traverse them from a separate
  # root container with the host cache mounted read-only; never chmod/chown the factor under test.
  docker run --rm \
    --mount "type=bind,src=$E5_CACHE_ROOT,dst=/cache,readonly" \
    --entrypoint find "$IMAGE" /cache -type f -printf '%P\t%s\n' | sort \
    >"$RESULT_DIR/$label-head-cache-manifest.txt"
  ssh -o BatchMode=yes "$WORKER_SSH" \
    "docker run --rm --mount $(printf '%q' "type=bind,src=$E5_CACHE_ROOT,dst=/cache,readonly") --entrypoint find $(printf '%q' "$IMAGE") /cache -type f -printf '%P\\t%s\\n' | sort" \
    >"$RESULT_DIR/$label-worker-cache-manifest.txt"
  [ -s "$RESULT_DIR/$label-head-cache-manifest.txt" ] || {
    echo "persistent cache remained empty on head" >&2
    exit 2
  }
  [ -s "$RESULT_DIR/$label-worker-cache-manifest.txt" ] || {
    echo "persistent cache remained empty on worker" >&2
    exit 2
  }
  docker run --rm \
    --mount "type=bind,src=$E5_CACHE_ROOT,dst=/cache,readonly" \
    --entrypoint du "$IMAGE" -sk /cache >"$RESULT_DIR/$label-head-cache-size-kib.txt"
  ssh -o BatchMode=yes "$WORKER_SSH" \
    "docker run --rm --mount $(printf '%q' "type=bind,src=$E5_CACHE_ROOT,dst=/cache,readonly") --entrypoint du $(printf '%q' "$IMAGE") -sk /cache" \
    >"$RESULT_DIR/$label-worker-cache-size-kib.txt"
}

verify_reproducibility
prepare_new_cache_roots

# Cold and prime runs establish the two states. The six comparison boots are ordered B-A-A-B-B-A
# so ordinary drift and an already-warm model-page cache cannot masquerade as a JIT-cache win.
run_arm e5-baseline-cold 0 0
run_arm e5-cache-prime 1 1
capture_cache_state e5-cache-prime
run_arm e5-baseline-warm-1 0 1
run_arm e5-cache-warm-1 1 1
run_arm e5-cache-warm-2 1 0
run_arm e5-baseline-warm-2 0 0
run_arm e5-baseline-warm-3 0 0
run_arm e5-cache-warm-3 1 0
capture_cache_state e5-cache-final
stop_arm

python3 "$REPO_DIR/benchmarks/evaluate_boot_cache.py" \
  "$RESULT_DIR/e5-baseline-cold-timing.json" \
  "$RESULT_DIR/e5-cache-prime-timing.json" \
  "$RESULT_DIR/e5-baseline-warm-1.json" \
  "$RESULT_DIR/e5-cache-warm-1.json" \
  --baseline-timing "$RESULT_DIR/e5-baseline-warm-1-timing.json" \
  --baseline-timing "$RESULT_DIR/e5-baseline-warm-2-timing.json" \
  --baseline-timing "$RESULT_DIR/e5-baseline-warm-3-timing.json" \
  --warm-timing "$RESULT_DIR/e5-cache-warm-1-timing.json" \
  --warm-timing "$RESULT_DIR/e5-cache-warm-2-timing.json" \
  --warm-timing "$RESULT_DIR/e5-cache-warm-3-timing.json" \
  --task open-ended | tee "$RESULT_DIR/decision.txt"
