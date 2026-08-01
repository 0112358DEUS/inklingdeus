#!/bin/bash
# E6: same-session MEMFRAC=0.85 vs 0.68 under chat-templated C8/C16 load.
# Requires an already-active earlyoom service; this script never installs or enables it.
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
CHAMPION_SPECULATOR=${CHAMPION_SPECULATOR:-dspark}
CHAMPION_BLOCK=${CHAMPION_BLOCK:-5}
RESULT_DIR=${RESULT_DIR:-artifacts/e6-memfrac}
READY_TIMEOUT=${READY_TIMEOUT:-900}
MIN_AVAILABLE_KIB=${MIN_AVAILABLE_KIB:-12582912}
REPS=${REPS:-8}
TOKENS=${TOKENS:-160}
REPO_DIR=$(cd "$(dirname "$0")/.." && pwd)
# shellcheck source=scripts/champion-profile.sh
source "$REPO_DIR/scripts/champion-profile.sh"
resolve_champion_profile "$CHAMPION_SPECULATOR" "$CHAMPION_BLOCK"
HEAD_RESULT_DIR="$REPO_DIR/$RESULT_DIR"
WORKER_RESULT_DIR="$WORKER_REPO/$RESULT_DIR"
HEAD_GUARD_PID=
WORKER_GUARD_PID=
LAST_STABLE=false
LAST_MEMORY_EVENT=false

case "$RESULT_DIR" in
  /*|*..*) echo "RESULT_DIR must be a safe relative path" >&2; exit 2 ;;
esac
[ ! -e "$HEAD_RESULT_DIR" ] || {
  echo "refusing pre-existing result directory: $HEAD_RESULT_DIR" >&2
  exit 2
}
ssh -o BatchMode=yes "$WORKER_SSH" "test ! -e $(printf '%q' "$WORKER_RESULT_DIR")" || {
  echo "refusing pre-existing worker result directory: $WORKER_RESULT_DIR" >&2
  exit 2
}
mkdir -p "$HEAD_RESULT_DIR"
ssh -o BatchMode=yes "$WORKER_SSH" "mkdir -p $(printf '%q' "$WORKER_RESULT_DIR")"

stop_guards() {
  if [ -n "$HEAD_GUARD_PID" ]; then
    kill "$HEAD_GUARD_PID" >/dev/null 2>&1 || true
    wait "$HEAD_GUARD_PID" 2>/dev/null || true
    HEAD_GUARD_PID=
  fi
  if [ -n "$WORKER_GUARD_PID" ]; then
    ssh -o BatchMode=yes "$WORKER_SSH" "kill $(printf '%q' "$WORKER_GUARD_PID")" \
      >/dev/null 2>&1 || true
    WORKER_GUARD_PID=
  fi
}

stop_arm() {
  docker rm -f inkling-sglang >/dev/null 2>&1 || true
  ssh -o BatchMode=yes "$WORKER_SSH" docker rm -f inkling-sglang >/dev/null 2>&1 || true
}

cleanup() {
  stop_guards
  stop_arm
}
trap cleanup EXIT

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

run_earlyoom_preflight() {
  "$REPO_DIR/scripts/earlyoom-preflight.sh" \
    | tee "$HEAD_RESULT_DIR/head-earlyoom-preflight.txt"
  ssh -o BatchMode=yes "$WORKER_SSH" \
    "$(printf '%q' "$WORKER_REPO/scripts/earlyoom-preflight.sh")" \
    | tee "$HEAD_RESULT_DIR/worker-earlyoom-preflight.txt"
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
  while [ "$SECONDS" -lt "$deadline" ]; do
    if curl -fsS http://127.0.0.1:30000/health >/dev/null 2>&1; then
      return 0
    fi
    sleep 5
  done
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

start_guards() {
  local label=$1
  nohup env CONTAINER=inkling-sglang MIN_AVAILABLE_KIB="$MIN_AVAILABLE_KIB" \
    TRIP_FILE="$HEAD_RESULT_DIR/$label-head-memory-trip.txt" \
    LOG_FILE="$HEAD_RESULT_DIR/$label-head-memory.log" \
    "$REPO_DIR/scripts/host-memory-guard.sh" </dev/null >/dev/null 2>&1 &
  HEAD_GUARD_PID=$!
  WORKER_GUARD_PID=$(ssh -o BatchMode=yes "$WORKER_SSH" \
    "nohup env CONTAINER=inkling-sglang MIN_AVAILABLE_KIB=$(printf '%q' "$MIN_AVAILABLE_KIB") TRIP_FILE=$(printf '%q' "$WORKER_RESULT_DIR/$label-worker-memory-trip.txt") LOG_FILE=$(printf '%q' "$WORKER_RESULT_DIR/$label-worker-memory.log") $(printf '%q' "$WORKER_REPO/scripts/host-memory-guard.sh") </dev/null >/dev/null 2>&1 & worker_pid=\$!; disown \$worker_pid; echo \$worker_pid")
}

collect_host_evidence() {
  local label=$1 started=$2
  ssh -o BatchMode=yes "$WORKER_SSH" \
    "if test -f $(printf '%q' "$WORKER_RESULT_DIR/$label-worker-memory.log"); then cat $(printf '%q' "$WORKER_RESULT_DIR/$label-worker-memory.log"); fi" \
    >"$HEAD_RESULT_DIR/$label-worker-memory.log"
  ssh -o BatchMode=yes "$WORKER_SSH" \
    "if test -f $(printf '%q' "$WORKER_RESULT_DIR/$label-worker-memory-trip.txt"); then cat $(printf '%q' "$WORKER_RESULT_DIR/$label-worker-memory-trip.txt"); fi" \
    >"$HEAD_RESULT_DIR/$label-worker-memory-trip.txt"
  journalctl -k --since "@$started" --no-pager \
    >"$HEAD_RESULT_DIR/$label-head-kernel.log"
  journalctl -u earlyoom --since "@$started" --no-pager \
    >"$HEAD_RESULT_DIR/$label-head-earlyoom.log"
  ssh -o BatchMode=yes "$WORKER_SSH" \
    "journalctl -k --since @$(printf '%q' "$started") --no-pager" \
    >"$HEAD_RESULT_DIR/$label-worker-kernel.log"
  ssh -o BatchMode=yes "$WORKER_SSH" \
    "journalctl -u earlyoom --since @$(printf '%q' "$started") --no-pager" \
    >"$HEAD_RESULT_DIR/$label-worker-earlyoom.log"

  for node in head worker; do
    if [ -s "$HEAD_RESULT_DIR/$label-$node-memory.log" ]; then
      awk '
        {
          for (i=1; i<=NF; i++) {
            if ($i ~ /^mem_available_kib=/) {
              split($i, parts, "=")
              if (minimum == "" || parts[2] < minimum) minimum=parts[2]
            }
          }
        }
        END {if (minimum != "") print minimum}
      ' "$HEAD_RESULT_DIR/$label-$node-memory.log" \
        >"$HEAD_RESULT_DIR/$label-$node-min-available-kib.txt"
    fi
  done
}

has_memory_event() {
  local label=$1
  [ -s "$HEAD_RESULT_DIR/$label-head-memory-trip.txt" ] && return 0
  [ -s "$HEAD_RESULT_DIR/$label-worker-memory-trip.txt" ] && return 0
  grep -Eiq 'Out of memory|oom-kill|Killed process' \
    "$HEAD_RESULT_DIR/$label-head-kernel.log" "$HEAD_RESULT_DIR/$label-worker-kernel.log" \
    && return 0
  grep -Eiq 'sending SIGTERM|sending SIGKILL' \
    "$HEAD_RESULT_DIR/$label-head-earlyoom.log" "$HEAD_RESULT_DIR/$label-worker-earlyoom.log" \
    && return 0
  return 1
}

containers_running() {
  docker inspect -f '{{.State.Running}}' inkling-sglang 2>/dev/null | grep -qx true \
    && ssh -o BatchMode=yes "$WORKER_SSH" \
      "docker inspect -f '{{.State.Running}}' inkling-sglang 2>/dev/null | grep -qx true"
}

run_arm() {
  local label=$1 memfrac=$2
  local started benchmark_rc=1 post_t4_rc=1 running=0 stable=false memory_event=false
  stop_guards
  stop_arm
  wait_stopped
  started=$(date +%s)
  start_guards "$label"
  ssh -f -o BatchMode=yes "$WORKER_SSH" \
    "cd $(printf '%q' "$WORKER_REPO") && exec env MASTER_IP=$(printf '%q' "$MASTER_IP") IF=$(printf '%q' "$IF") HCA=$(printf '%q' "$HCA") GID=$(printf '%q' "$GID") MODELS=$(printf '%q' "$MODELS") IMAGE=$(printf '%q' "$IMAGE") LOG=$(printf '%q' "$WORKER_RESULT_DIR/$label-worker.log") ./scripts/locked-experiment-launch.sh 1 $(printf '%q' "$FP4GEMM") $(printf '%q' "$PROFILE_BLOCK") $(printf '%q' "$PROFILE_SPEC") $(printf '%q' "$memfrac") 0 $(printf '%q' "$PROFILE_EXTRA_ARGS") '' </dev/null >/dev/null 2>&1"
  sleep 3
  nohup env MASTER_IP="$MASTER_IP" IF="$IF" HCA="$HCA" GID="$GID" \
    MODELS="$MODELS" IMAGE="$IMAGE" LOG="$HEAD_RESULT_DIR/$label-head.log" \
    "$REPO_DIR/scripts/locked-experiment-launch.sh" \
    0 "$FP4GEMM" "$PROFILE_BLOCK" "$PROFILE_SPEC" "$memfrac" 0 \
    "$PROFILE_EXTRA_ARGS" "" </dev/null >/dev/null 2>&1 &

  if wait_ready && lossless_gate | tee "$HEAD_RESULT_DIR/$label-lossless-pre.txt"; then
    if INKLING_URL=http://127.0.0.1:30000 \
      python3 "$REPO_DIR/benchmarks/concurrency_bench.py" "$label" \
      --task open-ended --concurrency 8 16 --reps "$REPS" --tokens "$TOKENS" \
      --output "$HEAD_RESULT_DIR/$label.json" \
      2>&1 | tee "$HEAD_RESULT_DIR/$label-benchmark.log"; then
      benchmark_rc=0
    else
      benchmark_rc=$?
    fi
    if [ "$benchmark_rc" -eq 0 ] \
      && lossless_gate | tee "$HEAD_RESULT_DIR/$label-lossless-post.txt"; then
      post_t4_rc=0
    fi
  else
    printf 'boot or pre-benchmark T4 failed\n' \
      | tee "$HEAD_RESULT_DIR/$label-boot-or-t4-failure.txt" >&2
  fi

  if containers_running; then
    running=1
    docker stats --no-stream inkling-sglang >"$HEAD_RESULT_DIR/$label-head-docker-stats.txt"
    ssh -o BatchMode=yes "$WORKER_SSH" docker stats --no-stream inkling-sglang \
      >"$HEAD_RESULT_DIR/$label-worker-docker-stats.txt"
  fi
  stop_guards
  collect_host_evidence "$label" "$started"
  if has_memory_event "$label"; then
    memory_event=true
  fi
  if [ "$benchmark_rc" -eq 0 ] && [ "$post_t4_rc" -eq 0 ] \
    && [ "$running" -eq 1 ] && [ "$memory_event" = false ]; then
    stable=true
  fi
  printf '{"schema_version":1,"label":"%s","mem_fraction":%s,"stable":%s,"memory_event":%s,"benchmark_exit":%s,"post_t4_pass":%s,"both_containers_running":%s}\n' \
    "$label" "$memfrac" "$stable" "$memory_event" "$benchmark_rc" \
    "$([ "$post_t4_rc" -eq 0 ] && echo true || echo false)" \
    "$([ "$running" -eq 1 ] && echo true || echo false)" \
    >"$HEAD_RESULT_DIR/$label-status.json"
  stop_arm
  LAST_STABLE=$stable
  LAST_MEMORY_EVENT=$memory_event
  return 0
}

verify_reproducibility
run_earlyoom_preflight

# Baseline first, then the one-variable lower-memory candidate in the same session.
run_arm e6-memfrac-085 0.85
if [ "$LAST_STABLE" = false ] && [ "$LAST_MEMORY_EVENT" = false ]; then
  echo "baseline failed without OOM/early-memory evidence; experiment is invalid" >&2
  exit 3
fi
run_arm e6-memfrac-068 0.68

python3 "$REPO_DIR/benchmarks/evaluate_memfrac.py" \
  "$HEAD_RESULT_DIR/e6-memfrac-085-status.json" \
  "$HEAD_RESULT_DIR/e6-memfrac-068-status.json" \
  "$HEAD_RESULT_DIR/e6-memfrac-085.json" \
  "$HEAD_RESULT_DIR/e6-memfrac-068.json" \
  --output "$HEAD_RESULT_DIR/decision.json"
