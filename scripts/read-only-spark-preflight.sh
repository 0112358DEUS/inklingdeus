#!/bin/bash
# Read-only readiness audit for the head/worker pair. Starts no containers and changes no files.
set -euo pipefail

WORKER_SSH=${WORKER_SSH:?set passwordless worker SSH}
WORKER_REPO=${WORKER_REPO:?set the absolute inklingdeus repo path on the worker}
MASTER_IP=${MASTER_IP:?set the head IP on the primary RoCE link}
IF=${IF:?set the RoCE netdev used by both nodes}
HCA=${HCA:?set the comma-separated HCA list intended for NCCL}
MODELS=${MODELS:?set the identical model directory path present on both nodes}
IMAGE=${IMAGE:-local/sglang-inkling:gb10-kvquant}
REPO_DIR=$(cd "$(dirname "$0")/.." && pwd)

failed=0
pass() { printf 'PASS %s\n' "$*"; }
fail() { printf 'FAIL %s\n' "$*" >&2; failed=1; }

if ssh -o BatchMode=yes -o ConnectTimeout=7 "$WORKER_SSH" true; then
  pass "passwordless worker SSH"
else
  fail "passwordless worker SSH"
  exit 2
fi

for command in python3 git docker nvidia-smi ip ibv_devinfo show_gids; do
  if command -v "$command" >/dev/null 2>&1; then
    pass "head command $command"
  else
    fail "head command missing: $command"
  fi
  if ssh -o BatchMode=yes "$WORKER_SSH" "command -v $(printf '%q' "$command") >/dev/null 2>&1"; then
    pass "worker command $command"
  else
    fail "worker command missing: $command"
  fi
done

printf 'head_identity '
hostname
id -un
uname -m
printf 'worker_identity '
ssh -o BatchMode=yes "$WORKER_SSH" 'hostname; id -un; uname -m'

if nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader; then
  pass "head NVIDIA runtime"
else
  fail "head NVIDIA runtime"
fi
if ssh -o BatchMode=yes "$WORKER_SSH" \
  'nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader'; then
  pass "worker NVIDIA runtime"
else
  fail "worker NVIDIA runtime"
fi

head_sha=$(git -C "$REPO_DIR" rev-parse HEAD)
worker_sha=$(ssh -o BatchMode=yes "$WORKER_SSH" \
  "git -C $(printf '%q' "$WORKER_REPO") rev-parse HEAD")
if [ "$head_sha" = "$worker_sha" ]; then
  pass "repo SHA $head_sha"
else
  fail "repo SHA mismatch head=$head_sha worker=$worker_sha"
fi

head_payload=$(python3 "$REPO_DIR/scripts/repo_fingerprint.py" --digest-only)
worker_payload=$(ssh -o BatchMode=yes "$WORKER_SSH" \
  "python3 $(printf '%q' "$WORKER_REPO/scripts/repo_fingerprint.py") --root $(printf '%q' "$WORKER_REPO") --digest-only")
if [ "$head_payload" = "$worker_payload" ]; then
  pass "repo payload $head_payload"
else
  fail "repo payload mismatch head=$head_payload worker=$worker_payload"
fi

for node in head worker; do
  if [ "$node" = head ]; then
    prefix=()
  else
    prefix=(ssh -o BatchMode=yes "$WORKER_SSH")
  fi
  if ${prefix[@]+"${prefix[@]}"} test -d "$MODELS/inkling-small-nvfp4"; then
    pass "$node target weights"
  else
    fail "$node target weights missing"
  fi
  if ${prefix[@]+"${prefix[@]}"} test -d "$MODELS/dspark-draft"; then
    pass "$node DSpark weights"
  else
    fail "$node DSpark weights missing"
  fi
  if ${prefix[@]+"${prefix[@]}"} docker image inspect "$IMAGE" >/dev/null 2>&1; then
    pass "$node image $IMAGE"
  else
    fail "$node image missing: $IMAGE"
  fi
  if ${prefix[@]+"${prefix[@]}"} test -e /dev/infiniband; then
    pass "$node RDMA device tree"
  else
    fail "$node /dev/infiniband missing"
  fi
  if ${prefix[@]+"${prefix[@]}"} ip link show "$IF" >/dev/null 2>&1; then
    pass "$node netdev $IF"
  else
    fail "$node netdev missing: $IF"
  fi
  if [ "$node" = head ]; then
    if ip -4 addr show "$IF" | grep -F "$MASTER_IP" >/dev/null; then
      pass "$node MASTER_IP on $IF"
    else
      fail "$node MASTER_IP not found on $IF"
    fi
  fi
  IFS=',' read -r -a hcas <<<"$HCA"
  for hca in "${hcas[@]}"; do
    if ${prefix[@]+"${prefix[@]}"} ibv_devinfo -d "$hca" >/dev/null 2>&1; then
      pass "$node HCA $hca"
    else
      fail "$node HCA missing: $hca"
    fi
  done
  if ${prefix[@]+"${prefix[@]}"} docker ps --filter name=^/inkling-sglang$ --format '{{.ID}}' \
    | grep -q .; then
    fail "$node is not idle: inkling-sglang is running"
  else
    pass "$node has no inkling-sglang container running"
  fi
done

printf 'head_memory '; awk '/^MemTotal:|^MemAvailable:/ {printf "%s=%s%s ", $1, $2, $3} END {print ""}' /proc/meminfo
printf 'worker_memory '
ssh -o BatchMode=yes "$WORKER_SSH" \
  "awk '/^MemTotal:|^MemAvailable:/ {printf \"%s=%s%s \", \$1, \$2, \$3} END {print \"\"}' /proc/meminfo"

if systemctl is-active --quiet earlyoom 2>/dev/null; then
  pass "head earlyoom active (required only for E6)"
else
  printf 'WARN head earlyoom inactive (E6 cannot run)\n' >&2
fi
if ssh -o BatchMode=yes "$WORKER_SSH" 'systemctl is-active --quiet earlyoom' 2>/dev/null; then
  pass "worker earlyoom active (required only for E6)"
else
  printf 'WARN worker earlyoom inactive (E6 cannot run)\n' >&2
fi

if [ "$failed" -ne 0 ]; then
  echo "read-only Spark preflight FAILED" >&2
  exit 2
fi
echo "read-only Spark preflight PASS"
