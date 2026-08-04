#!/bin/bash
# Dual-control exact writer + fixed-tile allocation gate for SparkFlash FP4.
set -euo pipefail

WORKER_SSH=${WORKER_SSH:?set passwordless worker SSH}
WORKER_REPO=${WORKER_REPO:?set the absolute repository path on the worker}
IMAGE=${IMAGE:-local/sglang-inkling:sparkflash-fp4-dev}
RESULT_DIR=${RESULT_DIR:-artifacts/fa4-fp4-pool-gate}
REPO_DIR=$(cd "$(dirname "$0")/.." && pwd)

mkdir -p "$RESULT_DIR"
ssh -o BatchMode=yes "$WORKER_SSH" \
  "mkdir -p $(printf '%q' "$WORKER_REPO/$RESULT_DIR")"

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
local_image_payload=$(IMAGE="$IMAGE" "$REPO_DIR/scripts/sparkflash-fp4-image-fingerprint.sh")
worker_image_payload=$(ssh -o BatchMode=yes "$WORKER_SSH" \
  "IMAGE=$(printf '%q' "$IMAGE") $(printf '%q' "$WORKER_REPO/scripts/sparkflash-fp4-image-fingerprint.sh")")
[ "$local_image_payload" = "$worker_image_payload" ] || {
  echo "image payload mismatch: head=$local_image_payload worker=$worker_image_payload" >&2
  exit 2
}
{
  printf 'repo_sha=%s\nrepo_payload=%s\nimage_payload=%s\n' \
    "$local_sha" "$local_payload" "$local_image_payload"
  printf 'head_image_id='
  docker image inspect "$IMAGE" --format '{{.Id}}'
  printf 'worker_image_id='
  ssh -o BatchMode=yes "$WORKER_SSH" \
    docker image inspect "$IMAGE" --format '{{.Id}}'
  echo 'reproducibility PASS'
} | tee "$RESULT_DIR/identity.txt"

docker run --rm --gpus all -v "$REPO_DIR:/repo:ro" \
  --entrypoint python3 "$IMAGE" /repo/benchmarks/fa4_sm121_fp4_pool_gate.py \
  >"$RESULT_DIR/control1-pool-gate.log" 2>&1
ssh -o BatchMode=yes "$WORKER_SSH" \
  "docker run --rm --gpus all -v $(printf '%q' "$WORKER_REPO:/repo:ro") --entrypoint python3 $(printf '%q' "$IMAGE") /repo/benchmarks/fa4_sm121_fp4_pool_gate.py" \
  >"$RESULT_DIR/control2-pool-gate.log" 2>&1

python3 - "$RESULT_DIR/control1-pool-gate.log" \
  "$RESULT_DIR/control2-pool-gate.log" <<'PY' | tee "$RESULT_DIR/contract.txt"
import json
import sys
from pathlib import Path

expected = {
    "capture_stream_selection",
    "ordinary_target_writer",
    "radix_move",
    "dspark_prefix_valid_writer",
    "prefix_valid_mask",
    "swa_wrapper_full_writer",
    "swa_wrapper_local_writer",
    "swa_radix_move",
    "no_pool_sized_bf16_allocation",
}
for path_text in sys.argv[1:]:
    path = Path(path_text)
    text = path.read_text(encoding="utf-8", errors="replace")
    if (
        "FA4 FP4 POOL GATE PASS writers=4 allocation=fixed-tile stream-select=1"
        not in text
    ):
        raise SystemExit(f"{path}: missing final pool-gate PASS")
    records = []
    for line in text.splitlines():
        if line.startswith("{"):
            records.append(json.loads(line))
    by_gate = {record["gate"]: record for record in records}
    if set(by_gate) != expected:
        raise SystemExit(
            f"{path}: gate set={sorted(by_gate)}, expected={sorted(expected)}"
        )
    for gate, record in by_gate.items():
        false_fields = [key for key, value in record.items() if value is False]
        if false_fields:
            raise SystemExit(f"{path}: {gate} false fields={false_fields}")
    allocation = by_gate["no_pool_sized_bf16_allocation"]
    if allocation["peak_delta_bytes"] >= allocation["peak_limit_bytes"]:
        raise SystemExit(f"{path}: allocation threshold failed: {allocation}")
    if allocation["storage_ratio"] != 0.28125:
        raise SystemExit(f"{path}: unexpected FP4 storage ratio: {allocation}")
    print(
        f"{path.name} PASS writers=4 pool_tokens={allocation['pool_tokens']} "
        f"peak_delta_bytes={allocation['peak_delta_bytes']} "
        f"storage_ratio={allocation['storage_ratio']}"
    )
print("DUAL-CONTROL FA4 FP4 POOL CONTRACT PASS")
PY

echo "PASS: dual-control FP4 payload+scale writers and fixed-tile allocation" \
  | tee "$RESULT_DIR/decision.txt"
