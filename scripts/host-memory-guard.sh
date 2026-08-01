#!/bin/bash
# Experiment-scoped early-OOM guard: remove only the Inkling container before host memory is exhausted.
set -euo pipefail

CONTAINER=${CONTAINER:-inkling-sglang}
MIN_AVAILABLE_KIB=${MIN_AVAILABLE_KIB:-12582912}
POLL_SECONDS=${POLL_SECONDS:-1}
WAIT_FOR_CONTAINER_SECONDS=${WAIT_FOR_CONTAINER_SECONDS:-900}
TRIP_FILE=${TRIP_FILE:?set the trip-evidence path}
LOG_FILE=${LOG_FILE:?set the memory-sample log path}

case "$MIN_AVAILABLE_KIB:$POLL_SECONDS:$WAIT_FOR_CONTAINER_SECONDS" in
  *[!0-9:]*|:*|*::*) echo "numeric guard settings must be positive integers" >&2; exit 2 ;;
esac
if [ "$MIN_AVAILABLE_KIB" -le 0 ] || [ "$POLL_SECONDS" -le 0 ] \
  || [ "$WAIT_FOR_CONTAINER_SECONDS" -le 0 ]; then
  echo "numeric guard settings must be positive integers" >&2
  exit 2
fi
mkdir -p "$(dirname "$TRIP_FILE")" "$(dirname "$LOG_FILE")"

deadline=$((SECONDS + WAIT_FOR_CONTAINER_SECONDS))
until docker inspect -f '{{.State.Running}}' "$CONTAINER" 2>/dev/null | grep -qx true; do
  if [ "$SECONDS" -ge "$deadline" ]; then
    echo "container did not start before guard timeout" >&2
    exit 3
  fi
  sleep 1
done

while docker inspect -f '{{.State.Running}}' "$CONTAINER" 2>/dev/null | grep -qx true; do
  available=$(awk '/^MemAvailable:/ {print $2}' /proc/meminfo)
  pressure=$(awk '/^some / {for (i=1; i<=NF; i++) if ($i ~ /^avg10=/) {split($i,a,"="); print a[2]}}' \
    /proc/pressure/memory 2>/dev/null || true)
  printf '%s mem_available_kib=%s pressure_some_avg10=%s\n' \
    "$(date --iso-8601=seconds)" "$available" "${pressure:-unavailable}" >>"$LOG_FILE"
  if [ "$available" -lt "$MIN_AVAILABLE_KIB" ]; then
    printf '%s TRIPPED mem_available_kib=%s threshold_kib=%s container=%s\n' \
      "$(date --iso-8601=seconds)" "$available" "$MIN_AVAILABLE_KIB" "$CONTAINER" \
      | tee "$TRIP_FILE" >&2
    docker rm -f "$CONTAINER" >/dev/null 2>&1 || true
    exit 42
  fi
  sleep "$POLL_SECONDS"
done
