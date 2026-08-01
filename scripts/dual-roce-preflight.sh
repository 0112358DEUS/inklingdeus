#!/bin/bash
# Verify both RoCE twins before E1's single-HCA vs dual-HCA serving A/B.
# Run on the head node with passwordless SSH to the worker.
set -euo pipefail

WORKER_SSH=${WORKER_SSH:?set WORKER_SSH, for example control2@<control2-host>}
HCAS_CSV=${HCAS_CSV:-rocep1s0f1,roceP2p1s0f1}
HEAD_IPS_CSV=${HEAD_IPS_CSV:?set one head IPv4 address per HCA, comma-separated}
WORKER_IPS_CSV=${WORKER_IPS_CSV:?set one worker IPv4 address per HCA, comma-separated}
MIN_GBITS=${MIN_GBITS:-100}
RESULT_DIR=${RESULT_DIR:-artifacts/e1-dual-roce}
DURATION=${DURATION:-10}
BASE_PORT=${BASE_PORT:-18515}

IFS=, read -r -a HCAS <<<"$HCAS_CSV"
IFS=, read -r -a HEAD_IPS <<<"$HEAD_IPS_CSV"
IFS=, read -r -a WORKER_IPS <<<"$WORKER_IPS_CSV"

if [ "${#HCAS[@]}" -ne 2 ] || [ "${#HEAD_IPS[@]}" -ne 2 ] || [ "${#WORKER_IPS[@]}" -ne 2 ]; then
  echo "E1 requires exactly two HCAs and one IPv4 address per HCA on each node" >&2
  exit 2
fi

mkdir -p "$RESULT_DIR"

require_command() {
  command -v "$1" >/dev/null || { echo "missing local command: $1" >&2; exit 2; }
  ssh -o BatchMode=yes "$WORKER_SSH" "command -v $1 >/dev/null" || {
    echo "missing worker command or SSH access: $1" >&2
    exit 2
  }
}

for command_name in ib_write_bw ibdev2netdev ibv_devinfo show_gids ethtool; do
  require_command "$command_name"
done

check_endpoint() {
  local where=$1 hca=$2 expected_ip=$3
  local command
  command="netdev=\$(ibdev2netdev | awk -v dev='$hca' '\$1 == dev {print \$5; exit}');
    test -n \"\$netdev\";
    ibv_devinfo -d '$hca' | grep -q 'PORT_ACTIVE';
    ip -4 -o addr show dev \"\$netdev\" | grep -q ' $expected_ip/';
    show_gids | awk -v dev='$hca' -v ip='$expected_ip' '\$1 == dev && \$5 == ip && \$6 == \"v2\" {found=1} END {exit !found}';
    speed=\$(ethtool \"\$netdev\" | awk -F: '/^[[:space:]]*Speed:/ {gsub(/[^0-9]/, \"\", \$2); print \$2}');
    test \"\$speed\" -ge 200000;
    printf '%s hca=%s netdev=%s ip=%s speed_mbps=%s\\n' '$where' '$hca' \"\$netdev\" '$expected_ip' \"\$speed\""
  if [ "$where" = head ]; then
    bash -c "$command"
  else
    ssh -o BatchMode=yes "$WORKER_SSH" "bash -c $(printf '%q' "$command")"
  fi
}

parse_average_gbits() {
  awk '
    /^[[:space:]]*[0-9]+[[:space:]]+[0-9]+/ && NF >= 4 {value=$4}
    END {if (value == "") exit 1; print value}
  ' "$1"
}

for index in 0 1; do
  hca=${HCAS[$index]}
  head_ip=${HEAD_IPS[$index]}
  worker_ip=${WORKER_IPS[$index]}
  port=$((BASE_PORT + index))
  server_log="$RESULT_DIR/${hca}-server.log"
  client_log="$RESULT_DIR/${hca}-client.log"

  check_endpoint head "$hca" "$head_ip"
  check_endpoint worker "$hca" "$worker_ip"

  ssh -o BatchMode=yes "$WORKER_SSH" \
    "timeout $((DURATION + 20)) ib_write_bw -d '$hca' --report_gbits -q 4 -R --force-link IB -D '$DURATION' -p '$port'" \
    >"$server_log" 2>&1 &
  server_pid=$!
  sleep 2
  if ! ib_write_bw -d "$hca" --report_gbits -q 4 -R --force-link IB \
    -D "$DURATION" -p "$port" "$worker_ip" >"$client_log" 2>&1; then
    wait "$server_pid" || true
    echo "ib_write_bw failed for $hca; inspect $client_log and $server_log" >&2
    exit 1
  fi
  wait "$server_pid"

  average=$(parse_average_gbits "$client_log")
  if ! awk -v value="$average" -v minimum="$MIN_GBITS" 'BEGIN {exit !(value >= minimum)}'; then
    echo "$hca averaged $average Gb/s, below the $MIN_GBITS Gb/s E1 gate" >&2
    exit 1
  fi
  echo "$hca PASS average_gbits=$average minimum_gbits=$MIN_GBITS"
done

echo "E1 raw-link preflight PASS: both twins meet the bandwidth gate"
