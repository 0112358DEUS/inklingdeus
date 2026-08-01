#!/bin/bash
# Read-only E6 preflight: require host-level earlyoom protection and readable OOM evidence.
set -euo pipefail

command -v earlyoom >/dev/null 2>&1 || {
  echo "earlyoom is not installed; do not start E6" >&2
  exit 2
}
systemctl is-active --quiet earlyoom || {
  echo "earlyoom.service is not active; do not start E6" >&2
  exit 2
}
journalctl -u earlyoom -n 0 --no-pager >/dev/null || {
  echo "earlyoom journal is not readable by this user" >&2
  exit 2
}
journalctl -k -n 0 --no-pager >/dev/null || {
  echo "kernel journal is not readable by this user" >&2
  exit 2
}

printf 'earlyoom preflight PASS\n'
printf 'exec_start=%s\n' "$(systemctl show -p ExecStart --value earlyoom)"
awk '/^MemTotal:|^MemAvailable:/ {print}' /proc/meminfo
