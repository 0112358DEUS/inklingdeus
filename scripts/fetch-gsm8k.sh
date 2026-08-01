#!/bin/bash
# Fetch the official OpenAI GSM8K test split and verify its immutable bytes.
set -euo pipefail

DEST=${1:-$HOME/.cache/inklingdeus/datasets/gsm8k-test.jsonl}
URL=https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/test.jsonl
EXPECTED_SHA256=3730d312f6e3440559ace48831e51066acaca737f6eabec99bccb9e4b3c39d14
EXPECTED_LINES=1319

mkdir -p "$(dirname "$DEST")"
if [ -f "$DEST" ]; then
  actual=$(sha256sum "$DEST" | awk '{print $1}')
  [ "$actual" = "$EXPECTED_SHA256" ] || {
    echo "existing GSM8K file has the wrong SHA256: $DEST" >&2
    exit 2
  }
else
  tmp="$DEST.tmp.$$"
  trap 'rm -f "$tmp"' EXIT
  curl -fsSL "$URL" -o "$tmp"
  actual=$(sha256sum "$tmp" | awk '{print $1}')
  [ "$actual" = "$EXPECTED_SHA256" ] || {
    echo "downloaded GSM8K SHA256 mismatch: $actual" >&2
    exit 2
  }
  mv "$tmp" "$DEST"
  trap - EXIT
fi

[ "$(wc -l <"$DEST" | tr -d ' ')" = "$EXPECTED_LINES" ] || {
  echo "GSM8K item-count mismatch" >&2
  exit 2
}
printf 'GSM8K dataset PASS path=%s sha256=%s items=%s\n' \
  "$DEST" "$EXPECTED_SHA256" "$EXPECTED_LINES"
