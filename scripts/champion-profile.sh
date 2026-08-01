#!/bin/bash
# Resolve the serving factor inherited by experiments after E3/E4.
# shellcheck disable=SC2034  # Globals are consumed by scripts that source this library.

resolve_champion_profile() {
  local speculator=${1:?speculator profile is required}
  local block=${2:?DSpark block is required}
  case "$block" in
    *[!0-9]*|0|'') echo "CHAMPION_BLOCK must be a positive integer" >&2; return 2 ;;
  esac
  PROFILE_BLOCK=$block
  case "$speculator" in
    dspark)
      PROFILE_SPEC=1
      PROFILE_EXTRA_ARGS=
      ;;
    mtp-width1)
      PROFILE_SPEC=0
      PROFILE_EXTRA_ARGS="--speculative-algorithm EAGLE --speculative-num-steps 1 --speculative-eagle-topk 1 --speculative-num-draft-tokens 2 --enable-multi-layer-eagle --speculative-use-rejection-sampling"
      ;;
    *)
      echo "CHAMPION_SPECULATOR must be dspark or mtp-width1" >&2
      return 2
      ;;
  esac
}
