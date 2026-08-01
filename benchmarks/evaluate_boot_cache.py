#!/usr/bin/env python3
"""Fail-closed decision for the E5 persistent-JIT-cache experiment."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path
from typing import Any

import compare_ab


def load_timing(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    required = {"label", "time_to_health_seconds", "time_to_t4_seconds"}
    missing = required.difference(payload)
    if missing:
        raise ValueError(f"{path}: missing timing fields: {', '.join(sorted(missing))}")
    for field in ("time_to_health_seconds", "time_to_t4_seconds"):
        value = payload[field]
        if not isinstance(value, (int, float)) or value <= 0:
            raise ValueError(f"{path}: {field} must be positive")
    if payload["time_to_t4_seconds"] < payload["time_to_health_seconds"]:
        raise ValueError(f"{path}: T4 cannot precede health")
    return payload


def decision(
    baseline_seconds: float,
    warm_seconds: float,
    baseline_throughput: dict[str, float],
    warm_throughput: dict[str, float],
    *,
    baseline_boot_se: float = 0.0,
    warm_boot_se: float = 0.0,
    maximum_warm_seconds: float = 240.0,
    minimum_saved_seconds: float = 60.0,
    maximum_warm_ratio: float = 0.8,
) -> tuple[str, str]:
    delta_tps = warm_throughput["mean"] - baseline_throughput["mean"]
    combined_se = math.sqrt(
        baseline_throughput["se"] ** 2 + warm_throughput["se"] ** 2
    )
    if delta_tps <= -combined_se:
        return "REJECT", "warm-cache serving throughput loses at least one combined SE"

    saved_seconds = baseline_seconds - warm_seconds
    warm_ratio = warm_seconds / baseline_seconds
    boot_bars_do_not_overlap = warm_seconds + warm_boot_se < baseline_seconds - baseline_boot_se
    if (
        warm_seconds < maximum_warm_seconds
        and saved_seconds >= minimum_saved_seconds
        and warm_ratio <= maximum_warm_ratio
        and boot_bars_do_not_overlap
    ):
        return "ACCEPT", "warm time-to-T4 clears the target with material boot savings"
    if warm_seconds >= baseline_seconds:
        return "REJECT", "persistent caches do not improve time-to-T4"
    return "INCONCLUSIVE", "boot improves, but not enough to accept the cache change"


def summarize_timings(timings: list[dict[str, Any]]) -> dict[str, float | int]:
    if len(timings) != 3:
        raise ValueError(f"boot comparison requires exactly 3 runs per arm, got {len(timings)}")
    values = [float(payload["time_to_t4_seconds"]) for payload in timings]
    return {
        "n": len(values),
        "mean": statistics.fmean(values),
        "se": statistics.stdev(values) / math.sqrt(len(values)),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cold_timing", type=Path)
    parser.add_argument("prime_timing", type=Path)
    parser.add_argument("baseline_benchmark", type=Path)
    parser.add_argument("warm_benchmark", type=Path)
    parser.add_argument("--baseline-timing", action="append", type=Path, required=True)
    parser.add_argument("--warm-timing", action="append", type=Path, required=True)
    parser.add_argument("--task", default="open-ended")
    parser.add_argument("--maximum-warm-seconds", type=float, default=240.0)
    parser.add_argument("--minimum-saved-seconds", type=float, default=60.0)
    parser.add_argument("--maximum-warm-ratio", type=float, default=0.8)
    args = parser.parse_args()

    cold_timing = load_timing(args.cold_timing)
    prime_timing = load_timing(args.prime_timing)
    baseline_boot = summarize_timings([load_timing(path) for path in args.baseline_timing])
    warm_boot = summarize_timings([load_timing(path) for path in args.warm_timing])
    baseline_benchmark = compare_ab.load(args.baseline_benchmark)
    warm_benchmark = compare_ab.load(args.warm_benchmark)
    compare_ab.check_comparable(baseline_benchmark, warm_benchmark, args.task)
    baseline_tps = baseline_benchmark["summaries"][args.task]["tokens_per_second"]
    warm_tps = warm_benchmark["summaries"][args.task]["tokens_per_second"]

    verdict, reason = decision(
        float(baseline_boot["mean"]),
        float(warm_boot["mean"]),
        baseline_tps,
        warm_tps,
        baseline_boot_se=float(baseline_boot["se"]),
        warm_boot_se=float(warm_boot["se"]),
        maximum_warm_seconds=args.maximum_warm_seconds,
        minimum_saved_seconds=args.minimum_saved_seconds,
        maximum_warm_ratio=args.maximum_warm_ratio,
    )
    saved = float(baseline_boot["mean"]) - float(warm_boot["mean"])
    ratio = float(warm_boot["mean"]) / float(baseline_boot["mean"])
    delta_tps = warm_tps["mean"] - baseline_tps["mean"]
    combined_se = math.sqrt(baseline_tps["se"] ** 2 + warm_tps["se"] ** 2)
    print(
        f"time_to_T4 cold={cold_timing['time_to_t4_seconds']:.1f}s "
        f"prime={prime_timing['time_to_t4_seconds']:.1f}s "
        f"baseline_warm={baseline_boot['mean']:.1f}+/-{baseline_boot['se']:.1f}s "
        f"cache_warm={warm_boot['mean']:.1f}+/-{warm_boot['se']:.1f}s "
        f"saved={saved:.1f}s warm_ratio={ratio:.3f}"
    )
    print(
        f"throughput baseline={baseline_tps['mean']:.3f}+/-{baseline_tps['se']:.3f} "
        f"warm={warm_tps['mean']:.3f}+/-{warm_tps['se']:.3f} "
        f"delta={delta_tps:+.3f} combined_se={combined_se:.3f}"
    )
    print(f"{verdict}: {reason}")
    return {"ACCEPT": 0, "REJECT": 2, "INCONCLUSIVE": 3}[verdict]


if __name__ == "__main__":
    raise SystemExit(main())
