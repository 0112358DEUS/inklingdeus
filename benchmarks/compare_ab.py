#!/usr/bin/env python3
"""Fail-closed comparison for two chat_bench.py result files."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


def load(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if payload.get("schema_version") != 1:
        raise ValueError(f"{path}: unsupported or missing schema_version")
    return payload


def check_comparable(a: dict[str, Any], b: dict[str, Any], task: str) -> None:
    comparable_fields = (
        "model",
        "url",
        "tasks",
        "prompts_per_task",
        "repetitions_per_prompt",
        "samples_per_task",
        "max_tokens",
        "endpoint",
    )
    differences = [
        field
        for field in comparable_fields
        if a["plan"].get(field) != b["plan"].get(field)
    ]
    if differences:
        raise ValueError(f"A/B plans differ in: {', '.join(differences)}")
    for name, payload in (("A", a), ("B", b)):
        if task not in payload["summaries"]:
            raise ValueError(f"arm {name} has no {task!r} summary")
        if payload["summaries"][task]["n"] != 32:
            raise ValueError(f"arm {name} must have exactly n=32 for {task!r}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("arm_a", type=Path)
    parser.add_argument("arm_b", type=Path)
    parser.add_argument("--task", default="open-ended")
    parser.add_argument("--minimum-gain", type=float, default=0.5)
    args = parser.parse_args()

    arm_a = load(args.arm_a)
    arm_b = load(args.arm_b)
    check_comparable(arm_a, arm_b, args.task)

    a = arm_a["summaries"][args.task]["tokens_per_second"]
    b = arm_b["summaries"][args.task]["tokens_per_second"]
    delta = b["mean"] - a["mean"]
    combined_se = math.sqrt(a["se"] ** 2 + b["se"] ** 2)
    error_bars_do_not_overlap = b["mean"] - b["se"] > a["mean"] + a["se"]

    print(
        f"A={a['mean']:.3f}+/-{a['se']:.3f} tok/s  "
        f"B={b['mean']:.3f}+/-{b['se']:.3f} tok/s  "
        f"delta={delta:+.3f}  combined_se={combined_se:.3f}"
    )
    if delta >= args.minimum_gain and error_bars_do_not_overlap:
        print("ACCEPT: gain meets the threshold and 1-SE error bars do not overlap")
        return 0
    if delta <= -combined_se:
        print("REJECT: throughput loss is at least one combined standard error")
        return 2
    print("INCONCLUSIVE: do not accept or quote a serving improvement")
    return 3


if __name__ == "__main__":
    raise SystemExit(main())
