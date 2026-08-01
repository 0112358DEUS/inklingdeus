#!/usr/bin/env python3
"""Select E3's DSpark block size from exact-n32 chat benchmark results."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


EXPECTED_BLOCKS = (5, 6, 7)
COMPARABLE_FIELDS = (
    "url",
    "model",
    "tasks",
    "prompts_per_task",
    "repetitions_per_prompt",
    "samples_per_task",
    "max_tokens",
    "endpoint",
)


def load(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        result = json.load(handle)
    if result.get("schema_version") != 1:
        raise ValueError(f"{path}: unsupported or missing schema_version")
    return result


def validate(results: list[dict[str, Any]], task: str) -> dict[int, dict[str, Any]]:
    by_block = {result["plan"].get("block_size"): result for result in results}
    if len(by_block) != 3 or set(by_block) != set(EXPECTED_BLOCKS):
        raise ValueError(f"expected exactly block sizes {EXPECTED_BLOCKS}")
    baseline_plan = by_block[7]["plan"]
    for block, result in by_block.items():
        drift = [
            field
            for field in COMPARABLE_FIELDS
            if result["plan"].get(field) != baseline_plan.get(field)
        ]
        if drift:
            raise ValueError(f"block {block} plan drift: {', '.join(drift)}")
        summary = result["summaries"].get(task)
        if summary is None or summary.get("n") != 32:
            raise ValueError(f"block {block} must have exactly n=32 for {task!r}")
        if not summary.get("histogram_complete"):
            raise ValueError(f"block {block} lacks a complete acceptance histogram")
        positions = summary.get("accept_by_position", [])
        if len(positions) < block:
            raise ValueError(
                f"block {block} histogram has only {len(positions)} draft positions"
            )
    return by_block


def decide(
    by_block: dict[int, dict[str, Any]], task: str, minimum_gain: float
) -> dict[str, Any]:
    baseline = by_block[7]["summaries"][task]["tokens_per_second"]
    candidates = []
    for block in (5, 6):
        throughput = by_block[block]["summaries"][task]["tokens_per_second"]
        delta = throughput["mean"] - baseline["mean"]
        combined_se = math.sqrt(throughput["se"] ** 2 + baseline["se"] ** 2)
        nonoverlap = (
            throughput["mean"] - throughput["se"]
            > baseline["mean"] + baseline["se"]
        )
        candidates.append(
            {
                "block_size": block,
                "mean": throughput["mean"],
                "se": throughput["se"],
                "delta_vs_7": delta,
                "combined_se": combined_se,
                "loses_one_se": delta <= -combined_se,
                "accepted": delta >= minimum_gain and nonoverlap,
            }
        )

    accepted = [candidate for candidate in candidates if candidate["accepted"]]
    winner = max(accepted, key=lambda candidate: candidate["mean"]) if accepted else None
    baseline_positions = by_block[7]["summaries"][task]["accept_by_position"]
    tail = {
        item["position"]: item["accept_rate"]
        for item in baseline_positions
        if item["position"] in (5, 6, 7)
    }
    return {
        "task": task,
        "minimum_gain": minimum_gain,
        "baseline": {"block_size": 7, **baseline},
        "candidates": candidates,
        "block7_tail_accept_rate": tail,
        "tail_6_7_below_5pct": tail.get(6, 1.0) < 0.05
        and tail.get(7, 1.0) < 0.05,
        "accepted_block_size": winner["block_size"] if winner else None,
        "verdict": "ACCEPT" if winner else "RETAIN_BLOCK_7",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", nargs=3, type=Path)
    parser.add_argument("--task", default="open-ended")
    parser.add_argument("--minimum-gain", type=float, default=0.5)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    by_block = validate([load(path) for path in args.results], args.task)
    decision = decide(by_block, args.task, args.minimum_gain)
    print(json.dumps(decision, indent=2))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(decision, indent=2) + "\n", encoding="utf-8")
    return 0 if decision["verdict"] == "ACCEPT" else 3


if __name__ == "__main__":
    raise SystemExit(main())
