#!/usr/bin/env python3
"""Fail-closed decision for E6 MEMFRAC=0.85 vs 0.68 at C8/C16."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


def load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def validate_result(path: Path) -> dict[str, Any]:
    payload = load_json(path)
    if payload.get("schema_version") != 1:
        raise ValueError(f"{path}: unsupported or missing schema_version")
    plan = payload.get("plan", {})
    if plan.get("endpoint") != "/v1/chat/completions":
        raise ValueError(f"{path}: serving result is not chat-templated")
    if plan.get("concurrencies") != [8, 16]:
        raise ValueError(f"{path}: expected exact C8/C16 plan")
    if plan.get("samples_per_concurrency") != 32:
        raise ValueError(f"{path}: expected exact n=32 per concurrency")
    for concurrency in (8, 16):
        summary = payload.get("summaries", {}).get(str(concurrency), {})
        if summary.get("n") != 32:
            raise ValueError(f"{path}: C{concurrency} must contain exactly n=32")
        aggregate = summary.get("aggregate_tokens_per_second", {})
        for field in ("mean", "se"):
            if not isinstance(aggregate.get(field), (int, float)):
                raise ValueError(f"{path}: C{concurrency} aggregate {field} missing")
    return payload


def check_comparable(baseline: dict[str, Any], candidate: dict[str, Any]) -> None:
    fields = (
        "url",
        "model",
        "task",
        "prompts",
        "repetitions_per_prompt",
        "samples_per_concurrency",
        "concurrencies",
        "max_tokens",
        "warmups",
        "endpoint",
    )
    drift = [field for field in fields if baseline["plan"].get(field) != candidate["plan"].get(field)]
    if drift:
        raise ValueError(f"concurrency plans differ in: {', '.join(drift)}")


def decide(
    baseline_status: dict[str, Any],
    candidate_status: dict[str, Any],
    baseline: dict[str, Any] | None,
    candidate: dict[str, Any] | None,
) -> dict[str, Any]:
    baseline_stable = baseline_status.get("stable") is True
    candidate_stable = candidate_status.get("stable") is True
    baseline_memory_event = baseline_status.get("memory_event") is True

    if not candidate_stable:
        return {"verdict": "REJECT_0.68", "reason": "candidate did not remain stable and lossless"}
    if candidate is None:
        raise ValueError("stable candidate is missing C8/C16 results")
    if not baseline_stable:
        if not baseline_memory_event:
            return {
                "verdict": "INVALID",
                "reason": "baseline failed without host-memory or OOM evidence",
            }
        return {
            "verdict": "ACCEPT_0.68",
            "reason": "0.85 triggered host-memory protection while 0.68 completed losslessly",
        }
    if baseline is None:
        raise ValueError("stable baseline is missing C8/C16 results")

    check_comparable(baseline, candidate)
    comparisons = {}
    regressions = []
    for concurrency in (8, 16):
        a = baseline["summaries"][str(concurrency)]["aggregate_tokens_per_second"]
        b = candidate["summaries"][str(concurrency)]["aggregate_tokens_per_second"]
        delta = b["mean"] - a["mean"]
        combined_se = math.sqrt(a["se"] ** 2 + b["se"] ** 2)
        comparisons[str(concurrency)] = {
            "baseline_mean": a["mean"],
            "baseline_se": a["se"],
            "candidate_mean": b["mean"],
            "candidate_se": b["se"],
            "delta": delta,
            "combined_se": combined_se,
        }
        if delta <= -combined_se:
            regressions.append(concurrency)
    if regressions:
        return {
            "verdict": "RETAIN_0.85",
            "reason": f"0.68 loses at least one combined SE at C{regressions}",
            "comparisons": comparisons,
        }
    return {
        "verdict": "RETAIN_0.85",
        "reason": "0.85 completed C8/C16 safely; lowering the pool has no compensating win",
        "comparisons": comparisons,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline_status", type=Path)
    parser.add_argument("candidate_status", type=Path)
    parser.add_argument("baseline_result", type=Path)
    parser.add_argument("candidate_result", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    baseline_status = load_json(args.baseline_status)
    candidate_status = load_json(args.candidate_status)
    baseline = validate_result(args.baseline_result) if args.baseline_result.exists() else None
    candidate = validate_result(args.candidate_result) if args.candidate_result.exists() else None
    result = decide(baseline_status, candidate_status, baseline, candidate)
    print(f"{result['verdict']}: {result['reason']}")
    if result.get("comparisons"):
        for concurrency, comparison in result["comparisons"].items():
            print(
                f"C{concurrency} baseline={comparison['baseline_mean']:.2f}"
                f"+/-{comparison['baseline_se']:.2f} candidate="
                f"{comparison['candidate_mean']:.2f}+/-{comparison['candidate_se']:.2f} "
                f"delta={comparison['delta']:+.2f}"
            )
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return {
        "ACCEPT_0.68": 0,
        "RETAIN_0.85": 0,
        "REJECT_0.68": 2,
        "INVALID": 3,
    }[result["verdict"]]


if __name__ == "__main__":
    raise SystemExit(main())
