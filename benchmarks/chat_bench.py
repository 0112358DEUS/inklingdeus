#!/usr/bin/env python3
"""Chat-templated, statistically powered Inkling serving benchmark.

Every reported task result is 4 fixed prompts x 8 repetitions = n=32 by
default. Requests go through /v1/chat/completions so the model's chat template
is applied; this deliberately avoids the raw-/generate echo effect documented
in docs/MEASUREMENT-PROTOCOL.md.

Examples:
  python3 benchmarks/chat_bench.py baseline --task open-ended
  python3 benchmarks/chat_bench.py baseline --task all --output baseline.json
  INKLING_URL=http://192.168.192.3:30000 python3 benchmarks/chat_bench.py baseline
"""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import sys
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


DEFAULT_URL = os.environ.get("INKLING_URL", "http://localhost:30000").rstrip("/")
DEFAULT_MODEL = os.environ.get("INKLING_MODEL", "inkling-small")

TASKS: dict[str, tuple[str, ...]] = {
    "gsm8k": (
        "A school library had 240 books. It donated one quarter of them, then bought 36 new books. How many books does it have now? Show your reasoning step by step.",
        "A baker makes 18 trays with 14 rolls on each tray. She sells 167 rolls. How many rolls remain? Show your reasoning step by step.",
        "Mina saves $12 each week for 7 weeks, then spends $29 on a gift. How much money does she have left? Show your reasoning step by step.",
        "A bus travels 135 kilometers in the morning and 88 kilometers in the afternoon for 4 days. What total distance does it travel? Show your reasoning step by step.",
    ),
    "code": (
        "Explain what this Python function does, including its time complexity and one edge case:\n\ndef dedupe(items):\n    return list(dict.fromkeys(items))",
        "Explain the behavior of this JavaScript snippet and why the output may surprise someone:\n\nfor (var i = 0; i < 3; i++) setTimeout(() => console.log(i), 0);",
        "Explain what this SQL query computes and identify one possible performance issue:\n\nSELECT customer_id, COUNT(*) FROM orders WHERE created_at >= CURRENT_DATE - INTERVAL '30 days' GROUP BY customer_id ORDER BY COUNT(*) DESC;",
        "Explain how this shell pipeline works and name one filename-related pitfall:\n\nfind . -type f -name '*.log' -print0 | xargs -0 gzip",
    ),
    "chat": (
        "I'm starting a new job next week and feel nervous. Give me practical advice for the first day in a warm, conversational tone.",
        "Help me plan a simple vegetarian dinner for four people using common pantry ingredients. Keep it relaxed and practical.",
        "I need to tell a friend I cannot attend their birthday without sounding cold. Suggest a kind message and explain the tone.",
        "My home office feels distracting. Talk me through three small changes I can make today without buying expensive equipment.",
    ),
    "open-ended": (
        "Write a thoughtful short passage about why old maps remain fascinating even when modern navigation is more accurate.",
        "Describe how a city park changes from dawn to evening, focusing on people, sound, and light.",
        "Explore the idea that constraints can improve creativity, using concrete examples and a balanced conclusion.",
        "Write an engaging explanation of why rivers have shaped the growth of civilizations and still matter to cities today.",
    ),
}


@dataclass(frozen=True)
class Sample:
    task: str
    seed: int
    repetition: int
    completion_tokens: int
    elapsed_seconds: float
    tokens_per_second: float
    accept_length: float
    correct_drafts_histogram: list[int]


def mean_se(values: list[float]) -> tuple[float, float]:
    if not values:
        raise ValueError("cannot summarize an empty sample")
    mean = statistics.mean(values)
    se = statistics.stdev(values) / math.sqrt(len(values)) if len(values) > 1 else 0.0
    return mean, se


def request_body(model: str, prompt: str, max_tokens: int) -> dict[str, Any]:
    return {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0,
        "stream": False,
        "return_meta_info": True,
    }


def run_request(
    *, url: str, model: str, prompt: str, max_tokens: int, timeout: float
) -> tuple[int, float, float, list[int]]:
    req = urllib.request.Request(
        f"{url}/v1/chat/completions",
        data=json.dumps(request_body(model, prompt, max_tokens)).encode(),
        headers={"Content-Type": "application/json"},
    )
    started = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            payload = json.load(response)
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode(errors="replace")
        raise RuntimeError(f"HTTP {exc.code} from chat endpoint: {detail}") from exc
    elapsed = time.perf_counter() - started

    try:
        completion_tokens = int(payload["usage"]["completion_tokens"])
        meta = payload["choices"][0]["meta_info"]
    except (KeyError, IndexError, TypeError, ValueError) as exc:
        raise RuntimeError(
            "chat response lacks usage/completion_tokens or choices[0].meta_info; "
            "the benchmark requires SGLang return_meta_info support"
        ) from exc

    if completion_tokens <= 0:
        raise RuntimeError("chat response reported zero completion tokens")
    if not isinstance(meta, dict):
        raise RuntimeError("choices[0].meta_info is not an object")

    accept = meta.get("spec_accept_length")
    if accept is None:
        verify_count = meta.get("spec_verify_ct")
        if not verify_count:
            raise RuntimeError(
                "chat response has no spec_accept_length or usable spec_verify_ct; "
                "refusing to report an acceptance result"
            )
        accept = completion_tokens / float(verify_count)

    histogram = meta.get("spec_correct_drafts_histogram")
    if histogram is None:
        # Backward-compatible name used by earlier SGLang builds.
        histogram = meta.get("spec_accept_histogram")
    if histogram is None:
        histogram = []
    if not isinstance(histogram, list) or any(
        not isinstance(value, int) or value < 0 for value in histogram
    ):
        raise RuntimeError("speculative acceptance histogram is malformed")

    return completion_tokens, elapsed, float(accept), histogram


def summarize(
    samples: list[Sample], expected_block_size: int | None = None
) -> dict[str, Any]:
    tps = [sample.tokens_per_second for sample in samples]
    accepts = [sample.accept_length for sample in samples]
    tps_mean, tps_se = mean_se(tps)
    accept_mean, accept_se = mean_se(accepts)
    result = {
        "n": len(samples),
        "tokens_per_second": {
            "mean": tps_mean,
            "se": tps_se,
            "min": min(tps),
            "max": max(tps),
        },
        "accept_length": {
            "mean": accept_mean,
            "se": accept_se,
            "min": min(accepts),
            "max": max(accepts),
        },
    }
    histogram_complete = all(sample.correct_drafts_histogram for sample in samples)
    result["histogram_complete"] = histogram_complete
    if histogram_complete:
        width = max(len(sample.correct_drafts_histogram) for sample in samples)
        if expected_block_size is not None:
            # SGLang omits trailing zero bins when no verify step reaches the tail.
            width = max(width, expected_block_size + 1)
        histogram = [0] * width
        for sample in samples:
            for index, count in enumerate(sample.correct_drafts_histogram):
                histogram[index] += count
        verify_steps = sum(histogram)
        if verify_steps <= 0:
            raise RuntimeError("speculative acceptance histogram contains no verify steps")
        result["correct_drafts_histogram"] = histogram
        result["verify_steps"] = verify_steps
        result["accept_by_position"] = [
            {
                "position": position,
                "accepted_steps": sum(histogram[position:]),
                "accept_rate": sum(histogram[position:]) / verify_steps,
            }
            for position in range(1, len(histogram))
        ]
    return result


def format_summary(name: str, result: dict[str, Any]) -> str:
    tps = result["tokens_per_second"]
    accept = result["accept_length"]
    return (
        f"{name:12s} n={result['n']:3d}  "
        f"tok/s {tps['mean']:.2f} +/- {tps['se']:.2f}  "
        f"accept {accept['mean']:.3f} +/- {accept['se']:.3f}"
    )


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("label", help="configuration label stored in the result")
    parser.add_argument(
        "--task",
        choices=(*TASKS, "all", "pooled-open"),
        default="open-ended",
        help="task class to measure; each selected class is independently n=4*reps",
    )
    parser.add_argument("--reps", type=int, default=8, help="repetitions per prompt")
    parser.add_argument("--tokens", type=int, default=160, help="maximum completion tokens")
    parser.add_argument("--url", default=DEFAULT_URL, help="SGLang base URL")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="served model name")
    parser.add_argument("--timeout", type=float, default=300, help="request timeout in seconds")
    parser.add_argument("--output", type=Path, help="write full samples and summaries as JSON")
    parser.add_argument(
        "--block-size",
        type=int,
        help="record the DSpark block size in the result plan (required by E3)",
    )
    parser.add_argument(
        "--require-histogram",
        action="store_true",
        help="fail if SGLang does not return a verify-step acceptance histogram",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print the request plan and one representative body without making requests",
    )
    args = parser.parse_args(argv)
    if args.reps < 1 or args.tokens < 1 or args.timeout <= 0:
        parser.error("--reps, --tokens, and --timeout must be positive")
    if args.block_size is not None and args.block_size < 1:
        parser.error("--block-size must be positive")
    args.url = args.url.rstrip("/")
    return args


def selected_tasks(name: str) -> tuple[str, ...]:
    if name == "all":
        return tuple(TASKS)
    if name == "pooled-open":
        return ("code", "chat", "open-ended")
    return (name,)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv if argv is not None else sys.argv[1:])
    tasks = selected_tasks(args.task)
    plan = {
        "label": args.label,
        "url": args.url,
        "model": args.model,
        "tasks": list(tasks),
        "prompts_per_task": 4,
        "repetitions_per_prompt": args.reps,
        "samples_per_task": 4 * args.reps,
        "max_tokens": args.tokens,
        "warmups": 2,
        "endpoint": "/v1/chat/completions",
        "block_size": args.block_size,
    }
    if args.dry_run:
        print(json.dumps({"plan": plan, "example": request_body(args.model, TASKS[tasks[0]][0], args.tokens)}, indent=2))
        return 0

    # Exactly two warm-up calls, excluded from every reported result.
    warmup_prompts = (TASKS[tasks[0]][0], TASKS[tasks[0]][1])
    for prompt in warmup_prompts:
        run_request(
            url=args.url,
            model=args.model,
            prompt=prompt,
            max_tokens=min(32, args.tokens),
            timeout=args.timeout,
        )

    samples: list[Sample] = []
    summaries: dict[str, dict[str, Any]] = {}
    for task in tasks:
        task_samples: list[Sample] = []
        for seed, prompt in enumerate(TASKS[task]):
            seed_samples: list[Sample] = []
            for repetition in range(args.reps):
                completion_tokens, elapsed, accept, histogram = run_request(
                    url=args.url,
                    model=args.model,
                    prompt=prompt,
                    max_tokens=args.tokens,
                    timeout=args.timeout,
                )
                sample = Sample(
                    task=task,
                    seed=seed,
                    repetition=repetition,
                    completion_tokens=completion_tokens,
                    elapsed_seconds=elapsed,
                    tokens_per_second=completion_tokens / elapsed,
                    accept_length=accept,
                    correct_drafts_histogram=histogram,
                )
                if args.require_histogram and not histogram:
                    raise RuntimeError(
                        "SGLang returned no spec_correct_drafts_histogram; "
                        "E3 must not infer position acceptance from request averages"
                    )
                samples.append(sample)
                task_samples.append(sample)
                seed_samples.append(sample)
            seed_summary = summarize(seed_samples, args.block_size)
            print(format_summary(f"{task}/{seed + 1}", seed_summary), flush=True)
        summaries[task] = summarize(task_samples, args.block_size)
        print(format_summary(task, summaries[task]), flush=True)

    if len(tasks) > 1:
        pooled_samples = [sample for sample in samples if sample.task in tasks]
        summaries["pooled"] = summarize(pooled_samples, args.block_size)
        print(format_summary("pooled", summaries["pooled"]), flush=True)

    result = {
        "schema_version": 1,
        "created_at_unix": time.time(),
        "plan": plan,
        "summaries": summaries,
        "samples": [asdict(sample) for sample in samples],
    }
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
