#!/usr/bin/env python3
"""Chat-templated exact-n concurrency benchmark for Inkling serving.

Each requested concurrency receives 4 fixed prompts x 8 repetitions = n=32
responses by default. Results include per-request latency/rate plus aggregate
tokens/s by fixed-size wave. This supersedes the legacy raw `/generate` probe.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import math
import statistics
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import chat_bench


@dataclass(frozen=True)
class ConcurrentSample:
    concurrency: int
    wave: int
    seed: int
    repetition: int
    completion_tokens: int
    elapsed_seconds: float
    tokens_per_second: float
    accept_length: float


@dataclass(frozen=True)
class Wave:
    concurrency: int
    wave: int
    requests: int
    completion_tokens: int
    elapsed_seconds: float
    aggregate_tokens_per_second: float


def work_items(task: str, repetitions: int) -> list[tuple[int, int, str]]:
    return [
        (seed, repetition, prompt)
        for repetition in range(repetitions)
        for seed, prompt in enumerate(chat_bench.TASKS[task])
    ]


def mean_se(values: list[float]) -> tuple[float, float]:
    if not values:
        raise ValueError("cannot summarize an empty sample")
    mean = statistics.fmean(values)
    se = statistics.stdev(values) / math.sqrt(len(values)) if len(values) > 1 else 0.0
    return mean, se


def summarize(samples: list[ConcurrentSample], waves: list[Wave]) -> dict[str, Any]:
    request_tps_mean, request_tps_se = mean_se(
        [sample.tokens_per_second for sample in samples]
    )
    latency_mean, latency_se = mean_se([sample.elapsed_seconds for sample in samples])
    accept_mean, accept_se = mean_se([sample.accept_length for sample in samples])
    wave_tps_mean, wave_tps_se = mean_se(
        [wave.aggregate_tokens_per_second for wave in waves]
    )
    total_tokens = sum(wave.completion_tokens for wave in waves)
    total_wall = sum(wave.elapsed_seconds for wave in waves)
    return {
        "n": len(samples),
        "wave_n": len(waves),
        "request_tokens_per_second": {"mean": request_tps_mean, "se": request_tps_se},
        "request_latency_seconds": {"mean": latency_mean, "se": latency_se},
        "accept_length": {"mean": accept_mean, "se": accept_se},
        "aggregate_tokens_per_second": {
            "mean": wave_tps_mean,
            "se": wave_tps_se,
            "overall": total_tokens / total_wall,
        },
        "completion_tokens": total_tokens,
        "wall_seconds": total_wall,
    }


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("label")
    parser.add_argument("--task", choices=tuple(chat_bench.TASKS), default="open-ended")
    parser.add_argument("--concurrency", type=int, nargs="+", default=[1, 2, 4, 8, 16])
    parser.add_argument("--reps", type=int, default=8)
    parser.add_argument("--tokens", type=int, default=160)
    parser.add_argument("--url", default=chat_bench.DEFAULT_URL)
    parser.add_argument("--model", default=chat_bench.DEFAULT_MODEL)
    parser.add_argument("--timeout", type=float, default=600)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if args.reps < 1 or args.tokens < 1 or args.timeout <= 0:
        parser.error("--reps, --tokens, and --timeout must be positive")
    if not args.concurrency or any(value < 1 for value in args.concurrency):
        parser.error("--concurrency values must be positive")
    if len(set(args.concurrency)) != len(args.concurrency):
        parser.error("--concurrency values must be unique")
    samples_per_level = len(chat_bench.TASKS[args.task]) * args.reps
    if any(samples_per_level % value for value in args.concurrency):
        parser.error("each concurrency must divide the per-level sample count exactly")
    args.url = args.url.rstrip("/")
    return args


def run_level(args: argparse.Namespace, concurrency: int) -> tuple[list[ConcurrentSample], list[Wave]]:
    items = work_items(args.task, args.reps)
    samples: list[ConcurrentSample] = []
    waves: list[Wave] = []
    for wave_index, offset in enumerate(range(0, len(items), concurrency)):
        batch = items[offset : offset + concurrency]
        started = time.perf_counter()
        with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as executor:
            futures = [
                executor.submit(
                    chat_bench.run_request,
                    url=args.url,
                    model=args.model,
                    prompt=prompt,
                    max_tokens=args.tokens,
                    timeout=args.timeout,
                )
                for _, _, prompt in batch
            ]
            results = [future.result() for future in futures]
        wall = time.perf_counter() - started
        wave_tokens = 0
        for (seed, repetition, _), result in zip(batch, results):
            completion_tokens, elapsed, accept, _ = result
            wave_tokens += completion_tokens
            samples.append(
                ConcurrentSample(
                    concurrency=concurrency,
                    wave=wave_index,
                    seed=seed,
                    repetition=repetition,
                    completion_tokens=completion_tokens,
                    elapsed_seconds=elapsed,
                    tokens_per_second=completion_tokens / elapsed,
                    accept_length=accept,
                )
            )
        waves.append(
            Wave(
                concurrency=concurrency,
                wave=wave_index,
                requests=len(batch),
                completion_tokens=wave_tokens,
                elapsed_seconds=wall,
                aggregate_tokens_per_second=wave_tokens / wall,
            )
        )
        print(
            f"C{concurrency} wave={wave_index + 1}/{len(items) // concurrency} "
            f"tokens={wave_tokens} wall={wall:.2f}s aggregate={wave_tokens / wall:.2f} tok/s",
            flush=True,
        )
    return samples, waves


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv if argv is not None else sys.argv[1:])
    sample_count = len(chat_bench.TASKS[args.task]) * args.reps
    plan = {
        "label": args.label,
        "url": args.url,
        "model": args.model,
        "task": args.task,
        "prompts": len(chat_bench.TASKS[args.task]),
        "repetitions_per_prompt": args.reps,
        "samples_per_concurrency": sample_count,
        "concurrencies": args.concurrency,
        "max_tokens": args.tokens,
        "warmups": 2,
        "endpoint": "/v1/chat/completions",
    }
    if args.dry_run:
        print(json.dumps({"plan": plan}, indent=2))
        return 0

    for prompt in chat_bench.TASKS[args.task][:2]:
        chat_bench.run_request(
            url=args.url,
            model=args.model,
            prompt=prompt,
            max_tokens=min(32, args.tokens),
            timeout=args.timeout,
        )

    all_samples: list[ConcurrentSample] = []
    all_waves: list[Wave] = []
    summaries: dict[str, dict[str, Any]] = {}
    for concurrency in args.concurrency:
        samples, waves = run_level(args, concurrency)
        if len(samples) != sample_count:
            raise RuntimeError(
                f"C{concurrency} produced n={len(samples)}, expected exactly {sample_count}"
            )
        all_samples.extend(samples)
        all_waves.extend(waves)
        summary = summarize(samples, waves)
        summaries[str(concurrency)] = summary
        aggregate = summary["aggregate_tokens_per_second"]
        print(
            f"C{concurrency} n={summary['n']} aggregate "
            f"{aggregate['mean']:.2f} +/- {aggregate['se']:.2f} tok/s",
            flush=True,
        )

    result = {
        "schema_version": 1,
        "created_at_unix": time.time(),
        "plan": plan,
        "summaries": summaries,
        "samples": [asdict(sample) for sample in all_samples],
        "waves": [asdict(wave) for wave in all_waves],
    }
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
