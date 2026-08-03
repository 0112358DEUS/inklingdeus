#!/usr/bin/env python3
"""Token-measured needle-in-a-haystack checks at 512K and 1M chat context."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Callable


ADJECTIVES = (
    "amber",
    "brisk",
    "cedar",
    "distant",
    "evening",
    "frosted",
    "granite",
    "harbor",
    "indigo",
    "juniper",
    "kindled",
    "lunar",
)
NOUNS = (
    "archive",
    "bridge",
    "courtyard",
    "district",
    "estuary",
    "foundry",
    "garden",
    "harvest",
    "island",
    "junction",
    "kiln",
    "library",
)
VERBS = (
    "catalogued",
    "described",
    "examined",
    "followed",
    "gathered",
    "indexed",
    "mapped",
    "noted",
    "observed",
    "recorded",
    "surveyed",
    "traced",
)


def filler_record(index: int, seed: int) -> str:
    a = ADJECTIVES[(index * 5 + seed) % len(ADJECTIVES)]
    b = NOUNS[(index * 7 + seed * 3) % len(NOUNS)]
    c = VERBS[(index * 11 + seed * 5) % len(VERBS)]
    d = ADJECTIVES[(index * 13 + seed * 7) % len(ADJECTIVES)]
    e = NOUNS[(index * 17 + seed * 11) % len(NOUNS)]
    checksum = hashlib.sha256(f"{seed}:{index}".encode()).hexdigest()[:10]
    return (
        f"Archive record {index:06d}: The {a} {b} was {c} beside the {d} {e}. "
        f"Clerks filed the observation under checksum {checksum}; no action was required.\n"
    )


def secret_for(target: int, depth: float, seed: int) -> str:
    digest = hashlib.sha256(f"{target}:{depth:.3f}:{seed}".encode()).hexdigest()[:8].upper()
    return f"QZV-{digest}-KAPPA"


def build_messages(
    *, record_count: int, depth: float, secret: str, seed: int
) -> tuple[list[dict[str, str]], int]:
    needle_index = min(record_count, max(0, round(record_count * depth)))
    records = [filler_record(index, seed) for index in range(record_count)]
    needle = (
        f"CRITICAL ARCHIVE NOTE: The exact vault access code is {secret}. "
        "Preserve this value exactly.\n"
    )
    records.insert(needle_index, needle)
    prompt = (
        "Read the archive below. One critical note gives a vault access code. "
        "At the end, answer with that exact code only.\n\n"
        + "".join(records)
        + "\nQuestion: What is the exact vault access code? Answer with the code only."
    )
    return [{"role": "user", "content": prompt}], needle_index


def calibrate_record_count(
    *,
    target_tokens: int,
    tolerance_tokens: int,
    count_for_records: Callable[[int], int],
    initial_tokens_per_record: float = 36.0,
    max_iterations: int = 5,
) -> tuple[int, int]:
    record_count = max(1, round(target_tokens / initial_tokens_per_record))
    previous: tuple[int, int] | None = None
    for _ in range(max_iterations):
        token_count = count_for_records(record_count)
        if abs(token_count - target_tokens) <= tolerance_tokens:
            return record_count, token_count
        if previous is not None and previous[0] != record_count:
            slope = (token_count - previous[1]) / (record_count - previous[0])
        else:
            slope = token_count / record_count
        if slope <= 0:
            raise RuntimeError("token calibration observed a non-positive record slope")
        adjustment = round((target_tokens - token_count) / slope)
        if adjustment == 0:
            adjustment = 1 if token_count < target_tokens else -1
        previous = (record_count, token_count)
        record_count = max(1, record_count + adjustment)
    token_count = count_for_records(record_count)
    if abs(token_count - target_tokens) > tolerance_tokens:
        raise RuntimeError(
            f"could not calibrate target {target_tokens}: got {token_count} with "
            f"{record_count} records"
        )
    return record_count, token_count


def post_json(url: str, path: str, body: dict[str, Any], timeout: float) -> dict[str, Any]:
    request = urllib.request.Request(
        f"{url}{path}",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.load(response)
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode(errors="replace")
        raise RuntimeError(f"{path}: HTTP {exc.code}: {detail}") from exc


def tokenize_messages(
    *, url: str, model: str, messages: list[dict[str, str]], timeout: float
) -> int:
    payload = post_json(
        url,
        "/v1/tokenize",
        {
            "model": model,
            "messages": messages,
            "reasoning_effort": "none",
        },
        timeout,
    )
    count = payload.get("count")
    tokens = payload.get("tokens")
    if not isinstance(count, int) or not isinstance(tokens, list) or count != len(tokens):
        raise RuntimeError("/v1/tokenize returned an inconsistent count/token list")
    max_model_len = payload.get("max_model_len")
    if isinstance(max_model_len, int) and 0 < max_model_len < count:
        raise RuntimeError(
            f"tokenizer reports max_model_len={max_model_len}, below measured input {count}"
        )
    return count


def run_case(
    *,
    url: str,
    model: str,
    target_tokens: int,
    depth: float,
    seed: int,
    tolerance_tokens: int,
    max_tokens: int,
    timeout: float,
) -> dict[str, Any]:
    secret = secret_for(target_tokens, depth, seed)

    def count_for_records(record_count: int) -> int:
        messages, _ = build_messages(
            record_count=record_count,
            depth=depth,
            secret=secret,
            seed=seed,
        )
        return tokenize_messages(url=url, model=model, messages=messages, timeout=timeout)

    record_count, measured_tokens = calibrate_record_count(
        target_tokens=target_tokens,
        tolerance_tokens=tolerance_tokens,
        count_for_records=count_for_records,
    )
    messages, needle_index = build_messages(
        record_count=record_count,
        depth=depth,
        secret=secret,
        seed=seed,
    )
    final_count = tokenize_messages(url=url, model=model, messages=messages, timeout=timeout)
    if final_count != measured_tokens:
        raise RuntimeError("tokenizer count changed for identical NIAH messages")
    started = time.perf_counter()
    payload = post_json(
        url,
        "/v1/chat/completions",
        {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": 0,
            "stream": False,
            "reasoning_effort": "none",
        },
        timeout,
    )
    elapsed = time.perf_counter() - started
    try:
        message = payload["choices"][0]["message"]
        content = message.get("content") or ""
        prompt_tokens = int(payload["usage"]["prompt_tokens"])
    except (KeyError, IndexError, TypeError, ValueError) as exc:
        raise RuntimeError("malformed NIAH chat response") from exc
    if prompt_tokens != final_count:
        raise RuntimeError(
            f"tokenize/chat prompt count mismatch: tokenize={final_count}, chat={prompt_tokens}"
        )
    passed = secret in content.upper()
    return {
        "target_tokens": target_tokens,
        "measured_prompt_tokens": final_count,
        "tolerance_tokens": tolerance_tokens,
        "depth": depth,
        "record_count": record_count,
        "needle_record_index": needle_index,
        "secret": secret,
        "passed": passed,
        "content": content,
        "reasoning_content": message.get("reasoning_content"),
        "elapsed_seconds": elapsed,
        "completion_tokens": payload["usage"].get("completion_tokens"),
    }


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default=os.environ.get("INKLING_URL", "http://localhost:30000"))
    parser.add_argument("--model", default=os.environ.get("INKLING_MODEL", "inkling-small"))
    parser.add_argument("--targets", type=int, nargs="+", default=[512_000, 1_000_000])
    parser.add_argument("--depths", type=float, nargs="+", default=[0.1, 0.5, 0.9])
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--timeout", type=float, default=7200)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="resume an identity-matched partial output checkpoint",
    )
    args = parser.parse_args(argv)
    if any(target < 1 for target in args.targets) or args.max_tokens < 1 or args.timeout <= 0:
        parser.error("targets, max tokens, and timeout must be positive")
    if any(not 0 < depth < 1 for depth in args.depths):
        parser.error("depths must be strictly between 0 and 1")
    if any(target + args.max_tokens > 1_048_576 for target in args.targets):
        parser.error("target plus output budget exceeds the served 1,048,576-token context")
    args.url = args.url.rstrip("/")
    return args


def case_key(result: dict[str, Any]) -> tuple[int, float, int]:
    return (
        int(result["target_tokens"]),
        float(result["depth"]),
        int(result["seed"]),
    )


def build_payload(
    *,
    plan: dict[str, Any],
    model: str,
    results: list[dict[str, Any]],
) -> dict[str, Any]:
    by_target = {
        str(target): {
            "n": sum(result["target_tokens"] == target for result in results),
            "completed": sum(result["target_tokens"] == target for result in results),
            "passed": sum(
                result["target_tokens"] == target and result["passed"]
                for result in results
            ),
        }
        for target in plan["targets"]
    }
    complete = len(results) == plan["cases"]
    return {
        "schema_version": 2,
        "plan": plan,
        "model": model,
        "complete": complete,
        "completed_cases": len(results),
        "all_passed": (
            all(result["passed"] for result in results) if complete else None
        ),
        "by_target": by_target,
        "results": results,
    }


def write_checkpoint(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def load_checkpoint(
    *,
    path: Path,
    plan: dict[str, Any],
    model: str,
    expected_keys: list[tuple[int, float, int]],
) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 2:
        raise RuntimeError("NIAH checkpoint schema mismatch")
    if payload.get("plan") != plan or payload.get("model") != model:
        raise RuntimeError("NIAH checkpoint plan/model mismatch")
    results = payload.get("results")
    if not isinstance(results, list):
        raise RuntimeError("NIAH checkpoint results are malformed")
    actual_keys = [case_key(result) for result in results]
    if actual_keys != expected_keys[: len(actual_keys)]:
        raise RuntimeError("NIAH checkpoint is not an exact case-order prefix")
    if payload.get("completed_cases") != len(results):
        raise RuntimeError("NIAH checkpoint completed-case count mismatch")
    return results


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv if argv is not None else sys.argv[1:])
    plan = {
        "targets": args.targets,
        "depths": args.depths,
        "cases": len(args.targets) * len(args.depths),
        "endpoint": "/v1/chat/completions",
        "tokenizer_endpoint": "/v1/tokenize",
        "reasoning_effort": "none",
        "max_tokens": args.max_tokens,
    }
    if args.dry_run:
        print(json.dumps(plan, indent=2))
        return 0

    ordered_cases = [
        (target, depth, seed)
        for target in args.targets
        for seed, depth in enumerate(args.depths)
    ]
    if args.output.exists():
        if not args.resume:
            raise RuntimeError(
                f"refusing to overwrite existing NIAH output without --resume: {args.output}"
            )
        results = load_checkpoint(
            path=args.output,
            plan=plan,
            model=args.model,
            expected_keys=ordered_cases,
        )
        print(
            f"NIAH resume completed={len(results)} total={len(ordered_cases)}",
            flush=True,
        )
    else:
        results = []

    for case_index, (target, depth, seed) in enumerate(ordered_cases):
        if case_index < len(results):
            continue
        tolerance = max(256, round(target * 0.001))
        result = run_case(
            url=args.url,
            model=args.model,
            target_tokens=target,
            depth=depth,
            seed=seed,
            tolerance_tokens=tolerance,
            max_tokens=args.max_tokens,
            timeout=args.timeout,
        )
        result["seed"] = seed
        results.append(result)
        write_checkpoint(
            args.output,
            build_payload(plan=plan, model=args.model, results=results),
        )
        print(
            f"NIAH target={target} measured={result['measured_prompt_tokens']} "
            f"depth={depth:.2f} passed={result['passed']} "
            f"elapsed={result['elapsed_seconds']:.1f}s checkpointed=true",
            flush=True,
        )

    payload = build_payload(plan=plan, model=args.model, results=results)
    write_checkpoint(args.output, payload)
    all_passed = bool(payload["all_passed"])
    print(f"NIAH verdict={'PASS' if all_passed else 'FAIL'}")
    return 0 if all_passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
