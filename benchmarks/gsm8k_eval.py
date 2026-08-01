#!/usr/bin/env python3
"""Evaluate the complete official GSM8K test split through chat completions."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import re
import sys
import urllib.error
import urllib.request
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any


EXPECTED_SHA256 = "3730d312f6e3440559ace48831e51066acaca737f6eabec99bccb9e4b3c39d14"
EXPECTED_ITEMS = 1319
OFFICIAL_DSPARK_REFERENCE_PCT = 95.83
MINIMUM_ACCEPTABLE_PCT = OFFICIAL_DSPARK_REFERENCE_PCT - 1.0
NUMBER_RE = re.compile(r"-?\$?[0-9][0-9,]*(?:\.[0-9]+)?")
HASH_ANSWER_RE = re.compile(r"####\s*(-?\$?[0-9][0-9,]*(?:\.[0-9]+)?)")


def normalize_number(text: str) -> Decimal | None:
    candidate = text.strip().replace("$", "").replace(",", "")
    try:
        return Decimal(candidate)
    except InvalidOperation:
        return None


def extract_reference(answer: str) -> Decimal:
    match = HASH_ANSWER_RE.search(answer)
    if not match:
        raise ValueError("reference answer has no #### numeric answer")
    value = normalize_number(match.group(1))
    if value is None:
        raise ValueError("reference answer is not numeric")
    return value


def extract_prediction(content: str) -> tuple[Decimal | None, str]:
    hash_matches = HASH_ANSWER_RE.findall(content)
    if hash_matches:
        return normalize_number(hash_matches[-1]), "hash"
    matches = NUMBER_RE.findall(content)
    if matches:
        return normalize_number(matches[-1]), "last-number-fallback"
    return None, "missing"


def load_dataset(path: Path) -> list[dict[str, str]]:
    raw = path.read_bytes()
    actual_sha = hashlib.sha256(raw).hexdigest()
    if actual_sha != EXPECTED_SHA256:
        raise ValueError(f"{path}: SHA256 {actual_sha} != {EXPECTED_SHA256}")
    items = [json.loads(line) for line in raw.decode("utf-8").splitlines() if line]
    if len(items) != EXPECTED_ITEMS:
        raise ValueError(f"{path}: expected {EXPECTED_ITEMS} items, got {len(items)}")
    for index, item in enumerate(items):
        if not isinstance(item.get("question"), str) or not isinstance(item.get("answer"), str):
            raise ValueError(f"{path}: malformed item {index}")
        extract_reference(item["answer"])
    return items


def request_one(
    *,
    index: int,
    item: dict[str, str],
    url: str,
    model: str,
    max_tokens: int,
    reasoning_effort: str,
    timeout: float,
) -> dict[str, Any]:
    prompt = (
        "Solve this grade-school math problem. Show the necessary reasoning, then end with a new "
        "line in exactly the form `#### <numeric answer>`.\n\n"
        f"{item['question']}"
    )
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0,
        "stream": False,
        "reasoning_effort": reasoning_effort,
    }
    request = urllib.request.Request(
        f"{url}/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            payload = json.load(response)
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode(errors="replace")
        raise RuntimeError(f"item {index}: HTTP {exc.code}: {detail}") from exc
    try:
        message = payload["choices"][0]["message"]
        content = message.get("content") or ""
        usage = payload["usage"]
    except (KeyError, IndexError, TypeError) as exc:
        raise RuntimeError(f"item {index}: malformed chat response") from exc
    predicted, extraction = extract_prediction(content)
    reference = extract_reference(item["answer"])
    return {
        "index": index,
        "question_sha256": hashlib.sha256(item["question"].encode()).hexdigest(),
        "reference": str(reference),
        "prediction": str(predicted) if predicted is not None else None,
        "correct": predicted == reference,
        "extraction": extraction,
        "content": content,
        "reasoning_content": message.get("reasoning_content"),
        "usage": usage,
    }


def load_resume(path: Path, items: list[dict[str, str]]) -> dict[int, dict[str, Any]]:
    if not path.exists():
        return {}
    completed: dict[int, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            record = json.loads(line)
            index = record.get("index")
            if not isinstance(index, int) or not 0 <= index < len(items):
                raise ValueError(f"{path}:{line_number}: invalid index")
            expected_hash = hashlib.sha256(items[index]["question"].encode()).hexdigest()
            if record.get("question_sha256") != expected_hash or index in completed:
                raise ValueError(f"{path}:{line_number}: resume identity mismatch or duplicate")
            completed[index] = record
    return completed


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--url", default=os.environ.get("INKLING_URL", "http://localhost:30000"))
    parser.add_argument("--model", default=os.environ.get("INKLING_MODEL", "inkling-small"))
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--reasoning-effort", default="max")
    parser.add_argument("--timeout", type=float, default=1200)
    parser.add_argument("--responses", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if args.concurrency < 1 or args.max_tokens < 1 or args.timeout <= 0:
        parser.error("concurrency, max tokens, and timeout must be positive")
    args.url = args.url.rstrip("/")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv if argv is not None else sys.argv[1:])
    items = load_dataset(args.dataset)
    if args.dry_run:
        print(
            json.dumps(
                {
                    "items": len(items),
                    "dataset_sha256": EXPECTED_SHA256,
                    "concurrency": args.concurrency,
                    "reasoning_effort": args.reasoning_effort,
                    "minimum_acceptable_pct": MINIMUM_ACCEPTABLE_PCT,
                },
                indent=2,
            )
        )
        return 0

    completed = load_resume(args.responses, items)
    pending = [(index, item) for index, item in enumerate(items) if index not in completed]
    args.responses.parent.mkdir(parents=True, exist_ok=True)
    with args.responses.open("a", encoding="utf-8") as output:
        for offset in range(0, len(pending), args.concurrency):
            batch = pending[offset : offset + args.concurrency]
            with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as executor:
                futures = [
                    executor.submit(
                        request_one,
                        index=index,
                        item=item,
                        url=args.url,
                        model=args.model,
                        max_tokens=args.max_tokens,
                        reasoning_effort=args.reasoning_effort,
                        timeout=args.timeout,
                    )
                    for index, item in batch
                ]
                records = [future.result() for future in futures]
            for record in sorted(records, key=lambda value: value["index"]):
                output.write(json.dumps(record, ensure_ascii=False) + "\n")
                output.flush()
                completed[record["index"]] = record
            correct = sum(record["correct"] for record in completed.values())
            print(
                f"GSM8K completed={len(completed)}/{len(items)} "
                f"running_accuracy={100 * correct / len(completed):.2f}%",
                flush=True,
            )

    if len(completed) != EXPECTED_ITEMS:
        raise RuntimeError(f"incomplete run: {len(completed)}/{EXPECTED_ITEMS}")
    correct = sum(record["correct"] for record in completed.values())
    accuracy = 100 * correct / EXPECTED_ITEMS
    fallback_count = sum(
        record.get("extraction") == "last-number-fallback" for record in completed.values()
    )
    summary = {
        "schema_version": 1,
        "dataset": "openai/gsm8k test",
        "dataset_sha256": EXPECTED_SHA256,
        "n": EXPECTED_ITEMS,
        "correct": correct,
        "accuracy_pct": accuracy,
        "official_dspark_reference_pct": OFFICIAL_DSPARK_REFERENCE_PCT,
        "minimum_acceptable_pct": MINIMUM_ACCEPTABLE_PCT,
        "within_one_point": accuracy >= MINIMUM_ACCEPTABLE_PCT,
        "last_number_fallback_count": fallback_count,
        "endpoint": "/v1/chat/completions",
        "model": args.model,
        "reasoning_effort": args.reasoning_effort,
        "max_tokens": args.max_tokens,
    }
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(
        f"GSM8K n={EXPECTED_ITEMS} accuracy={accuracy:.2f}% "
        f"threshold={MINIMUM_ACCEPTABLE_PCT:.2f}%"
    )
    return 0 if summary["within_one_point"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
