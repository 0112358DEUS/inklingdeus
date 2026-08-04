#!/usr/bin/env python3
"""Inkling structured tool-call and post-tool token-leak regression suite."""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


FORBIDDEN_TOKENS = (
    "<|end_message|>",
    "<|content_model_end_sampling|>",
    "<|content_invoke_tool_json|>",
    "<|message_model|>",
)
CASES = (
    {
        "name": "get_weather",
        "description": "Get weather for a city",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
        },
        "prompt": "Use the weather tool for London.",
        "required_arguments": {"city": "London"},
        "tool_result": {"city": "London", "temperature_c": 18, "condition": "clear"},
        "followup": "Summarize the weather result in one sentence without another tool call.",
    },
    {
        "name": "convert_currency",
        "description": "Convert a money amount between currencies",
        "parameters": {
            "type": "object",
            "properties": {
                "amount": {"type": "number"},
                "from_currency": {"type": "string"},
                "to_currency": {"type": "string"},
            },
            "required": ["amount", "from_currency", "to_currency"],
        },
        "prompt": "Use the conversion tool to convert 100 USD to EUR.",
        "required_arguments": {"amount": 100, "from_currency": "USD", "to_currency": "EUR"},
        "tool_result": {"amount": 100, "from_currency": "USD", "to_currency": "EUR", "result": 92},
        "followup": "State the conversion result plainly without another tool call.",
    },
    {
        "name": "lookup_inventory",
        "description": "Look up stock for a product code",
        "parameters": {
            "type": "object",
            "properties": {"sku": {"type": "string"}},
            "required": ["sku"],
        },
        "prompt": "Use the inventory tool for SKU XR-17.",
        "required_arguments": {"sku": "XR-17"},
        "tool_result": {"sku": "XR-17", "in_stock": True, "quantity": 24},
        "followup": "Tell me the inventory result in one sentence without another tool call.",
    },
    {
        "name": "get_calendar_event",
        "description": "Look up a calendar event by date",
        "parameters": {
            "type": "object",
            "properties": {"date": {"type": "string"}},
            "required": ["date"],
        },
        "prompt": "Use the calendar tool for 2030-04-05.",
        "required_arguments": {"date": "2030-04-05"},
        "tool_result": {"date": "2030-04-05", "event": "Design review", "time": "14:00"},
        "followup": "Summarize the calendar result without another tool call.",
    },
)


def leaked_tokens(value: Any) -> list[str]:
    if value is None:
        return []
    text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
    return [token for token in FORBIDDEN_TOKENS if token in text]


def tool_schema(case: dict[str, Any]) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": case["name"],
            "description": case["description"],
            "parameters": case["parameters"],
        },
    }


def post_chat(
    *, url: str, model: str, messages: list[dict[str, Any]], tools: list[dict[str, Any]],
    tool_choice: Any, timeout: float,
) -> dict[str, Any]:
    body = {
        "model": model,
        "messages": messages,
        "tools": tools,
        "tool_choice": tool_choice,
        "max_tokens": 256,
        "temperature": 0,
        "stream": False,
        "reasoning_effort": "none",
        "parallel_tool_calls": False,
    }
    request = urllib.request.Request(
        f"{url}/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.load(response)
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode(errors="replace")
        raise RuntimeError(f"HTTP {exc.code}: {detail}") from exc


def validate_tool_message(message: dict[str, Any], case: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    failures = leaked_tokens(message.get("content")) + leaked_tokens(message.get("reasoning_content"))
    calls = message.get("tool_calls")
    if not isinstance(calls, list) or len(calls) != 1:
        failures.append("expected exactly one structured tool call")
        return {}, failures
    call = calls[0]
    function = call.get("function", {})
    if function.get("name") != case["name"]:
        failures.append(f"wrong tool name: {function.get('name')!r}")
    try:
        arguments = json.loads(function.get("arguments", ""))
    except (json.JSONDecodeError, TypeError):
        arguments = {}
        failures.append("tool arguments are not valid JSON")
    for key, expected in case["required_arguments"].items():
        if arguments.get(key) != expected:
            failures.append(f"argument {key}={arguments.get(key)!r}, expected {expected!r}")
    if not isinstance(call.get("id"), str) or not call["id"]:
        failures.append("tool call id is missing")
    return call, failures


def run_flow(
    *, case: dict[str, Any], repetition: int, url: str, model: str, timeout: float
) -> dict[str, Any]:
    tools = [tool_schema(case)]
    messages: list[dict[str, Any]] = [{"role": "user", "content": case["prompt"]}]
    first = post_chat(
        url=url,
        model=model,
        messages=messages,
        tools=tools,
        tool_choice={"type": "function", "function": {"name": case["name"]}},
        timeout=timeout,
    )
    try:
        first_message = first["choices"][0]["message"]
    except (KeyError, IndexError, TypeError) as exc:
        raise RuntimeError("malformed tool-call response") from exc
    call, failures = validate_tool_message(first_message, case)
    if not call:
        return {
            "case": case["name"],
            "repetition": repetition,
            "passed": False,
            "failures": failures,
            "tool_response": first,
            "tool_message": first_message,
            "post_tool_response": None,
            "post_tool_message": None,
        }

    messages.extend(
        [
            {
                "role": "assistant",
                "content": first_message.get("content"),
                "tool_calls": first_message["tool_calls"],
            },
            {
                "role": "tool",
                "tool_call_id": call["id"],
                "name": case["name"],
                "content": json.dumps(case["tool_result"], separators=(",", ":")),
            },
            {"role": "user", "content": case["followup"]},
        ]
    )
    second = post_chat(
        url=url,
        model=model,
        messages=messages,
        tools=tools,
        tool_choice="none",
        timeout=timeout,
    )
    try:
        second_message = second["choices"][0]["message"]
    except (KeyError, IndexError, TypeError) as exc:
        raise RuntimeError("malformed post-tool response") from exc
    failures.extend(leaked_tokens(second_message.get("content")))
    failures.extend(leaked_tokens(second_message.get("reasoning_content")))
    if second_message.get("tool_calls"):
        failures.append("post-tool assistant turn emitted another tool call despite tool_choice=none")
    if not (second_message.get("content") or "").strip():
        failures.append("post-tool assistant content is empty")
    return {
        "case": case["name"],
        "repetition": repetition,
        "passed": not failures,
        "failures": failures,
        "tool_response": first,
        "tool_message": first_message,
        "post_tool_response": second,
        "post_tool_message": second_message,
    }


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default=os.environ.get("INKLING_URL", "http://localhost:30000"))
    parser.add_argument("--model", default=os.environ.get("INKLING_MODEL", "inkling-small"))
    parser.add_argument("--reps", type=int, default=4)
    parser.add_argument("--timeout", type=float, default=600)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if args.reps < 1 or args.timeout <= 0:
        parser.error("reps and timeout must be positive")
    args.url = args.url.rstrip("/")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv if argv is not None else sys.argv[1:])
    plan = {
        "cases": [case["name"] for case in CASES],
        "repetitions_per_case": args.reps,
        "responses": len(CASES) * args.reps * 2,
        "post_tool_turns": len(CASES) * args.reps,
        "endpoint": "/v1/chat/completions",
        "forbidden_tokens": FORBIDDEN_TOKENS,
    }
    if args.dry_run:
        print(json.dumps(plan, indent=2))
        return 0

    results = []
    for repetition in range(args.reps):
        for case in CASES:
            result = run_flow(
                case=case,
                repetition=repetition,
                url=args.url,
                model=args.model,
                timeout=args.timeout,
            )
            results.append(result)
            print(
                f"tool case={case['name']} repetition={repetition + 1}/{args.reps} "
                f"passed={result['passed']}",
                flush=True,
            )

    passed = sum(result["passed"] for result in results)
    payload = {
        "schema_version": 1,
        "plan": plan,
        "model": args.model,
        "flows": len(results),
        "passed_flows": passed,
        "all_passed": passed == len(results),
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"tool regression flows={len(results)} passed={passed}")
    return 0 if payload["all_passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
