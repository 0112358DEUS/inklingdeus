# Q2/Q3 — depth quality and tool-call regression

Status: **READY — NOT RUN**. The optimization loop stopped after E3 met its serving-performance
success threshold; no NIAH, GSM8K, or tool-call result is claimed.

## Q2 — quality at depth

The suite runs two complementary gates on the locked champion:

1. **NIAH at 512K and 1M.** Deterministic synthetic archive records place a unique secret at 10%,
   50%, and 90% depth. `/v1/tokenize` measures the rendered chat prompt; calibration must land
   within 0.1% (minimum 256 tokens) of 512,000 or 1,000,000 tokens, and the chat response's own
   `usage.prompt_tokens` must match exactly. All six secrets must be recovered.
2. **Complete GSM8K test split.** All 1,319 official OpenAI test items run through
   `/v1/chat/completions`; the dataset bytes are pinned to SHA256
   `3730d312f6e3440559ace48831e51066acaca737f6eabec99bccb9e4b3c39d14`. Responses are append-only
   and identity-checked so a long run can resume. The gate is at least **94.83%**, within one point
   of SGLang's published **95.83%** NVFP4 + DSpark reference.

The source bytes come from the
[official OpenAI GSM8K repository](https://github.com/openai/grade-school-math/blob/master/grade_school_math/data/test.jsonl).
The comparison value is pinned from SGLang's current
[Inkling-Small benchmark configuration](https://github.com/sgl-project/sglang/blob/main/docs_new/src/snippets/configs/thinkingmachines/inkling-small-benchmarks.jsx).
Different hardware and prompt methodology remain explicit in the result; the threshold is a gate,
not a claim that the upstream run has already been reproduced.

## Q3 — tool calls and parser-token leaks

Four forced structured tools (weather, currency, inventory, calendar) each run four times. Every
flow has a tool-call response and an assistant response after a supplied tool result: **32 chat
responses total, including 16 critical post-tool turns**. The suite requires valid JSON arguments,
the requested tool name and fields, a non-empty final assistant response, no second call under
`tool_choice=none`, and zero occurrences of:

- `<|end_message|>`
- `<|content_model_end_sampling|>`
- `<|content_invoke_tool_json|>`
- `<|message_model|>`

Any leak in content or reasoning content fails Q3.

## Ready-to-run sequence

Fetch the immutable dataset first. This changes only the chosen dataset-cache path; it does not
touch the serving configuration.

```bash
./scripts/fetch-gsm8k.sh "$HOME/.cache/inklingdeus/datasets/gsm8k-test.jsonl"

export WORKER_SSH=control2@<control2-host>
export WORKER_REPO=/home/control2/code/inklingdeus
export MASTER_IP=<control1-primary-roce-ip>
export IF=<control1-primary-roce-netdev>
export HCA=<unchanged-HCA-list>
export MODELS=<same-absolute-model-path-on-both-nodes>
export GSM8K_DATA="$HOME/.cache/inklingdeus/datasets/gsm8k-test.jsonl"
export RESULT_DIR=artifacts/quality-gates-<run-id>
export CHAMPION_BLOCK=5
export CHAMPION_SPECULATOR=<dspark-or-mtp-width1>

./scripts/run-quality-gates.sh
```

The runner checks exact repo SHA, runnable worktree, and image identity; starts the locked champion;
and runs T4 before the
suite and after NIAH, GSM8K, and tool calls. Reusing a result directory is allowed only for the same
repo and image-payload identity, which preserves GSM8K resume safety.
