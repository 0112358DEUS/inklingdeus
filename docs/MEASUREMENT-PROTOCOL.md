# Measurement protocol — read this before quoting any number

**The target forward pass on this stack is nondeterministic at temperature 0.** Same prompt,
same seed, same config → different output text run to run. Proven spec-independent: it persists
with speculation OFF and is *not* fixed by `--enable-deterministic-inference` (the culprit is
kernel-level: triton split-KV attention reduction order and/or marlin MoE reduction on sm_121a,
neither covered by the deterministic path).

## Why that wrecks naive benchmarking

DSpark acceptance depends heavily on *which* continuation a run lands on:

| Continuation style | typical accept |
|---|---|
| repetitive / list-like / templated | 4.0 – 5.7 |
| novel coherent prose | 1.5 – 2.5 |

So a single 10-run probe on one prompt has a noise band wider than most config effects. During
this campaign that trap produced, on **identical** configs and images: 7.31, 2.44, 4.51, 3.09 —
and sent two separate agent sessions chasing a "regression" that never existed. An early headline
of "64.6 tok/s" was one lucky draw from a distribution whose mean was ~30.

## Two traps that produced published-but-wrong numbers here

**1. The echo trap inflates acceptance ~60%.** Feeding untemplated text to `/generate` makes this
model regurgitate the prompt, and repetitive output drafts trivially. A raw-continuation probe
measured 3.44 accept / 34.3 tok/s where a chat-templated, serving-representative gate measures
2.27 / 23.9. Always benchmark through the model's own chat template.

**2. Acceptance is task-dependent by more than 2×.** On this exact serve: GSM8K-style 4.81,
code 2.46, chat 2.20, open-ended prose 2.15. A single pooled number hides that. Report per class,
and say which class you measured — a "faster" config may simply have been measured on easier text.

## The protocol

Use [`benchmarks/chat_bench.py`](../benchmarks/chat_bench.py) for serving comparisons:

- **4 fixed, topic-distinct prompts × 8 reps = 32 samples per task class**
- chat requests only (`/v1/chat/completions` with `return_meta_info=true`); never substitute raw
  `/generate`, whose untemplated inputs fall into repetition traps that inflate acceptance
- 2 warm-up calls first (cold first-request always reads low)
- reports **mean ± standard error** and range, per-seed and overall

```bash
# One task class: n=32. Use the same class for both A/B arms.
python3 benchmarks/chat_bench.py "my-config-label" --task open-ended --reps 8 \
  --output "my-config-label.json"

# Full quality/task profile: n=32 independently for each class.
python3 benchmarks/chat_bench.py "my-config-label" --task all --reps 8
```

[`benchmarks/accept_probe.py`](../benchmarks/accept_probe.py) is deliberately retained as a
**legacy raw-continuation probe** so old 3.44 accept / 34.3 tok/s figures remain reproducible. It
must not be used for serving claims or optimization acceptance decisions.

Concurrency claims use [`benchmarks/concurrency_bench.py`](../benchmarks/concurrency_bench.py):
the same four chat prompts and eight repetitions produce exact n=32 at each requested level, with
aggregate tokens/s summarized across fixed-size waves. The retired raw `/generate` C1/C4/C8
script is not comparable to real chat serving.

Quality gates are scored separately from speed A/Bs. [`benchmarks/niah_eval.py`](../benchmarks/niah_eval.py)
records tokenizer-measured chat input length and requires all predefined depths to pass;
[`benchmarks/gsm8k_eval.py`](../benchmarks/gsm8k_eval.py) scores the complete checksum-pinned test
split; [`benchmarks/tool_call_regression.py`](../benchmarks/tool_call_regression.py) requires every
structured and post-tool flow to pass. Do not turn a hand-picked subset into a percentage claim.

**Rules**
1. Never quote a single-run number. Ever.
2. Compare configs only via non-overlapping error bars. A +0.3 accept difference with ±0.18 se
   on each side is *suggestive*, not proven.
3. Re-measure after every reboot — boot-to-boot means shift.
4. Always state the task class. Chat-template traffic accepts differently from raw continuation.
