# MiaAI dual-Spark benchmark crosswalk

This project uses the MiaAI-Lab result as a **diagnostic reference**, not as an adoption gate.
The source is pinned to
[`ce6bfbe90d6e680e7b8172ebd1922fa148158e26`](https://github.com/MiaAI-Lab/Inkling-Small-NVFP4-Dual-DGX-Sparks/tree/ce6bfbe90d6e680e7b8172ebd1922fa148158e26).
That snapshot contains a README, start/stop wrappers, and one screenshot; it does not contain the
benchmark program or raw result records.

## What the published result shows

The pinned [`bench.png`](https://github.com/MiaAI-Lab/Inkling-Small-NVFP4-Dual-DGX-Sparks/blob/ce6bfbe90d6e680e7b8172ebd1922fa148158e26/bench.png)
labels itself a 512-token decode benchmark and displays this one concurrency curve:

| Concurrency | Aggregate tok/s | Stream tok/s (displayed) | TTFT |
|---:|---:|---:|---:|
| 1 | 33.9 | 33.9 | 3.37 s |
| 2 | 48.5 | 24.9 | 6.99 s |
| 3 | 56.6 | 24.1 | 9.72 s |
| 4 | 65.6 | 18.8 | 9.34 s |
| 6 | 78.7 | 16.7 | 8.49 s |
| 8 | 74.9 | 11.9 | 6.21 s |

The strongest useful targets are therefore **C1 aggregate >=33.9 tok/s** and **C8 aggregate
>=74.9 tok/s** under a 512-output-token compatibility curve. The screenshot's C6 aggregate is
higher than C8, so the whole curve matters; a C1-only comparison can miss saturation.

The README phrase "~34 tok/s per user at moderate batch" is not supported by the displayed stream
column: 33.9 appears at C1, while the displayed values are 24.9 at C2 and 11.9 at C8. This project
will quote the table, not the phrase.

## Runtime crosswalk

MiaAI-Lab's pinned
[`start_sglang.sh`](https://github.com/MiaAI-Lab/Inkling-Small-NVFP4-Dual-DGX-Sparks/blob/ce6bfbe90d6e680e7b8172ebd1922fa148158e26/start_sglang.sh)
is a deployment wrapper around the same public drowzeys `kvquant` image lineage, not a separate
faster inference engine. Its relevant settings are triton attention with FP32 reduction, page-1 FP4
KV, marlin MoE, `flashinfer_trtllm` FP4 GEMM, DSpark block 7, and decode graphs.

Important differences from the adopted champion here:

- MiaAI-Lab uses DSpark block 7; the gated E3 result here promoted block 5.
- Its wrapper does not set the adopted continuous-decode-steps value 2.
- It references an image tag rather than freezing an image digest or patched-payload fingerprint.
- The screenshot does not disclose prompt text, endpoint/template, sampling parameters, warmups,
  token-counting rule, repetitions, errors, or T4 bracketing.

Those gaps make the two headline rates non-equivalent. In particular, this project's old raw
continuation diagnostic measured 34.3 +/- 1.7 tok/s, close to MiaAI-Lab's C1 33.9, while the
authoritative open-ended chat-templated n=32 champion result is 26.225 +/- 0.323 tok/s. That is a
measurement-contract difference, not evidence of a 7.7 tok/s regression.

## How it is used here

`benchmarks/concurrency_bench.py` can run a reproducible **Mia-shaped** curve with all six displayed
concurrency levels. Forty-eight samples per level are required so every level has complete waves:

```bash
python3 benchmarks/concurrency_bench.py mia-shaped-512 \
  --task open-ended \
  --concurrency 1 2 3 4 6 8 \
  --reps 12 \
  --tokens 512 \
  --output artifacts/mia-shaped-512/curve.json
```

Before any comparison, record the exact repo SHA, image IDs/payload fingerprints, rendered launch,
and a T4 pass. Run the curve at least twice in the same unchanged session, retain both raw JSON
files, and pass T4 again afterward. Report each C-level mean, standard error, overall aggregate,
request rate, latency, and acceptance. Never pool warmup requests into the result.

This is deliberately called Mia-shaped rather than a reproduction: it uses this repository's fixed
chat prompts and OpenAI chat endpoint, and cannot reproduce the undisclosed MiaAI-Lab prompt/tool.
If their benchmark source and raw records become available, pin them and add an exact-compatibility
lane rather than changing this one.

## Decision boundary

- Use C1 33.9 and C8 aggregate 74.9 as diagnostic targets for saturation/profiling.
- Do not adopt a launcher, backend, or kernel from this curve alone.
- Keep N1 authoritative at chat-templated open-ended n=32, >=32 tok/s, with exact T4 and replicated
  evidence.
- Keep N3 authoritative at FA4 + DSpark with T4, depth/GSM8K quality, and tool regression green.
