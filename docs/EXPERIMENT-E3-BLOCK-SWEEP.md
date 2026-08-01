# E3 — DSpark block-size sweep with accept-by-position evidence

Status: **COMPLETE — ACCEPT block 5** on 2026-08-02.

## Live result

All three arms ran in one session with the same image, model, single-HCA transport, prompts, and
exact `n=32` chat-templated open-ended plan. Byte-exact T4 passed before and after every arm.

| block | throughput | accept length | verdict |
|---:|---:|---:|---|
| 7 | 24.747 +/- 0.208 tok/s | 2.176 +/- 0.016 | baseline |
| 5 | **26.007 +/- 0.334 tok/s** | 2.093 +/- 0.026 | **ACCEPT** |
| 6 | 25.079 +/- 0.294 tok/s | 2.106 +/- 0.023 | below threshold |

Block 5 gained **1.260 tok/s** over block 7 with combined SE **0.394**. The gain exceeded the
predeclared +0.5 tok/s threshold and the 1-SE bars did not overlap. Block 6 gained only
0.332 tok/s and was not accepted. Block-7 acceptance at draft positions 5, 6, and 7 was 4.667%,
3.691%, and 2.036%, respectively, supporting the tail-width hypothesis.

Evidence is in `artifacts/e3-block-sweep/`. The immutable result digests are:

- block-7 JSON: `85d4821ad6b35c1ba56fb1bc2c7eca9f60e4874cfa7c112119d881f34f818772`
- block-5 JSON: `444e727646385b33115a47de47ae37fdf39eec459f463d897479d330e8eda447`
- block-6 JSON: `8e2d3121e60ddb7181555ce8698f13f66ec3025ff941effeb691aa9766954ced`
- decision JSON: `954f6a00c6db3937caaf5a69749bae597f29dc8dcd0ece9c814387218033ac08`
- each T4 record: `aac69468d03ab55a8da2d9f15e7939103b479a8e93d9b7839e12953299119de7`

## Hypothesis and gates

- **Hypothesis:** positions 6–7 contribute little acceptance while still widening every verify;
  block 5 or 6 can therefore improve decode throughput without changing output quality.
- **Expected effect:** +0.5 to +2 tok/s if the block-7 tail is mostly dead.
- **Only changed serving variable:** `BLOCK=7`, `BLOCK=5`, or `BLOCK=6`.
- **Instrumentation:** SGLang's returned `spec_correct_drafts_histogram` counts verify steps by
  accepted-draft length. Position `p` acceptance is derived exactly as
  `sum(histogram[p:]) / sum(histogram)`; the bonus token is excluded.
- **Kill criteria:** any missing/malformed histogram, any T4 mismatch before or after a run, any
  arm other than exact n=32, or a candidate that loses at least one combined SE.
- **Acceptance:** a smaller block gains at least 0.5 tok/s over block 7 and the 1-SE throughput
  bars do not overlap. Tail acceptance is diagnostic evidence, not a substitute for throughput.

## Ready-to-run sequence

```bash
export WORKER_SSH=control2@<control2-host>
export WORKER_REPO=/home/control2/code/inklingdeus
export MASTER_IP=<control1-primary-roce-ip>
export IF=<control1-primary-roce-netdev>
export HCA=<unchanged-HCA-list-for-all-three-arms>
export MODELS=<same-absolute-model-path-on-both-nodes>

./scripts/run-e3-block-sweep.sh
```

The runner measures the inherited block-7 baseline first, then blocks 5 and 6, all in one session.
`benchmarks/select_block_size.py` validates plan identity, exact sample counts, and histogram width
before it can emit `ACCEPT`; otherwise the verdict is `RETAIN_BLOCK_7`.
