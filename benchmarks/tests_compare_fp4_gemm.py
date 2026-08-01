#!/usr/bin/env python3
"""GPU numerical gate for E2 dense NVFP4 GEMM backends.

Run inside the baked SGLang image on an idle GB10. The test builds identical
serialized ModelOpt-NVFP4 linear layers, lets each backend perform its own
required weight repack, and compares deterministic outputs over decode and
small-batch shapes. It intentionally exercises the same ModelOptFp4LinearMethod
used by the target model without loading 156 GB of weights.
"""

from __future__ import annotations

import sys

import torch

from sglang.srt.layers.quantization import fp4_utils
from sglang.srt.layers.quantization.fp4_utils import Fp4GemmRunnerBackend
from sglang.srt.layers.quantization.modelopt_quant import (
    ModelOptFp4Config,
    ModelOptFp4LinearMethod,
)


BACKENDS = ("flashinfer_trtllm", "marlin")
M_VALUES = (1, 2, 8, 32)
K = 256
N = 256
ATOL = 0.125
RTOL = 0.02


class LinearFixture(torch.nn.Module):
    pass


def serialized_fixture() -> LinearFixture:
    config = ModelOptFp4Config(
        is_checkpoint_nvfp4_serialized=True,
        group_size=16,
        exclude_modules=[],
    )
    method = ModelOptFp4LinearMethod(config)
    layer = LinearFixture()
    method.create_weights(
        layer,
        input_size_per_partition=K,
        output_partition_sizes=[N],
        input_size=K,
        output_size=N,
        params_dtype=torch.bfloat16,
    )
    generator = torch.Generator(device="cpu").manual_seed(20260801)
    layer.weight.data.copy_(
        torch.randint(0, 256, layer.weight.shape, dtype=torch.uint8, generator=generator)
    )
    # Positive, exactly representable E4M3 block scales across several exponents.
    exponents = torch.randint(
        -3, 4, layer.weight_scale.shape, dtype=torch.int32, generator=generator
    )
    layer.weight_scale.data.copy_(torch.pow(2.0, exponents.float()).to(torch.float8_e4m3fn))
    layer.input_scale.data.fill_(1.0)
    layer.weight_scale_2.data.fill_(1.0)
    layer.quant_method = method
    return layer


def prepare(source: LinearFixture, backend: str) -> LinearFixture:
    fp4_utils.FP4_GEMM_RUNNER_BACKEND = Fp4GemmRunnerBackend(backend)
    # vLLM's BasevLLMParameter deliberately does not implement Python deepcopy.
    # Each caller provides a fresh fixture built from the same local RNG seed, so
    # both backends still receive byte-identical serialized checkpoint tensors.
    layer = source.cuda()
    layer.quant_method.process_weights_after_loading(layer)
    return layer


def run(layer: LinearFixture, backend: str, x: torch.Tensor) -> torch.Tensor:
    fp4_utils.FP4_GEMM_RUNNER_BACKEND = Fp4GemmRunnerBackend(backend)
    result = layer.quant_method.apply(layer, x)
    torch.cuda.synchronize()
    return result


def main() -> int:
    if not torch.cuda.is_available():
        print("E2 GPU gate requires CUDA", file=sys.stderr)
        return 2
    major, minor = torch.cuda.get_device_capability()
    if major < 12:
        print(f"E2 GPU gate requires GB10-class SM12x, got sm_{major}{minor}", file=sys.stderr)
        return 2

    torch.manual_seed(20260801)
    layers = {
        backend: prepare(serialized_fixture(), backend) for backend in BACKENDS
    }
    failures: list[str] = []
    for m in M_VALUES:
        x = torch.randn(m, K, device="cuda", dtype=torch.bfloat16) * 0.25
        outputs = {}
        for backend in BACKENDS:
            try:
                first = run(layers[backend], backend, x)
                second = run(layers[backend], backend, x)
            except Exception as exc:  # The gate must report every backend, not crash early.
                detail = f"{type(exc).__name__}: {exc}"
                print(f"M={m:2d} backend={backend} ERROR {detail}")
                failures.append(f"{backend} M={m} raised {detail}")
                continue
            deterministic = torch.equal(first, second)
            finite = bool(torch.isfinite(first).all())
            print(
                f"M={m:2d} backend={backend} "
                f"repeatable={deterministic} finite={finite}"
            )
            if not deterministic:
                failures.append(f"{backend} M={m} is not bitwise repeatable")
            if not finite:
                failures.append(f"{backend} M={m} produced non-finite output")
            outputs[backend] = first

        if set(outputs) != set(BACKENDS):
            print(f"M={m:2d} cross_backend=SKIPPED missing={sorted(set(BACKENDS) - set(outputs))}")
            continue
        baseline = outputs["flashinfer_trtllm"].float()
        candidate = outputs["marlin"].float()
        max_abs = (baseline - candidate).abs().max().item()
        close = torch.allclose(baseline, candidate, rtol=RTOL, atol=ATOL)
        bitwise = torch.equal(outputs["flashinfer_trtllm"], outputs["marlin"])
        print(
            f"M={m:2d} bitwise_cross_backend={bitwise} "
            f"max_abs={max_abs:.6f} allclose={close}"
        )
        if not close:
            failures.append(
                f"M={m} cross-backend mismatch max_abs={max_abs:.6f} "
                f"rtol={RTOL} atol={ATOL}"
            )

    if failures:
        print("E2 dense FP4 GEMM numerical gate FAIL", file=sys.stderr)
        for failure in failures:
            print(f"- {failure}", file=sys.stderr)
        return 1
    print("E2 dense FP4 GEMM numerical gate PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
