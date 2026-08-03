#!/usr/bin/env python3
"""Validate the champion launcher's dry-run command contract."""

from __future__ import annotations

import os
import shlex
import subprocess
from pathlib import Path


def render(root: Path, overrides: dict[str, str] | None = None) -> list[str]:
    env = os.environ.copy()
    env.update(
        {
            "DRY_RUN": "1",
            "MASTER_IP": "192.0.2.10",
            "IF": "enp1s0f1np1",
            "HCA": "rocep1s0f1,roceP2p1s0f1",
            "GID": "3",
            "MODELS": "/models/inkling",
            "PERSIST_JIT_CACHE": "0",
        }
    )
    if overrides:
        env.update(overrides)
    rendered = subprocess.run(
        [str(root / "scripts/nvfp4-kv-boot.sh"), "0"],
        cwd=root,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    return shlex.split(rendered)


def render_locked(
    root: Path,
    *,
    fp4_backend: str = "flashinfer_trtllm",
    block: str = "5",
    spec: str = "1",
    memfrac: str = "0.85",
    persist_jit: str = "0",
    extra_args: str = "",
    jit_cache_root: str = "",
) -> list[str]:
    env = os.environ.copy()
    env.update(
        {
            "DRY_RUN": "1",
            "MASTER_IP": "192.0.2.10",
            "IF": "enp1s0f1np1",
            "HCA": "rocep1s0f1,roceP2p1s0f1",
            "GID": "3",
            "MODELS": "/models/inkling",
        }
    )
    rendered = subprocess.run(
        [
            str(root / "scripts/locked-experiment-launch.sh"),
            "0",
            fp4_backend,
            block,
            spec,
            memfrac,
            persist_jit,
            extra_args,
            jit_cache_root,
        ],
        cwd=root,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    return shlex.split(rendered)


def render_bf16(root: Path, overrides: dict[str, str]) -> list[str]:
    env = os.environ.copy()
    env.update(
        {
            "DRY_RUN": "1",
            "MASTER_IP": "192.0.2.10",
            "IF": "enp1s0f1np1",
            "HCA": "rocep1s0f1",
            "GID": "3",
            "MODELS": "/models/inkling",
            "PERSIST_JIT_CACHE": "0",
        }
    )
    env.update(overrides)
    rendered = subprocess.run(
        [str(root / "scripts/inkling-sglang-launch.sh"), "0"],
        cwd=root,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    return shlex.split(rendered)


def value_after(command: list[str], flag: str) -> str:
    return command[command.index(flag) + 1]


def resolve_profile(root: Path, speculator: str, block: str) -> tuple[str, str, str]:
    script = (
        'source "$1"; resolve_champion_profile "$2" "$3"; '
        'printf "%s\\n%s\\n%s\\n" "$PROFILE_BLOCK" "$PROFILE_SPEC" "$PROFILE_EXTRA_ARGS"'
    )
    output = subprocess.run(
        [
            "bash",
            "-c",
            script,
            "profile-check",
            str(root / "scripts/champion-profile.sh"),
            speculator,
            block,
        ],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    assert len(output) == 3
    return output[0], output[1], output[2]


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    command = render(root)
    required_pairs = {
        "--context-length": "1048576",
        "--kv-cache-dtype": "fp4_mx_block16",
        "--attention-backend": "triton",
        "--moe-runner-backend": "marlin",
        "--fp4-gemm-backend": "flashinfer_trtllm",
        "--page-size": "1",
        "--speculative-dspark-block-size": "5",
        "--num-continuous-decode-steps": "2",
    }
    for flag, value in required_pairs.items():
        assert value_after(command, flag) == value, (flag, value_after(command, flag))
    required_flags = {
        "--triton-attention-reduce-in-fp32",
        "--disable-piecewise-cuda-graph",
        "--disable-prefill-cuda-graph",
    }
    missing = required_flags.difference(command)
    assert not missing, f"missing champion flags: {sorted(missing)}"
    joined = " ".join(command)
    assert "SGLANG_RAGGED_VERIFY_MODE" not in joined
    assert "NCCL_IB_HCA=rocep1s0f1,roceP2p1s0f1" in joined

    override_cds = render(root, {"CONTINUOUS_DECODE_STEPS": "4"})
    assert override_cds.count("--num-continuous-decode-steps") == 1
    assert value_after(override_cds, "--num-continuous-decode-steps") == "4"

    e2_command = render_locked(root, fp4_backend="marlin")
    assert value_after(e2_command, "--fp4-gemm-backend") == "marlin"
    assert value_after(e2_command, "--speculative-dspark-block-size") == "5"
    assert value_after(e2_command, "--num-continuous-decode-steps") == "2"

    e3_command = render_locked(root, block="5")
    assert value_after(e3_command, "--speculative-dspark-block-size") == "5"
    assert value_after(e3_command, "--fp4-gemm-backend") == "flashinfer_trtllm"

    mtp_args = " ".join(
        (
            "--speculative-algorithm EAGLE",
            "--speculative-num-steps 1",
            "--speculative-eagle-topk 1",
            "--speculative-num-draft-tokens 2",
            "--enable-multi-layer-eagle",
            "--speculative-use-rejection-sampling",
        )
    )
    e4_command = render_locked(root, spec="0", extra_args=mtp_args)
    e4_pairs = {
        "--speculative-algorithm": "EAGLE",
        "--speculative-num-steps": "1",
        "--speculative-eagle-topk": "1",
        "--speculative-num-draft-tokens": "2",
    }
    for flag, value in e4_pairs.items():
        assert e4_command.count(flag) == 1
        assert value_after(e4_command, flag) == value
    assert "--enable-multi-layer-eagle" in e4_command
    assert "--speculative-use-rejection-sampling" in e4_command
    forbidden_e4 = {
        "DSPARK",
        "--speculative-draft-model-path",
        "--speculative-draft-model-quantization",
        "--speculative-dspark-block-size",
    }
    assert not forbidden_e4.intersection(e4_command)
    assert value_after(e4_command, "--fp4-gemm-backend") == "flashinfer_trtllm"
    assert value_after(e4_command, "--context-length") == "1048576"

    cache_root = "/var/cache/inkling-e5"
    e5_command = render_locked(root, persist_jit="1", jit_cache_root=cache_root)
    volumes = {
        e5_command[index + 1]
        for index, value in enumerate(e5_command)
        if value == "-v"
    }
    expected_cache_volumes = {
        f"{cache_root}/triton:/root/.triton",
        f"{cache_root}/flashinfer:/root/.cache/flashinfer",
        f"{cache_root}/sglang:/root/.cache/sglang",
        f"{cache_root}/torch_extensions:/root/.cache/torch_extensions",
        f"{cache_root}/torchinductor:/tmp/torchinductor_root",
        f"{cache_root}/cuda:/root/.nv/ComputeCache",
    }
    assert expected_cache_volumes.issubset(volumes)
    default_volumes = {
        command[index + 1]
        for index, value in enumerate(command)
        if value == "-v"
    }
    assert not expected_cache_volumes.intersection(default_volumes)

    e6_command = render_locked(root, memfrac="0.68")
    assert value_after(e6_command, "--mem-fraction-static") == "0.68"
    assert value_after(e6_command, "--speculative-dspark-block-size") == "5"
    assert value_after(e6_command, "--context-length") == "1048576"
    assert value_after(e6_command, "--fp4-gemm-backend") == "flashinfer_trtllm"

    dspark_profile = resolve_profile(root, "dspark", "5")
    assert dspark_profile == ("5", "1", "")
    mtp_profile = resolve_profile(root, "mtp-width1", "5")
    assert mtp_profile[:2] == ("5", "0")
    inherited_mtp = render_locked(
        root,
        block=mtp_profile[0],
        spec=mtp_profile[1],
        extra_args=mtp_profile[2],
    )
    assert value_after(inherited_mtp, "--speculative-algorithm") == "EAGLE"
    assert "--speculative-dspark-block-size" not in inherited_mtp

    e7_command = render_bf16(
        root,
        {
            "IMAGE": "local/sglang-inkling:fa4-sm121-dev",
            "ATTN": "fa4",
            "PAGE": "128",
            "CTX": "65536",
            "SPEC": "0",
            "GRAPHS": "1",
            "MEMFRAC": "0.85",
            "CONTINUOUS_DECODE_STEPS": "2",
        },
    )
    e7_pairs = {
        "--attention-backend": "fa4",
        "--page-size": "128",
        "--context-length": "65536",
        "--moe-runner-backend": "marlin",
        "--mem-fraction-static": "0.85",
        "--num-continuous-decode-steps": "2",
    }
    for flag, value in e7_pairs.items():
        assert e7_command.count(flag) == 1
        assert value_after(e7_command, flag) == value
    assert "--kv-cache-dtype" not in e7_command
    assert "--speculative-algorithm" not in e7_command
    assert "--disable-piecewise-cuda-graph" in e7_command
    assert "--disable-prefill-cuda-graph" in e7_command
    print("launch render PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
