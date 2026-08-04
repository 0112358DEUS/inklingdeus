#!/usr/bin/env python3
"""Exact writer and allocation gates for the SM121 native FP4 KV pool."""

from __future__ import annotations

import gc
import inspect
import json
import os
from types import SimpleNamespace

import torch

from sglang.kernels.ops.attention.flash_attn.cute import flash_attn_varlen_func
from sglang.srt.layers.quantization.kvfp4_tensor import (
    FP4MXBlock16KVQuantizeUtil,
)
from sglang.srt.mem_cache.kv_quant_pools import MHATokenToKVPoolFP4Native
from sglang.srt.mem_cache.memory_pool import KVWriteLoc
from sglang.srt.mem_cache.swa_memory_pool import SWAKVPool
from sglang.srt.models.dspark import DSparkDraftMixin


DEVICE = torch.device("cuda")
DTYPE = torch.bfloat16
H, D, PAGE = 4, 128, 128


def emit(gate: str, **values: object) -> None:
    print(json.dumps({"gate": gate, **values}, sort_keys=True), flush=True)


def make_pool(
    size: int = 512, *, enable_alt_stream: bool = False
) -> MHATokenToKVPoolFP4Native:
    return MHATokenToKVPoolFP4Native(
        size=size,
        page_size=PAGE,
        dtype=torch.float4_e2m1fn_x2,
        head_num=H,
        head_dim=D,
        v_head_dim=D,
        layer_num=1,
        device="cuda",
        enable_memory_saver=False,
        enable_alt_stream=enable_alt_stream,
        enable_kv_cache_copy=True,
    )


def capture_stream_selection_gate() -> None:
    name = "SGLANG_FP4_KV_CAPTURE_SINGLE_STREAM"
    previous = os.environ.pop(name, None)
    try:
        default_pool = make_pool(enable_alt_stream=True)
        default_enabled = default_pool.alt_stream is not None
        del default_pool
        gc.collect()
        torch.cuda.empty_cache()

        os.environ[name] = "1"
        candidate_pool = make_pool(enable_alt_stream=True)
        candidate_disabled = candidate_pool.alt_stream is None
        del candidate_pool
        gc.collect()
        torch.cuda.empty_cache()
    finally:
        if previous is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = previous
    if not default_enabled or not candidate_disabled:
        raise SystemExit(
            "capture stream selection failed: "
            f"{default_enabled=} {candidate_disabled=}"
        )
    emit(
        "capture_stream_selection",
        default_alt_stream=default_enabled,
        candidate_single_stream=candidate_disabled,
    )


def quantize(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    payload, scales = FP4MXBlock16KVQuantizeUtil.batched_quantize(x)
    return payload.view(torch.uint8), scales.view(torch.uint8)


def expected_rows(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    payload, scales = quantize(x)
    return payload.reshape(x.shape[0], H, D // 2), scales.reshape(
        x.shape[0], H * D // 16
    )


def assert_rows(
    pool: MHATokenToKVPoolFP4Native,
    loc: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    label: str,
) -> None:
    kp, ks = expected_rows(k)
    vp, vs = expected_rows(v)
    kb, vb = pool.get_kv_buffer(0)
    ksb, vsb = pool.get_kv_scale_buffer(0)
    checks = {
        "k_payload": torch.equal(kb[loc], kp),
        "v_payload": torch.equal(vb[loc], vp),
        "k_scale": torch.equal(ksb[loc], ks),
        "v_scale": torch.equal(vsb[loc], vs),
    }
    if not all(checks.values()):
        raise SystemExit(f"{label} byte mismatch: {checks}")
    emit(label, rows=int(loc.numel()), **checks)


def ordinary_and_move_gate() -> None:
    pool = make_pool()
    layer = SimpleNamespace(layer_id=0)
    loc = torch.tensor([8, 9, 10], device=DEVICE, dtype=torch.int64)
    k = torch.randn(3, H, D, device=DEVICE, dtype=DTYPE)
    v = torch.randn_like(k)
    k_before, v_before = k.clone(), v.clone()
    pool.set_kv_buffer(layer, loc, k, v)
    if not torch.equal(k, k_before) or not torch.equal(v, v_before):
        raise SystemExit("ordinary writer mutated its BF16 inputs")
    assert_rows(pool, loc, k, v, label="ordinary_target_writer")

    kb, vb = pool.get_kv_buffer(0)
    ksb, vsb = pool.get_kv_scale_buffer(0)
    snapshots = tuple(x[loc].clone() for x in (kb, vb, ksb, vsb))
    dst = torch.tensor([28, 29, 30], device=DEVICE, dtype=torch.int64)
    pool.move_kv_cache(dst, loc)
    checks = [
        torch.equal(buf[dst], expected)
        for buf, expected in zip((kb, vb, ksb, vsb), snapshots)
    ]
    if not all(checks):
        raise SystemExit(f"radix move dropped payload or scale bytes: {checks}")
    emit(
        "radix_move",
        rows=int(dst.numel()),
        k_payload=checks[0],
        v_payload=checks[1],
        k_scale=checks[2],
        v_scale=checks[3],
    )


class _FakeSelfAttention:
    def __init__(self) -> None:
        self.attn = SimpleNamespace(layer_id=0, k_scale=None, v_scale=None)
        self.num_kv_heads = H
        self.head_dim = D

    def kv_proj_only(self, hidden: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        width = H * D
        return hidden[:, :width], hidden[:, width : 2 * width]

    def apply_k_norm(self, k: torch.Tensor) -> torch.Tensor:
        return k

    def apply_k_rope(
        self, positions: torch.Tensor, k: torch.Tensor
    ) -> torch.Tensor:
        del positions
        return k


class _FakeDSpark:
    def __init__(self) -> None:
        self.layers = [SimpleNamespace(self_attn=_FakeSelfAttention())]

    def project_target_hidden(self, target_hidden: torch.Tensor) -> torch.Tensor:
        return target_hidden

    def _fused_kv_write_bundle(self, pool):
        del pool
        return None

    def _stacked_ctx_kv_params(self):
        return None


def dspark_prefix_valid_gate() -> None:
    pool = make_pool()
    fake = _FakeDSpark()
    loc_2d = torch.tensor(
        [[48, 49, 50], [56, 57, 58]], device=DEVICE, dtype=torch.int64
    )
    commit_lens = torch.tensor([2, 1], device=DEVICE, dtype=torch.int32)
    hidden = torch.randn(6, 2 * H * D, device=DEVICE, dtype=DTYPE)
    positions = torch.arange(6, device=DEVICE, dtype=torch.int64)
    DSparkDraftMixin.write_target_hidden_kv(
        fake,
        target_hidden=hidden,
        pool=pool,
        positions=positions,
        cache_loc=loc_2d.reshape(-1),
        cache_loc_2d=loc_2d,
        commit_lens=commit_lens,
    )
    selected_rows = torch.tensor([0, 1, 3], device=DEVICE, dtype=torch.int64)
    selected_locs = torch.tensor([48, 49, 56], device=DEVICE, dtype=torch.int64)
    width = H * D
    k = hidden[:, :width].view(-1, H, D)[selected_rows]
    v = hidden[:, width : 2 * width].view(-1, H, D)[selected_rows]
    assert_rows(
        pool,
        selected_locs,
        k,
        v,
        label="dspark_prefix_valid_writer",
    )
    kb, vb = pool.get_kv_buffer(0)
    ksb, vsb = pool.get_kv_scale_buffer(0)
    rejected = torch.tensor([50, 57, 58], device=DEVICE, dtype=torch.int64)
    if any(torch.count_nonzero(x[rejected]).item() for x in (kb, vb, ksb, vsb)):
        raise SystemExit("prefix-valid writer changed an uncommitted destination row")
    emit("prefix_valid_mask", rejected_rows=int(rejected.numel()), all_zero=True)


def swa_gate() -> None:
    pool = SWAKVPool(
        size=256,
        size_swa=256,
        page_size=PAGE,
        dtype=torch.float4_e2m1fn_x2,
        head_num=H,
        head_dim=D,
        v_head_dim=D,
        swa_attention_layer_ids=[1],
        full_attention_layer_ids=[0],
        device="cuda",
        token_to_kv_pool_class=MHATokenToKVPoolFP4Native,
        enable_alt_stream=False,
        enable_kv_cache_copy=True,
    )
    full_loc = torch.tensor([40, 41], device=DEVICE, dtype=torch.int64)
    swa_loc = torch.tensor([8, 9], device=DEVICE, dtype=torch.int64)
    write_loc = KVWriteLoc(loc=full_loc, swa_loc=swa_loc)
    k_full = torch.randn(2, H, D, device=DEVICE, dtype=DTYPE)
    v_full = torch.randn_like(k_full)
    k_swa = torch.randn(2, H, D, device=DEVICE, dtype=DTYPE)
    v_swa = torch.randn_like(k_swa)
    pool.set_kv_buffer(SimpleNamespace(layer_id=0), write_loc, k_full, v_full)
    pool.set_kv_buffer(SimpleNamespace(layer_id=1), write_loc, k_swa, v_swa)
    assert_rows(
        pool.full_kv_pool,
        full_loc,
        k_full,
        v_full,
        label="swa_wrapper_full_writer",
    )
    assert_rows(
        pool.swa_kv_pool,
        swa_loc,
        k_swa,
        v_swa,
        label="swa_wrapper_local_writer",
    )

    mapping = torch.arange(384, device=DEVICE, dtype=torch.int64)
    mapping[40] = 8
    mapping[50] = 18
    pool.register_mapping(mapping)
    full_buffers = (*pool.full_kv_pool.get_kv_buffer(0), *pool.full_kv_pool.get_kv_scale_buffer(0))
    swa_buffers = (*pool.swa_kv_pool.get_kv_buffer(0), *pool.swa_kv_pool.get_kv_scale_buffer(0))
    full_expected = tuple(x[40].clone() for x in full_buffers)
    swa_expected = tuple(x[8].clone() for x in swa_buffers)
    pool.move_kv_cache(
        torch.tensor([50], device=DEVICE, dtype=torch.int64),
        torch.tensor([40], device=DEVICE, dtype=torch.int64),
    )
    full_checks = [torch.equal(x[50], e) for x, e in zip(full_buffers, full_expected)]
    swa_checks = [torch.equal(x[18], e) for x, e in zip(swa_buffers, swa_expected)]
    if not all(full_checks + swa_checks):
        raise SystemExit(
            f"SWA radix move dropped payload or scale bytes: "
            f"full={full_checks} swa={swa_checks}"
        )
    emit("swa_radix_move", full=all(full_checks), swa=all(swa_checks))


def allocation_gate() -> None:
    size = 262_144
    pool = make_pool(size=size)
    layer = SimpleNamespace(layer_id=0)
    loc = torch.arange(PAGE, 2 * PAGE, device=DEVICE, dtype=torch.int64)
    k = torch.randn(PAGE, H, D, device=DEVICE, dtype=DTYPE)
    v = torch.randn_like(k)
    pool.set_kv_buffer(layer, loc, k, v)
    kb, vb = pool.get_kv_buffer(0)
    ks, vs = pool.get_kv_scale_buffer(0)
    if kb.dtype != torch.uint8 or vb.dtype != torch.uint8:
        raise SystemExit(f"raw pool accessor returned {kb.dtype}/{vb.dtype}, expected uint8")
    if "batched_dequantize" in inspect.getsource(pool._get_key_buffer):
        raise SystemExit("raw key accessor contains whole-pool dequantization")

    q = torch.randn(1, 16, D, device=DEVICE, dtype=DTYPE)
    cu_q = torch.tensor([0, 1], device=DEVICE, dtype=torch.int32)
    seqused_k = torch.tensor([PAGE], device=DEVICE, dtype=torch.int32)
    page_table = torch.tensor([[1]], device=DEVICE, dtype=torch.int32)
    k_pages = kb.view(-1, PAGE, H, D // 2)
    v_pages = vb.view(-1, PAGE, H, D // 2)
    ks_pages = ks.view(-1, PAGE, H, D // 16)
    vs_pages = vs.view(-1, PAGE, H, D // 16)

    def attention_call():
        return flash_attn_varlen_func(
            q,
            k_pages,
            v_pages,
            cu_seqlens_q=cu_q,
            seqused_k=seqused_k,
            page_table=page_table,
            max_seqlen_q=1,
            max_seqlen_k=PAGE,
            causal=False,
            window_size=(None, None),
            num_splits=1,
            pack_gqa=True,
            sfk=ks_pages,
            sfv=vs_pages,
            kv_fp4=True,
        )

    with torch.inference_mode():
        warm, _ = attention_call()
    torch.cuda.synchronize()
    del warm
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    baseline = torch.cuda.memory_allocated()
    with torch.inference_mode():
        out, _ = attention_call()
    torch.cuda.synchronize()
    peak_delta = torch.cuda.max_memory_allocated() - baseline
    if not torch.isfinite(out).all():
        raise SystemExit("allocation-gate attention returned non-finite output")
    limit = 64 * 1024 * 1024
    if peak_delta >= limit:
        raise SystemExit(
            f"allocation gate exceeded fixed-tile limit: peak_delta={peak_delta} "
            f"limit={limit}"
        )
    logical_bf16_bytes = (size + PAGE) * H * D * 2 * 2
    raw_bytes = kb.numel() + vb.numel() + ks.numel() + vs.numel()
    emit(
        "no_pool_sized_bf16_allocation",
        pool_tokens=size,
        raw_bytes=raw_bytes,
        logical_bf16_bytes=logical_bf16_bytes,
        storage_ratio=raw_bytes / logical_bf16_bytes,
        peak_delta_bytes=peak_delta,
        peak_limit_bytes=limit,
        finite=True,
    )


def main() -> int:
    torch.manual_seed(20260805)
    capture_stream_selection_gate()
    ordinary_and_move_gate()
    dspark_prefix_valid_gate()
    swa_gate()
    allocation_gate()
    print(
        "FA4 FP4 POOL GATE PASS writers=4 allocation=fixed-tile stream-select=1",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
