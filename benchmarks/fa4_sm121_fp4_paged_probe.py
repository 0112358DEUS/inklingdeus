#!/usr/bin/env python3
"""SM121 FA4 page-128 fused fp4_mx_block16 numerical gate."""

from __future__ import annotations

import json
import math

import torch
from sglang.kernels.ops.attention.flash_attn.cute import flash_attn_varlen_func
from sglang.srt.layers.quantization.kvfp4_tensor import (
    FP4MXBlock16KVQuantizeUtil,
)


def quantize_pages(x: torch.Tensor, page_size: int):
    num_pages, _, heads, dim = x.shape
    flat = x.reshape(-1, heads, dim)
    payload, scales = FP4MXBlock16KVQuantizeUtil.batched_quantize(flat)
    payload = payload.view(torch.uint8).reshape(
        num_pages, page_size, heads, dim // 2
    )
    scales = scales.view(torch.uint8).reshape(
        num_pages, page_size, heads, dim // 16
    )
    return payload, scales


def dequantize_pages(payload: torch.Tensor, scales: torch.Tensor, dim: int):
    pages, page_size, heads, _ = payload.shape
    flat_payload = payload.reshape(-1, heads, dim // 2)
    flat_scales = scales.reshape(-1, heads * dim // 16)
    result = FP4MXBlock16KVQuantizeUtil.batched_dequantize(
        flat_payload, flat_scales
    )
    return result.reshape(pages, page_size, heads, dim)


def reference(q, k, v, length, local, h_q, h_kv, head_dim):
    k_seq = k.reshape(-1, h_kv, head_dim)[:length].float()
    v_seq = v.reshape(-1, h_kv, head_dim)[:length].float()
    k_expanded = k_seq.repeat_interleave(h_q // h_kv, dim=1)
    v_expanded = v_seq.repeat_interleave(h_q // h_kv, dim=1)
    if local and length > 512:
        k_expanded = k_expanded[-512:]
        v_expanded = v_expanded[-512:]
    scores = torch.einsum("hd,lhd->hl", q[0].float(), k_expanded)
    scores *= 1.0 / math.sqrt(head_dim)
    return torch.einsum(
        "hl,lhd->hd", torch.softmax(scores, dim=-1), v_expanded
    )


def main() -> int:
    torch.manual_seed(20260804)
    device = torch.device("cuda")
    dtype = torch.bfloat16
    h_q, h_kv, head_dim, page_size = 16, 4, 128, 128
    results = []
    for length in (1, 127, 128, 129, 511, 512, 513):
        num_pages = (length + page_size - 1) // page_size
        q = torch.randn(1, h_q, head_dim, device=device, dtype=dtype)
        k = torch.randn(
            num_pages, page_size, h_kv, head_dim, device=device, dtype=dtype
        )
        v = torch.randn_like(k)
        kp, ks = quantize_pages(k, page_size)
        vp, vs = quantize_pages(v, page_size)
        # Repeat quantization must be byte-identical before the attention path
        # is allowed to consume the payload.
        kp2, ks2 = quantize_pages(k, page_size)
        vp2, vs2 = quantize_pages(v, page_size)
        if not all(
            torch.equal(a, b)
            for a, b in ((kp, kp2), (ks, ks2), (vp, vp2), (vs, vs2))
        ):
            raise SystemExit("FP4 QUANTIZER REPEATABILITY FAIL")
        kdq = dequantize_pages(kp, ks, head_dim)
        vdq = dequantize_pages(vp, vs, head_dim)
        cu_q = torch.tensor([0, 1], device=device, dtype=torch.int32)
        seqused_k = torch.tensor([length], device=device, dtype=torch.int32)
        page_table = torch.arange(
            num_pages, device=device, dtype=torch.int32
        ).view(1, -1)
        for local in (False, True):
            with torch.inference_mode():
                out, _ = flash_attn_varlen_func(
                    q,
                    kp,
                    vp,
                    cu_seqlens_q=cu_q,
                    seqused_k=seqused_k,
                    page_table=page_table,
                    max_seqlen_q=1,
                    max_seqlen_k=length,
                    causal=local,
                    window_size=(511, 0) if local else (None, None),
                    num_splits=1,
                    pack_gqa=True,
                    sfk=ks,
                    sfv=vs,
                    kv_fp4=True,
                )
            expected = reference(
                q, kdq, vdq, length, local, h_q, h_kv, head_dim
            )
            diff = (out[0].float() - expected).abs()
            record = {
                "length": length,
                "mode": "swa512" if local else "full",
                "finite": bool(torch.isfinite(out).all().item()),
                "max_abs": float(diff.max().item()),
                "mean_abs": float(diff.mean().item()),
                "payload_bytes": int(kp.numel() + vp.numel()),
                "scale_bytes": int(ks.numel() + vs.numel()),
            }
            print(json.dumps(record, sort_keys=True), flush=True)
            if not record["finite"] or record["max_abs"] > 0.05:
                raise SystemExit(f"FA4 FP4 NUMERICS FAIL: {record}")
            results.append(record)
    print(f"FA4 PAGED FP4 NUMERICS PASS cases={len(results)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
