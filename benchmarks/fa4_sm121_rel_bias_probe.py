#!/usr/bin/env python3
"""Inkling TP2 FA4 score-mod relative-bias numerics on real SM121 hardware."""

from __future__ import annotations

import json
import math

import torch
from sglang.kernels.ops.attention.flash_attn.cute import flash_attn_varlen_func
from sglang.srt.models.inkling_common.attn import (
    get_inkling_relative_attention_score_mod,
)


def reference(q, k, v, rel_bias, length, local, h_q, h_kv, head_dim):
    k_seq = k.reshape(-1, h_kv, head_dim)[:length].float()
    v_seq = v.reshape(-1, h_kv, head_dim)[:length].float()
    k_expanded = k_seq.repeat_interleave(h_q // h_kv, dim=1)
    v_expanded = v_seq.repeat_interleave(h_q // h_kv, dim=1)
    kv_indices = torch.arange(length, device=q.device)
    if local and length > 512:
        k_expanded = k_expanded[-512:]
        v_expanded = v_expanded[-512:]
        kv_indices = kv_indices[-512:]
    scores = torch.einsum("hd,lhd->hl", q[0].float(), k_expanded)
    scores *= 1.0 / math.sqrt(head_dim)
    rel_dist = (length - 1) - kv_indices
    valid = rel_dist < rel_bias.shape[-1]
    safe_dist = rel_dist.clamp(max=rel_bias.shape[-1] - 1)
    bias = rel_bias[0].float().index_select(-1, safe_dist)
    scores += torch.where(valid.unsqueeze(0), bias, torch.zeros_like(bias))
    return torch.einsum("hl,lhd->hd", torch.softmax(scores, dim=-1), v_expanded)


def main() -> int:
    torch.manual_seed(20260803)
    device = torch.device("cuda")
    dtype = torch.bfloat16
    h_q, h_kv, head_dim, page_size, rel_extent = 16, 4, 128, 128, 1024
    score_mod = get_inkling_relative_attention_score_mod(rel_extent)
    results = []
    for length in (1, 127, 128, 129, 511, 512, 513):
        num_pages = (length + page_size - 1) // page_size
        q = torch.randn(1, h_q, head_dim, device=device, dtype=dtype)
        k = torch.randn(num_pages, page_size, h_kv, head_dim, device=device, dtype=dtype)
        v = torch.randn_like(k)
        rel_bias = torch.randn(1, h_q, rel_extent, device=device, dtype=dtype)
        cu_q = torch.tensor([0, 1], device=device, dtype=torch.int32)
        seqused_k = torch.tensor([length], device=device, dtype=torch.int32)
        page_table = torch.arange(num_pages, device=device, dtype=torch.int32).view(1, -1)
        for local in (False, True):
            with torch.inference_mode():
                out, _ = flash_attn_varlen_func(
                    q,
                    k,
                    v,
                    cu_seqlens_q=cu_q,
                    seqused_k=seqused_k,
                    page_table=page_table,
                    max_seqlen_q=1,
                    max_seqlen_k=length,
                    causal=local,
                    window_size=(511, 0) if local else (None, None),
                    num_splits=1,
                    pack_gqa=True,
                    score_mod=score_mod,
                    aux_tensors=[rel_bias],
                )
            expected = reference(
                q, k, v, rel_bias, length, local, h_q, h_kv, head_dim
            )
            diff = (out[0].float() - expected).abs()
            record = {
                "length": length,
                "mode": "swa512" if local else "full",
                "finite": bool(torch.isfinite(out).all().item()),
                "max_abs": float(diff.max().item()),
                "mean_abs": float(diff.mean().item()),
            }
            print(json.dumps(record, sort_keys=True), flush=True)
            if not record["finite"] or record["max_abs"] > 0.05:
                raise SystemExit(f"FA4 REL BIAS NUMERICS FAIL: {record}")
            results.append(record)
    print(f"FA4 REL BIAS NUMERICS PASS cases={len(results)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
