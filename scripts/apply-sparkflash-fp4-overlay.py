#!/usr/bin/env python3
"""Apply the small SGLang-side SparkFlash FP4 adapter, fail-closed."""

from __future__ import annotations

import sys
from pathlib import Path


def replace_exact(path: Path, before: str, after: str, expected: int = 1) -> None:
    text = path.read_text(encoding="utf-8")
    count = text.count(before)
    if count != expected:
        raise SystemExit(
            f"overlay source drift: {path} expected {expected} matches, got {count}"
        )
    path.write_text(text.replace(before, after), encoding="utf-8")


def main() -> int:
    if len(sys.argv) != 2:
        raise SystemExit("usage: apply-sparkflash-fp4-overlay.py SGLANG_PACKAGE_ROOT")
    root = Path(sys.argv[1]).resolve()
    wrapper = root / "kernels/ops/attention/flash_attention_v4.py"
    backend = root / "srt/layers/attention/flashattention_backend.py"

    replace_exact(
        wrapper,
        """    sfv: Optional[
        torch.Tensor
    ] = None,  # MXFP8 UE8M0 per-32-elem block scales (in-kernel V dequant)
    rel_bias: Optional[torch.Tensor] = None,
""",
        """    sfv: Optional[
        torch.Tensor
    ] = None,  # MXFP8 UE8M0 per-32-elem block scales (in-kernel V dequant)
    kv_fp4: bool = False,
    rel_bias: Optional[torch.Tensor] = None,
""",
    )
    replace_exact(
        wrapper,
        """    if sfv is not None:
        sf_kwargs["sfv"] = sfv

    descale_kwargs = {}
""",
        """    if sfv is not None:
        sf_kwargs["sfv"] = sfv
    fp4_kwargs = {"kv_fp4": True} if kv_fp4 else {}

    descale_kwargs = {}
""",
    )
    replace_exact(
        wrapper,
        """        return_lse=return_softmax_lse,
        **sf_kwargs,
        **descale_kwargs,
""",
        """        return_lse=return_softmax_lse,
        **sf_kwargs,
        **fp4_kwargs,
        **descale_kwargs,
""",
    )
    replace_exact(
        wrapper,
        """    sfq: Optional[torch.Tensor] = None,
    sfk: Optional[torch.Tensor] = None,
    sfv: Optional[torch.Tensor] = None,
    rel_bias: Optional[torch.Tensor] = None,
""",
        """    sfq: Optional[torch.Tensor] = None,
    sfk: Optional[torch.Tensor] = None,
    sfv: Optional[torch.Tensor] = None,
    kv_fp4: bool = False,
    rel_bias: Optional[torch.Tensor] = None,
""",
    )
    replace_exact(
        wrapper,
        """        sfq=sfq,
        sfk=sfk,
        sfv=sfv,
        rel_bias=rel_bias,
""",
        """        sfq=sfq,
        sfk=sfk,
        sfv=sfv,
        kv_fp4=kv_fp4,
        rel_bias=rel_bias,
""",
    )

    replace_exact(
        backend,
        """            and layer.head_dim <= 256
            and not self.kv_cache_is_mxfp8
        ):
""",
        """            and layer.head_dim <= 256
            and not self.kv_cache_is_mxfp8
            and self.kv_cache_dtype_str not in ("nvfp4", "fp4_mx_block16")
        ):
""",
        expected=1,
    )
    replace_exact(
        backend,
        """            and layer.head_dim <= 256
            and self.fa_impl_ver != 4
            and not self.kv_cache_is_mxfp8
        ):
""",
        """            and layer.head_dim <= 256
            and self.fa_impl_ver != 4
            and not self.kv_cache_is_mxfp8
            and self.kv_cache_dtype_str not in ("nvfp4", "fp4_mx_block16")
        ):
""",
    )

    cache_before = """            key_cache, value_cache = self.token_to_kv_pool.get_kv_buffer(layer.layer_id)

            key_cache = key_cache.view(
                -1, self.page_size, layer.tp_k_head_num, layer.head_dim
            )
            value_cache = value_cache.view(
                -1, self.page_size, layer.tp_v_head_num, layer.v_head_dim
            )
"""
    cache_after = """            key_cache, value_cache = self.token_to_kv_pool.get_kv_buffer(layer.layer_id)

            if key_cache.dtype == torch.uint8:
                if not hasattr(self.token_to_kv_pool, "get_kv_scale_buffer"):
                    raise RuntimeError("raw FP4 KV payload is missing scale buffers")
                key_scale, value_scale = self.token_to_kv_pool.get_kv_scale_buffer(
                    layer.layer_id
                )
                key_cache = key_cache.view(
                    -1, self.page_size, layer.tp_k_head_num, layer.head_dim // 2
                )
                value_cache = value_cache.view(
                    -1, self.page_size, layer.tp_v_head_num, layer.v_head_dim // 2
                )
                kwargs["sfk"] = key_scale.view(
                    -1, self.page_size, layer.tp_k_head_num, layer.head_dim // 16
                )
                kwargs["sfv"] = value_scale.view(
                    -1, self.page_size, layer.tp_v_head_num, layer.v_head_dim // 16
                )
                kwargs["kv_fp4"] = True
            else:
                key_cache = key_cache.view(
                    -1, self.page_size, layer.tp_k_head_num, layer.head_dim
                )
                value_cache = value_cache.view(
                    -1, self.page_size, layer.tp_v_head_num, layer.v_head_dim
                )
"""
    replace_exact(backend, cache_before, cache_after)

    decode_before = cache_before.replace(
        "layer.layer_id)\n\n            key_cache", "layer.layer_id)\n            key_cache"
    )
    decode_after = cache_after.replace(
        "layer.layer_id)\n\n            if key_cache", "layer.layer_id)\n            if key_cache"
    )
    replace_exact(backend, decode_before, decode_after)

    print("SPARKFLASH FP4 SGLANG OVERLAY PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
