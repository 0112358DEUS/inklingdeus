# SM120 paged-KV FA4 provenance

`third_party/inkling_sm120_fa4/` is an unchanged copy of the BSD-3-Clause vendor bundle from
`eugr/spark-vllm-docker` commit `552b618c6e0b092d45a7290916547a7ccc1a078d`, directory
`mods/inkling-sm12-paged-kv/vendor/inkling_sm120_fa4`.

The bundle records its upstream as `SecondNatureComputing/flash-attn-4-sm120` commit
`60117041e10fcc6f19882afd274318c755a5ef6e`. It also records two mechanical CUTLASS DSL 4.6
migrations from `vllm-project/tml-fa4` commit
`b206834606ed5b5f21f8eed6b0683f528ea9cf7d`: `cute.core.ThrMma` to `cute.ThrMma` and
`cute.make_fragment` to `cute.make_rmem_tensor`.

The original `LICENSE`, `AUTHORS`, and `UPSTREAM_COMMIT` are retained. `SHA256SUMS` covers every
vendored source file and must pass before the bundle is copied into a development image. The
champion image is never modified; E7 uses a separate `fa4-sm121-dev` tag.
