# Licensing and provenance

This repository is a fork of
[drowzeys/keys-1M-CTX-Inkling-Small-NVFP4-Dspark-NVFP4-KV-Cache-SGlang-SM121-optimized-on-Two-DGX-Sparks](https://github.com/drowzeys/keys-1M-CTX-Inkling-Small-NVFP4-Dspark-NVFP4-KV-Cache-SGlang-SM121-optimized-on-Two-DGX-Sparks)
(an independent field port, not affiliated with LMSYS, Thinking Machines, RadixArk, or NVIDIA).

Two distinct bodies of work live here:

1. **`patches/files/` and `patches/kv-quant/`** contain modified copies of source files from
   [SGLang](https://github.com/sgl-project/sglang), which is licensed under the
   **Apache License 2.0** (Copyright the SGLang Team). Those files remain under Apache-2.0 —
   see [`patches/LICENSE`](patches/LICENSE) for the full text. The modifications (GB10/sm_121
   fixes, the NVFP4 KV-cache implementation, the DSpark draft-width and conv-state-commit
   fixes) are documented per file in `patches/all-patches.diff` and `docs/`.

2. **Everything else** (documentation, scripts, benchmarks, journal) is original work of the
   upstream repository's author. No explicit license was published upstream at the time of
   forking; it is reproduced here under GitHub's terms for public forks. If the upstream
   author publishes a license, it applies.
