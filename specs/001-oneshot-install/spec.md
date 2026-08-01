# Spec: Lossless Inkling-Small + DSpark serving on 2× DGX Spark

## Goal
An OpenAI-compatible endpoint serving thinkingmachines/Inkling-Small-NVFP4 with the RadixArk
DSpark speculator, TP=2 across two GB10 nodes, that is (a) byte-exact lossless vs non-speculative
decoding at temp 0, and (b) at the measured block-5 champion throughput: 26.007 ± 0.334 tok/s on
chat-templated open-ended serving (`n=32`). The old ~34 tok/s raw-continuation probe is diagnostic
only (see README's task-class table and its 2026-08-01 correction).

## Success criteria
- SC1: server logs `Initialized DSpark draft runner ... gamma=5` (spec config not silently dropped)
- SC2: T4 lossless probe matches byte-exact
- SC3: chat-templated open-ended probe (`benchmarks/chat_bench.py --task open-ended`, n=32) is
  within the README champion error band, currently 26.007 ± 0.334 tok/s and 2.093 ± 0.026 accept. The
  legacy raw `/generate` probe is diagnostic only (see docs/MEASUREMENT-PROTOCOL.md).
- SC4: chat C1 medians within ±15% of README table
- SC5: survives C8 concurrent load without scheduler death (no wedge/EngineDead)

## Non-goals
- fa4 attention, TRT-LLM/cutlass FP4 MoE (arch-impossible / numerically unsafe on sm_121)
- Training or evaluating the draft beyond its 64K adaptation window. The baked draft-context cap
  keeps short-request acceptance stable even when the target advertises the full 1M context.
- vLLM path (Inkling MTP works there but caps at k=1 on 0.26-line builds; Lamport op needs MNNVL)
