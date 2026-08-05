---
title: "Streaming a 753B-Parameter Mixture-of-Experts Model on an 8 GB Laptop"
subtitle: "I/O concurrency, not bandwidth, is the binding constraint"
author: Udit Jain
date: August 2026
---

## Abstract

We run GLM-5.2 — a 753-billion-parameter mixture-of-experts model with 19,200 routed
experts — on a MacBook Pro with **8 GB of unified memory**. The quantized checkpoint is
**143 GiB, eighteen times the machine's total RAM**. No experts are pruned and no
distillation is used; weights are memory-mapped and streamed from NVMe, with the operating
system's page cache acting as a sliding window over the model.

We report a measurement-driven optimization of the streaming path yielding a **2.30×
speedup** (32,955 → 14,324 ms/token, n=16, noise floor 0.5%). The dominant finding is that
the binding constraint is neither compute nor raw disk bandwidth but **I/O concurrency**:
each compute thread blocks on its own page fault, so thread count effectively sets NVMe
queue depth. A second contribution exploits an asymmetry in MoE inference — routed experts
cannot be prefetched because the router has not yet run, but **attention weights can**,
since they occupy static addresses and constitute 72% of per-token bytes. A graph-level
lookahead prefetcher for these weights yields **31%** in a controlled A/B.

We make no quality claim. We report why: benchmarking is structurally blocked, not merely
slow, because **MoE prefill does not amortize**.

## 1. Setting

| | |
|---|---|
| Model | GLM-5.2, 753B params, 79 blocks, 256 experts/layer, 8 active, MLA attention |
| Quantization | 1.63 bits/weight; experts `IQ1_S_R4`, attention `Q6_0`; imatrix-calibrated |
| Checkpoint | 143 GiB |
| Machine | Apple M3, 8 GB unified memory, NVMe measured at 2.48 GB/s sequential |
| Engine | `ik_llama.cpp` (mainline llama.cpp cannot parse arch `glm-dsa`) |

Feasibility rests on sparsity: **40B of 753B parameters are active per token (3.7%)**.

## 2. Byte accounting

Per token the model touches **14.5 GiB** of weights:

| component | size | share |
|---|---|---|
| Dense trunk (attention, shared experts, embeddings) | 10.4 GiB | **72%** |
| Routed experts (8 of 256 × 75 layers) | 4.25 GiB | 28% |

The trunk dominates despite being **5% of the file**, because it is read in full every
token while only 3.1% of experts fire.

Measured disk traffic is **24.4 GiB/token** against 14.5 GiB of unique bytes — a **1.68×
re-read factor**, since the working set is roughly 3× the available page cache.

## 3. The binding constraint is I/O concurrency

Thread count scaling, all else fixed:

```
t2 40,623   t4 28,208   t6 20,227   t8 17,710   t12 16,591   t16 16,799   t24 19,638   t32 24,353
```

Near-linear scaling to t12 is inconsistent with a bandwidth limit and with a compute limit.
The mechanism is queue depth: each thread blocks on an independent page fault. Consistent
with this, **dedicated prefetch worker threads buy concurrency more cheaply than compute
threads** — 4 compute threads with 8 prefetch workers (15,512 ms/tok) outperform 8 compute
threads alone (17,710 ms/tok) — and with prefetch active the compute-thread optimum shifts
from t12 to t16.

Two further observations support this reading. Metal offload has **no effect** (the workload
is not compute-bound). Reducing active experts via `-ser` makes throughput **worse**, not
better, confirming that experts are not the bottleneck.

## 4. Trunk lookahead prefetching

MoE inference presents an asymmetry. Routed experts cannot be prefetched: their identities
depend on a router that has not yet executed. **Dense trunk weights can be**: their addresses
are static and independent of routing.

We add `ggml_trunk_prefetch_lookahead()`, which walks K nodes ahead in the computation graph
and populates `MUL_MAT` operand weights resident in the mapping, skipping `MUL_MAT_ID`
(routed experts). Controlled A/B, identical configuration, only K varied:

| K | ms/token |
|---|---|
| 0 (disabled) | 17,318 |
| 8 | 12,638 |
| 16 | 12,347 |
| 32 | **11,957** |

**A 31% improvement.** K is **non-monotonic** under thread variation (at t16: K=16 → 13,756,
K=32 → 14,077, K=64 → 13,176, K=128 → 14,150). We attribute this to contention: when the
working set exceeds page cache, aggressive lookahead evicts pages still required by the
current layer. Lookahead depth must be tuned against cache headroom, not maximized.

## 5. A latent portability defect

`llama_get_device_memory()` provides branches for RPC, CUDA, SYCL, Vulkan and CANN, but none
for Metal, falling through to `return 1;` — one byte. This is below the 1024 MiB safety
margin, so available device memory is clamped to zero and the loader reports
`offloaded 0/N layers` on **every Apple Silicon machine, silently, regardless of `-ngl`**.
We supply `ggml_backend_metal_get_device_memory()` and the missing branch. The fix is
independent of this work.

For this model the correction does not enable offload — even with all experts forced to CPU
the trunk requires 17.4 GiB against 4.4 GiB available — but it establishes that as a
**measured** result rather than an artifact.

## 6. Why no benchmark

**MoE prefill does not amortize.** For dense models, batching amortizes weight reads across
the batch. For MoE, each token in a batch routes to different experts, so a 512-token batch
touches most of the 256 experts per layer: batching reads *more* weight bytes, not fewer.
A wikitext perplexity run completed **zero chunks in 13 minutes**.

At measured rates, HumanEval requires ~115 hours, MMLU ~180 days, SWE-bench Verified ~500
days. **No standard benchmark is reachable**, and this follows from architecture, not tuning.

Consequently we make no quality claim. GLM-5.2's published figures are for the unquantized
model; at 1.63 bpw — more aggressive than any quantization the primary distributor ships —
degradation should be assumed. Six spot checks are released as anecdote, explicitly labelled.

## 7. Limits, and what is not bounded

Streaming one token per forward pass with no expert reuse bounds throughput at
4.25 GiB ÷ 2.31 GiB/s = **0.54 tok/s**. This bounds *that construction*, not the task.
Untested and unexcluded:

1. **Multi-token prediction.** The checkpoint contains `nextn` tensors. Verifying k
   speculated tokens in one forward pass multiplies the bound directly by k.
2. **Expert caching.** Routing skew was never measured; if the router is concentrated, a
   RAM-resident cache changes the arithmetic.
3. **Locality-optimized layout.** Observed reads are 13–113 KB against a device delivering
   2,480 MB/s sequentially — 32% of hardware capability. Co-activated experts stored
   adjacently would convert scattered reads to sequential ones.

A structural threshold exists at **~15 GiB of page cache**, where one token's working set
becomes resident and the 1.68× re-read vanishes — placing a 16 GB machine on the far side
of a discontinuity rather than a gradient.

## 8. Reproducibility

Throughput degrades with accumulated swap, which evicts the page cache the method depends
on: 14.3 s/token at fresh boot (≈5 GB cache), 19.5 s/token at 4.9 GB swap (3.3 GB cache),
71 s/token at 6.3 GB swap (1.6 GB cache) — **identical build and flags**. Measurements
must be taken from a clean boot. Additionally, short generations flatter results: identical
flags produced 12,392 ms/token at n=5 versus 13,756 at n=8, as early tokens exploit residual
post-load cache warmth. All figures reported here use n=16.

Code, patches and samples: `https://github.com/uditjainstjis/glm52-on-8gb`
