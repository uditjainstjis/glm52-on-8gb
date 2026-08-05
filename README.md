# GLM-5.2 · 753B params · 1.63bpw · **2.5 GB RAM**

**A 753-billion-parameter mixture-of-experts model running inference on a MacBook with 8 GB of memory.**
All 19,200 experts. Nothing pruned. Nothing distilled.

```
$ glm "What is 17 multiplied by 23?"

  391

    [2 tok · 29s · 14.3 s/tok · 0.070 tok/s]
```
<sub>Real output, verbatim. More in <code>samples/</code> — including the failures.</sub>

| | |
|---|---|
| **Model** | GLM-5.2 — 753B params, 256 experts/layer × 75 layers = **19,200 experts** |
| **Quantization** | **1.63 bits/weight** (experts `IQ1_S_R4`, attention `Q6_0`), imatrix-calibrated |
| **Peak RSS** | **2.45 GB** |
| **On disk** | 143 GiB — **18× the machine's total RAM** |
| **Machine** | MacBook Pro, Apple M3, **8 GB** unified memory |
| **Speed** | **14.3 s/token** (2.30× faster than the stock config) |

> **Full memory ledger, so nothing is hidden:** 2.45 GB process RSS **+ 3.26 GB OS page cache**
> (where the mmap'd weights actually live) + 1.62 GB kernel = **7.33 GB of 8 GB in use**.
> RSS alone is the number comparable to prior art; it is not the whole story, and the whole
> story is above.

For reference, [Kimi-K3-in-C](https://github.com/FareedKhan-dev/kimi-k3-in-c) reports 8.24 GB
peak RSS and 32.69 s/token — measured on a **228 GiB EPYC 7763 server**, per its own
`PERFORMANCE.md`. This runs on a laptop that has 8 GB, at 14.3 s/token.

---

## The constraint is the point

This model's own documentation says it needs a **256 GB unified-memory Mac**. Unsloth's smallest published full-model quant is **217 GB**. Serving it at FP16 takes **~1,642 GB of VRAM**.

It runs in **8 GB**.

That works because **only 3.7% of the parameters are active per token** (40B of 753B), and the weights are memory-mapped — the OS pages in what the router selects and evicts the rest. 8 GB is the *window*, not the model.

### What it actually costs

Nothing here is free, and the honest ledger is:

| | |
|---|---|
| **Disk** | 143 GiB, and it is read continuously |
| **Bytes per token** | **14.5 GiB** — 10.4 GiB trunk + 4.25 GiB routed experts |
| **Disk read per token** | **24.4 GiB measured** (1.68× re-read; the working set is ~3× the page cache) |
| **RAM in use** | 7.33 / 8 GB — 2.45 GB process + 3.26 GB page cache + 1.62 GB kernel |
| **Speed** | ~16 s/token → a 500-word page takes ~2.5 hours |
| **First token** | ~26 s |

**The bottleneck is I/O concurrency, not compute and not bandwidth.** Each thread blocks on its own page fault, so thread count effectively sets the NVMe queue depth.

---

## Speed: 2.30× over the stock configuration

All measurements at n=16 tokens, both terms in the same environment, noise floor 0.5% (3 repeats, SD 29 ms).

```
baseline   -t 4                                    32,955 ms/token
tuned      -t 16, 8 prefetch workers, lookahead 64  14,324 ms/token
                                                    ─────────────
                                                        2.30×
```

For scale: [Kimi-K3-in-C](https://github.com/FareedKhan-dev/kimi-k3-in-c) reports 32.69 s/token — measured on a **228 GiB EPYC 7763 server**, per its own `PERFORMANCE.md`.

### Where the gain came from

| lever | effect |
|---|---|
| Porting the MoE prefetch subsystem to macOS | it was `#if defined(__linux__)` — no-op stubs on every Mac |
| I/O concurrency: 16 threads + 8 prefetch workers | prefetch workers buy queue depth more cheaply than compute threads |
| **Graph-level trunk lookahead** (new) | **31%**, clean A/B (K=0 → 17,318 ms/tok, K=32 → 11,957) |

The trunk lookahead exploits an asymmetry: **routed experts can't be prefetched** (the router hasn't run), **but attention weights can** — they sit at static addresses and are 72% of per-token bytes.

⚠️ **K is non-monotonic.** At 16 threads: K=16 → 13,756, K=32 → 14,077 (worse), K=64 → 13,176 (best), K=128 → 14,150. Lookahead competes with live data for a page cache 3× too small. Tune it per thread count.

---

## Patches

Requires [`ik_llama.cpp`](https://github.com/ikawrakow/ik_llama.cpp) — mainline llama.cpp **cannot read this file** (arch `glm-dsa`, quant types 133/219 outside its 0–43 range).

`patches/ik_llama_macos_moe.patch` — 138 insertions, 6 files:

1. **Bug fix — Metal offload is silently dead on every Apple Silicon Mac.** `llama_get_device_memory()` has branches for RPC/CUDA/SYCL/Vulkan/CANN but none for Metal, so it falls through to `return 1;`. One byte is below the 1024 MiB safety margin → clamped to 0 → `offloaded 0/N layers`, regardless of `-ngl`. **This fix is useful independent of anything here.**
2. **Port** — MoE expert prefetch compiled to no-op stubs on macOS (`mincore` signature differs; `MADV_POPULATE_READ` doesn't exist, emulated by touching pages).
3. **Feature** — `ggml_trunk_prefetch_lookahead()`, env `GGML_TRUNK_LOOKAHEAD`.

---

## Reproduce

```bash
# 1. engine
git clone --depth 1 https://github.com/ikawrakow/ik_llama.cpp && cd ik_llama.cpp
git apply /path/to/patches/ik_llama_macos_moe.patch
cmake -B build -DGGML_METAL=ON -DCMAKE_BUILD_TYPE=Release && cmake --build build -j4

# 2. weights — 143 GiB
hf download sokann/GLM-5.2-GGUF-1.630bpw GLM-5.2-GGUF-1.630bpw-muzzy-imatrix.gguf --local-dir model

# 3. run
GGML_TRUNK_LOOKAHEAD=64 ./build/bin/llama-cli \
  -m model/GLM-5.2-GGUF-1.630bpw-muzzy-imatrix.gguf \
  -ngl 0 -t 16 --prefetch-experts --prefetch-experts-threads 8 \
  -c 512 --temp 1.0 --top-p 0.95 --repeat-penalty 1.05 -p "..."
```

`scripts/chat.sh` runs it as a server (**model loads once**, ~12 min, then stays resident); `scripts/glm` is a terminal client with live streaming and a tok/s meter.

---

## What this is not

**No benchmark was run, and no quality claim is made here.**

GLM-5.2's published figures (74.4 FrontierSWE, 81.0 Terminal-Bench 2.1, 62.1 SWE-bench Pro) are for the **unquantized** model. This is 1.63 bits/weight — more aggressive than any quant Unsloth ships (their smallest is ~2.3 bpw effective, documented at ~82% of full accuracy). **Assume it is worse. It has not been measured.**

Benchmarking is not merely slow, it is structurally blocked: **MoE prefill does not amortize.** A 512-token batch routes each token to different experts, touching most of the 256 per layer — so batching reads *more* weight bytes, not fewer. HumanEval would take ~115 hours; SWE-bench Verified, ~500 days.

Spot checks (6 prompts, in `samples/`) show correct content with unreliable termination — correct algorithms and correct factual sequences, with occasional spurious tokens. **That is an anecdote, not a measurement, and it is presented as one.**

---

## Known limits

- **~0.54 tok/s** is the ceiling for one-token-per-forward-pass streaming with no expert reuse (4.25 GiB ÷ 2.31 GiB/s). **This is not a bound on the task** — MTP speculative decoding (the `nextn` tensors are in the file), expert caching, and locality-optimized layout are all untested.
- **Trunk requantize** Q6_0→Q4_K is only **1.28×** (14.3 → 11.2 s/token). Not worth it.
- **Metal offload is impossible here**: even with every expert on CPU, the trunk needs 17.4 GiB against 4.4 GiB available.
- **16 GB is the interesting threshold.** One token's working set is 14.5 GiB; above that the 1.68× re-read vanishes and every token after the first is served from RAM.


---

## Reproducibility caveat

Throughput degrades sharply as macOS swap accumulates. Measured on the same build, same flags:

| state | page cache | speed |
|---|---|---|
| fresh boot | ~5 GB | **14.3 s/token** |
| after hours of use (4.9 GB swap) | 3.3 GB | 19.5 s/token |
| 6.3 GB swap | 1.6 GB | 71 s/token |

The page cache is where the mmap'd weights live, so swap pressure starves the exact
mechanism this depends on. **Reboot before measuring.** Running as a resident server is a
net loss on 8 GB — it holds ~2.45 GB permanently and accelerates the swap spiral.
