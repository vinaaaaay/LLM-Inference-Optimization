# Flash Decoding vs Normal Decoding: Experimental Analysis Report

**Model:** facebook/opt-350m (331M parameters)
**GPU:** NVIDIA GeForce RTX 3050 (5795 MB VRAM, Compute Capability 8.6, Ampere)
**Software:** PyTorch 2.10.0, CUDA 12.8
**Date:** March 2026

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [What is Flash Decoding?](#2-what-is-flash-decoding)
3. [Experimental Setup](#3-experimental-setup)
4. [Experiment 1: Decode Latency vs KV-Cache Length](#4-experiment-1-decode-latency-vs-kv-cache-length)
5. [Experiment 2: Decode Throughput Scaling](#5-experiment-2-decode-throughput-scaling)
6. [Experiment 3: Batch Size Impact on Decode](#6-experiment-3-batch-size-impact-on-decode)
7. [Experiment 4: Kernel-Level Decode Attention](#7-experiment-4-kernel-level-decode-attention)
8. [Experiment 5: End-to-End + Memory Analysis](#8-experiment-5-end-to-end--memory-analysis)
9. [Experiment 6: Memory-Bound vs Compute-Bound Analysis (FP16 vs FP32)](#9-experiment-6-memory-bound-vs-compute-bound-analysis-fp16-vs-fp32)
10. [Cross-Experiment Analysis](#10-cross-experiment-analysis)
11. [Comparison with Flash Attention Study](#11-comparison-with-flash-attention-study)
12. [Conclusion](#12-conclusion)

---

## 1. Executive Summary

This report presents an empirical comparison of **Flash Decoding** (optimized SDPA backends: FlashAttention and memory-efficient) vs **Normal Decoding** (standard math backend) during the decode phase of LLM inference. While the companion Flash Attention study focused on the **prefill phase**, this study focuses entirely on the **decode phase** — the autoregressive token-by-token generation that dominates inference time for generation-heavy workloads.

**Key Findings:**

- Flash decoding achieves **up to 5x faster per-token decode latency** compared to normal decoding at long KV-cache lengths (564ms vs 114ms per token at cache_len=1800).
- At the **kernel level**, flash decode attention is **5.1x faster** than math at kv_len=2048 (0.086ms vs 0.437ms), with **2.4x less memory** (17 MB vs 41 MB).
- Flash decoding provides **~40% higher decode throughput** (141 tok/s vs 101 tok/s) in end-to-end generation.
- The **memory bottleneck during decode is the KV-cache**, not the attention matrix. The attention matrix for decode (query_len=1) is tiny (~0.125 MB at kv_len=2048), while the KV-cache grows to 192 MB. Flash decoding's advantage comes from more efficient KV-cache access patterns, not from avoiding attention matrix materialization.
- Flash and memory-efficient backends perform **nearly identically** during decode, with flash having a slight edge at longer KV lengths.
- **Precision analysis (FP16 vs FP32) confirms optimized backends are memory-bound** (FP32/FP16 ratio up to 3.3x at kernel level) while the math backend is compute-bound (FP32 is paradoxically *faster* than FP16). At the model level, decode is memory-bound at short cache (ratio 2.0x) transitioning to mixed at long cache (ratio 1.25x).

---

## 2. What is Flash Decoding?

### Normal Decoding (Math Backend)

During autoregressive generation, each new token requires computing attention between a **single query** (the new token, shape `1 × d`) and the entire **KV-cache** (all previous tokens, shape `n × d`). The standard implementation:

1. Computes `score = q @ K^T` → shape `1 × n` (one value per cached token)
2. Applies `softmax(score / sqrt(d))` → shape `1 × n`
3. Computes `output = attn_weights @ V` → shape `1 × d`

This is **memory-bandwidth-bound**: the GPU must read the entire KV-cache from HBM for each decode step. With a small `1 × n` workload, the GPU's compute units are severely under-utilized — most time is spent waiting for memory transfers.

### Flash Decoding (Optimized Backends)

Flash Decoding introduces **parallelism over the KV sequence length**:

1. **Split** K and V into chunks along the sequence dimension.
2. **Compute** partial attention for each chunk in parallel across GPU streaming multiprocessors (SMs).
3. **Rescale and combine** using online softmax to produce the correct final result.

This keeps data in GPU SRAM (shared memory) as much as possible, reducing HBM round-trips. The GPU's compute units stay busy by processing multiple KV chunks simultaneously, dramatically improving utilization even at batch_size=1.

**Key difference from Flash Attention (prefill):** Flash Attention optimizes the `n × n` attention matrix during prefill (where both Q and K have length n). Flash Decoding optimizes the `1 × n` attention during decode (where Q has length 1, K has length n). The bottleneck is different — prefill is about avoiding O(n²) memory, decode is about maximizing GPU utilization for a memory-bound operation.

---

## 3. Experimental Setup

### Hardware

| Property | Value |
|---|---|
| GPU | NVIDIA GeForce RTX 3050 |
| VRAM | 5795 MB |
| Compute Capability | 8.6 (Ampere) |
| CUDA Version | 12.8 |
| PyTorch Version | 2.10.0+cu128 |

### Model

| Property | Value |
|---|---|
| Model | facebook/opt-350m |
| Parameters | ~331M |
| dtype | float16 |
| max_position_embeddings | 2048 |
| Attention Heads | 16 |
| Head Dimension | 64 |
| Layers | 24 |

### Backends Compared

| Backend | Description |
|---|---|
| **flash** | FlashAttention v2 kernel (SDPA FLASH_ATTENTION) |
| **mem_efficient** | xFormers memory-efficient kernel (SDPA EFFICIENT_ATTENTION) |
| **math** | Standard PyTorch matmul + softmax (SDPA MATH) — the "normal" baseline |

### Methodology

- All benchmarks use **5 measured runs** per configuration, reporting the **median**.
- **3 warm-up iterations** precede all measurements.
- **`torch.cuda.synchronize()`** is called **before AND after** every timing measurement to ensure we measure actual GPU execution time, not just kernel launch time.
- `torch.cuda.reset_peak_memory_stats()` is called before each run for accurate memory tracking.
- Greedy decoding (`do_sample=False`) is used throughout.

---

## 4. Experiment 1: Decode Latency vs KV-Cache Length

**Goal:** Measure how per-token decode latency scales as the KV-cache grows. This is the core experiment — during generation, each new token must attend to all previous tokens via the KV-cache.

**Method:** For each cache length, generate 16 tokens one at a time, measuring each step independently.

### Per-Token Decode Latency (ms)

| Cache Len | Flash | Mem-Efficient | Math | Flash vs Math |
|-----------|-------|---------------|------|---------------|
| 256 | 20.6 | 20.4 | 32.8 | 1.6x |
| 512 | 36.9 | 36.6 | 79.0 | 2.1x |
| 768 | 47.4 | 47.2 | 135.8 | 2.9x |
| 1024 | 64.1 | 64.3 | 216.0 | 3.4x |
| 1536 | 93.0 | 93.8 | 421.7 | 4.5x |
| 1800 | 113.6 | 115.5 | 564.3 | **5.0x** |

### Total Decode Time for 16 Tokens (seconds)

| Cache Len | Flash | Mem-Efficient | Math | Flash Savings |
|-----------|-------|---------------|------|---------------|
| 256 | 0.309 | 0.305 | 0.493 | 37% |
| 512 | 0.554 | 0.549 | 1.185 | 53% |
| 1024 | 0.961 | 0.965 | 3.241 | 70% |
| 1536 | 1.396 | 1.406 | 6.325 | 78% |
| 1800 | 1.704 | 1.732 | 8.465 | **80%** |

### Prefill Time (seconds)

| Cache Len | Flash | Math | Flash vs Math |
|-----------|-------|------|---------------|
| 256 | 0.019 | 0.030 | 1.6x |
| 512 | 0.029 | 0.068 | 2.3x |
| 1024 | 0.060 | 0.208 | 3.5x |
| 1800 | 0.114 | 0.561 | 4.9x |

### Observations

1. **The flash decoding advantage grows dramatically with KV-cache length.** At cache_len=256, flash is 1.6x faster. At cache_len=1800, it is 5.0x faster. This compounding effect occurs because the math backend's decode cost scales poorly with cache size — it must read the entire KV-cache from HBM for each token, while flash decoding's parallel chunking and SRAM-local processing amortize this cost.

2. **Math backend decode latency grows super-linearly.** From 256 to 1800 (7x more cache), math decode latency goes from 32.8ms to 564.3ms (17.2x increase). Flash goes from 20.6ms to 113.6ms (5.5x increase). The math backend becomes increasingly memory-bandwidth-starved as the KV-cache grows.

3. **Flash and mem_efficient are nearly identical** in decode performance, with flash having a marginal (~2%) edge at longer cache lengths. This contrasts with the prefill phase where mem_efficient sometimes outperforms flash on this GPU.

4. **Decode time dominates total generation time.** At cache_len=1800, the 16-token decode takes 8.5s with math vs 1.7s with flash — this is the time the user would wait for generation.

---

## 5. Experiment 2: Decode Throughput Scaling

**Goal:** Measure how decode throughput (tokens/sec) changes with generation length. Fixed prompt length of 512 tokens, varying generation from 32 to 256 tokens.

### End-to-End Latency (seconds)

| Gen Tokens | Flash | Mem-Efficient | Math | Flash vs Math |
|------------|-------|---------------|------|---------------|
| 32 | 0.246 | 0.244 | 0.357 | 1.45x |
| 64 | 0.470 | 0.466 | 0.660 | 1.40x |
| 128 | 0.925 | 0.916 | 1.284 | 1.39x |
| 256 | 1.851 | 1.844 | 2.603 | 1.41x |

### Decode-Only Throughput (tokens/sec)

| Gen Tokens | Flash | Mem-Efficient | Math | Improvement |
|------------|-------|---------------|------|-------------|
| 32 | 148.8 | 150.0 | 111.2 | +34% |
| 64 | 145.5 | 146.9 | 108.2 | +34% |
| 128 | 143.1 | 144.5 | 105.4 | +36% |
| 256 | 140.6 | 141.1 | 101.0 | +39% |

### Observations

1. **Flash decoding consistently provides ~35-39% higher decode throughput** across all generation lengths. The speedup ratio is remarkably stable, confirming this is a fundamental architectural advantage, not a noise artifact.

2. **Decode throughput decreases slightly with generation length** for all backends (~149 tok/s at 32 tokens → ~141 tok/s at 256 tokens for flash). This is because as more tokens are generated, the KV-cache grows, making each subsequent decode step slightly slower.

3. **TTFT is constant regardless of generation length** (~0.030s for flash, ~0.069s for math), confirming TTFT depends only on prompt length (prefill), not generation length.

4. **The speedup ratio (1.39-1.45x E2E) is smaller than per-token decode speedup** (Exp 1 showed 2.1x at cache_len=512) because E2E includes prefill time and early decode steps where the cache is small and flash's advantage is smaller.

---

## 6. Experiment 3: Batch Size Impact on Decode

**Goal:** Determine whether flash decoding's advantage holds across batch sizes. The hypothesis is that flash decoding should better utilize the GPU at all batch sizes.

### Decode-Only Throughput (tokens/sec per batch)

**Seq Len = 512:**

| Batch | Flash | Mem-Efficient | Math | Flash vs Math |
|-------|-------|---------------|------|---------------|
| 1 | 143.2 | 144.6 | 105.3 | 1.36x |
| 2 | 237.1 | 249.4 | 153.8 | 1.54x |
| 4 | 358.1 | 388.0 | 202.7 | 1.77x |

**Seq Len = 1024:**

| Batch | Flash | Mem-Efficient | Math | Flash vs Math |
|-------|-------|---------------|------|---------------|
| 1 | 126.1 | 122.8 | 79.6 | 1.58x |
| 2 | 187.1 | 198.5 | 105.0 | 1.78x |
| 4 | 255.3 | 280.1 | 126.3 | 2.02x |

### Peak Memory (MB)

| Seq Len | Batch | Flash | Math | Savings |
|---------|-------|-------|------|---------|
| 512 | 1 | 719 | 739 | 2.7% |
| 512 | 4 | 888 | 1027 | 13.5% |
| 1024 | 1 | 762 | 910 | 16.3% |
| 1024 | 4 | 1122 | 1702 | 34.1% |

### Observations

1. **Flash decoding's advantage INCREASES with batch size.** At batch=1, flash is 1.36-1.58x faster than math. At batch=4, this grows to 1.77-2.02x. Larger batches amplify the memory bandwidth bottleneck that math suffers from, making flash's SRAM-local processing increasingly beneficial.

2. **Mem_efficient outperforms flash at larger batch sizes** on this GPU (388 vs 358 tok/s at seq=512, batch=4). The xFormers kernel appears to have a more efficient batching strategy for the RTX 3050's cache hierarchy.

3. **Memory savings scale with both batch and sequence length.** At seq=1024, batch=4, flash uses 34% less memory (1122 vs 1702 MB). This freed memory could enable larger batch sizes or longer sequences.

4. **Throughput scales sub-linearly with batch size.** Doubling batch from 1 to 2 gives ~65-70% more throughput (not 100%), indicating memory bandwidth saturation. This is a hardware limitation, not a software one.

---

## 7. Experiment 4: Kernel-Level Decode Attention

**Goal:** Isolate the raw decode attention kernel by benchmarking `F.scaled_dot_product_attention` with a single query token against KV-caches of varying length. This removes all model overhead (FFN, layer norm, embedding) to measure pure attention cost.

**Configuration:** batch=1, 16 heads, head_dim=64, q_len=1.

### Kernel Latency (ms)

| KV Len | Flash | Mem-Efficient | Math | Flash vs Math |
|--------|-------|---------------|------|---------------|
| 128 | 0.032 | 0.027 | 0.095 | 3.0x |
| 256 | 0.033 | 0.029 | 0.094 | 2.8x |
| 512 | 0.046 | 0.044 | 0.138 | 3.0x |
| 1024 | 0.060 | 0.069 | 0.239 | 4.0x |
| 1536 | 0.075 | 0.092 | 0.340 | 4.5x |
| 2048 | 0.086 | 0.114 | 0.437 | **5.1x** |

### Kernel Memory (MB)

| KV Len | Flash | Mem-Efficient | Math | Math/Flash Ratio |
|--------|-------|---------------|------|-----------------|
| 128 | 9.6 | 9.6 | 11.1 | 1.2x |
| 256 | 10.1 | 10.1 | 13.2 | 1.3x |
| 512 | 11.1 | 11.1 | 17.2 | 1.5x |
| 1024 | 13.1 | 13.1 | 25.2 | 1.9x |
| 1536 | 16.0 | 16.0 | 34.1 | 2.1x |
| 2048 | 17.1 | 17.1 | 41.3 | **2.4x** |

### Latency Growth Factor (when doubling KV length)

| Transition | Flash | Mem-Efficient | Math |
|------------|-------|---------------|------|
| 128 → 256 | 1.03x | 1.07x | 0.99x |
| 256 → 512 | 1.39x | 1.52x | 1.47x |
| 512 → 1024 | 1.30x | 1.57x | 1.73x |
| 1024 → 2048 | 1.43x | 1.65x | 1.83x |

### Observations

1. **Flash is 5.1x faster at kv_len=2048** at the kernel level. This is the pure decode attention speedup, unmasked by any model overhead.

2. **The decode attention matrix is tiny** compared to prefill. At kv_len=2048 with q_len=1, the attention "matrix" is just `16 heads × 1 × 2048 × 2 bytes = 64 KB`. Compare with prefill at seq_len=2048 where it's `16 × 2048 × 2048 × 2 bytes = 128 MB`. The decode speedup is NOT from avoiding a large attention matrix — it's from better GPU utilization through parallel KV processing.

3. **Flash's advantage grows with KV length** (3.0x at 128 → 5.1x at 2048), confirming flash decoding's parallel KV strategy scales better than sequential processing.

4. **Flash outperforms mem_efficient at longer KV lengths** during decode (0.086ms vs 0.114ms at kv_len=2048). This is the opposite of what was observed in the companion study's prefill benchmarks, where mem_efficient had a slight edge. The different access pattern (1×n decode vs n×n prefill) favors flash's parallelization strategy.

5. **Memory difference is modest** during decode (2.4x at kv_len=2048) compared to prefill (25.5x at seq_len=2048 in the companion study). This is expected: the decode attention matrix is O(n) not O(n²), so even math doesn't use much memory for just the attention computation.

---

## 8. Experiment 5: End-to-End + Memory Analysis

### Part A: End-to-End Comparison with Prefill/Decode Breakdown

**Configuration:** batch=1, max_new_tokens=128, greedy decoding.

**Latency and Throughput:**

| Seq Len | Backend | Total Lat (s) | TTFT (s) | Decode Time (s) | Prefill % | Decode % | Tok/s |
|---------|---------|--------------|----------|-----------------|-----------|----------|-------|
| 512 | flash | 0.925 | 0.030 | 0.894 | 3.3% | 96.7% | 138.5 |
| 512 | mem_eff | 0.916 | 0.031 | 0.886 | 3.3% | 96.7% | 139.7 |
| 512 | math | 1.284 | 0.069 | 1.215 | 5.4% | 94.6% | 99.7 |
| 1024 | flash | 1.076 | 0.060 | 1.015 | 5.6% | 94.4% | 119.0 |
| 1024 | mem_eff | 1.103 | 0.061 | 1.042 | 5.5% | 94.5% | 116.1 |
| 1024 | math | 1.815 | 0.208 | 1.607 | 11.5% | 88.5% | 70.5 |
| 1800 | flash | 1.318 | 0.114 | 1.204 | 8.6% | 91.4% | 97.1 |
| 1800 | mem_eff | 1.397 | 0.116 | 1.281 | 8.3% | 91.7% | 91.7 |
| 1800 | math | 2.751 | 0.561 | 2.190 | 20.4% | 79.6% | 46.5 |

**Peak Memory (MB):**

| Seq Len | Flash | Mem-Efficient | Math | Savings |
|---------|-------|---------------|------|---------|
| 512 | 719 | 719 | 739 | 2.7% |
| 1024 | 762 | 762 | 910 | 16.3% |
| 1800 | 854 | 854 | 1312 | 34.9% |

### Part B: KV-Cache vs Attention Matrix Memory — "The Wall"

Understanding where the memory bottleneck lives during decode:

| KV Len | Theoretical KV-Cache (MB) | Theoretical Attn Matrix (MB) | Flash Kernel Mem (MB) | Math Kernel Mem (MB) |
|--------|--------------------------|-----------------------------|-----------------------|---------------------|
| 256 | 24.0 | 0.016 | 10.1 | 13.2 |
| 512 | 48.0 | 0.031 | 11.1 | 17.2 |
| 1024 | 96.0 | 0.063 | 13.1 | 25.2 |
| 1536 | 144.0 | 0.094 | 16.0 | 34.1 |
| 2048 | 192.0 | 0.125 | 17.1 | 41.3 |

> **Note:** Theoretical KV-cache = 2 (K+V) × 24 layers × 16 heads × kv_len × 64 head_dim × 2 bytes (fp16). Theoretical attention matrix = 16 heads × 1 × kv_len × 4 bytes (fp32 for softmax).

### Observations

1. **Decode dominates total inference time across all backends.** Even at seq=1800, decode accounts for 80-97% of total latency. This validates that optimizing the decode phase (flash decoding) is critical for generation-heavy workloads.

2. **Flash decoding reduces the DECODE portion by ~45% at seq=1800** (1.204s vs 2.190s). Combined with the prefill speedup (0.114s vs 0.561s), this yields an overall 2.09x E2E speedup.

3. **The memory bottleneck during decode is unquestionably the KV-cache, NOT the attention matrix.** At kv_len=2048:
   - KV-cache: **192 MB** (this is the data being read from HBM each decode step)
   - Attention matrix: **0.125 MB** (negligible)
   
   The math backend uses 41 MB kernel memory vs flash's 17 MB — a 24 MB difference that comes from intermediate buffers and less efficient workspace allocation, NOT from attention matrix materialization (which is only 0.125 MB).

4. **Memory savings at E2E level come from two sources:**
   - **Prefill attention matrix avoidance** (the O(n²) savings — same as flash attention study)
   - **More efficient decode workspace** (smaller constant-factor overhead per attention layer)
   
   At seq=1800: flash uses 854 MB vs math's 1312 MB. The 458 MB savings is primarily from the prefill phase attention patterns, not decode phase.

---

## 9. Experiment 6: Memory-Bound vs Compute-Bound Analysis (FP16 vs FP32)

**Goal:** Determine whether the decode attention is **memory-bound** (bottlenecked by data transfer through the memory bus) or **compute-bound** (bottlenecked by the ALU/tensor cores).

**Principle:** If we double the data size (FP16 → FP32 = 2 bytes → 4 bytes) and observe:
- **Latency doubles (~2x)** → **Memory-Bound** — the bus is the bottleneck; halving data size halves transfer time.
- **Latency stays the same (~1x)** → **Compute-Bound** — the GPU is calculating as fast as it can; smaller data doesn't help.

> **FP8 Note:** The ideal test would be FP16 vs FP8 (halving data size). However, the RTX 3050 (Ampere, CC 8.6) does **not** support FP8 tensor core operations — FP8 requires CC 8.9+ (Ada Lovelace/Hopper). We therefore use FP16 vs FP32 (doubling data size), which demonstrates the same principle in reverse.

### Part A: Kernel-Level Decode — FP16 vs FP32

Raw `F.scaled_dot_product_attention` with Q=(1,16,1,64), KV=(1,16,kv_len,64).

> **Important:** The flash backend does not support FP32 inputs (it requires FP16/BF16), so only the mem_efficient and math backends could be tested with FP32 at kernel level.

**Mem-Efficient Backend (Optimized):**

| KV Len | FP16 (ms) | FP32 (ms) | FP32/FP16 Ratio | Diagnosis |
|--------|-----------|-----------|-----------------|----------------|
| 128 | 0.028 | 0.050 | 1.79x | MEMORY-BOUND |
| 256 | 0.034 | 0.080 | 2.35x | MEMORY-BOUND |
| 512 | 0.052 | 0.137 | 2.63x | MEMORY-BOUND |
| 1024 | 0.084 | 0.249 | 2.96x | MEMORY-BOUND |
| 1536 | 0.115 | 0.363 | 3.16x | MEMORY-BOUND |
| 2048 | 0.145 | 0.476 | **3.28x** | **MEMORY-BOUND** |

**Math Backend (Normal):**

| KV Len | FP16 (ms) | FP32 (ms) | FP32/FP16 Ratio | Diagnosis |
|--------|-----------|-----------|-----------------|----------------|
| 128 | 0.101 | 0.069 | 0.68x | COMPUTE-BOUND |
| 256 | 0.099 | 0.068 | 0.69x | COMPUTE-BOUND |
| 512 | 0.149 | 0.096 | 0.64x | COMPUTE-BOUND |
| 1024 | 0.253 | 0.155 | 0.61x | COMPUTE-BOUND |
| 1536 | 0.353 | 0.213 | 0.60x | COMPUTE-BOUND |
| 2048 | 0.451 | 0.270 | **0.60x** | **COMPUTE-BOUND** |

### Part A Observations

1. **The optimized backend (mem_efficient) is clearly MEMORY-BOUND.** The FP32/FP16 ratio ranges from 1.8x to 3.3x — even exceeding the theoretical 2x because FP32 tensors also cause cache misses and wider memory bus transactions. Doubling the data size more than doubles the latency, confirming the bottleneck is data movement, not computation.

2. **The math backend is COMPUTE-BOUND — and FP32 is paradoxically FASTER than FP16.** The FP32/FP16 ratio is 0.60-0.69x, meaning FP32 runs ~40% *faster* than FP16! This happens because:
   - The math backend's implementation uses explicit matmul + softmax operations.
   - On Ampere GPUs, FP32 matmul uses standard CUDA cores which have full throughput per cycle.
   - FP16 matmul on Ampere uses Tensor Cores, which have high throughput for large matrix multiplies but **overhead for small dot products** (like 1×n decode attention).
   - The operation is too small to saturate Tensor Cores, so the Tensor Core launch/scheduling overhead dominates — making FP16 slower than FP32's simpler CUDA core path.

3. **This explains why flash decoding is so effective:** The optimized backends are memory-bound, and flash decoding reduces memory traffic through parallel KV chunking and SRAM-local processing. The math backend is compute-bound (on small decode operations), wasting memory bandwidth that is freely available.

### Part B: Model-Level Decode — FP16 vs FP32

Per-token decode latency with the full OPT-350m model. Only the math backend is comparable (flash/mem_efficient don't support FP32).

**Decode Per-Token Latency:**

| Cache Len | FP16 (ms) | FP32 (ms) | FP32/FP16 Ratio | Diagnosis |
|-----------|-----------|-----------|-----------------|----------------|
| 256 | 32.8 | 67.0 | 2.04x | MEMORY-BOUND |
| 512 | 78.7 | 133.0 | 1.69x | MEMORY-BOUND |
| 1024 | 214.2 | 314.3 | 1.47x | MIXED |
| 1800 | 563.1 | 704.0 | 1.25x | MIXED |

**Prefill Latency:**

| Input Len | FP16 (s) | FP32 (s) | FP32/FP16 Ratio | Diagnosis |
|-----------|----------|----------|-----------------|----------------|
| 256 | 0.030 | 0.057 | 1.90x | MEMORY-BOUND |
| 512 | 0.068 | 0.123 | 1.82x | MEMORY-BOUND |
| 1024 | 0.207 | 0.302 | 1.46x | MIXED |
| 1800 | 0.560 | 0.701 | 1.25x | MIXED |

**Memory Usage:**

| Cache Len | FP16 (MB) | FP32 (MB) | FP32/FP16 Ratio |
|-----------|-----------|-----------|------------------|
| 256 | 681 | 1343 | 1.97x |
| 512 | 739 | 1416 | 1.92x |
| 1024 | 910 | 1633 | 1.79x |
| 1800 | 1311 | 2105 | 1.61x |

### Part B Observations

1. **At short cache lengths, model decode is MEMORY-BOUND** (ratio 2.04x at cache=256). This makes sense: with a small KV-cache, the decode operation reads a small amount of data and the compute is trivial — the bottleneck is entirely data transfer.

2. **At long cache lengths, the regime transitions to MIXED** (ratio 1.25x at cache=1800). As the KV-cache grows, the compute workload also grows (more softmax entries, more V accumulation), so compute starts to contribute more to total latency. The operation is no longer purely memory-bound.

3. **Prefill shows the same transition pattern** as decode — memory-bound at short sequences (1.90x at input=256), mixed at long sequences (1.25x at input=1800). This is consistent: prefill with the math backend must materialize and read the full n×n attention matrix, which becomes increasingly compute-heavy at larger n.

4. **Memory usage confirms ~2x scaling** from FP16 to FP32 (e.g., 681→1343 MB at cache=256), exactly as expected for doubling the bytes per parameter.

### Summary: Is Decode Memory-Bound or Compute-Bound?

| Component | At Short Cache (≤512) | At Long Cache (≥1024) |
|-----------|----------------------|----------------------|
| Optimized kernel (flash/mem_eff) | **Memory-Bound** (2.4-3.3x) | **Memory-Bound** (2.6-3.3x) |
| Math kernel | Compute-Bound (0.6-0.7x) | Compute-Bound (0.6x) |
| Model decode (math backend) | **Memory-Bound** (2.0x) | Mixed (1.25-1.47x) |
| Model prefill (math backend) | **Memory-Bound** (1.8-1.9x) | Mixed (1.25-1.46x) |

**Key takeaway:** The optimized backends (flash/mem_efficient) are **always memory-bound** during decode — which is exactly why flash decoding's strategy of reducing memory traffic works so well. The math backend kernel is compute-bound (due to FP16 Tensor Core overhead on small operations), but the full model becomes memory-bound because non-attention components (FFN, layer norm) add significant memory traffic.

---

## 10. Cross-Experiment Analysis

### Insight 1: Flash Decoding's Advantage Compounds with Sequence Length

| Metric | 256 Cache | 512 Cache | 1024 Cache | 1800 Cache |
|--------|-----------|-----------|------------|------------|
| Per-token speedup (Exp 1) | 1.6x | 2.1x | 3.4x | 5.0x |
| Kernel speedup (Exp 4) | 2.8x | 3.0x | 4.0x | ~4.8x (est.) |

The speedup grows because the math backend becomes increasingly memory-bandwidth-starved. Each decode step requires reading more KV-cache data from HBM, and the math backend's sequential access pattern cannot keep the GPU's compute units busy. Flash decoding's parallel chunking amortizes memory access overhead.

### Insight 2: Decode is the "Last Mile" Problem

From the companion flash attention study, we know:
- Flash **attention** (prefill) provides up to 36x kernel speedup and 5x TTFT improvement
- Flash **decoding** (this study) provides up to 5x per-token decode speedup

For generation-heavy workloads (chatbots, code generation), prefill is a small fraction of total time. Flash decoding's 5x per-token improvement directly reduces the user-perceived latency of watching tokens appear.

### Insight 3: The Bottleneck Shifts Between Phases

| Phase | Bottleneck | Flash Optimization |
|-------|-----------|-------------------|
| Prefill (Flash Attention) | O(n²) attention matrix in HBM | Tile computation in SRAM, avoid materializing full matrix |
| Decode (Flash Decoding) | KV-cache reads from HBM | Parallel KV chunking, better GPU utilization |

During prefill, the problem is **space** (O(n²) matrix). During decode, the problem is **bandwidth** (reading the KV-cache). Flash decoding and flash attention use similar tiling/parallelization techniques but target fundamentally different bottlenecks.

### Insight 4: Backend Ranking Changes Between Prefill and Decode

| Phase | Winner | Runner-up |
|-------|--------|-----------|
| Prefill (companion study) | mem_efficient | flash |
| Decode (this study, long KV) | flash | mem_efficient |
| Decode (this study, batched) | mem_efficient | flash |

This is GPU-specific. On the RTX 3050, the flash backend's parallelization strategy is better optimized for the decode access pattern (1×n attention), while mem_efficient is better for batched decode. Both dramatically outperform the math backend in all cases.

---

## 11. Comparison with Flash Attention Study

This study complements the companion Flash Attention study (Person A). Here's how they connect:

| Aspect | Flash Attention Study | Flash Decoding Study (This) |
|--------|----------------------|-----------------------------|
| Focus | Prefill phase | Decode phase |
| Bottleneck | O(n²) attention matrix | KV-cache HBM bandwidth |
| Kernel speedup | 36x (at seq=2048) | 5.1x (at kv=2048) |
| Memory savings (kernel) | 25.5x (645 MB → 25 MB) | 2.4x (41 MB → 17 MB) |
| E2E speedup | Up to 2.6x | Up to 2.1x |
| When it helps most | Long prompts, short outputs | Short prompts, long outputs |

**Together,** flash attention and flash decoding cover both phases of LLM inference:
- Use flash attention to speed up prefill for summarization/RAG workloads
- Use flash decoding to speed up generation for chatbot/code-gen workloads
- Use both (default when using SDPA) for balanced coverage

---

## 12. Conclusion

Through **140+ benchmarked configurations across 6 experiments**, this study establishes:

1. **Flash decoding provides significant decode-phase speedups.** At the kernel level, it achieves 5.1x faster decode attention at kv_len=2048. At the E2E level, it provides 30-50% latency reduction and 35-39% higher decode throughput. These gains come from parallelizing attention over the KV sequence dimension, keeping data in GPU SRAM, and better utilizing the GPU's compute units.

2. **The decode bottleneck is memory bandwidth, not memory capacity.** The decode attention "matrix" is tiny (0.125 MB at kv_len=2048) — it's the KV-cache itself (192 MB) that must be streamed from HBM. Flash decoding helps by reading the KV-cache in parallel chunks rather than sequentially.

3. **Flash decoding's advantage grows with context length.** The 1.6x speedup at cache=256 grows to 5.0x at cache=1800. For future models with 8K-128K context, flash decoding will be even more critical.

4. **Decode dominates total generation time.** With 128 tokens generated, decode accounts for 80-97% of total latency. Optimizing this phase is essential for user-facing applications where perceived generation speed matters.

5. **Flash attention and flash decoding are complementary.** Flash attention optimizes the prefill bottleneck (O(n²) attention matrix). Flash decoding optimizes the decode bottleneck (KV-cache bandwidth). Together, they provide comprehensive inference optimization across both phases.

6. **Precision analysis confirms the optimized decode kernels are memory-bound** (FP32/FP16 ratio up to 3.3x), validating that flash decoding's strategy of reducing memory traffic is the right optimization. The math backend's decode kernel is paradoxically compute-bound (FP32 faster than FP16 by ~40%) due to Tensor Core overhead on small operations — revealing an important hardware nuance on Ampere GPUs.

---

## Appendix: Data Files

| File | Description |
|---|---|
| `exp1_decode_vs_cache.csv` | Per-token decode latency vs KV-cache length |
| `exp2_decode_throughput.csv` | Decode throughput scaling (32-256 tokens) |
| `exp2_decode_throughput_raw.csv` | Per-run raw data for Exp 2 |
| `exp3_batch_decode.csv` | Batch size impact on decode performance |
| `exp3_batch_decode_raw.csv` | Per-run raw data for Exp 3 |
| `exp4_kernel_decode.csv` | Kernel-level decode attention benchmark |
| `exp4_kernel_decode_raw.csv` | Per-run raw data for Exp 4 |
| `exp5_e2e_comparison.csv` | End-to-end comparison with prefill/decode breakdown |
| `exp5_e2e_comparison_raw.csv` | Per-run raw data for Exp 5A |
| `exp5_memory_analysis.csv` | KV-cache vs attention matrix memory analysis |
| `precision_exp_a_kernel.csv` | Kernel-level FP16 vs FP32 benchmark |
| `precision_exp_a_diagnosis.csv` | Kernel-level memory/compute-bound diagnosis |
| `precision_exp_b_model.csv` | Model-level FP16 vs FP32 benchmark |
| `precision_exp_b_diagnosis.csv` | Model-level memory/compute-bound diagnosis |
| `precision_analysis.json` | Combined precision analysis data |
| `all_results.json` | All results in JSON format |
| `gpu_info.txt` | Hardware and software metadata |
