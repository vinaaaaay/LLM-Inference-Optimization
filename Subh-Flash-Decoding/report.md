# Flash Decoding vs Normal Decoding: Experimental Analysis Report (Qwen3-0.6B)

**Model:** Qwen/Qwen3-0.6B (0.6B parameters)  
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

This report presents an empirical comparison of **Flash Decoding** (optimized SDPA backends: FlashAttention and memory-efficient) vs **Normal Decoding** (standard math backend) during the decode phase of LLM inference using the **Qwen3-0.6B** model. While typical attention optimization focus is on the prefill phase, this study concentrates entirely on the **decode phase** — the autoregressive token-by-token generation that dominates inference time for long-form generation.

**Key Findings:**

- **Critical Latency Speedup**: Flash decoding achieves up to **3.2x faster per-token decode latency** compared to normal decoding at long KV-cache lengths (257.3ms vs 834.6ms per token at cache_len=1800).
- **Kernel Efficiency**: At the kernel level, flash decode attention is **3.2x faster** than math at kv_len=2048 (0.314ms vs 1.009ms), with **2.5x less memory** (32 MB vs 80 MB).
- **Throughput Gains**: Flash decoding provides **~40% higher decode throughput** (65.5 tok/s vs 46.6 tok/s) in end-to-end generation for long sequences.
- **The Bottleneck**: The memory bottleneck during decode is the **KV-cache**, not the attention matrix. The attention matrix for decode (query_len=1) is tiny (~0.125 MB at kv_len=2048), while the KV-cache for Qwen3-0.6B grows to **224 MB**. Flash decoding's advantage comes from more efficient KV-cache access patterns (parallelization), not from avoiding attention matrix materialization.
- **Backend Parity**: Flash and memory-efficient backends perform nearly identically during decode on this hardware, with flash having a slight edge at the longest KV lengths.
- **Hard Hardware Diagnosis**: Precision analysis (FP16 vs FP32) confirms optimized backends are **memory-bound** (FP32/FP16 ratio up to 3.0x at kernel level) while the math backend is **compute-bound** during decode (FP32 is paradoxically faster than FP16 due to Tensor Core overhead on tiny operations).

---

## 2. What is Flash Decoding?

### Normal Decoding (Math Backend)
During autoregressive generation, each new token requires computing attention between a single query (the new token, shape $1 \times d$) and the entire KV-cache (all previous tokens, shape $n \times d$). The standard implementation:

1.  Computes **score** = $q \times K^T \rightarrow$ shape $1 \times n$ (one value per cached token)
2.  Applies **softmax(score / sqrt(d))** $\rightarrow$ shape $1 \times n$
3.  Computes **output** = $attn\_weights \times V \rightarrow$ shape $1 \times d$

This is **memory-bandwidth-bound**: the GPU must read the entire KV-cache from HBM for each decode step. With a small $1 \times n$ workload, the GPU's compute units are severely under-utilized — most time is spent waiting for memory transfers of the KV-cache.

### Flash Decoding (Optimized Backends)
Flash Decoding introduces **parallelism over the KV sequence length**:

1.  **Split** K and V into chunks along the sequence dimension.
2.  **Compute partial attention** for each chunk in parallel across GPU streaming multiprocessors (SMs).
3.  **Rescale and combine** using online softmax to produce the correct final result.

This keeps data in GPU SRAM (shared memory) as much as possible, reducing HBM round-trips. The GPU's compute units stay busy by processing multiple KV chunks simultaneously, dramatically improving utilization even at $batch\_size=1$.

**Key difference from Flash Attention (prefill):** Flash Attention optimizes the $n \times n$ attention matrix during prefill (where both $Q$ and $K$ have length $n$). Flash Decoding optimizes the $1 \times n$ attention during decode (where $Q$ has length 1). The bottleneck is different — prefill is about avoiding $O(n^2)$ memory, decode is about maximizing GPU utilization for a memory-bound operation.

---

## 3. Experimental Setup

### Hardware
- **GPU:** NVIDIA GeForce RTX 3050 (Ampere)
- **VRAM:** 5795 MB
- **Compute Capability:** 8.6

### Model
- **Model:** Qwen/Qwen3-0.6B
- **Dtype:** float16
- **Parameters:** ~0.6B
- **Layers:** 28
- **Heads:** 16 (Query), 8 (KV - Grouped Query Attention)
- **Head Dim:** 128

### Methodology
- All benchmarks use **3-5 measured runs** per configuration (median reported).
- **3 warm-up iterations** precede all measurements.
- `torch.cuda.synchronize()` used before and after every measurement.
- Greedy decoding (`do_sample=False`) is used throughout.

---

## 4. Experiment 1: Decode Latency vs KV-Cache Length

**Goal:** Measure how per-token decode latency scales as the KV-cache grows. This is the core bottleneck of autoregressive generation.

### Per-Token Decode Latency (ms)

| Cache Len | Flash | Mem-Efficient | Math | Flash vs Math |
|-----------|-------|---------------|------|---------------|
| 256 | 43.9 | 43.0 | 60.3 | 1.4x |
| 512 | 76.0 | 77.8 | 134.1 | 1.8x |
| 1024 | 144.3 | 149.7 | 344.7 | 2.4x |
| 1536 | 217.3 | 228.2 | 644.0 | 3.0x |
| 1800 | 257.3 | 272.0 | 834.6 | **3.2x** |

### Total Decode Time for 16 Tokens (seconds)

| Cache Len | Flash | Mem-Efficient | Math | Flash Savings |
|-----------|-------|---------------|------|---------------|
| 512 | 1.140 | 1.167 | 2.011 | 43% |
| 1024 | 2.164 | 2.245 | 5.171 | 58% |
| 1800 | 3.860 | 4.080 | 12.519 | **69%** |

### Prefill Time (seconds)

| Cache Len | Flash | Math | Flash vs Math |
|-----------|-------|------|---------------|
| 512 | 0.071 | 0.124 | 1.8x |
| 1024 | 0.144 | 0.335 | 2.3x |
| 1800 | 0.257 | 0.829 | **3.2x** |

#### Observations
- The **Flash Decoding advantage grows dramatically** with KV-cache length. At $cache\_len=256$, flash is 1.4x faster. At $cache\_len=1800$, it is **3.2x faster**. This compounding effect occurs because the math backend's decode cost scales poorly — it must read the entire KV-cache from HBM for each token.
- **Math backend decode latency grows super-linearly**. From 256 to 1800 (7x more cache), math decode latency goes from 60.3ms to 834.6ms (13.8x increase). Flash goes from 43.9ms to 257.3ms (5.8x increase).
- Decode time dominates total generation time for the user. At $cache\_len=1800$, 16 tokens take **12.5s with math vs 3.8s with flash**.

---

## 5. Experiment 2: Decode Throughput Scaling

**Goal:** Measure how decode throughput (tokens/sec) changes with generation length.

### Decode-Only Throughput (tokens/sec)

| Gen Tokens | Flash | Mem-Efficient | Math | Improvement |
|------------|-------|---------------|------|-------------|
| 32 | 69.73 | 69.93 | 51.78 | +34.7% |
| 128 | 67.13 | 67.67 | 48.86 | +37.4% |
| 256 | 65.53 | 65.92 | 46.62 | **+40.6%** |

#### Observations
- Flash decoding consistently provides **~35-40% higher decode throughput**.
- Throughput decreases slightly as generation length increases because the KV-cache grows token-by-token, increasing the data requirement for each step.

---

## 6. Experiment 3: Batch Size Impact on Decode

**Goal:** Determine whether flash decoding's advantage holds across batch sizes.

### Decode-Only Throughput (tokens/sec per batch) - Seq Len = 1024

| Batch | Flash | Mem-Efficient | Math | Flash vs Math |
|-------|-------|---------------|------|---------------|
| 1 | 56.7 | 56.8 | 35.9 | 1.58x |
| 2 | 80.2 | 82.8 | 45.6 | 1.76x |
| 4 | 104.7 | 110.1 | 53.0 | **1.98x** |

### Peak Memory (MB) - Seq Len = 1024

| Batch | Flash | Math | Savings |
|-------|-------|------|---------|
| 1 | 1580 | 1752 | 9.8% |
| 4 | 1996 | 2672 | **25.3%** |

#### Observations
- **Advantages increase with batch size**. Flash decoding goes from 1.58x speedup (batch 1) to **1.98x speedup (batch 4)**. Larger batches amplify the memory bandwidth bottleneck of the math backend.
- Memory savings are significant at scale, reaching **25% VRAM reduction** with batch size 4.

---

## 7. Experiment 4: Kernel-Level Decode Attention

**Goal:** Isolate the raw decode attention kernel by benchmarking synthetic single-query attention.

### Kernel Latency (ms) (Q_len=1)

| KV Len | Flash | Mem-Efficient | Math | Flash vs Math |
|--------|-------|---------------|------|---------------|
| 512 | 0.107 | 0.103 | 0.289 | 2.7x |
| 1024 | 0.177 | 0.175 | 0.529 | 3.0x |
| 2048 | 0.314 | 0.313 | 1.009 | **3.2x** |

### Kernel Memory (MB)

| KV Len | Flash | Math | Math/Flash Ratio |
|--------|-------|------|-----------------|
| 512 | 14.1 | 26.2 | 1.9x |
| 2048 | 32.2 | 80.3 | **2.5x** |

#### Observations
- Flash is **3.2x faster** at $kv\_len=2048$ at the pure kernel level.
- The decode speedup is **not from avoiding a large matrix** (the materialization is only ~0.125 MB); it's from better HBM bandwidth utilization through parallelized chunking.

---

## 8. Experiment 5: End-to-End + Memory Analysis

### Part A: End-to-End Breakdown (batch=1)

| Seq Len | Backend | Total Lat (s) | TTFT (s) | Decode (s) | Tok/s |
|---------|---------|--------------|----------|------------|-------|
| 1800 | flash | 3.040 | 0.258 | 2.782 | 42.1 |
| 1800 | math | 5.880 | 0.833 | 5.047 | 21.8 |

### Part B: The "Memory Wall" - KV-Cache vs Attention Matrix

| KV Len | Theoretical KV-Cache (MB) | Theoretical Attn Matrix (MB) | Flash Kernel Mem (MB) | Math Kernel Mem (MB) |
|--------|--------------------------|-----------------------------|-----------------------|----------------------|
| 512 | 56.0 | 0.031 | 14.1 | 26.2 |
| 1024 | 112.0 | 0.063 | 20.2 | 44.2 |
| 2048 | 224.0 | 0.125 | 32.2 | 80.3 |

**Note:** Theoretical KV-cache for Qwen3-0.6B = 2 (K+V) x 28 layers x 8 KV heads x context x 128 dim x 2 bytes.

---

## 9. Experiment 6: Memory-Bound vs Compute-Bound Analysis (FP16 vs FP32)

**Goal:** Determine if decode is bottlenecked by the memory bus (data transfer) or the arithmetic units.

### Part A: Kernel-Level — FP16 vs FP32 (Mem-Efficient Backend)

| KV Len | FP16 (ms) | FP32 (ms) | FP32/FP16 Ratio | Diagnosis |
|--------|-----------|-----------|-----------------|-----------|
| 128 | 0.051 | 0.091 | 1.78x | **MEMORY-BOUND** |
| 2048 | 0.361 | 1.098 | **3.04x** | **MEMORY-BOUND** |

### Part B: Math Kernel Performance Paradox

| KV Len | FP16 (ms) | FP32 (ms) | Ratio | Diagnosis |
|--------|-----------|-----------|-------|-----------|
| 2048 | 1.224 | 0.941 | **0.77x** | **COMPUTE-BOUND** |

#### Observations
- **Optimized Kernels are Memory-Bound**: The ratio of ~3.0x confirms that doubling the data size nearly triples the latency, proving the bottleneck is memory bandwidth.
- **Math Kernel is Compute-Bound**: Paradoxically, FP32 is faster than FP16 for the math kernel. This is because FP16 Tensor Cores have high launch overhead on tiny dot products, making the simpler CUDA core path of FP32 faster for small decodes.

---

## 10. Cross-Experiment Analysis

1.  **Advantage Compounds**: The per-token speedup grows from 1.4x (256 cache) to 3.2x (1800 cache) as the memory bandwidth bottleneck of the math backend worsens.
2.  **Decode is the "Last Mile"**: While Flash Attention speeds up prefill by 36x, Flash Decoding's 3.2x speedup directly reduces the user-perceived "token streaming" wait time.
3.  **Bottleneck Shift**: In prefill, we optimize to avoid $O(n^2)$ space. In decode, we optimize to maximize HBM bandwidth utilization through parallelism.

---

## 11. Comparison with Flash Attention Study

| Aspect | Flash Attention (Prefill) | Flash Decoding (Decode) |
|--------|--------------------------|-------------------------|
| Bottleneck | $O(n^2)$ Matrix Space | HBM Bandwidth |
| Kernel Speedup | ~36x (at 2048) | **3.2x (at 2048)** |
| Memory Savings | 25x | **2.5x** |
| When it helps | Long prompts / Summarization | Long generations / Chatbots |

---

## 12. Conclusion

This study establishes that **Flash Decoding is essential for the Qwen3-0.6B model**. It provides up to **3.2x per-token acceleration** and reduces end-to-end memory usage by **25%** at scale. By parallelizing the KV-cache access, it overcomes the hardware "Memory Wall," transforming a memory-starved process into a highly efficient generation engine.

**Appendix: Data Files**
- `exp1_decode_vs_cache.csv` 
- `exp2_decode_throughput.csv`
- `exp3_batch_decode.csv`
- `exp4_kernel_decode.csv`
- `exp5_e2e_comparison.csv`
- `precision_analysis.json`
- `all_results.json`
