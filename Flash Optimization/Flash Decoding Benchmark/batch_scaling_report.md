# Batch Scaling: Flash Decoding vs Normal Decode

## Experiment Setup

| Parameter | Value |
|-----------|-------|
| **GPU** | NVIDIA GeForce RTX 3050 |
| **VRAM** | 5795 MB |
| **Compute Capability** | 8.6 |
| **PyTorch** | 2.10.0+cu128 |
| **CUDA** | 12.8 |
| **Model** | Qwen/Qwen3-0.6B |
| **Precision** | fp16 |
| **KV Length (kernel)** | 1024 |
| **Prompt Length (E2E)** | 512 |
| **Max New Tokens (E2E)** | 64 |
| **Batch Sizes** | 1, 2, 4, 8, 16, 32 |
| **Kernel Benchmark Runs** | 20 |
| **E2E Benchmark Runs** | 5 |
| **Peak BW (spec)** | 192.0 GB/s |
| **Peak FP16 (spec)** | 4.4 TFLOPS |
| **Date** | 2026-03-19 22:16:25 |

## 1. Kernel-Level Arithmetic Intensity vs Batch Size

Fixed KV-cache length = 1024, single-token decode query (q_len=1)

| Batch | Mode | Total FLOPs | DRAM Bytes | **AI (FLOPs/byte)** | Latency (ms) | Achieved BW (GB/s) | BW Util (%) | Eff. TFLOPS | Compute Util (%) |
|-------|------|-------------|------------|---------------------|--------------|--------------------:|:-----------:|:-----------:|:----------------:|
| 1 | Normal | 8,470,528 | 8,658,944 | **0.9782** | 0.5406 | 16.02 | 8.34 | 0.0157 | 0.36 |
| 1 | Flash | 8,470,528 | 8,396,800 | **1.0088** | 0.1823 | 46.06 | 23.99 | 0.0465 | 1.06 |
| 2 | Normal | 16,941,056 | 17,317,888 | **0.9782** | 1.0158 | 17.05 | 8.88 | 0.0167 | 0.38 |
| 2 | Flash | 16,941,056 | 16,793,600 | **1.0088** | 0.3982 | 42.17 | 21.97 | 0.0425 | 0.97 |
| 4 | Normal | 33,882,112 | 34,635,776 | **0.9782** | 1.9814 | 17.48 | 9.10 | 0.0171 | 0.39 |
| 4 | Flash | 33,882,112 | 33,587,200 | **1.0088** | 0.7660 | 43.85 | 22.84 | 0.0442 | 1.01 |
| 8 | Normal | 67,764,224 | 69,271,552 | **0.9782** | 3.8891 | 17.81 | 9.28 | 0.0174 | 0.40 |
| 8 | Flash | 67,764,224 | 67,174,400 | **1.0088** | 1.3443 | 49.97 | 26.03 | 0.0504 | 1.15 |
| 16 | Normal | 135,528,448 | 138,543,104 | **0.9782** | 8.0876 | 17.13 | 8.92 | 0.0168 | 0.38 |
| 16 | Flash | 135,528,448 | 134,348,800 | **1.0088** | 2.5108 | 53.51 | 27.87 | 0.0540 | 1.23 |
| 32 | Normal | 271,056,896 | 277,086,208 | **0.9782** | 15.5493 | 17.82 | 9.28 | 0.0174 | 0.40 |
| 32 | Flash | 271,056,896 | 268,697,600 | **1.0088** | 5.0621 | 53.08 | 27.65 | 0.0535 | 1.22 |

## 2. Arithmetic Intensity Improvement (Flash vs Normal)

| Batch | Normal AI | Flash AI | **AI Improvement (%)** | Normal Latency (ms) | Flash Latency (ms) | **Speedup** |
|-------|----------|---------|------------------------|---------------------|--------------------|-------------|
| 1 | 0.9782 | 1.0088 | **+3.13%** | 0.5406 | 0.1823 | **2.97×** |
| 2 | 0.9782 | 1.0088 | **+3.13%** | 1.0158 | 0.3982 | **2.55×** |
| 4 | 0.9782 | 1.0088 | **+3.13%** | 1.9814 | 0.7660 | **2.59×** |
| 8 | 0.9782 | 1.0088 | **+3.13%** | 3.8891 | 1.3443 | **2.89×** |
| 16 | 0.9782 | 1.0088 | **+3.13%** | 8.0876 | 2.5108 | **3.22×** |
| 32 | 0.9782 | 1.0088 | **+3.13%** | 15.5493 | 5.0621 | **3.07×** |

## 3. Memory Bandwidth Utilization vs Batch Size

| Batch | Normal BW (GB/s) | Flash BW (GB/s) | Normal Util (%) | Flash Util (%) |
|-------|------------------|-----------------|:---------------:|:--------------:|
| 1 | 16.02 | 46.06 | 8.34 | 23.99 |
| 2 | 17.05 | 42.17 | 8.88 | 21.97 |
| 4 | 17.48 | 43.85 | 9.10 | 22.84 |
| 8 | 17.81 | 49.97 | 9.28 | 26.03 |
| 16 | 17.13 | 53.51 | 8.92 | 27.87 |
| 32 | 17.82 | 53.08 | 9.28 | 27.65 |

## 4. Effective FLOPs vs Batch Size

| Batch | Normal Eff. TFLOPS | Flash Eff. TFLOPS | Normal Comp. Util (%) | Flash Comp. Util (%) |
|-------|--------------------:|------------------:|:---------------------:|:--------------------:|
| 1 | 0.0157 | 0.0465 | 0.36 | 1.06 |
| 2 | 0.0167 | 0.0425 | 0.38 | 0.97 |
| 4 | 0.0171 | 0.0442 | 0.39 | 1.01 |
| 8 | 0.0174 | 0.0504 | 0.40 | 1.15 |
| 16 | 0.0168 | 0.0540 | 0.38 | 1.23 |
| 32 | 0.0174 | 0.0535 | 0.40 | 1.22 |

## 5. End-to-End Tokens/sec vs Batch Size

Model: `Qwen/Qwen3-0.6B`, Prompt: 512 tokens, Generate: 64 tokens

| Batch | Mode | Tokens/sec | Latency (s) | Peak Memory (MB) |
|-------|------|------------|-------------|------------------|
| 1 | Normal | 43.31 | 1.4777 | 1560.05 |
| 1 | Flash | 61.58 | 1.0392 | 1527.78 |
| 2 | Normal | 59.62 | 2.1468 | 1677.33 |
| 2 | Flash | 96.36 | 1.3283 | 1580.31 |
| 4 | Normal | 74.53 | 3.4350 | 1911.88 |
| 4 | Flash | 134.21 | 1.9075 | 1718.85 |
| 8 | Normal | 84.83 | 6.0359 | 2380.99 |
| 8 | Flash | 169.14 | 3.0271 | 1995.93 |
| 16 | Normal | 93.84 | 10.9117 | 3319.22 |
| 16 | Flash | 191.80 | 5.3388 | 2550.09 |
| 32 | Flash | 208.03 | 9.8449 | 3658.41 |

### Tokens/sec Improvement

| Batch | Normal tok/s | Flash tok/s | **Improvement (%)** |
|-------|------------|-----------|---------------------|
| 1 | 43.31 | 61.58 | **+42.18%** |
| 2 | 59.62 | 96.36 | **+61.62%** |
| 4 | 74.53 | 134.21 | **+80.08%** |
| 8 | 84.83 | 169.14 | **+99.39%** |
| 16 | 93.84 | 191.80 | **+104.39%** |

## Analysis

### Batch Size Impact on Arithmetic Intensity

Arithmetic intensity (FLOPs/byte) for decode attention is **nearly constant**
across batch sizes because both FLOPs and DRAM traffic scale linearly with batch.
The key difference remains the attention matrix materialization:

- **Normal decode**: AI ≈ 0.978 (constant) — the materialized attention matrix
  adds ~3% extra DRAM traffic regardless of batch size.
- **Flash Decoding**: AI ≈ 1.009 (constant) — avoids materialization entirely.

### Batch Size Impact on Bandwidth Utilization

Larger batches increase GPU occupancy, enabling:
- Better memory coalescing and hiding of memory latency
- Higher achieved bandwidth (closer to the 192 GB/s peak)
- Flash Decoding benefits more because the fused kernel can
  overlap compute with memory access across batch elements.

### Practical Takeaway

Flash Decoding provides a **consistent speedup** across all batch sizes,
with the latency advantage becoming more pronounced at larger batches where
the GPU can better amortize kernel launch overhead and pipeline memory accesses.

---
*Report generated: 2026-03-19 22:16:25*