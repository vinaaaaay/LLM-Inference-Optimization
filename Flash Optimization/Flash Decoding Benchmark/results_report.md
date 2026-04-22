# Flash Decoding Arithmetic Intensity Benchmark

## Experiment Summary

| Parameter | Value |
|-----------|-------|
| **GPU** | NVIDIA GeForce RTX 3050 |
| **VRAM** | 5795 MB |
| **Compute Capability** | 8.6 |
| **PyTorch** | 2.10.0+cu128 |
| **CUDA** | 12.8 |
| **Model** | Qwen/Qwen3-0.6B |
| **Precision** | fp16 |
| **Batch Size** | 1 |
| **Layers** | 28 |
| **Q Heads / KV Heads** | 16 / 8 |
| **Head Dim** | 128 |
| **Peak BW (spec)** | 192.0 GB/s |
| **Peak FP16 (spec)** | 4.4 TFLOPS |
| **Benchmark Runs** | 20 |
| **Date** | 2026-03-19 21:49:21 |

## Kernel-Level Decode Attention Results

### Arithmetic Intensity Comparison

| KV Length | Mode | Total FLOPs | DRAM Bytes | Arith. Intensity (FLOPs/byte) | Latency (ms) | Achieved BW (GB/s) | BW Util. (%) | Achieved TFLOPS | Compute Util. (%) |
|-----------|------|-------------|------------|-------------------------------|--------------|--------------------|--------------|-----------------|--------------------|
| 256 | Normal | 2,117,632 | 2,170,880 | 0.9755 | 0.1758 | 12.35 | 6.43 | 0.0120 | 0.27 |
| 256 | Flash | 2,117,632 | 2,105,344 | 1.0058 | 0.0717 | 29.36 | 15.29 | 0.0295 | 0.67 |
| 512 | Normal | 4,235,264 | 4,333,568 | 0.9773 | 0.2994 | 14.47 | 7.54 | 0.0141 | 0.32 |
| 512 | Flash | 4,235,264 | 4,202,496 | 1.0078 | 0.1116 | 37.66 | 19.61 | 0.0380 | 0.86 |
| 1024 | Normal | 8,470,528 | 8,658,944 | 0.9782 | 0.5386 | 16.08 | 8.37 | 0.0157 | 0.36 |
| 1024 | Flash | 8,470,528 | 8,396,800 | 1.0088 | 0.1823 | 46.06 | 23.99 | 0.0465 | 1.06 |
| 2048 | Normal | 16,941,056 | 17,309,696 | 0.9787 | 1.0312 | 16.79 | 8.74 | 0.0164 | 0.37 |
| 2048 | Flash | 16,941,056 | 16,785,408 | 1.0093 | 0.3270 | 51.33 | 26.74 | 0.0518 | 1.18 |

### Arithmetic Intensity Improvement (Flash Decoding vs Normal)

| KV Length | Normal AI (FLOPs/byte) | Flash AI (FLOPs/byte) | % Improvement | DRAM Saved (bytes) | DRAM Saved (%) |
|-----------|------------------------|-----------------------|---------------|--------------------|----------------|
| 256 | 0.9755 | 1.0058 | 3.11% | 65,536 | 3.02% |
| 512 | 0.9773 | 1.0078 | 3.12% | 131,072 | 3.02% |
| 1024 | 0.9782 | 1.0088 | 3.13% | 262,144 | 3.03% |
| 2048 | 0.9787 | 1.0093 | 3.13% | 524,288 | 3.03% |

### Memory Bandwidth Utilization

| KV Length | Mode | Achieved BW (GB/s) | Peak BW (GB/s) | Utilization (%) |
|-----------|------|--------------------:|:--------------:|:---------------:|
| 256 | Normal | 12.35 | 192.0 | 6.43 |
| 256 | Flash | 29.36 | 192.0 | 15.29 |
| 512 | Normal | 14.47 | 192.0 | 7.54 |
| 512 | Flash | 37.66 | 192.0 | 19.61 |
| 1024 | Normal | 16.08 | 192.0 | 8.37 |
| 1024 | Flash | 46.06 | 192.0 | 23.99 |
| 2048 | Normal | 16.79 | 192.0 | 8.74 |
| 2048 | Flash | 51.33 | 192.0 | 26.74 |

### DRAM Traffic Breakdown

| KV Length | Mode | Q (bytes) | K (bytes) | V (bytes) | Attn Matrix (bytes) | Output (bytes) | Total (bytes) |
|-----------|------|-----------|-----------|-----------|---------------------|----------------|---------------|
| 256 | Normal | 4,096 | 1,048,576 | 1,048,576 | 65,536 | 4,096 | 2,170,880 |
| 256 | Flash | 4,096 | 1,048,576 | 1,048,576 | 0 | 4,096 | 2,105,344 |
| 512 | Normal | 4,096 | 2,097,152 | 2,097,152 | 131,072 | 4,096 | 4,333,568 |
| 512 | Flash | 4,096 | 2,097,152 | 2,097,152 | 0 | 4,096 | 4,202,496 |
| 1024 | Normal | 4,096 | 4,194,304 | 4,194,304 | 262,144 | 4,096 | 8,658,944 |
| 1024 | Flash | 4,096 | 4,194,304 | 4,194,304 | 0 | 4,096 | 8,396,800 |
| 2048 | Normal | 4,096 | 8,388,608 | 8,388,608 | 524,288 | 4,096 | 17,309,696 |
| 2048 | Flash | 4,096 | 8,388,608 | 8,388,608 | 0 | 4,096 | 16,785,408 |

### FLOPs Breakdown

| KV Length | QK^T FLOPs | Softmax FLOPs | Attn×V FLOPs | Total FLOPs |
|-----------|------------|---------------|--------------|-------------|
| 256 | 1,048,576 | 20,480 | 1,048,576 | 2,117,632 |
| 512 | 2,097,152 | 40,960 | 2,097,152 | 4,235,264 |
| 1024 | 4,194,304 | 81,920 | 4,194,304 | 8,470,528 |
| 2048 | 8,388,608 | 163,840 | 8,388,608 | 16,941,056 |

## End-to-End Model Decode Results

Model: `Qwen/Qwen3-0.6B`, Prompt length: 512, Max new tokens: 64

| Mode | Latency (s) | Tokens/sec | Tokens Generated | Peak Memory (MB) |
|------|-------------|------------|------------------|------------------|
| Normal | 1.3917 | 45.99 | 64 | 1560.05 |
| Flash | 1.0039 | 63.75 | 64 | 1527.78 |

**Tokens/sec improvement: +38.62%** (Flash Decoding vs Normal)

## Analysis

### Why Flash Decoding Improves Arithmetic Intensity

During the **decode phase** of LLM inference (autoregressive token generation),
the query length is always 1 while the KV-cache grows with each generated token.
This creates an extremely **memory-bound** operation:

- **Normal decode (MATH backend)**: Materializes the attention score matrix
  `(B, H, 1, S)` to global DRAM memory. Even though this matrix is small,
  the write-then-read pattern adds significant memory traffic, especially
  since softmax requires fp32 precision for numerical stability.

- **Flash Decoding (FLASH_ATTENTION backend)**: Uses a fused kernel that
  keeps attention scores in SRAM (shared memory/registers). The attention
  matrix is **never written to DRAM**, eliminating the materialization overhead.
  The kernel splits the KV-cache across thread blocks, each computing a
  partial softmax, then reduces the results — all in on-chip memory.

### Roofline Model Perspective

The RTX 3050 has a ridge point at:
  - Ridge Point = Peak FLOPS / Peak BW = 4400000000000 / 192000000000 = **22.92 FLOPs/byte**

Decode attention is well below this ridge point for both modes, confirming
it is **memory-bound**. Flash Decoding improves arithmetic intensity by
reducing DRAM traffic, moving the operation closer to (but still below)
the ridge point.

---
*Report generated: 2026-03-19 21:49:21*