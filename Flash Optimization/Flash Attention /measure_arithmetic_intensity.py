#!/usr/bin/env python3
"""
Experimentally measure arithmetic intensity of Normal vs Flash Attention Prefill
Arithmetic Intensity = FLOPs / Memory Bytes Transferred
"""

import torch
import torch.nn.functional as F
from torch.profiler import profile, ProfilerActivity
import math
import time

def normal_attention(Q, K, V, scale):
    """Standard attention - materializes n×n attention matrix"""
    # S = Q @ K^T
    S = torch.matmul(Q, K.transpose(-2, -1)) * scale
    # P = softmax(S)
    P = F.softmax(S, dim=-1)
    # O = P @ V
    O = torch.matmul(P, V)
    return O

def flash_attention(Q, K, V, scale):
    """Flash Attention using PyTorch SDPA with flash backend"""
    with torch.backends.cuda.sdp_kernel(
        enable_flash=True,
        enable_math=False,
        enable_mem_efficient=False
    ):
        O = F.scaled_dot_product_attention(Q, K, V, scale=scale)
    return O

def measure_with_profiler(func, Q, K, V, scale, name, warmup=5, iterations=20):
    """Measure using PyTorch profiler"""
    # Warmup
    for _ in range(warmup):
        _ = func(Q, K, V, scale)
    torch.cuda.synchronize()

    # Profile
    with profile(
        activities=[ProfilerActivity.CUDA],
        record_shapes=True,
        with_flops=True,
    ) as prof:
        for _ in range(iterations):
            _ = func(Q, K, V, scale)
        torch.cuda.synchronize()

    # Extract metrics
    total_flops = 0
    total_cuda_time = 0

    for event in prof.key_averages():
        if event.device_type == torch.autograd.DeviceType.CUDA:
            total_cuda_time += event.cuda_time_total
            if event.flops:
                total_flops += event.flops

    return {
        'total_flops': total_flops,
        'cuda_time_us': total_cuda_time,
        'iterations': iterations
    }

def measure_bandwidth(func, Q, K, V, scale, warmup=10, iterations=50):
    """Measure achieved memory bandwidth and compute throughput"""
    # Warmup
    for _ in range(warmup):
        _ = func(Q, K, V, scale)
    torch.cuda.synchronize()

    # Measure time
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    for _ in range(iterations):
        _ = func(Q, K, V, scale)
    end.record()
    torch.cuda.synchronize()

    elapsed_ms = start.elapsed_time(end) / iterations
    return elapsed_ms

def calculate_theoretical_metrics(batch, heads, seq_len, head_dim, dtype_bytes=2):
    """Calculate theoretical FLOPs and memory for comparison"""
    n, d = seq_len, head_dim

    # FLOPs per head: Q@K^T (2n²d) + softmax (~5n²) + P@V (2n²d) ≈ 4n²d
    flops_per_head = 4 * n * n * d
    total_flops = batch * heads * flops_per_head

    # Normal attention memory (per head):
    # Read Q,K,V: 3*n*d, Write S: n², Read S: n², Write P: n², Read P,V: n²+n*d, Write O: n*d
    # Simplified: 4*n*d + 4*n² elements
    normal_mem_elements = 4 * n * d + 4 * n * n
    normal_mem_bytes = batch * heads * normal_mem_elements * dtype_bytes

    # Flash attention memory (per head):
    # Read Q,K,V: 3*n*d, Write O: n*d = 4*n*d elements
    flash_mem_elements = 4 * n * d
    flash_mem_bytes = batch * heads * flash_mem_elements * dtype_bytes

    return {
        'total_flops': total_flops,
        'normal_mem_bytes': normal_mem_bytes,
        'flash_mem_bytes': flash_mem_bytes,
        'normal_ai_theoretical': total_flops / normal_mem_bytes,
        'flash_ai_theoretical': total_flops / flash_mem_bytes
    }

def main():
    print("=" * 70)
    print("ARITHMETIC INTENSITY MEASUREMENT: Normal vs Flash Attention Prefill")
    print("=" * 70)

    device = torch.device("cuda")
    dtype = torch.float16

    # Get GPU info
    gpu_name = torch.cuda.get_device_name(0)
    gpu_props = torch.cuda.get_device_properties(0)

    # RTX 3050 specs (adjust if different GPU)
    peak_tflops = 6.7  # TF16 TFLOPS
    peak_bandwidth = 168  # GB/s
    ridge_point = (peak_tflops * 1e12) / (peak_bandwidth * 1e9)

    print(f"\nGPU: {gpu_name}")
    print(f"Peak Compute: {peak_tflops} TFLOPS (FP16)")
    print(f"Peak Bandwidth: {peak_bandwidth} GB/s")
    print(f"Ridge Point: {ridge_point:.1f} FLOP/byte")

    # Test configurations
    batch = 1
    heads = 16
    head_dim = 128
    seq_lengths = [256, 512, 1024, 1800]

    print(f"\nConfiguration: batch={batch}, heads={heads}, head_dim={head_dim}")
    print("-" * 70)

    results = []

    for seq_len in seq_lengths:
        print(f"\n>>> Sequence Length: {seq_len}")

        # Create tensors
        Q = torch.randn(batch, heads, seq_len, head_dim, device=device, dtype=dtype)
        K = torch.randn(batch, heads, seq_len, head_dim, device=device, dtype=dtype)
        V = torch.randn(batch, heads, seq_len, head_dim, device=device, dtype=dtype)
        scale = 1.0 / math.sqrt(head_dim)

        # Theoretical values
        theory = calculate_theoretical_metrics(batch, heads, seq_len, head_dim)

        print(f"\n  Theoretical FLOPs: {theory['total_flops']/1e9:.2f} GFLOPs")
        print(f"  Normal Memory (theoretical): {theory['normal_mem_bytes']/1e6:.2f} MB")
        print(f"  Flash Memory (theoretical): {theory['flash_mem_bytes']/1e6:.2f} MB")

        # Measure execution time
        normal_time_ms = measure_bandwidth(normal_attention, Q, K, V, scale)

        try:
            flash_time_ms = measure_bandwidth(flash_attention, Q, K, V, scale)
            flash_available = True
        except Exception as e:
            print(f"  Flash attention unavailable: {e}")
            flash_available = False
            flash_time_ms = float('inf')

        # Calculate experimental metrics
        flops = theory['total_flops']

        # Normal attention
        normal_tflops = (flops / 1e12) / (normal_time_ms / 1000)
        # Estimate memory from time assuming we're memory-bound for normal at long seqs
        normal_mem_estimated = theory['normal_mem_bytes']
        normal_ai_experimental = flops / normal_mem_estimated

        # Also calculate AI from achieved throughput
        # If compute-bound: achieved_tflops close to peak
        # If memory-bound: achieved_bandwidth close to peak
        normal_compute_util = (normal_tflops / peak_tflops) * 100

        print(f"\n  --- Normal Attention ---")
        print(f"  Execution time: {normal_time_ms:.3f} ms")
        print(f"  Achieved: {normal_tflops:.2f} TFLOPS ({normal_compute_util:.1f}% of peak)")
        print(f"  Theoretical AI: {theory['normal_ai_theoretical']:.1f} FLOP/byte")

        # For experimental AI, use: AI = achieved_TFLOPS / achieved_bandwidth
        # Since achieved_BW = mem_bytes / time, and achieved_TFLOPS = flops / time
        # AI_exp = flops / mem_bytes (same as theoretical if we use theoretical mem)

        # Better approach: measure what limits us
        # achieved_compute = flops / time
        # If achieved_compute << peak, we're memory bound
        # estimated_bandwidth = achieved_compute / ridge_point (if memory bound)

        normal_estimated_bw = (normal_tflops * 1e12) / (ridge_point * 1e9)  # GB/s if at ridge
        print(f"  Estimated bandwidth usage: {normal_estimated_bw:.1f} GB/s")

        if flash_available:
            flash_tflops = (flops / 1e12) / (flash_time_ms / 1000)
            flash_compute_util = (flash_tflops / peak_tflops) * 100
            flash_estimated_bw = (flash_tflops * 1e12) / (ridge_point * 1e9)

            print(f"\n  --- Flash Attention ---")
            print(f"  Execution time: {flash_time_ms:.3f} ms")
            print(f"  Achieved: {flash_tflops:.2f} TFLOPS ({flash_compute_util:.1f}% of peak)")
            print(f"  Theoretical AI: {theory['flash_ai_theoretical']:.1f} FLOP/byte")
            print(f"  Estimated bandwidth usage: {flash_estimated_bw:.1f} GB/s")

            speedup = normal_time_ms / flash_time_ms
            print(f"\n  Speedup (Flash vs Normal): {speedup:.2f}x")

        # Store results
        results.append({
            'seq_len': seq_len,
            'normal_time_ms': normal_time_ms,
            'flash_time_ms': flash_time_ms if flash_available else None,
            'normal_tflops': normal_tflops,
            'flash_tflops': flash_tflops if flash_available else None,
            'normal_ai_theory': theory['normal_ai_theoretical'],
            'flash_ai_theory': theory['flash_ai_theoretical'],
            'speedup': speedup if flash_available else None
        })

        # Clean up
        del Q, K, V
        torch.cuda.empty_cache()

    # Summary table
    print("\n" + "=" * 70)
    print("SUMMARY: Arithmetic Intensity Comparison")
    print("=" * 70)
    print(f"{'Seq Len':<10} {'Normal AI':<12} {'Flash AI':<12} {'Speedup':<10} {'Normal TFLOPS':<14} {'Flash TFLOPS':<14}")
    print(f"{'':10} {'(FLOP/byte)':<12} {'(FLOP/byte)':<12} {'':10} {'(achieved)':<14} {'(achieved)':<14}")
    print("-" * 70)

    for r in results:
        flash_ai = f"{r['flash_ai_theory']:.1f}" if r['flash_time_ms'] else "N/A"
        flash_tflops = f"{r['flash_tflops']:.2f}" if r['flash_tflops'] else "N/A"
        speedup = f"{r['speedup']:.2f}x" if r['speedup'] else "N/A"
        print(f"{r['seq_len']:<10} {r['normal_ai_theory']:<12.1f} {flash_ai:<12} {speedup:<10} {r['normal_tflops']:<14.2f} {flash_tflops:<14}")

    print("-" * 70)
    print(f"Ridge Point: {ridge_point:.1f} FLOP/byte")
    print("Above ridge point = compute-bound, Below = memory-bound")

    # Memory analysis
    print("\n" + "=" * 70)
    print("MEMORY TRAFFIC ANALYSIS")
    print("=" * 70)
    print(f"{'Seq Len':<10} {'Normal Mem':<15} {'Flash Mem':<15} {'Memory Saved':<15}")
    print("-" * 70)
    for seq_len in seq_lengths:
        theory = calculate_theoretical_metrics(batch, heads, seq_len, head_dim)
        saved = theory['normal_mem_bytes'] - theory['flash_mem_bytes']
        saved_pct = (saved / theory['normal_mem_bytes']) * 100
        print(f"{seq_len:<10} {theory['normal_mem_bytes']/1e6:<15.2f} {theory['flash_mem_bytes']/1e6:<15.2f} {saved_pct:<15.1f}%")

if __name__ == "__main__":
    main()
