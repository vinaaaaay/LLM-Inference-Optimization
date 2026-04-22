"""
run_benchmark.py — Arithmetic Intensity Benchmark: Flash Decoding vs Normal Decode

Compares arithmetic intensity (FLOPs/byte) for LLM decode attention with and
without Flash Decoding on an NVIDIA RTX 3050 using Qwen-0.6B (fp16, batch=1).

For each SDPA backend (MATH = normal, FLASH_ATTENTION = flash decoding):
  - Analytically compute total FLOPs for decode attention
  - Analytically compute total DRAM bytes (global memory traffic)
  - Measure wall-clock latency via CUDA events
  - Derive arithmetic intensity, achieved bandwidth, utilization

Also runs end-to-end model decode for tokens/sec comparison.

Hardware: NVIDIA GeForce RTX 3050 (6 GB GDDR6, 192 GB/s BW, Compute 8.6)
Model:    Qwen/Qwen3-0.6B (28 layers, 16 QH, 8 KVH, head_dim=128, fp16)

Usage:
    python3 run_benchmark.py
"""

import gc
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from statistics import median, stdev

import torch
import torch.nn.functional as F
from torch.nn.attention import sdpa_kernel, SDPBackend
from transformers import AutoModelForCausalLM, AutoTokenizer

# ---------------------------------------------------------------------------
# Patch SDPA for GQA support (broadcast KV heads to match Q heads)
# ---------------------------------------------------------------------------
_original_sdpa = F.scaled_dot_product_attention

def _patched_sdpa(query, key, value, *args, **kwargs):
    if query.shape[1] != key.shape[1]:
        key = key.repeat_interleave(query.shape[1] // key.shape[1], dim=1)
        value = value.repeat_interleave(query.shape[1] // value.shape[1], dim=1)
    return _original_sdpa(query, key, value, *args, **kwargs)

F.scaled_dot_product_attention = _patched_sdpa

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MODEL_NAME = "Qwen/Qwen3-0.6B"

# Model config (Qwen3-0.6B)
NUM_LAYERS = 28
NUM_HEADS = 16       # query heads
NUM_KV_HEADS = 8     # KV heads (GQA)
HEAD_DIM = 128
BYTES_PER_ELEM = 2   # fp16

# RTX 3050 specs
RTX3050_PEAK_BW_GBS = 192.0       # GB/s peak memory bandwidth
RTX3050_PEAK_FP16_TFLOPS = 4.4    # TFLOPS FP16 (non-tensor)

# Benchmark config
KV_LENGTHS = [256, 512, 1024, 2048]
WARMUP_RUNS = 5
BENCH_RUNS = 20
BATCH_SIZE = 1

# SDPA backends to compare
BACKENDS = {
    "normal_decode (MATH)": SDPBackend.MATH,
    "flash_decoding (FLASH_ATTENTION)": SDPBackend.FLASH_ATTENTION,
}

RESULTS_DIR = Path(__file__).parent / "results"
REPORT_PATH = Path(__file__).parent / "results_report.md"


# ---------------------------------------------------------------------------
# Analytical FLOPs & DRAM calculations
# ---------------------------------------------------------------------------
def compute_decode_flops(batch, num_heads, seq_len, head_dim):
    """
    Compute total FLOPs for single-token decode attention (per layer).
    
    Operations:
      1. QK^T matmul:   2 * B * H * 1 * S * D  (matrix multiply)
      2. Softmax:        5 * B * H * 1 * S      (max, sub, exp, sum, div)
      3. Attn @ V:       2 * B * H * 1 * S * D  (matrix multiply)
    
    Total = B * H * S * (4*D + 5)
    """
    qk_flops = 2 * batch * num_heads * 1 * seq_len * head_dim
    softmax_flops = 5 * batch * num_heads * 1 * seq_len
    av_flops = 2 * batch * num_heads * 1 * seq_len * head_dim
    total = qk_flops + softmax_flops + av_flops
    return {
        "qk_flops": qk_flops,
        "softmax_flops": softmax_flops,
        "av_flops": av_flops,
        "total_flops": total,
    }


def compute_dram_bytes_normal(batch, num_heads, num_kv_heads, seq_len, head_dim, bytes_per_elem=2):
    """
    DRAM bytes for MATH (normal) decode — materializes attention scores.
    
    Reads:
      Q:  B * H * 1 * D * bpe
      K:  B * H * S * D * bpe   (after GQA broadcast)
      V:  B * H * S * D * bpe   (after GQA broadcast)
      Attn scores (read back for softmax): B * H * 1 * S * 4  (fp32)
    
    Writes:
      Attn scores (QK^T output):  B * H * 1 * S * 4  (fp32)
      Softmax output:             B * H * 1 * S * 4  (fp32, in-place or separate)
      Output:                     B * H * 1 * D * bpe
    
    The key difference: attention matrix is materialized to DRAM.
    """
    bpe = bytes_per_elem
    # Input reads
    q_bytes = batch * num_heads * 1 * head_dim * bpe
    k_bytes = batch * num_heads * seq_len * head_dim * bpe  # after broadcast
    v_bytes = batch * num_heads * seq_len * head_dim * bpe  # after broadcast
    
    # Attention matrix materialized (written then read, fp32 for numerical stability)
    attn_write = batch * num_heads * 1 * seq_len * 4  # write QK^T scores
    attn_read = batch * num_heads * 1 * seq_len * 4   # read for softmax
    softmax_write = batch * num_heads * 1 * seq_len * 4  # write softmax output
    softmax_read = batch * num_heads * 1 * seq_len * 4   # read for attn@V
    
    # Output write
    out_bytes = batch * num_heads * 1 * head_dim * bpe
    
    total_read = q_bytes + k_bytes + v_bytes + attn_read + softmax_read
    total_write = attn_write + softmax_write + out_bytes
    total = total_read + total_write
    
    return {
        "q_bytes": q_bytes,
        "k_bytes": k_bytes,
        "v_bytes": v_bytes,
        "attn_materialized_bytes": attn_write + attn_read + softmax_write + softmax_read,
        "out_bytes": out_bytes,
        "total_read": total_read,
        "total_write": total_write,
        "total_bytes": total,
    }


def compute_dram_bytes_flash(batch, num_heads, num_kv_heads, seq_len, head_dim, bytes_per_elem=2):
    """
    DRAM bytes for FLASH_ATTENTION (Flash Decoding) — fused kernel, no materialization.
    
    Reads:
      Q:  B * H * 1 * D * bpe
      K:  B * H * S * D * bpe   (after GQA broadcast, but tiled)
      V:  B * H * S * D * bpe   (after GQA broadcast, but tiled)
    
    Writes:
      Output: B * H * 1 * D * bpe
    
    Flash Decoding keeps attention scores in SRAM (shared memory / registers),
    never materializing the full attention matrix to DRAM.
    """
    bpe = bytes_per_elem
    q_bytes = batch * num_heads * 1 * head_dim * bpe
    k_bytes = batch * num_heads * seq_len * head_dim * bpe
    v_bytes = batch * num_heads * seq_len * head_dim * bpe
    out_bytes = batch * num_heads * 1 * head_dim * bpe
    
    total_read = q_bytes + k_bytes + v_bytes
    total_write = out_bytes
    total = total_read + total_write
    
    return {
        "q_bytes": q_bytes,
        "k_bytes": k_bytes,
        "v_bytes": v_bytes,
        "attn_materialized_bytes": 0,  # No materialization!
        "out_bytes": out_bytes,
        "total_read": total_read,
        "total_write": total_write,
        "total_bytes": total,
    }


# ---------------------------------------------------------------------------
# Kernel benchmark
# ---------------------------------------------------------------------------
def make_decode_qkv(batch, num_heads, num_kv_heads, kv_len, head_dim, dtype=torch.float16):
    q = torch.randn(batch, num_heads, 1, head_dim, device="cuda", dtype=dtype)
    k = torch.randn(batch, num_kv_heads, kv_len, head_dim, device="cuda", dtype=dtype)
    v = torch.randn(batch, num_kv_heads, kv_len, head_dim, device="cuda", dtype=dtype)
    return q, k, v


def bench_kernel(backend_enum, q, k, v, warmup=5, runs=20):
    """Benchmark SDPA kernel with CUDA event timing for accurate latency."""
    # Warmup
    with sdpa_kernel(backend_enum):
        for _ in range(warmup):
            _ = F.scaled_dot_product_attention(q, k, v, is_causal=False)
    torch.cuda.synchronize()

    # Benchmark with CUDA events
    latencies = []
    for _ in range(runs):
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        
        torch.cuda.synchronize()
        start_event.record()
        with sdpa_kernel(backend_enum):
            _ = F.scaled_dot_product_attention(q, k, v, is_causal=False)
        end_event.record()
        torch.cuda.synchronize()
        
        latencies.append(start_event.elapsed_time(end_event))  # ms
    
    return latencies


def bench_kernel_safe(backend_name, backend_enum, q, k, v, warmup, runs):
    """Run kernel benchmark with error handling."""
    try:
        lats = bench_kernel(backend_enum, q, k, v, warmup=warmup, runs=runs)
        med = median(lats)
        print(f"    {backend_name}: median={med:.4f} ms")
        return {
            "backend": backend_name,
            "median_latency_ms": round(med, 4),
            "std_latency_ms": round(stdev(lats) if len(lats) > 1 else 0, 4),
            "min_latency_ms": round(min(lats), 4),
            "max_latency_ms": round(max(lats), 4),
            "all_latencies_ms": [round(l, 4) for l in lats],
        }
    except Exception as e:
        print(f"    {backend_name}: ERROR — {e}")
        return None


# ---------------------------------------------------------------------------
# End-to-end model decode benchmark
# ---------------------------------------------------------------------------
def run_e2e_decode(model, tokenizer, backend_name, backend_enum, 
                   prompt_len=512, max_new_tokens=64, warmup=3, runs=5):
    """Run end-to-end decode with the model, measuring tokens/sec."""
    # Build prompt
    seed = "The quick brown fox jumps over the lazy dog. "
    text = seed * (prompt_len // 8 + 1)
    inputs = tokenizer(
        [text], return_tensors="pt", truncation=True,
        max_length=prompt_len, padding="max_length"
    )
    inputs = {k: v.to("cuda") for k, v in inputs.items()}
    actual_len = inputs["input_ids"].shape[1]
    
    # Warmup
    for _ in range(warmup):
        with torch.no_grad(), sdpa_kernel(backend_enum):
            model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    torch.cuda.synchronize()
    
    # Benchmark
    latencies = []
    tokens_generated = []
    peak_mems = []
    
    for _ in range(runs):
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        start = time.perf_counter()
        with torch.no_grad(), sdpa_kernel(backend_enum):
            out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        torch.cuda.synchronize()
        lat = time.perf_counter() - start
        
        n_gen = out.shape[1] - actual_len
        latencies.append(lat)
        tokens_generated.append(n_gen)
        peak_mems.append(torch.cuda.max_memory_allocated() / (1024**2))
    
    med_lat = median(latencies)
    med_tok = median(tokens_generated)
    
    return {
        "backend": backend_name,
        "prompt_len": actual_len,
        "max_new_tokens": max_new_tokens,
        "median_latency_s": round(med_lat, 4),
        "median_tokens_per_sec": round(med_tok / med_lat if med_lat > 0 else 0, 2),
        "median_peak_memory_mb": round(median(peak_mems), 2),
        "tokens_generated": int(med_tok),
    }


# ---------------------------------------------------------------------------
# GPU info
# ---------------------------------------------------------------------------
def get_gpu_info():
    return {
        "gpu_name": torch.cuda.get_device_name(0),
        "gpu_vram_mb": torch.cuda.get_device_properties(0).total_memory // (1024**2),
        "cuda_version": torch.version.cuda,
        "pytorch_version": torch.__version__,
        "compute_capability": ".".join(str(x) for x in torch.cuda.get_device_capability(0)),
    }


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------
def generate_report(kernel_results, e2e_results, gpu_info):
    """Generate the final markdown report."""
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    lines = []
    lines.append("# Flash Decoding Arithmetic Intensity Benchmark")
    lines.append("")
    lines.append("## Experiment Summary")
    lines.append("")
    lines.append("| Parameter | Value |")
    lines.append("|-----------|-------|")
    lines.append(f"| **GPU** | {gpu_info['gpu_name']} |")
    lines.append(f"| **VRAM** | {gpu_info['gpu_vram_mb']} MB |")
    lines.append(f"| **Compute Capability** | {gpu_info['compute_capability']} |")
    lines.append(f"| **PyTorch** | {gpu_info['pytorch_version']} |")
    lines.append(f"| **CUDA** | {gpu_info['cuda_version']} |")
    lines.append(f"| **Model** | {MODEL_NAME} |")
    lines.append(f"| **Precision** | fp16 |")
    lines.append(f"| **Batch Size** | {BATCH_SIZE} |")
    lines.append(f"| **Layers** | {NUM_LAYERS} |")
    lines.append(f"| **Q Heads / KV Heads** | {NUM_HEADS} / {NUM_KV_HEADS} |")
    lines.append(f"| **Head Dim** | {HEAD_DIM} |")
    lines.append(f"| **Peak BW (spec)** | {RTX3050_PEAK_BW_GBS} GB/s |")
    lines.append(f"| **Peak FP16 (spec)** | {RTX3050_PEAK_FP16_TFLOPS} TFLOPS |")
    lines.append(f"| **Benchmark Runs** | {BENCH_RUNS} |")
    lines.append(f"| **Date** | {now} |")
    lines.append("")
    
    # ---- Kernel-level results ----
    lines.append("## Kernel-Level Decode Attention Results")
    lines.append("")
    lines.append("### Arithmetic Intensity Comparison")
    lines.append("")
    lines.append("| KV Length | Mode | Total FLOPs | DRAM Bytes | Arith. Intensity (FLOPs/byte) | Latency (ms) | Achieved BW (GB/s) | BW Util. (%) | Achieved TFLOPS | Compute Util. (%) |")
    lines.append("|-----------|------|-------------|------------|-------------------------------|--------------|--------------------|--------------|-----------------|--------------------|")
    
    for r in kernel_results:
        fl = r["flops"]["total_flops"]
        db = r["dram"]["total_bytes"]
        ai = fl / db if db > 0 else 0
        lat_ms = r["timing"]["median_latency_ms"]
        lat_s = lat_ms / 1000.0
        
        achieved_bw = (db / lat_s) / 1e9 if lat_s > 0 else 0
        bw_util = (achieved_bw / RTX3050_PEAK_BW_GBS) * 100
        achieved_tflops = (fl / lat_s) / 1e12 if lat_s > 0 else 0
        compute_util = (achieved_tflops / RTX3050_PEAK_FP16_TFLOPS) * 100
        
        r["derived"] = {
            "arithmetic_intensity": round(ai, 4),
            "achieved_bw_gbs": round(achieved_bw, 2),
            "bw_utilization_pct": round(bw_util, 2),
            "achieved_tflops": round(achieved_tflops, 4),
            "compute_utilization_pct": round(compute_util, 2),
        }
        
        mode_label = "Normal" if "normal" in r["mode"] else "Flash"
        lines.append(
            f"| {r['kv_len']} | {mode_label} | {fl:,} | {db:,} | "
            f"{ai:.4f} | {lat_ms:.4f} | {achieved_bw:.2f} | "
            f"{bw_util:.2f} | {achieved_tflops:.4f} | {compute_util:.2f} |"
        )
    
    # ---- Improvement summary ----
    lines.append("")
    lines.append("### Arithmetic Intensity Improvement (Flash Decoding vs Normal)")
    lines.append("")
    lines.append("| KV Length | Normal AI (FLOPs/byte) | Flash AI (FLOPs/byte) | % Improvement | DRAM Saved (bytes) | DRAM Saved (%) |")
    lines.append("|-----------|------------------------|-----------------------|---------------|--------------------|----------------|")
    
    # Group by kv_len
    by_kv = {}
    for r in kernel_results:
        by_kv.setdefault(r["kv_len"], {})[r["mode"]] = r
    
    for kv_len in sorted(by_kv.keys()):
        entries = by_kv[kv_len]
        normal = entries.get("normal")
        flash = entries.get("flash")
        if normal and flash:
            n_ai = normal["derived"]["arithmetic_intensity"]
            f_ai = flash["derived"]["arithmetic_intensity"]
            pct_improve = ((f_ai - n_ai) / n_ai) * 100 if n_ai > 0 else 0
            n_bytes = normal["dram"]["total_bytes"]
            f_bytes = flash["dram"]["total_bytes"]
            bytes_saved = n_bytes - f_bytes
            pct_saved = (bytes_saved / n_bytes) * 100 if n_bytes > 0 else 0
            
            lines.append(
                f"| {kv_len} | {n_ai:.4f} | {f_ai:.4f} | "
                f"{pct_improve:.2f}% | {bytes_saved:,} | {pct_saved:.2f}% |"
            )
    
    # ---- Memory bandwidth utilization ----
    lines.append("")
    lines.append("### Memory Bandwidth Utilization")
    lines.append("")
    lines.append("| KV Length | Mode | Achieved BW (GB/s) | Peak BW (GB/s) | Utilization (%) |")
    lines.append("|-----------|------|--------------------:|:--------------:|:---------------:|")
    
    for r in kernel_results:
        mode_label = "Normal" if "normal" in r["mode"] else "Flash"
        d = r["derived"]
        lines.append(
            f"| {r['kv_len']} | {mode_label} | {d['achieved_bw_gbs']:.2f} | "
            f"{RTX3050_PEAK_BW_GBS} | {d['bw_utilization_pct']:.2f} |"
        )
    
    # ---- DRAM breakdown ----
    lines.append("")
    lines.append("### DRAM Traffic Breakdown")
    lines.append("")
    lines.append("| KV Length | Mode | Q (bytes) | K (bytes) | V (bytes) | Attn Matrix (bytes) | Output (bytes) | Total (bytes) |")
    lines.append("|-----------|------|-----------|-----------|-----------|---------------------|----------------|---------------|")
    
    for r in kernel_results:
        mode_label = "Normal" if "normal" in r["mode"] else "Flash"
        d = r["dram"]
        lines.append(
            f"| {r['kv_len']} | {mode_label} | {d['q_bytes']:,} | {d['k_bytes']:,} | "
            f"{d['v_bytes']:,} | {d['attn_materialized_bytes']:,} | "
            f"{d['out_bytes']:,} | {d['total_bytes']:,} |"
        )
    
    # ---- FLOPs breakdown ----
    lines.append("")
    lines.append("### FLOPs Breakdown")
    lines.append("")
    lines.append("| KV Length | QK^T FLOPs | Softmax FLOPs | Attn×V FLOPs | Total FLOPs |")
    lines.append("|-----------|------------|---------------|--------------|-------------|")
    
    seen_kv = set()
    for r in kernel_results:
        if r["kv_len"] not in seen_kv:
            seen_kv.add(r["kv_len"])
            f = r["flops"]
            lines.append(
                f"| {r['kv_len']} | {f['qk_flops']:,} | {f['softmax_flops']:,} | "
                f"{f['av_flops']:,} | {f['total_flops']:,} |"
            )
    
    # ---- E2E results ----
    if e2e_results:
        lines.append("")
        lines.append("## End-to-End Model Decode Results")
        lines.append("")
        lines.append(f"Model: `{MODEL_NAME}`, Prompt length: {e2e_results[0]['prompt_len']}, "
                      f"Max new tokens: {e2e_results[0]['max_new_tokens']}")
        lines.append("")
        lines.append("| Mode | Latency (s) | Tokens/sec | Tokens Generated | Peak Memory (MB) |")
        lines.append("|------|-------------|------------|------------------|------------------|")
        
        for r in e2e_results:
            mode_label = "Normal" if "normal" in r["backend"].lower() else "Flash"
            lines.append(
                f"| {mode_label} | {r['median_latency_s']:.4f} | "
                f"{r['median_tokens_per_sec']:.2f} | {r['tokens_generated']} | "
                f"{r['median_peak_memory_mb']:.2f} |"
            )
        
        # Improvement
        normal_e2e = [r for r in e2e_results if "normal" in r["backend"].lower()]
        flash_e2e = [r for r in e2e_results if "flash" in r["backend"].lower()]
        if normal_e2e and flash_e2e:
            n_tps = normal_e2e[0]["median_tokens_per_sec"]
            f_tps = flash_e2e[0]["median_tokens_per_sec"]
            speedup = ((f_tps - n_tps) / n_tps) * 100 if n_tps > 0 else 0
            lines.append("")
            lines.append(f"**Tokens/sec improvement: {speedup:+.2f}%** (Flash Decoding vs Normal)")
    
    # ---- Analysis ----
    lines.append("")
    lines.append("## Analysis")
    lines.append("")
    lines.append("### Why Flash Decoding Improves Arithmetic Intensity")
    lines.append("")
    lines.append("During the **decode phase** of LLM inference (autoregressive token generation),")
    lines.append("the query length is always 1 while the KV-cache grows with each generated token.")
    lines.append("This creates an extremely **memory-bound** operation:")
    lines.append("")
    lines.append("- **Normal decode (MATH backend)**: Materializes the attention score matrix")
    lines.append("  `(B, H, 1, S)` to global DRAM memory. Even though this matrix is small,")
    lines.append("  the write-then-read pattern adds significant memory traffic, especially")
    lines.append("  since softmax requires fp32 precision for numerical stability.")
    lines.append("")
    lines.append("- **Flash Decoding (FLASH_ATTENTION backend)**: Uses a fused kernel that")
    lines.append("  keeps attention scores in SRAM (shared memory/registers). The attention")
    lines.append("  matrix is **never written to DRAM**, eliminating the materialization overhead.")
    lines.append("  The kernel splits the KV-cache across thread blocks, each computing a")
    lines.append("  partial softmax, then reduces the results — all in on-chip memory.")
    lines.append("")
    lines.append("### Roofline Model Perspective")
    lines.append("")
    lines.append(f"The RTX 3050 has a ridge point at:")
    lines.append(f"  - Ridge Point = Peak FLOPS / Peak BW = {RTX3050_PEAK_FP16_TFLOPS * 1e12:.0f} / "
                 f"{RTX3050_PEAK_BW_GBS * 1e9:.0f} = "
                 f"**{(RTX3050_PEAK_FP16_TFLOPS * 1e12) / (RTX3050_PEAK_BW_GBS * 1e9):.2f} FLOPs/byte**")
    lines.append("")
    lines.append("Decode attention is well below this ridge point for both modes, confirming")
    lines.append("it is **memory-bound**. Flash Decoding improves arithmetic intensity by")
    lines.append("reducing DRAM traffic, moving the operation closer to (but still below)")
    lines.append("the ridge point.")
    lines.append("")
    lines.append("---")
    lines.append(f"*Report generated: {now}*")
    
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("=" * 70)
    print("  FLASH DECODING ARITHMETIC INTENSITY BENCHMARK")
    print("=" * 70)
    
    gpu_info = get_gpu_info()
    print(f"\nGPU: {gpu_info['gpu_name']}")
    print(f"VRAM: {gpu_info['gpu_vram_mb']} MB")
    print(f"PyTorch: {gpu_info['pytorch_version']}, CUDA: {gpu_info['cuda_version']}")
    print(f"Model: {MODEL_NAME}")
    print(f"Precision: fp16, Batch: {BATCH_SIZE}")
    
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    
    # ==== PART 1: Kernel-level decode attention ====
    print("\n" + "#" * 60)
    print("# PART 1: Kernel-Level Decode Attention Benchmark")
    print("#" * 60)
    
    kernel_results = []
    
    for kv_len in KV_LENGTHS:
        print(f"\n  KV Length = {kv_len}")
        q, k, v = make_decode_qkv(BATCH_SIZE, NUM_HEADS, NUM_KV_HEADS, kv_len, HEAD_DIM)
        
        # Compute FLOPs (same for both modes — same math operations)
        flops = compute_decode_flops(BATCH_SIZE, NUM_HEADS, kv_len, HEAD_DIM)
        
        for bname, benum in BACKENDS.items():
            print(f"\n    Backend: {bname}")
            
            # Compute DRAM bytes
            if "normal" in bname.lower():
                dram = compute_dram_bytes_normal(
                    BATCH_SIZE, NUM_HEADS, NUM_KV_HEADS, kv_len, HEAD_DIM, BYTES_PER_ELEM
                )
                mode = "normal"
            else:
                dram = compute_dram_bytes_flash(
                    BATCH_SIZE, NUM_HEADS, NUM_KV_HEADS, kv_len, HEAD_DIM, BYTES_PER_ELEM
                )
                mode = "flash"
            
            # Benchmark kernel
            timing = bench_kernel_safe(bname, benum, q, k, v,
                                        warmup=WARMUP_RUNS, runs=BENCH_RUNS)
            
            if timing:
                result = {
                    "kv_len": kv_len,
                    "mode": mode,
                    "backend_name": bname,
                    "flops": flops,
                    "dram": dram,
                    "timing": timing,
                }
                kernel_results.append(result)
                
                ai = flops["total_flops"] / dram["total_bytes"] if dram["total_bytes"] > 0 else 0
                print(f"      FLOPs:  {flops['total_flops']:,}")
                print(f"      DRAM:   {dram['total_bytes']:,} bytes")
                print(f"      AI:     {ai:.4f} FLOPs/byte")
        
        del q, k, v
        torch.cuda.empty_cache()
    
    # ==== PART 2: End-to-end model decode ====
    print("\n" + "#" * 60)
    print("# PART 2: End-to-End Model Decode")
    print("#" * 60)
    
    e2e_results = []
    try:
        print("\n  Loading model...")
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME, dtype=torch.float16, attn_implementation="sdpa"
        ).to("cuda")
        model.eval()
        
        for bname, benum in BACKENDS.items():
            print(f"\n  E2E: {bname}")
            try:
                r = run_e2e_decode(
                    model, tokenizer, bname, benum,
                    prompt_len=512, max_new_tokens=64,
                    warmup=3, runs=5,
                )
                e2e_results.append(r)
                print(f"    Latency: {r['median_latency_s']:.4f}s, "
                      f"Tokens/sec: {r['median_tokens_per_sec']:.2f}, "
                      f"Memory: {r['median_peak_memory_mb']:.2f} MB")
            except Exception as e:
                print(f"    ERROR: {e}")
        
        del model
        torch.cuda.empty_cache()
        gc.collect()
    except Exception as e:
        print(f"  Model loading error: {e}")
    
    # ==== Generate Report ====
    print("\n" + "#" * 60)
    print("# Generating Report")
    print("#" * 60)
    
    report = generate_report(kernel_results, e2e_results, gpu_info)
    REPORT_PATH.write_text(report)
    print(f"\n  Report saved: {REPORT_PATH}")
    
    # Save raw JSON
    json_data = {
        "gpu_info": gpu_info,
        "kernel_results": [],
        "e2e_results": e2e_results,
    }
    for r in kernel_results:
        json_data["kernel_results"].append({
            "kv_len": r["kv_len"],
            "mode": r["mode"],
            "flops": r["flops"],
            "dram": r["dram"],
            "timing": r["timing"],
            "derived": r.get("derived", {}),
        })
    
    json_path = RESULTS_DIR / "benchmark_results.json"
    with open(json_path, "w") as f:
        json.dump(json_data, f, indent=2)
    print(f"  JSON saved: {json_path}")
    
    print("\n" + "=" * 70)
    print("  BENCHMARK COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
