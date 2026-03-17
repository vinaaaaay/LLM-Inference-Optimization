"""
run_precision_analysis.py — Memory-Bound vs Compute-Bound Analysis

Determines whether the decode attention is memory-bound or compute-bound
by comparing performance across data precisions.

Principle:
  - If halving data size halves latency → Memory-Bound
    (bottleneck is data transfer through the memory bus)
  - If halving data size barely changes latency → Compute-Bound
    (bottleneck is the ALU/tensor cores, not data movement)

FP8 Note: Native FP8 (float8_e4m3fn) requires compute capability 8.9+
(Ada Lovelace / Hopper GPUs). The RTX 3050 (Ampere, CC 8.6) does NOT
support FP8 tensor core operations. Therefore, we use FP16 vs FP32
(2x data size difference) which demonstrates the same principle:
  - FP16: 2 bytes per element
  - FP32: 4 bytes per element (2x more data to move)
  - If FP32 is ~2x slower → memory-bound
  - If FP32 is ~1x (same speed) → compute-bound

We test both:
  Exp A: Kernel-level decode attention (mirrors Exp 4)
  Exp B: Per-token decode latency (mirrors Exp 1)

Hardware: NVIDIA GeForce RTX 3050 (5795 MB VRAM, CC 8.6, Ampere)
Model: facebook/opt-350m
"""

import argparse
import csv
import gc
import json
import sys
import time
from pathlib import Path
from statistics import median, stdev

import torch
import torch.nn.functional as F
from torch.nn.attention import sdpa_kernel, SDPBackend
from transformers import AutoModelForCausalLM, AutoTokenizer


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DEFAULT_MODEL = "facebook/opt-350m"
WARMUP_RUNS = 3
BENCHMARK_RUNS = 5

BACKENDS = {
    "flash": SDPBackend.FLASH_ATTENTION,
    "mem_efficient": SDPBackend.EFFICIENT_ATTENTION,
    "math": SDPBackend.MATH,
}

# OPT-350m attention config
NUM_HEADS = 16
HEAD_DIM = 64

# Precisions to test
PRECISIONS = {
    "float16": torch.float16,
    "float32": torch.float32,
}

# KV lengths for kernel test
KERNEL_KV_LENGTHS = [128, 256, 512, 1024, 1536, 2048]

# Cache lengths for decode test
DECODE_CACHE_LENGTHS = [256, 512, 1024, 1800]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def get_gpu_info():
    return {
        "gpu_name": torch.cuda.get_device_name(0),
        "gpu_vram_mb": torch.cuda.get_device_properties(0).total_memory // (1024 * 1024),
        "cuda_version": torch.version.cuda,
        "pytorch_version": torch.__version__,
        "compute_capability": ".".join(
            str(x) for x in torch.cuda.get_device_capability(0)
        ),
    }


def make_decode_qkv(batch_size, num_heads, kv_len, head_dim, dtype):
    q = torch.randn(batch_size, num_heads, 1, head_dim, device="cuda", dtype=dtype)
    k = torch.randn(batch_size, num_heads, kv_len, head_dim, device="cuda", dtype=dtype)
    v = torch.randn(batch_size, num_heads, kv_len, head_dim, device="cuda", dtype=dtype)
    return q, k, v


def bench_decode_kernel(backend_enum, q, k, v, warmup=3, runs=10):
    # Warm-up
    with sdpa_kernel(backend_enum):
        for _ in range(warmup):
            _ = F.scaled_dot_product_attention(q, k, v, is_causal=False)
    torch.cuda.synchronize()

    # Benchmark
    latencies = []
    peak_mems = []
    for _ in range(runs):
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        start = time.perf_counter()
        with sdpa_kernel(backend_enum):
            _ = F.scaled_dot_product_attention(q, k, v, is_causal=False)
        torch.cuda.synchronize()
        lat = time.perf_counter() - start
        peak_mem = torch.cuda.max_memory_allocated() / (1024 ** 2)
        latencies.append(lat)
        peak_mems.append(peak_mem)
    return latencies, peak_mems


def safe_kernel(backend_name, backend_enum, q, k, v, warmup, runs):
    try:
        lats, mems = bench_decode_kernel(backend_enum, q, k, v, warmup=warmup, runs=runs)
        med_lat = median(lats)
        med_mem = median(mems)
        print(f"    median_lat={med_lat*1000:.3f}ms  mem={med_mem:.1f}MB")
        return {
            "backend": backend_name,
            "median_latency_ms": round(med_lat * 1000, 3),
            "std_latency_ms": round(stdev(lats) * 1000 if len(lats) > 1 else 0, 3),
            "median_peak_memory_mb": round(med_mem, 2),
            "num_runs": runs,
            "all_latencies_ms": [round(l * 1000, 4) for l in lats],
            "all_peak_memories": mems,
        }
    except torch.cuda.OutOfMemoryError:
        print("    OOM — skipping")
        torch.cuda.empty_cache()
        gc.collect()
        return None
    except Exception as e:
        print(f"    ERROR — {e}")
        return None


def build_prompt(tokenizer, seq_len, batch_size):
    seed_text = "The quick brown fox jumps over the lazy dog. "
    repeated = seed_text * (seq_len // 8 + 1)
    encoded = tokenizer(
        [repeated] * batch_size,
        return_tensors="pt",
        truncation=True,
        max_length=seq_len,
        padding="max_length",
    )
    return {k: v.to("cuda") for k, v in encoded.items()}


def load_model(model_name, attn_impl="sdpa", dtype=torch.float16):
    print(f"  Loading model: attn='{attn_impl}', dtype={dtype} ...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=dtype,
        attn_implementation=attn_impl,
    ).to("cuda")
    model.eval()
    return model


def measure_decode_latency_per_token(model, inputs, max_new_tokens, backend_enum):
    with torch.no_grad():
        # Prefill
        torch.cuda.synchronize()
        prefill_start = time.perf_counter()
        with sdpa_kernel(backend_enum):
            outputs = model.generate(**inputs, max_new_tokens=1, do_sample=False)
        torch.cuda.synchronize()
        prefill_time = time.perf_counter() - prefill_start

        # Decode step by step
        decode_latencies = []
        current_ids = outputs
        for _ in range(max_new_tokens - 1):
            torch.cuda.synchronize()
            step_start = time.perf_counter()
            with sdpa_kernel(backend_enum):
                current_ids = model.generate(
                    input_ids=current_ids,
                    attention_mask=torch.ones_like(current_ids),
                    max_new_tokens=1,
                    do_sample=False,
                )
            torch.cuda.synchronize()
            step_lat = time.perf_counter() - step_start
            decode_latencies.append(step_lat)

    return prefill_time, decode_latencies


def print_table(results, title, cols):
    header = "  ".join(f"{c:>18}" for c in cols)
    print(f"\n{'=' * len(header)}")
    print(title)
    print(f"{'=' * len(header)}")
    print(header)
    print("-" * len(header))
    for r in results:
        row = "  ".join(f"{str(r.get(c, '')):>18}" for c in cols)
        print(row)
    print("=" * len(header))


def write_csv(results, filepath, fields):
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(results)
    print(f"  -> Saved: {filepath}")


# ---------------------------------------------------------------------------
# Exp A: Kernel-Level Decode — FP16 vs FP32
# ---------------------------------------------------------------------------
def run_exp_a_kernel_precision(results_dir):
    """
    Kernel-level decode attention with FP16 vs FP32.
    Q: (1, 16, 1, D), K/V: (1, 16, kv_len, D)
    """
    print("\n" + "#" * 60)
    print("# EXP A: Kernel-Level Decode — FP16 vs FP32")
    print("#" * 60)
    print("\nPrinciple: If FP32 (2x data) takes ~2x longer → Memory-Bound")
    print("           If FP32 takes ~1x (same) → Compute-Bound\n")

    all_results = []

    for kv_len in KERNEL_KV_LENGTHS:
        for prec_name, prec_dtype in PRECISIONS.items():
            for bname, benum in BACKENDS.items():
                print(f"  backend={bname}, kv_len={kv_len}, dtype={prec_name}")

                q, k, v = make_decode_qkv(1, NUM_HEADS, kv_len, HEAD_DIM, prec_dtype)
                r = safe_kernel(bname, benum, q, k, v,
                                warmup=WARMUP_RUNS, runs=BENCHMARK_RUNS)
                if r:
                    r["kv_len"] = kv_len
                    r["dtype"] = prec_name
                    r["bytes_per_elem"] = 2 if prec_name == "float16" else 4
                    # KV data size: 2 (K+V) * heads * kv_len * head_dim * bytes
                    kv_data_mb = (2 * NUM_HEADS * kv_len * HEAD_DIM * r["bytes_per_elem"]) / (1024**2)
                    r["kv_data_mb"] = round(kv_data_mb, 2)
                    all_results.append(r)

                del q, k, v
                torch.cuda.empty_cache()

    # Compute ratio analysis
    analysis = []
    for kv_len in KERNEL_KV_LENGTHS:
        for bname in BACKENDS:
            fp16_r = next((r for r in all_results
                          if r["kv_len"] == kv_len and r["backend"] == bname
                          and r["dtype"] == "float16"), None)
            fp32_r = next((r for r in all_results
                          if r["kv_len"] == kv_len and r["backend"] == bname
                          and r["dtype"] == "float32"), None)
            if fp16_r and fp32_r:
                ratio = fp32_r["median_latency_ms"] / fp16_r["median_latency_ms"]
                bound = "MEMORY-BOUND" if ratio > 1.5 else ("COMPUTE-BOUND" if ratio < 1.2 else "MIXED")
                analysis.append({
                    "backend": bname,
                    "kv_len": kv_len,
                    "fp16_ms": fp16_r["median_latency_ms"],
                    "fp32_ms": fp32_r["median_latency_ms"],
                    "fp32_fp16_ratio": round(ratio, 2),
                    "fp16_mem_mb": fp16_r["median_peak_memory_mb"],
                    "fp32_mem_mb": fp32_r["median_peak_memory_mb"],
                    "diagnosis": bound,
                })

    if all_results:
        cols = ["backend", "kv_len", "dtype", "median_latency_ms",
                "median_peak_memory_mb", "kv_data_mb"]
        print_table(all_results, "EXP A: Kernel Decode — All Results", cols)
        write_csv(all_results, results_dir / "precision_exp_a_kernel.csv",
                  cols + ["std_latency_ms", "bytes_per_elem", "num_runs"])

    if analysis:
        cols_a = ["backend", "kv_len", "fp16_ms", "fp32_ms",
                  "fp32_fp16_ratio", "diagnosis"]
        print_table(analysis, "EXP A: Memory-Bound vs Compute-Bound Diagnosis (Kernel)", cols_a)
        write_csv(analysis, results_dir / "precision_exp_a_diagnosis.csv",
                  cols_a + ["fp16_mem_mb", "fp32_mem_mb"])

    return all_results, analysis


# ---------------------------------------------------------------------------
# Exp B: Model-Level Decode — FP16 vs FP32 (mirrors Exp 1)
# ---------------------------------------------------------------------------
def run_exp_b_decode_precision(results_dir, model_name=DEFAULT_MODEL):
    """
    Per-token decode latency with the model loaded in FP16 vs FP32.
    """
    print("\n" + "#" * 60)
    print("# EXP B: Model-Level Decode — FP16 vs FP32")
    print("#" * 60)
    print("\nPrinciple: If FP32 (2x data) per-token decode takes ~2x longer → Memory-Bound")
    print("           If FP32 takes ~1x (same) → Compute-Bound\n")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    all_results = []
    decode_tokens = 16

    for prec_name, prec_dtype in PRECISIONS.items():
        try:
            model = load_model(model_name, "sdpa", dtype=prec_dtype)
        except Exception as e:
            print(f"  SKIP dtype={prec_name}: {e}")
            continue

        max_pos = getattr(model.config, "max_position_embeddings", None)

        for cache_len in DECODE_CACHE_LENGTHS:
            if max_pos and (cache_len + decode_tokens) > max_pos:
                print(f"  SKIP — cache({cache_len})+gen({decode_tokens}) > max_pos({max_pos})")
                continue

            # Use flash backend for the controlled comparison
            # (we want to isolate precision effect, not backend effect)
            for bname in ["flash", "math"]:
                benum = BACKENDS[bname]
                print(f"\n  dtype={prec_name}, backend={bname}, cache_len={cache_len}")

                try:
                    inputs = build_prompt(tokenizer, cache_len, 1)

                    # Warm-up
                    for _ in range(WARMUP_RUNS):
                        with torch.no_grad(), sdpa_kernel(benum):
                            model.generate(**inputs, max_new_tokens=decode_tokens, do_sample=False)
                    torch.cuda.synchronize()

                    # Measure
                    all_decode_lats = []
                    prefill_times = []
                    for _ in range(BENCHMARK_RUNS):
                        pfill, dlats = measure_decode_latency_per_token(
                            model, inputs, decode_tokens, benum
                        )
                        prefill_times.append(pfill)
                        all_decode_lats.append(dlats)

                    med_per_token_lats = []
                    for step in range(len(all_decode_lats[0])):
                        step_lats = [run[step] for run in all_decode_lats]
                        med_per_token_lats.append(median(step_lats))

                    avg_decode_lat = sum(med_per_token_lats) / len(med_per_token_lats) if med_per_token_lats else 0

                    # Peak memory
                    torch.cuda.reset_peak_memory_stats()
                    with torch.no_grad(), sdpa_kernel(benum):
                        model.generate(**inputs, max_new_tokens=decode_tokens, do_sample=False)
                    torch.cuda.synchronize()
                    peak_mem = torch.cuda.max_memory_allocated() / (1024**2)

                    result = {
                        "dtype": prec_name,
                        "backend": bname,
                        "cache_len": cache_len,
                        "median_prefill_s": round(median(prefill_times), 4),
                        "median_per_token_decode_ms": round(avg_decode_lat * 1000, 3),
                        "total_decode_s": round(sum(med_per_token_lats), 4),
                        "peak_memory_mb": round(peak_mem, 2),
                    }
                    all_results.append(result)
                    print(
                        f"    prefill={result['median_prefill_s']:.4f}s  "
                        f"decode/token={result['median_per_token_decode_ms']:.3f}ms  "
                        f"mem={result['peak_memory_mb']:.1f}MB"
                    )

                except torch.cuda.OutOfMemoryError:
                    print("    OOM — skipping")
                    torch.cuda.empty_cache()
                    gc.collect()
                except Exception as e:
                    print(f"    ERROR — {e}")

        del model
        torch.cuda.empty_cache()
        gc.collect()

    # Ratio analysis
    analysis = []
    for cache_len in DECODE_CACHE_LENGTHS:
        for bname in ["flash", "math"]:
            fp16_r = next((r for r in all_results
                          if r["cache_len"] == cache_len and r["backend"] == bname
                          and r["dtype"] == "float16"), None)
            fp32_r = next((r for r in all_results
                          if r["cache_len"] == cache_len and r["backend"] == bname
                          and r["dtype"] == "float32"), None)
            if fp16_r and fp32_r:
                decode_ratio = fp32_r["median_per_token_decode_ms"] / fp16_r["median_per_token_decode_ms"]
                prefill_ratio = fp32_r["median_prefill_s"] / fp16_r["median_prefill_s"]
                decode_bound = "MEMORY-BOUND" if decode_ratio > 1.5 else ("COMPUTE-BOUND" if decode_ratio < 1.2 else "MIXED")
                prefill_bound = "MEMORY-BOUND" if prefill_ratio > 1.5 else ("COMPUTE-BOUND" if prefill_ratio < 1.2 else "MIXED")
                analysis.append({
                    "backend": bname,
                    "cache_len": cache_len,
                    "fp16_decode_ms": fp16_r["median_per_token_decode_ms"],
                    "fp32_decode_ms": fp32_r["median_per_token_decode_ms"],
                    "decode_ratio": round(decode_ratio, 2),
                    "decode_diagnosis": decode_bound,
                    "fp16_prefill_s": fp16_r["median_prefill_s"],
                    "fp32_prefill_s": fp32_r["median_prefill_s"],
                    "prefill_ratio": round(prefill_ratio, 2),
                    "prefill_diagnosis": prefill_bound,
                    "fp16_mem_mb": fp16_r["peak_memory_mb"],
                    "fp32_mem_mb": fp32_r["peak_memory_mb"],
                    "mem_ratio": round(fp32_r["peak_memory_mb"] / fp16_r["peak_memory_mb"], 2),
                })

    if all_results:
        cols = ["dtype", "backend", "cache_len", "median_prefill_s",
                "median_per_token_decode_ms", "peak_memory_mb"]
        print_table(all_results, "EXP B: Model Decode — All Results", cols)
        write_csv(all_results, results_dir / "precision_exp_b_model.csv", cols)

    if analysis:
        cols_a = ["backend", "cache_len",
                  "fp16_decode_ms", "fp32_decode_ms", "decode_ratio", "decode_diagnosis",
                  "fp16_prefill_s", "fp32_prefill_s", "prefill_ratio", "prefill_diagnosis"]
        print_table(analysis, "EXP B: Memory-Bound vs Compute-Bound Diagnosis (Model)", cols_a)
        write_csv(analysis, results_dir / "precision_exp_b_diagnosis.csv",
                  cols_a + ["fp16_mem_mb", "fp32_mem_mb", "mem_ratio"])

    return all_results, analysis


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    project_root = Path(__file__).resolve().parent
    results_dir = project_root / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    if not torch.cuda.is_available():
        print("ERROR: CUDA not available.")
        sys.exit(1)

    gpu_info = get_gpu_info()
    print("=" * 60)
    print("FP16 vs FP32 PRECISION ANALYSIS")
    print("Memory-Bound vs Compute-Bound Diagnosis")
    print("=" * 60)
    for k, v in gpu_info.items():
        print(f"  {k}: {v}")
    print("=" * 60)

    print("\n⚠  FP8 (float8_e4m3fn) is NOT supported for SDPA/matmul")
    print("   on this GPU (Ampere CC 8.6 — needs CC 8.9+ for FP8).")
    print("   Using FP16 vs FP32 instead (same principle: 2x data size).\n")

    # Run both experiments
    kernel_results, kernel_analysis = run_exp_a_kernel_precision(results_dir)
    model_results, model_analysis = run_exp_b_decode_precision(results_dir)

    # Save combined analysis
    combined = {
        "note": "FP8 not supported on Ampere CC 8.6. Using FP16 vs FP32 (2x data size).",
        "principle": "If FP32 (2x data) is ~2x slower → MEMORY-BOUND. If ~1x → COMPUTE-BOUND.",
        "kernel_analysis": kernel_analysis,
        "model_analysis": model_analysis,
    }
    json_path = results_dir / "precision_analysis.json"
    with open(json_path, "w") as f:
        json.dump(combined, f, indent=2)
    print(f"\n  -> Saved: {json_path}")

    # Print final summary
    print("\n" + "=" * 60)
    print("PRECISION ANALYSIS COMPLETE")
    print("=" * 60)
    if kernel_analysis:
        print("\nKernel-Level Diagnosis:")
        for a in kernel_analysis:
            print(f"  {a['backend']:>14} kv={a['kv_len']:>5}  "
                  f"FP16={a['fp16_ms']:.3f}ms  FP32={a['fp32_ms']:.3f}ms  "
                  f"ratio={a['fp32_fp16_ratio']:.2f}x → {a['diagnosis']}")
    if model_analysis:
        print("\nModel-Level Decode Diagnosis:")
        for a in model_analysis:
            print(f"  {a['backend']:>14} cache={a['cache_len']:>5}  "
                  f"FP16={a['fp16_decode_ms']:.3f}ms  FP32={a['fp32_decode_ms']:.3f}ms  "
                  f"ratio={a['decode_ratio']:.2f}x → {a['decode_diagnosis']}")
        print("\nModel-Level Prefill Diagnosis:")
        for a in model_analysis:
            print(f"  {a['backend']:>14} input={a['cache_len']:>5}  "
                  f"FP16={a['fp16_prefill_s']:.4f}s  FP32={a['fp32_prefill_s']:.4f}s  "
                  f"ratio={a['prefill_ratio']:.2f}x → {a['prefill_diagnosis']}")
    print("=" * 60)


if __name__ == "__main__":
    main()
