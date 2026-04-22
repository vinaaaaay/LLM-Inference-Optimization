"""
run_flash_decoding.py — Flash Decoding vs Normal Decoding Comparison

Compares decode-phase performance of three PyTorch SDPA backends:
  - FLASH_ATTENTION  (FlashAttention v2 kernel — "flash decoding")
  - EFFICIENT_ATTENTION (xFormers memory-efficient kernel)
  - MATH (vanilla PyTorch matmul + softmax — "normal decoding")

Flash Decoding parallelizes attention over the KV-cache sequence length,
achieving higher GPU utilization during the decode phase where a single
query token attends to all cached keys/values.

Experiments:

  Exp 1 (Decode Latency vs KV-Cache Length):
      Measure per-token decode latency as KV-cache grows (256→1800).
      -> Core experiment: flash decoding should scale better with cache size.

  Exp 2 (Decode Throughput Scaling):
      Measure tokens/sec with varying generation lengths (32→256).
      -> Isolates pure decode throughput.

  Exp 3 (Batch Size Impact on Decode):
      Compare decode across batch sizes (1, 2, 4) at fixed seq lengths.
      -> Flash decoding should better utilize GPU at small batches.

  Exp 4 (Kernel-Level Decode Attention):
      Raw F.scaled_dot_product_attention with query shape (B, H, 1, D)
      against KV of varying lengths.
      -> Directly benchmarks single-token decode attention kernel.

  Exp 5 (End-to-End Comparison + Memory Analysis):
      Full inference with prefill/decode breakdown and KV-cache vs
      attention matrix memory analysis.

Hardware: NVIDIA GeForce RTX 3050 (5795 MB VRAM, Compute 8.6, Ampere)
Model: Qwen/Qwen3-0.6B (0.6B params, 16 query heads, 8 KV heads, head_dim=128)
Software: PyTorch 2.10.0, CUDA 12.8

Usage:
    python3 run_flash_decoding.py                         # all experiments
    python3 run_flash_decoding.py --experiments 1 2       # specific ones
    python3 run_flash_decoding.py --benchmark_runs 3      # quick test
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
# GQA Support for SDPA
# ---------------------------------------------------------------------------
# Many SDPA fused kernels require query, key, and value to have the same
# number of heads for dense inputs. We monkeypatch F.scaled_dot_product_attention
# to automatically broadcast KV heads if they don't match Q heads.
_original_sdpa = F.scaled_dot_product_attention

def _patched_sdpa(query, key, value, *args, **kwargs):
    if query.shape[1] != key.shape[1]:
        # Broadcast heads (GQA -> Multi-Head)
        key = key.repeat_interleave(query.shape[1] // key.shape[1], dim=1)
        value = value.repeat_interleave(query.shape[1] // value.shape[1], dim=1)
    return _original_sdpa(query, key, value, *args, **kwargs)

F.scaled_dot_product_attention = _patched_sdpa


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DEFAULT_MODEL = "Qwen/Qwen3-0.6B"
DEFAULT_SEQ_LENGTHS = [512, 1024, 1800]
DEFAULT_BATCH_SIZES = [1, 2, 4]
DEFAULT_MAX_NEW_TOKENS = 128
DEFAULT_WARMUP_RUNS = 3
DEFAULT_BENCHMARK_RUNS = 5

BACKENDS = {
    "flash": SDPBackend.FLASH_ATTENTION,
    "mem_efficient": SDPBackend.EFFICIENT_ATTENTION,
    "math": SDPBackend.MATH,
}

# Exp 1: KV-cache lengths to test decode latency against
DECODE_CACHE_LENGTHS = [256, 512, 768, 1024, 1536, 1800]

# Exp 2: generation lengths for throughput scaling
GEN_LENGTHS = [32, 64, 128, 256]

# Exp 4: kernel-level KV-cache lengths
KERNEL_KV_LENGTHS = [128, 256, 512, 1024, 1536, 2048]

# Model attention config (Qwen3-0.6B)
NUM_HEADS = 16
NUM_KV_HEADS = 8
HEAD_DIM = 128


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def get_gpu_info() -> dict:
    return {
        "gpu_name": torch.cuda.get_device_name(0),
        "gpu_vram_mb": torch.cuda.get_device_properties(0).total_memory // (1024 * 1024),
        "cuda_version": torch.version.cuda,
        "pytorch_version": torch.__version__,
        "compute_capability": ".".join(
            str(x) for x in torch.cuda.get_device_capability(0)
        ),
    }


def build_prompt(tokenizer, seq_len: int, batch_size: int):
    """Build a tokenized prompt of the given length."""
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


def load_model(model_name: str, attn_impl: str = "sdpa"):
    """Load model with specified attention implementation."""
    print(f"  Loading model: attn='{attn_impl}', dtype=float16 ...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.float16,
        attn_implementation=attn_impl,
    ).to("cuda")
    model.eval()
    return model


# ---------------------------------------------------------------------------
# Kernel-level decode benchmark helpers
# ---------------------------------------------------------------------------
def make_decode_qkv(batch_size, num_heads, num_kv_heads, kv_len, head_dim, dtype=torch.float16):
    """
    Create tensors simulating a single-token decode step with GQA support:
      Q: (batch, heads, 1, head_dim)      — the new token query
      K: (batch, kv_heads, kv_len, head_dim)  — cached keys
      V: (batch, kv_heads, kv_len, head_dim)  — cached values
    """
    q = torch.randn(batch_size, num_heads, 1, head_dim, device="cuda", dtype=dtype)
    k = torch.randn(batch_size, num_kv_heads, kv_len, head_dim, device="cuda", dtype=dtype)
    v = torch.randn(batch_size, num_kv_heads, kv_len, head_dim, device="cuda", dtype=dtype)
    return q, k, v


def bench_decode_kernel(backend_enum, q, k, v, warmup=3, runs=10):
    """
    Benchmark a single SDPA backend for decode-style attention
    (query length = 1, varying KV length).

    CRITICAL: torch.cuda.synchronize() is called BEFORE and AFTER every
    timing measurement to ensure we measure actual GPU execution time,
    not just kernel launch time.
    """
    # Warm-up (with sync to ensure kernels are compiled)
    # Warm-up (with sync to ensure kernels are compiled)
    with sdpa_kernel(backend_enum):
        for _ in range(warmup):
            _ = F.scaled_dot_product_attention(q, k, v, is_causal=False)
    torch.cuda.synchronize()

    # Benchmark
    latencies = []
    peak_mems = []
    for _ in range(runs):
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()  # sync BEFORE measurement
        start = time.perf_counter()
        with sdpa_kernel(backend_enum):
            _ = F.scaled_dot_product_attention(q, k, v, is_causal=False)
        torch.cuda.synchronize()  # sync AFTER measurement
        lat = time.perf_counter() - start
        peak_mem = torch.cuda.max_memory_allocated() / (1024 ** 2)
        latencies.append(lat)
        peak_mems.append(peak_mem)

    return latencies, peak_mems


# ---------------------------------------------------------------------------
# End-to-end generation benchmarks with forced backend
# ---------------------------------------------------------------------------
def measure_ttft(model, inputs, backend_enum):
    """Measure time-to-first-token with proper GPU synchronization."""
    torch.cuda.synchronize()
    start = time.perf_counter()
    with torch.no_grad(), sdpa_kernel(backend_enum):
        model.generate(**inputs, max_new_tokens=1, do_sample=False)
    torch.cuda.synchronize()
    return time.perf_counter() - start


def run_generation(model, inputs, max_new_tokens, backend_enum):
    """Run full generation with proper GPU synchronization."""
    torch.cuda.synchronize()
    start = time.perf_counter()
    with torch.no_grad(), sdpa_kernel(backend_enum):
        output_ids = model.generate(
            **inputs, max_new_tokens=max_new_tokens, do_sample=False
        )
    torch.cuda.synchronize()
    latency = time.perf_counter() - start
    return output_ids, latency


def measure_decode_latency_per_token(model, inputs, max_new_tokens, backend_enum):
    """
    Generate tokens one at a time and measure per-step decode latency.
    This isolates the decode phase from prefill.

    Uses torch.cuda.synchronize() before and after EVERY step.
    """
    with torch.no_grad():
        # Prefill: generate first token to populate KV cache
        torch.cuda.synchronize()
        prefill_start = time.perf_counter()
        with sdpa_kernel(backend_enum):
            outputs = model.generate(**inputs, max_new_tokens=1, do_sample=False)
        torch.cuda.synchronize()
        prefill_time = time.perf_counter() - prefill_start

        # Decode: generate remaining tokens one at a time
        decode_latencies = []
        current_ids = outputs
        for step in range(max_new_tokens - 1):
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


def benchmark_e2e(
    model, tokenizer, backend_name, backend_enum,
    seq_len, batch_size, max_new_tokens, warmup_runs, benchmark_runs,
):
    """End-to-end generation benchmark with forced SDPA backend."""
    inputs = build_prompt(tokenizer, seq_len, batch_size)
    actual_input_len = inputs["input_ids"].shape[1]

    # Warm-up
    for _ in range(warmup_runs):
        with torch.no_grad(), sdpa_kernel(backend_enum):
            model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    torch.cuda.synchronize()

    # TTFT
    ttft_values = []
    for _ in range(benchmark_runs):
        ttft_values.append(measure_ttft(model, inputs, backend_enum))

    # Full generation
    latencies, tokens_list, peak_mems = [], [], []
    for _ in range(benchmark_runs):
        torch.cuda.reset_peak_memory_stats()
        output_ids, latency = run_generation(
            model, inputs, max_new_tokens, backend_enum
        )
        peak_mem = torch.cuda.max_memory_allocated() / (1024 ** 2)
        num_gen = output_ids.shape[1] - actual_input_len
        latencies.append(latency)
        tokens_list.append(num_gen * batch_size)
        peak_mems.append(peak_mem)

    med_latency = median(latencies)
    med_tokens = median(tokens_list)

    return {
        "backend": backend_name,
        "seq_len": seq_len,
        "actual_input_len": actual_input_len,
        "batch_size": batch_size,
        "max_new_tokens": max_new_tokens,
        "median_latency_s": round(med_latency, 4),
        "std_latency_s": round(stdev(latencies) if len(latencies) > 1 else 0, 4),
        "median_tokens_per_sec": round(
            med_tokens / med_latency if med_latency > 0 else 0, 2
        ),
        "median_ttft_s": round(median(ttft_values), 4),
        "median_peak_memory_mb": round(median(peak_mems), 2),
        "num_runs": benchmark_runs,
        "all_latencies": latencies,
        "all_ttft": ttft_values,
        "all_peak_memories": peak_mems,
    }


def safe_e2e(
    model, tokenizer, backend_name, backend_enum,
    seq_len, batch_size, max_new_tokens, warmup_runs, benchmark_runs, max_pos,
):
    """Run E2E benchmark with OOM and error handling."""
    if max_pos and (seq_len + max_new_tokens) > max_pos:
        print(f"    SKIP — seq({seq_len})+gen({max_new_tokens}) > max_pos({max_pos})")
        return None
    try:
        r = benchmark_e2e(
            model, tokenizer, backend_name, backend_enum,
            seq_len, batch_size, max_new_tokens, warmup_runs, benchmark_runs,
        )
        print(
            f"    lat={r['median_latency_s']:.4f}s  "
            f"tok/s={r['median_tokens_per_sec']:.2f}  "
            f"ttft={r['median_ttft_s']:.4f}s  "
            f"mem={r['median_peak_memory_mb']:.1f}MB"
        )
        return r
    except torch.cuda.OutOfMemoryError:
        print("    OOM — skipping")
        torch.cuda.empty_cache()
        gc.collect()
        return None
    except Exception as e:
        print(f"    ERROR — {e}")
        return None


def safe_kernel(backend_name, backend_enum, q, k, v, warmup, runs):
    """Run kernel benchmark with error handling."""
    try:
        lats, mems = bench_decode_kernel(
            backend_enum, q, k, v, warmup=warmup, runs=runs
        )
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


# ---------------------------------------------------------------------------
# CSV / printing
# ---------------------------------------------------------------------------
def write_csv(results, filepath, fields):
    """Write summary CSV."""
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(results)
    print(f"  -> Saved: {filepath}")


def write_raw_csv(results, filepath, fields):
    """Write per-run raw CSV for kernel results."""
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in results:
            for i in range(r["num_runs"]):
                row = {}
                for field in fields:
                    if field == "run":
                        row["run"] = i + 1
                    elif field == "latency_ms":
                        row["latency_ms"] = r["all_latencies_ms"][i]
                    elif field == "peak_memory_mb":
                        row["peak_memory_mb"] = round(r["all_peak_memories"][i], 2)
                    else:
                        row[field] = r.get(field, "")
                writer.writerow(row)
    print(f"  -> Saved: {filepath}")


def write_raw_e2e_csv(results, filepath):
    """Write per-run raw CSV for E2E results."""
    filepath.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "backend", "seq_len", "batch_size", "max_new_tokens",
        "run", "latency_s", "ttft_s", "peak_memory_mb",
    ]
    with open(filepath, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in results:
            for i in range(r["num_runs"]):
                writer.writerow({
                    "backend": r["backend"],
                    "seq_len": r["seq_len"],
                    "batch_size": r["batch_size"],
                    "max_new_tokens": r["max_new_tokens"],
                    "run": i + 1,
                    "latency_s": round(r["all_latencies"][i], 4),
                    "ttft_s": round(r["all_ttft"][i], 4),
                    "peak_memory_mb": round(r["all_peak_memories"][i], 2),
                })
    print(f"  -> Saved: {filepath}")


def print_table(results, title, cols):
    """Print a formatted results table to console."""
    header = "  ".join(f"{c:>16}" for c in cols)
    print(f"\n{'=' * len(header)}")
    print(title)
    print(f"{'=' * len(header)}")
    print(header)
    print("-" * len(header))
    for r in results:
        row = "  ".join(f"{str(r.get(c, '')):>16}" for c in cols)
        print(row)
    print("=" * len(header))


# ---------------------------------------------------------------------------
# Experiments
# ---------------------------------------------------------------------------
def run_exp1_decode_latency_vs_cache(args, tokenizer, results_dir):
    """
    Exp 1: Decode Latency vs KV-Cache Length.

    For each backend, prefill with varying prompt lengths, then measure
    per-token decode latency. Flash decoding should show better scaling
    as KV-cache grows because it parallelizes across the KV sequence.
    """
    print("\n" + "#" * 60)
    print("# EXPERIMENT 1: Decode Latency vs KV-Cache Length")
    print("#" * 60)

    model = load_model(args.model, "sdpa")
    max_pos = getattr(model.config, "max_position_embeddings", None)

    all_results = []
    decode_tokens = 16  # Generate 16 tokens per config to measure decode

    for cache_len in DECODE_CACHE_LENGTHS:
        if max_pos and (cache_len + decode_tokens) > max_pos:
            print(f"  SKIP — cache({cache_len})+gen({decode_tokens}) > max_pos({max_pos})")
            continue

        for bname, benum in BACKENDS.items():
            print(f"\n  backend={bname}, cache_len={cache_len}")
            try:
                inputs = build_prompt(tokenizer, cache_len, 1)

                # Warm-up
                for _ in range(args.warmup_runs):
                    with torch.no_grad(), sdpa_kernel(benum):
                        model.generate(**inputs, max_new_tokens=decode_tokens, do_sample=False)
                torch.cuda.synchronize()

                # Measure per-token decode latency
                all_decode_lats = []
                prefill_times = []
                for _ in range(args.benchmark_runs):
                    pfill, dlats = measure_decode_latency_per_token(
                        model, inputs, decode_tokens, benum
                    )
                    prefill_times.append(pfill)
                    all_decode_lats.append(dlats)

                # Compute median per-token decode time
                med_per_token_lats = []
                for step in range(len(all_decode_lats[0])):
                    step_lats = [run[step] for run in all_decode_lats]
                    med_per_token_lats.append(median(step_lats))

                avg_decode_lat = sum(med_per_token_lats) / len(med_per_token_lats) if med_per_token_lats else 0

                result = {
                    "backend": bname,
                    "cache_len": cache_len,
                    "decode_tokens": decode_tokens,
                    "median_prefill_s": round(median(prefill_times), 4),
                    "median_per_token_decode_ms": round(avg_decode_lat * 1000, 3),
                    "total_decode_s": round(sum(med_per_token_lats), 4),
                    "num_runs": args.benchmark_runs,
                }
                all_results.append(result)
                print(
                    f"    prefill={result['median_prefill_s']:.4f}s  "
                    f"decode/token={result['median_per_token_decode_ms']:.3f}ms  "
                    f"total_decode={result['total_decode_s']:.4f}s"
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

    if all_results:
        cols = ["backend", "cache_len", "median_prefill_s",
                "median_per_token_decode_ms", "total_decode_s"]
        print_table(all_results, "EXP 1: Decode Latency vs KV-Cache Length", cols)
        write_csv(all_results, results_dir / "exp1_decode_vs_cache.csv",
                  cols + ["decode_tokens", "num_runs"])
    return all_results


def run_exp2_decode_throughput(args, tokenizer, results_dir):
    """
    Exp 2: Decode Throughput Scaling.

    Fixed prompt length (512), vary generation length to measure
    decode-only throughput (subtracting prefill time).
    """
    print("\n" + "#" * 60)
    print("# EXPERIMENT 2: Decode Throughput Scaling")
    print("#" * 60)

    model = load_model(args.model, "sdpa")
    max_pos = getattr(model.config, "max_position_embeddings", None)

    all_results = []
    seq_len = 512

    for gen_len in GEN_LENGTHS:
        for bname, benum in BACKENDS.items():
            print(f"\n  backend={bname}, seq={seq_len}, gen={gen_len}")
            r = safe_e2e(
                model, tokenizer, bname, benum,
                seq_len, 1, gen_len,
                args.warmup_runs, args.benchmark_runs, max_pos,
            )
            if r:
                # Compute decode-only throughput
                decode_time = r["median_latency_s"] - r["median_ttft_s"]
                gen_tokens = gen_len  # approximate
                decode_tok_per_sec = gen_tokens / decode_time if decode_time > 0 else 0
                r["decode_time_s"] = round(decode_time, 4)
                r["decode_tokens_per_sec"] = round(decode_tok_per_sec, 2)
                all_results.append(r)

    del model
    torch.cuda.empty_cache()
    gc.collect()

    if all_results:
        cols = [
            "backend", "seq_len", "max_new_tokens",
            "median_latency_s", "median_ttft_s", "decode_time_s",
            "median_tokens_per_sec", "decode_tokens_per_sec",
        ]
        print_table(all_results, "EXP 2: Decode Throughput Scaling", cols)
        write_csv(all_results, results_dir / "exp2_decode_throughput.csv",
                  cols + ["median_peak_memory_mb", "num_runs"])
        write_raw_e2e_csv(all_results, results_dir / "exp2_decode_throughput_raw.csv")
    return all_results


def run_exp3_batch_decode(args, tokenizer, results_dir):
    """
    Exp 3: Batch Size Impact on Decode.

    Compare decode performance across batch sizes at fixed seq lengths.
    Flash decoding should better utilize GPU even at small batch sizes.
    """
    print("\n" + "#" * 60)
    print("# EXPERIMENT 3: Batch Size Impact on Decode")
    print("#" * 60)

    model = load_model(args.model, "sdpa")
    max_pos = getattr(model.config, "max_position_embeddings", None)

    all_results = []
    test_seq_lengths = [512, 1024]

    for seq_len in test_seq_lengths:
        for batch_size in args.batch_sizes:
            for bname, benum in BACKENDS.items():
                print(f"\n  backend={bname}, seq={seq_len}, batch={batch_size}")
                r = safe_e2e(
                    model, tokenizer, bname, benum,
                    seq_len, batch_size, args.max_new_tokens,
                    args.warmup_runs, args.benchmark_runs, max_pos,
                )
                if r:
                    decode_time = r["median_latency_s"] - r["median_ttft_s"]
                    r["decode_time_s"] = round(decode_time, 4)
                    r["decode_tokens_per_sec"] = round(
                        (args.max_new_tokens * batch_size) / decode_time
                        if decode_time > 0 else 0, 2
                    )
                    all_results.append(r)

    del model
    torch.cuda.empty_cache()
    gc.collect()

    if all_results:
        cols = [
            "backend", "seq_len", "batch_size",
            "median_latency_s", "median_ttft_s", "decode_time_s",
            "decode_tokens_per_sec", "median_peak_memory_mb",
        ]
        print_table(all_results, "EXP 3: Batch Size Impact on Decode", cols)
        write_csv(all_results, results_dir / "exp3_batch_decode.csv",
                  cols + ["max_new_tokens", "num_runs"])
        write_raw_e2e_csv(all_results, results_dir / "exp3_batch_decode_raw.csv")
    return all_results


def run_exp4_kernel_decode(args, tokenizer, results_dir):
    """
    Exp 4: Kernel-Level Decode Attention.

    Benchmarks raw F.scaled_dot_product_attention with:
      Q: (1, 16, 1, 128)    — single token query
      K: (1, 16, kv_len, 128) — varying KV cache sizes
      V: (1, 16, kv_len, 128)

    This isolates the EXACT decode attention kernel cost, removing
    all model overhead.
    """
    print("\n" + "#" * 60)
    print("# EXPERIMENT 4: Kernel-Level Decode Attention")
    print("#" * 60)

    all_results = []
    batch_size = 1

    for kv_len in KERNEL_KV_LENGTHS:
        q, k, v = make_decode_qkv(batch_size, NUM_HEADS, NUM_KV_HEADS, kv_len, HEAD_DIM)
        for bname, benum in BACKENDS.items():
            print(f"\n  backend={bname}, kv_len={kv_len}, q_len=1")
            r = safe_kernel(bname, benum, q, k, v,
                            warmup=args.warmup_runs, runs=args.benchmark_runs)
            if r:
                r["kv_len"] = kv_len
                r["q_len"] = 1
                r["num_heads"] = NUM_HEADS
                r["num_kv_heads"] = NUM_KV_HEADS
                r["head_dim"] = HEAD_DIM
                r["batch_size"] = batch_size
                all_results.append(r)
        del q, k, v
        torch.cuda.empty_cache()

    if all_results:
        cols = [
            "backend", "kv_len", "q_len",
            "median_latency_ms", "std_latency_ms", "median_peak_memory_mb",
        ]
        print_table(all_results, "EXP 4: Kernel-Level Decode Attention", cols)
        write_csv(all_results, results_dir / "exp4_kernel_decode.csv",
                  cols + ["num_heads", "num_kv_heads", "head_dim", "num_runs"])
        write_raw_csv(
            all_results,
            results_dir / "exp4_kernel_decode_raw.csv",
            ["backend", "kv_len", "run", "latency_ms", "peak_memory_mb"],
        )
    return all_results


def run_exp5_e2e_memory(args, tokenizer, results_dir):
    """
    Exp 5: End-to-End Comparison + Memory Analysis.

    Full inference comparison with:
    - Prefill/decode latency breakdown
    - KV-cache vs attention matrix memory analysis
    """
    print("\n" + "#" * 60)
    print("# EXPERIMENT 5: End-to-End + Memory Analysis")
    print("#" * 60)

    # --- Part A: E2E comparison ---
    print("\n  --- Part A: End-to-end comparison ---")
    model = load_model(args.model, "sdpa")
    max_pos = getattr(model.config, "max_position_embeddings", None)

    e2e_results = []
    for seq_len in args.seq_lengths:
        for bname, benum in BACKENDS.items():
            print(f"\n  [e2e] backend={bname}, seq={seq_len}, batch=1")
            r = safe_e2e(
                model, tokenizer, bname, benum,
                seq_len, 1, args.max_new_tokens,
                args.warmup_runs, args.benchmark_runs, max_pos,
            )
            if r:
                # Decompose into prefill and decode
                decode_time = r["median_latency_s"] - r["median_ttft_s"]
                r["decode_time_s"] = round(decode_time, 4)
                r["prefill_pct"] = round(
                    (r["median_ttft_s"] / r["median_latency_s"]) * 100
                    if r["median_latency_s"] > 0 else 0, 1
                )
                r["decode_pct"] = round(100 - r["prefill_pct"], 1)
                e2e_results.append(r)

    del model
    torch.cuda.empty_cache()
    gc.collect()

    # --- Part B: KV-Cache vs Attention Matrix Memory ---
    print("\n  --- Part B: KV-Cache vs Attention Matrix Memory ---")
    memory_analysis = []
    mem_seq_lengths = [256, 512, 1024, 1536, 2048]
    batch_size = 1

    for seq_len in mem_seq_lengths:
        # Kernel-level memory (attention matrix cost)
        q, k, v = make_decode_qkv(batch_size, NUM_HEADS, NUM_KV_HEADS, seq_len, HEAD_DIM)
        for bname, benum in BACKENDS.items():
            print(f"\n  [memory] backend={bname}, kv_len={seq_len}")
            r = safe_kernel(bname, benum, q, k, v,
                            warmup=args.warmup_runs, runs=args.benchmark_runs)
            if r:
                r["kv_len"] = seq_len
                # Theoretical KV-cache size: 2 * num_layers * num_heads * seq_len * head_dim * 2 bytes
                # Qwen3-0.6B has 28 layers
                num_layers = 28
                kv_cache_size_mb = (
                    2 * num_layers * NUM_HEADS * seq_len * HEAD_DIM * 2
                ) / (1024 ** 2)
                r["theoretical_kv_cache_mb"] = round(kv_cache_size_mb, 2)
                # Theoretical math attention matrix: num_heads * 1 * seq_len * 4 bytes (float32 for softmax)
                # Note: Softmax is computed per query head.
                attn_matrix_mb = (NUM_HEADS * 1 * seq_len * 4) / (1024 ** 2)
                r["theoretical_attn_matrix_mb"] = round(attn_matrix_mb, 4)
                r["level"] = "kernel_decode"
                memory_analysis.append(r)
        del q, k, v
        torch.cuda.empty_cache()

    # --- Output ---
    if e2e_results:
        cols_e2e = [
            "backend", "seq_len", "median_latency_s", "median_ttft_s",
            "decode_time_s", "prefill_pct", "decode_pct",
            "median_tokens_per_sec", "median_peak_memory_mb",
        ]
        print_table(e2e_results, "EXP 5A: End-to-End Comparison", cols_e2e)
        write_csv(e2e_results, results_dir / "exp5_e2e_comparison.csv",
                  cols_e2e + ["std_latency_s", "num_runs"])
        write_raw_e2e_csv(e2e_results, results_dir / "exp5_e2e_comparison_raw.csv")

    if memory_analysis:
        cols_mem = [
            "backend", "kv_len", "median_latency_ms", "median_peak_memory_mb",
            "theoretical_kv_cache_mb", "theoretical_attn_matrix_mb",
        ]
        print_table(memory_analysis, "EXP 5B: Memory Analysis (KV-Cache vs Attn Matrix)", cols_mem)
        write_csv(memory_analysis, results_dir / "exp5_memory_analysis.csv",
                  cols_mem + ["num_runs"])

    return e2e_results + memory_analysis


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(
        description="Flash Decoding vs Normal Decoding benchmark suite."
    )
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--seq_lengths", nargs="+", type=int, default=DEFAULT_SEQ_LENGTHS)
    p.add_argument("--batch_sizes", nargs="+", type=int, default=DEFAULT_BATCH_SIZES)
    p.add_argument("--max_new_tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS)
    p.add_argument("--warmup_runs", type=int, default=DEFAULT_WARMUP_RUNS)
    p.add_argument("--benchmark_runs", type=int, default=DEFAULT_BENCHMARK_RUNS)
    p.add_argument("--results_dir", type=str, default=None)
    p.add_argument(
        "--experiments", nargs="+", type=int, default=[1, 2, 3, 4, 5, 6],
        choices=[1, 2, 3, 4, 5, 6],
        help="Which experiments to run (1-6, default: all)",
    )
    return p.parse_args()

def run_exp6_precision_analysis(args, tokenizer, results_dir):
    """
    Exp 6: Memory-Bound vs Compute-Bound (Precision Analysis).
    Compares FP16 vs FP32 to diagnose bottlenecks.
    """
    print("\n" + "#" * 60)
    print("# EXPERIMENT 6: Precision Analysis (Compute vs Memory Bound)")
    print("#" * 60)

    results_a = [] # Part A: Kernel
    results_b = [] # Part B: Model

    # Part A: Kernel-Level
    print("\n  --- Part A: Kernel-level FP16 vs FP32 ---")
    batch_size = 1
    # Test only Math and Mem-Efficient (Flash doesn't support FP32)
    test_backends = {k: v for k, v in BACKENDS.items() if k in ["math", "mem_efficient"]}

    for kv_len in KERNEL_KV_LENGTHS:
        for dtype_name, dtype in [("fp16", torch.float16), ("fp32", torch.float32)]:
            q, k, v = make_decode_qkv(batch_size, NUM_HEADS, NUM_KV_HEADS, kv_len, HEAD_DIM, dtype=dtype)
            for bname, benum in test_backends.items():
                print(f"    backend={bname}, kv_len={kv_len}, dtype={dtype_name}")
                r = safe_kernel(bname, benum, q, k, v, warmup=args.warmup_runs, runs=args.benchmark_runs)
                if r:
                    r["kv_len"] = kv_len
                    r["dtype"] = dtype_name
                    results_a.append(r)
            del q, k, v
            torch.cuda.empty_cache()

    # Diagnosis for Part A
    diag_a = []
    for kv_len in KERNEL_KV_LENGTHS:
        for bname in test_backends:
            fp16 = next((r for r in results_a if r["backend"] == bname and r["kv_len"] == kv_len and r["dtype"] == "fp16"), None)
            fp32 = next((r for r in results_a if r["backend"] == bname and r["kv_len"] == kv_len and r["dtype"] == "fp32"), None)
            if fp16 and fp32:
                ratio = fp32["median_latency_ms"] / fp16["median_latency_ms"]
                diagnosis = "MEMORY-BOUND" if ratio > 1.5 else "COMPUTE-BOUND"
                diag_a.append({
                    "backend": bname, "kv_len": kv_len,
                    "fp16_ms": fp16["median_latency_ms"], "fp32_ms": fp32["median_latency_ms"],
                    "ratio": round(ratio, 2), "diagnosis": diagnosis
                })

    # Part B: Model-Level (Math only for simplicity)
    print("\n  --- Part B: Model-level FP16 vs FP32 (Math backend) ---")
    for dtype_name, dtype in [("fp16", torch.float16), ("fp32", torch.float32)]:
        print(f"    Loading model: {args.model}, dtype={dtype_name} ...")
        model = AutoModelForCausalLM.from_pretrained(
            args.model, torch_dtype=dtype, attn_implementation="sdpa"
        ).to("cuda")

        for cache_len in args.seq_lengths:
            print(f"      backend=math, cache_len={cache_len}")
            try:
                inputs = build_prompt(tokenizer, cache_len, 1)
                # Warm-up
                for _ in range(args.warmup_runs):
                    with torch.no_grad(), sdpa_kernel(SDPBackend.MATH):
                        model.generate(**inputs, max_new_tokens=16, do_sample=False)
                torch.cuda.synchronize()

                # Measure per-token decode latency
                all_decode_lats = []
                for _ in range(args.benchmark_runs):
                    _, dlats = measure_decode_latency_per_token(
                        model, inputs, 16, SDPBackend.MATH
                    )
                    all_decode_lats.append(dlats)

                # Compute median per-token decode time
                med_per_token_lats = []
                for step in range(len(all_decode_lats[0])):
                    step_lats = [run[step] for run in all_decode_lats]
                    med_per_token_lats.append(median(step_lats))

                avg_decode_lat = sum(med_per_token_lats) / len(med_per_token_lats) if med_per_token_lats else 0
                results_b.append({
                    "dtype": dtype_name,
                    "cache_len": cache_len,
                    "median_per_token_decode_ms": round(avg_decode_lat * 1000, 3)
                })
            except Exception as e:
                print(f"      ERROR — {e}")

        del model
        torch.cuda.empty_cache()
        gc.collect()

    # Diagnosis for Part B
    diag_b = []
    for cache_len in args.seq_lengths:
        fp16 = next((r for r in results_b if r["dtype"] == "fp16" and r["cache_len"] == cache_len), None)
        fp32 = next((r for r in results_b if r["dtype"] == "fp32" and r["cache_len"] == cache_len), None)
        if fp16 and fp32:
            ratio = fp32["median_per_token_decode_ms"] / fp16["median_per_token_decode_ms"]
            diagnosis = "MEMORY-BOUND" if ratio > 1.5 else "COMPUTE-BOUND"
            diag_b.append({
                "cache_len": cache_len,
                "fp16_ms": fp16["median_per_token_decode_ms"], "fp32_ms": fp32["median_per_token_decode_ms"],
                "ratio": round(ratio, 2), "diagnosis": diagnosis
            })

    # Output
    if diag_a:
        print_table(diag_a, "EXP 6A: Kernel Diagnosis", ["backend", "kv_len", "ratio", "diagnosis"])
        write_csv(results_a, results_dir / "precision_exp_a_kernel.csv", ["backend", "kv_len", "dtype", "median_latency_ms"])
        write_csv(diag_a, results_dir / "precision_exp_a_diagnosis.csv", ["backend", "kv_len", "ratio", "diagnosis"])

    if diag_b:
        print_table(diag_b, "EXP 6B: Model Diagnosis", ["cache_len", "ratio", "diagnosis"])
        write_csv(results_b, results_dir / "precision_exp_b_model.csv", ["dtype", "cache_len", "median_per_token_decode_ms"])
        write_csv(diag_b, results_dir / "precision_exp_b_diagnosis.csv", ["cache_len", "ratio", "diagnosis"])

    # Final combined JSON for Exp 6
    with open(results_dir / "precision_analysis.json", "w") as f:
        json.dump({"kernel": diag_a, "model": diag_b}, f, indent=2)

    return results_a + results_b


def main():
    args = parse_args()
    project_root = Path(__file__).resolve().parent
    results_dir = Path(args.results_dir) if args.results_dir else project_root / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    if not torch.cuda.is_available():
        print("ERROR: CUDA not available.")
        sys.exit(1)

    gpu_info = get_gpu_info()
    print("=" * 60)
    print("GPU INFORMATION")
    print("=" * 60)
    for k, v in gpu_info.items():
        print(f"  {k}: {v}")
    print("=" * 60)

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    exp_map = {
        1: ("Decode Latency vs KV-Cache Length", run_exp1_decode_latency_vs_cache),
        2: ("Decode Throughput Scaling", run_exp2_decode_throughput),
        3: ("Batch Size Impact on Decode", run_exp3_batch_decode),
        4: ("Kernel-Level Decode Attention", run_exp4_kernel_decode),
        5: ("End-to-End + Memory Analysis", run_exp5_e2e_memory),
        6: ("Precision Analysis (Compute vs Memory Bound)", run_exp6_precision_analysis),
    }

    print(f"\nRunning experiments: {args.experiments}")
    total_results = {}

    for exp_num in sorted(args.experiments):
        name, runner = exp_map[exp_num]
        print(f"\n{'*' * 60}")
        print(f"* Starting Experiment {exp_num}: {name}")
        print(f"{'*' * 60}")
        total_results[exp_num] = runner(args, tokenizer, results_dir)

    # Save GPU info
    gpu_path = results_dir / "gpu_info.txt"
    with open(gpu_path, "w") as f:
        for k, v in gpu_info.items():
            f.write(f"{k}: {v}\n")

    # Summary
    print("\n" + "=" * 60)
    print("ALL FLASH DECODING EXPERIMENTS COMPLETE")
    print("=" * 60)
    for exp_num in sorted(args.experiments):
        name, _ = exp_map[exp_num]
        count = len(total_results.get(exp_num, []))
        print(f"  Exp {exp_num} ({name}): {count} configs benchmarked")
    print(f"\nResults directory: {results_dir}")
    print("=" * 60)

    # Save all results as JSON for report generation
    json_results = {}
    for exp_num, results in total_results.items():
        if results:
            cleaned = []
            for r in results:
                c = {k: v for k, v in r.items()
                     if not k.startswith("all_")}
                cleaned.append(c)
            json_results[str(exp_num)] = cleaned
    json_path = results_dir / "all_results.json"
    with open(json_path, "w") as f:
        json.dump(json_results, f, indent=2)
    print(f"  -> Saved: {json_path}")


if __name__ == "__main__":
    main()
