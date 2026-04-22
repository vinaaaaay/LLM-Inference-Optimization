"""
run_flash_attention.py — FlashAttention Deep-Dive Study (Person A)

Model: Qwen3-0.6B (GQA: 16 query heads / 8 KV heads, head_dim=128, 28 layers)

Isolates and compares the three PyTorch SDPA backends:
  - FLASH_ATTENTION  (FlashAttention v2 kernel)
  - EFFICIENT_ATTENTION (xFormers memory-efficient kernel)
  - MATH (vanilla PyTorch matmul + softmax)

Runs 5 experiments:

  Exp 1 (Backend Sweep):
      backends x seq_lengths x batch_sizes — end-to-end model generation.
      -> Direct comparison of each backend at inference level.

  Exp 2 (Kernel-Level Attention Benchmark):
      backends x seq_lengths — raw F.scaled_dot_product_attention calls.
      -> Isolates attention kernel cost, removes all other model overhead.
      -> Uses GQA-aware shapes: Q has num_query_heads, K/V have num_kv_heads.

  Exp 3 (Sequence Length Scaling):
      backends x fine-grained seq_lengths — kernel only.
      -> Shows O(n^2) math vs sub-quadratic flash/efficient scaling.

  Exp 4 (Head Dimension Scaling):
      backends x head_dims — kernel only.
      -> How performance changes with different head sizes (32, 64, 128).

  Exp 5 (Memory Scaling):
      backends x seq_lengths — peak memory for kernel + full model.
      -> Quantifies memory savings of flash/efficient vs math.

Usage:
    python3 inference/run_flash_attention.py                        # all experiments
    python3 inference/run_flash_attention.py --experiments 1 2      # specific ones
    python3 inference/run_flash_attention.py --experiments 2 --benchmark_runs 3
"""

import argparse
import csv
import gc
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
DEFAULT_MODEL = "/home/administrator/bin/GPU Analysis/Qwen3-0.6B"
DEFAULT_SEQ_LENGTHS = [256, 512, 1024, 1800]
DEFAULT_BATCH_SIZES = [1, 2, 4]
DEFAULT_MAX_NEW_TOKENS = 128
DEFAULT_WARMUP_RUNS = 3
DEFAULT_BENCHMARK_RUNS = 5

BACKENDS = {
    "flash": SDPBackend.FLASH_ATTENTION,
    "mem_efficient": SDPBackend.EFFICIENT_ATTENTION,
    "math": SDPBackend.MATH,
}

# Exp 2: kernel-level seq lengths
KERNEL_SEQ_LENGTHS = [256, 512, 1024, 2048]

# Exp 3: fine-grained scaling
SCALING_SEQ_LENGTHS = [128, 256, 512, 768, 1024, 1536, 2048]

# Exp 4: head dimensions
HEAD_DIMS = [32, 64, 128]

# Model attention config for Qwen3-0.6B (GQA: 16 query heads, 8 KV heads, head_dim=128)
NUM_QUERY_HEADS = 16
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
        "compute_capability": ".".join(str(x) for x in torch.cuda.get_device_capability(0)),
    }


def build_prompt(tokenizer, seq_len: int, batch_size: int):
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
    print(f"  Loading model: attn='{attn_impl}', dtype=float16 ...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.float16,
        attn_implementation=attn_impl,
    ).to("cuda")
    model.eval()
    return model


# ---------------------------------------------------------------------------
# Kernel-level benchmark helpers
# ---------------------------------------------------------------------------
def make_qkv(batch_size, num_query_heads, num_kv_heads, seq_len, head_dim, dtype=torch.float16):
    """Create random Q, K, V tensors on GPU with GQA-aware shapes.

    Q shape: (batch, num_query_heads, seq_len, head_dim)
    K/V shape: (batch, num_kv_heads, seq_len, head_dim)

    For SDPA, K/V are broadcast-expanded to match Q's head count internally.
    """
    q = torch.randn((batch_size, num_query_heads, seq_len, head_dim), device="cuda", dtype=dtype)
    k = torch.randn((batch_size, num_kv_heads, seq_len, head_dim), device="cuda", dtype=dtype)
    v = torch.randn((batch_size, num_kv_heads, seq_len, head_dim), device="cuda", dtype=dtype)
    return q, k, v


def bench_kernel(backend_enum, q, k, v, warmup=3, runs=10, is_causal=True):
    """Benchmark a single SDPA backend at the kernel level.

    Supports GQA: if Q and K/V have different head counts, uses enable_gqa=True.
    """
    use_gqa = q.shape[1] != k.shape[1]
    sdpa_kwargs = {"is_causal": is_causal}
    if use_gqa:
        sdpa_kwargs["enable_gqa"] = True

    # Warm-up
    with sdpa_kernel(backend_enum):
        for _ in range(warmup):
            _ = F.scaled_dot_product_attention(q, k, v, **sdpa_kwargs)
    torch.cuda.synchronize()

    # Benchmark
    latencies = []
    peak_mems = []
    for _ in range(runs):
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        start = time.perf_counter()
        with sdpa_kernel(backend_enum):
            _ = F.scaled_dot_product_attention(q, k, v, **sdpa_kwargs)
        torch.cuda.synchronize()
        lat = time.perf_counter() - start
        peak_mem = torch.cuda.max_memory_allocated() / (1024 ** 2)
        latencies.append(lat)
        peak_mems.append(peak_mem)

    return latencies, peak_mems


# ---------------------------------------------------------------------------
# End-to-end generation benchmark with forced backend
# ---------------------------------------------------------------------------
def measure_ttft(model, inputs, backend_enum):
    torch.cuda.synchronize()
    start = time.perf_counter()
    with torch.no_grad(), sdpa_kernel(backend_enum):
        model.generate(**inputs, max_new_tokens=1, do_sample=False)
    torch.cuda.synchronize()
    return time.perf_counter() - start


def run_generation(model, inputs, max_new_tokens, backend_enum):
    torch.cuda.synchronize()
    start = time.perf_counter()
    with torch.no_grad(), sdpa_kernel(backend_enum):
        output_ids = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    torch.cuda.synchronize()
    latency = time.perf_counter() - start
    return output_ids, latency


def benchmark_e2e(model, tokenizer, backend_name, backend_enum,
                  seq_len, batch_size, max_new_tokens, warmup_runs, benchmark_runs):
    """End-to-end generation benchmark with a forced SDPA backend."""
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
        output_ids, latency = run_generation(model, inputs, max_new_tokens, backend_enum)
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
        "median_tokens_per_sec": round(med_tokens / med_latency if med_latency > 0 else 0, 2),
        "median_ttft_s": round(median(ttft_values), 4),
        "median_peak_memory_mb": round(median(peak_mems), 2),
        "num_runs": benchmark_runs,
        "all_latencies": latencies,
        "all_ttft": ttft_values,
        "all_peak_memories": peak_mems,
    }


def safe_e2e(model, tokenizer, backend_name, backend_enum,
             seq_len, batch_size, max_new_tokens, warmup_runs, benchmark_runs, max_pos):
    if max_pos and (seq_len + max_new_tokens) > max_pos:
        print(f"    SKIP — seq({seq_len})+gen({max_new_tokens}) > max_pos({max_pos})")
        return None
    try:
        r = benchmark_e2e(model, tokenizer, backend_name, backend_enum,
                          seq_len, batch_size, max_new_tokens, warmup_runs, benchmark_runs)
        print(f"    lat={r['median_latency_s']:.4f}s  tok/s={r['median_tokens_per_sec']:.2f}  "
              f"ttft={r['median_ttft_s']:.4f}s  mem={r['median_peak_memory_mb']:.1f}MB")
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
    try:
        lats, mems = bench_kernel(backend_enum, q, k, v, warmup=warmup, runs=runs)
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
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(results)
    print(f"  -> Saved: {filepath}")


def write_raw_e2e(results, filepath):
    filepath.parent.mkdir(parents=True, exist_ok=True)
    fields = ["backend", "seq_len", "batch_size", "max_new_tokens",
              "run", "latency_s", "ttft_s", "peak_memory_mb"]
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


def write_raw_kernel(results, filepath, extra_fields=None):
    filepath.parent.mkdir(parents=True, exist_ok=True)
    base = ["backend"]
    if extra_fields:
        base += extra_fields
    base += ["run", "latency_ms", "peak_memory_mb"]
    with open(filepath, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=base)
        writer.writeheader()
        for r in results:
            for i in range(r["num_runs"]):
                row = {"backend": r["backend"], "run": i + 1,
                       "latency_ms": r["all_latencies_ms"][i],
                       "peak_memory_mb": round(r["all_peak_memories"][i], 2)}
                if extra_fields:
                    for ef in extra_fields:
                        row[ef] = r.get(ef, "")
                writer.writerow(row)
    print(f"  -> Saved: {filepath}")


def print_table(results, title, cols):
    header = "  ".join(f"{c:>14}" for c in cols)
    print(f"\n{'=' * len(header)}")
    print(title)
    print(f"{'=' * len(header)}")
    print(header)
    print("-" * len(header))
    for r in results:
        row = "  ".join(f"{str(r.get(c, '')):>14}" for c in cols)
        print(row)
    print("=" * len(header))


# ---------------------------------------------------------------------------
# Experiments
# ---------------------------------------------------------------------------
def run_exp1_backend_sweep(args, tokenizer, results_dir):
    """Exp 1: backends x seq_lengths x batch_sizes — end-to-end."""
    print("\n" + "#" * 60)
    print("# EXPERIMENT 1: Backend Sweep (end-to-end generation)")
    print("#" * 60)

    model = load_model(args.model, "sdpa")
    max_pos = getattr(model.config, "max_position_embeddings", None)
    if max_pos:
        print(f"  max_position_embeddings: {max_pos}")

    all_results = []
    combos = [(b, s, bs) for b in BACKENDS for s in args.seq_lengths for bs in args.batch_sizes]
    total = len(combos)

    for idx, (bname, seq_len, batch_size) in enumerate(combos, 1):
        benum = BACKENDS[bname]
        print(f"\n  [{idx}/{total}] backend={bname}, seq={seq_len}, batch={batch_size}")
        r = safe_e2e(model, tokenizer, bname, benum,
                     seq_len, batch_size, args.max_new_tokens,
                     args.warmup_runs, args.benchmark_runs, max_pos)
        if r:
            all_results.append(r)

    del model
    torch.cuda.empty_cache()
    gc.collect()

    if all_results:
        cols = ["backend", "seq_len", "batch_size", "median_latency_s",
                "median_tokens_per_sec", "median_ttft_s", "median_peak_memory_mb"]
        print_table(all_results, "EXP 1: Backend Sweep (E2E)", cols)
        write_csv(all_results, results_dir / "flash_exp1_backend_sweep.csv", cols + ["std_latency_s", "num_runs"])
        write_raw_e2e(all_results, results_dir / "flash_exp1_backend_sweep_raw.csv")
    return all_results


def run_exp2_kernel_benchmark(args, tokenizer, results_dir):
    """Exp 2: kernel-level attention benchmark across seq lengths."""
    print("\n" + "#" * 60)
    print("# EXPERIMENT 2: Kernel-Level Attention Benchmark")
    print("#" * 60)

    all_results = []
    batch_size = 1

    for seq_len in KERNEL_SEQ_LENGTHS:
        q, k, v = make_qkv(batch_size, NUM_QUERY_HEADS, NUM_KV_HEADS, seq_len, HEAD_DIM)
        for bname, benum in BACKENDS.items():
            print(f"\n  backend={bname}, seq={seq_len}, q_heads={NUM_QUERY_HEADS}, kv_heads={NUM_KV_HEADS}, d={HEAD_DIM}")
            r = safe_kernel(bname, benum, q, k, v,
                            warmup=args.warmup_runs, runs=args.benchmark_runs)
            if r:
                r["seq_len"] = seq_len
                r["num_query_heads"] = NUM_QUERY_HEADS
                r["num_kv_heads"] = NUM_KV_HEADS
                r["head_dim"] = HEAD_DIM
                r["batch_size"] = batch_size
                all_results.append(r)
        del q, k, v
        torch.cuda.empty_cache()

    if all_results:
        cols = ["backend", "seq_len", "num_query_heads", "num_kv_heads", "head_dim",
                "median_latency_ms", "std_latency_ms", "median_peak_memory_mb"]
        print_table(all_results, "EXP 2: Kernel-Level Attention", cols)
        write_csv(all_results, results_dir / "flash_exp2_kernel.csv", cols + ["batch_size", "num_runs"])
        write_raw_kernel(all_results, results_dir / "flash_exp2_kernel_raw.csv",
                         extra_fields=["seq_len", "num_query_heads", "num_kv_heads", "head_dim"])
    return all_results


def run_exp3_seq_scaling(args, tokenizer, results_dir):
    """Exp 3: fine-grained sequence length scaling — kernel only."""
    print("\n" + "#" * 60)
    print("# EXPERIMENT 3: Sequence Length Scaling (kernel-level)")
    print("#" * 60)

    all_results = []
    batch_size = 1

    for seq_len in SCALING_SEQ_LENGTHS:
        q, k, v = make_qkv(batch_size, NUM_QUERY_HEADS, NUM_KV_HEADS, seq_len, HEAD_DIM)
        for bname, benum in BACKENDS.items():
            print(f"\n  backend={bname}, seq={seq_len}")
            r = safe_kernel(bname, benum, q, k, v,
                            warmup=args.warmup_runs, runs=args.benchmark_runs)
            if r:
                r["seq_len"] = seq_len
                all_results.append(r)
        del q, k, v
        torch.cuda.empty_cache()

    if all_results:
        cols = ["backend", "seq_len", "median_latency_ms", "std_latency_ms", "median_peak_memory_mb"]
        print_table(all_results, "EXP 3: Sequence Length Scaling", cols)
        write_csv(all_results, results_dir / "flash_exp3_seq_scaling.csv", cols + ["num_runs"])
        write_raw_kernel(all_results, results_dir / "flash_exp3_seq_scaling_raw.csv",
                         extra_fields=["seq_len"])
    return all_results


def run_exp4_head_dim(args, tokenizer, results_dir):
    """Exp 4: head dimension scaling — kernel only."""
    print("\n" + "#" * 60)
    print("# EXPERIMENT 4: Head Dimension Scaling (kernel-level)")
    print("#" * 60)

    all_results = []
    batch_size = 1
    seq_len = 1024

    for head_dim in HEAD_DIMS:
        q, k, v = make_qkv(batch_size, NUM_QUERY_HEADS, NUM_KV_HEADS, seq_len, head_dim)
        for bname, benum in BACKENDS.items():
            print(f"\n  backend={bname}, seq={seq_len}, head_dim={head_dim}")
            r = safe_kernel(bname, benum, q, k, v,
                            warmup=args.warmup_runs, runs=args.benchmark_runs)
            if r:
                r["seq_len"] = seq_len
                r["head_dim"] = head_dim
                r["num_query_heads"] = NUM_QUERY_HEADS
                r["num_kv_heads"] = NUM_KV_HEADS
                all_results.append(r)
        del q, k, v
        torch.cuda.empty_cache()

    if all_results:
        cols = ["backend", "head_dim", "median_latency_ms", "std_latency_ms", "median_peak_memory_mb"]
        print_table(all_results, "EXP 4: Head Dimension Scaling", cols)
        write_csv(all_results, results_dir / "flash_exp4_head_dim.csv",
                  cols + ["seq_len", "num_query_heads", "num_kv_heads", "num_runs"])
        write_raw_kernel(all_results, results_dir / "flash_exp4_head_dim_raw.csv",
                         extra_fields=["head_dim"])
    return all_results


def run_exp5_memory_scaling(args, tokenizer, results_dir):
    """Exp 5: memory usage scaling — kernel + end-to-end."""
    print("\n" + "#" * 60)
    print("# EXPERIMENT 5: Memory Scaling (kernel + end-to-end)")
    print("#" * 60)

    # --- Part A: Kernel-level memory ---
    print("\n  --- Part A: Kernel-level memory ---")
    kernel_results = []
    batch_size = 1
    mem_seq_lengths = [256, 512, 1024, 1536, 2048]

    for seq_len in mem_seq_lengths:
        q, k, v = make_qkv(batch_size, NUM_QUERY_HEADS, NUM_KV_HEADS, seq_len, HEAD_DIM)
        for bname, benum in BACKENDS.items():
            print(f"\n  [kernel] backend={bname}, seq={seq_len}")
            torch.cuda.reset_peak_memory_stats()
            r = safe_kernel(bname, benum, q, k, v,
                            warmup=args.warmup_runs, runs=args.benchmark_runs)
            if r:
                r["seq_len"] = seq_len
                r["level"] = "kernel"
                kernel_results.append(r)
        del q, k, v
        torch.cuda.empty_cache()

    # --- Part B: End-to-end memory ---
    print("\n  --- Part B: End-to-end model memory ---")
    e2e_results = []
    model = load_model(args.model, "sdpa")
    max_pos = getattr(model.config, "max_position_embeddings", None)

    for seq_len in args.seq_lengths:
        for bname, benum in BACKENDS.items():
            print(f"\n  [e2e] backend={bname}, seq={seq_len}")
            r = safe_e2e(model, tokenizer, bname, benum,
                         seq_len, 1, args.max_new_tokens,
                         args.warmup_runs, args.benchmark_runs, max_pos)
            if r:
                r["level"] = "e2e"
                e2e_results.append(r)

    del model
    torch.cuda.empty_cache()
    gc.collect()

    # --- Output ---
    if kernel_results:
        cols_k = ["backend", "seq_len", "level", "median_latency_ms", "median_peak_memory_mb"]
        print_table(kernel_results, "EXP 5A: Kernel Memory Scaling", cols_k)
        write_csv(kernel_results, results_dir / "flash_exp5_memory_kernel.csv", cols_k + ["num_runs"])

    if e2e_results:
        cols_e = ["backend", "seq_len", "level", "median_latency_s",
                  "median_tokens_per_sec", "median_peak_memory_mb"]
        print_table(e2e_results, "EXP 5B: E2E Memory Scaling", cols_e)
        write_csv(e2e_results, results_dir / "flash_exp5_memory_e2e.csv",
                  cols_e + ["median_ttft_s", "num_runs"])

    return kernel_results + e2e_results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="FlashAttention deep-dive benchmark suite.")
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--seq_lengths", nargs="+", type=int, default=DEFAULT_SEQ_LENGTHS)
    p.add_argument("--batch_sizes", nargs="+", type=int, default=DEFAULT_BATCH_SIZES)
    p.add_argument("--max_new_tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS)
    p.add_argument("--warmup_runs", type=int, default=DEFAULT_WARMUP_RUNS)
    p.add_argument("--benchmark_runs", type=int, default=DEFAULT_BENCHMARK_RUNS)
    p.add_argument("--results_dir", type=str, default=None)
    p.add_argument("--experiments", nargs="+", type=int, default=[1, 2, 3, 4, 5],
                   choices=[1, 2, 3, 4, 5])
    return p.parse_args()


def main():
    args = parse_args()
    project_root = Path(__file__).resolve().parent.parent
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
        1: ("Backend Sweep (E2E)", run_exp1_backend_sweep),
        2: ("Kernel-Level Benchmark", run_exp2_kernel_benchmark),
        3: ("Sequence Length Scaling", run_exp3_seq_scaling),
        4: ("Head Dimension Scaling", run_exp4_head_dim),
        5: ("Memory Scaling", run_exp5_memory_scaling),
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
    print("ALL FLASH ATTENTION EXPERIMENTS COMPLETE")
    print("=" * 60)
    for exp_num in sorted(args.experiments):
        name, _ = exp_map[exp_num]
        count = len(total_results.get(exp_num, []))
        print(f"  Exp {exp_num} ({name}): {count} configs benchmarked")
    print(f"\nResults directory: {results_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
