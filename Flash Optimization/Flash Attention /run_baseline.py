"""
run_baseline.py — Baseline Inference Pipeline (Person A)

Model: Qwen3-0.6B (GQA: 16 query heads / 8 KV heads, head_dim=128, 28 layers)

Runs multiple experiment suites in a single invocation:

  Experiment 1 (Core Sweep):
      Sweep implementations x seq_lengths x batch_sizes
      -> Shows how attention backends compare across scale.

  Experiment 2 (Generation Length Scaling):
      Sweep implementations x max_new_tokens (fixed seq_len=512, batch=1)
      -> Reveals prefill vs decode cost; TTFT should stay flat.

  Experiment 3 (Input/Output Ratio):
      Sweep implementations x (input_len, gen_len) pairs (fixed batch=1)
      -> Isolates whether bottleneck is prefill or decode.

Experiment 4 (Precision / dtype):
      Sweep dtypes x seq_lengths (fixed batch=1, eager only)
      -> Shows memory-latency tradeoff of quantization.

  Experiment 5 (Decode Strategy):
      Greedy vs sampling (fixed seq_len=512, batch=1)
      -> Shows sampling overhead to validate attention is the bottleneck.

All results go into results/ as CSVs.

Usage:
    python3 inference/run_baseline.py                        # run all experiments
    python3 inference/run_baseline.py --experiments 1 2      # run specific experiments
    python3 inference/run_baseline.py --experiments 1 --benchmark_runs 3  # quick test
"""

import argparse
import csv
import gc
import sys
import time
from pathlib import Path
from statistics import median, stdev

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DEFAULT_MODEL = "/home/administrator/bin/GPU Analysis/Qwen3-0.6B"
DEFAULT_IMPLEMENTATIONS = ["eager", "sdpa"]
DEFAULT_SEQ_LENGTHS = [256, 512, 1024, 1800]
DEFAULT_BATCH_SIZES = [1, 2, 4]
DEFAULT_MAX_NEW_TOKENS = 128
DEFAULT_WARMUP_RUNS = 3
DEFAULT_BENCHMARK_RUNS = 5

# Experiment 2: generation length scaling
GEN_LENGTH_TOKENS = [32, 64, 128, 256]

# Experiment 3: input/output ratio pairs (input_len, gen_len)
IO_RATIO_CONFIGS = [
    (128, 512),    # short prompt, long generation (chatbot)
    (512, 128),    # medium prompt, medium generation
    (1024, 64),    # long prompt, short generation (summarization)
    (1800, 32),    # very long prompt, tiny generation (classification)
]

# Experiment 4: precision (skip float32 — Qwen3-0.6B at fp32 ~2.4GB weights, tight on 6GB GPU)
DTYPES = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}
DTYPE_SEQ_LENGTHS = [256, 512, 1024]

# Experiment 5: decode strategies
DECODE_STRATEGIES = {
    "greedy": {"do_sample": False},
    "sample_t0.7_topk50": {"do_sample": True, "temperature": 0.7, "top_k": 50},
    "sample_t1.0_topp0.9": {"do_sample": True, "temperature": 1.0, "top_p": 0.9},
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def get_gpu_info() -> dict:
    return {
        "gpu_name": torch.cuda.get_device_name(0),
        "gpu_vram_mb": torch.cuda.get_device_properties(0).total_memory // (1024 * 1024),
        "cuda_version": torch.version.cuda,
        "pytorch_version": torch.__version__,
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


def load_model(model_name: str, attn_impl: str, dtype=torch.float16):
    print(f"  Loading model: attn='{attn_impl}', dtype={dtype} ...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=dtype,
        attn_implementation=attn_impl,
    ).to("cuda")
    model.eval()
    return model


def measure_ttft(model, inputs, gen_kwargs: dict) -> float:
    kw = {**gen_kwargs, "max_new_tokens": 1}
    torch.cuda.synchronize()
    start = time.perf_counter()
    with torch.no_grad():
        model.generate(**inputs, **kw)
    torch.cuda.synchronize()
    return time.perf_counter() - start


def run_generation(model, inputs, gen_kwargs: dict):
    torch.cuda.synchronize()
    start = time.perf_counter()
    with torch.no_grad():
        output_ids = model.generate(**inputs, **gen_kwargs)
    torch.cuda.synchronize()
    latency = time.perf_counter() - start
    return output_ids, latency


def benchmark_single(
    model, tokenizer, seq_len, batch_size, max_new_tokens,
    warmup_runs, benchmark_runs, gen_kwargs=None,
):
    if gen_kwargs is None:
        gen_kwargs = {"do_sample": False}
    gen_kwargs["max_new_tokens"] = max_new_tokens

    inputs = build_prompt(tokenizer, seq_len, batch_size)
    actual_input_len = inputs["input_ids"].shape[1]

    # Warm-up
    warmup_kw = {**gen_kwargs}
    for _ in range(warmup_runs):
        with torch.no_grad():
            model.generate(**inputs, **warmup_kw)
    torch.cuda.synchronize()

    # TTFT
    ttft_values = []
    for _ in range(benchmark_runs):
        ttft_values.append(measure_ttft(model, inputs, gen_kwargs))

    # Full generation
    latencies, tokens_list, peak_mems = [], [], []
    for _ in range(benchmark_runs):
        torch.cuda.reset_peak_memory_stats()
        output_ids, latency = run_generation(model, inputs, gen_kwargs)
        peak_mem = torch.cuda.max_memory_allocated() / (1024 ** 2)
        num_gen = output_ids.shape[1] - actual_input_len
        latencies.append(latency)
        tokens_list.append(num_gen * batch_size)
        peak_mems.append(peak_mem)

    med_latency = median(latencies)
    med_tokens = median(tokens_list)
    std_latency = stdev(latencies) if len(latencies) > 1 else 0.0

    return {
        "seq_len": seq_len,
        "actual_input_len": actual_input_len,
        "batch_size": batch_size,
        "max_new_tokens": max_new_tokens,
        "median_latency_s": round(med_latency, 4),
        "std_latency_s": round(std_latency, 4),
        "median_tokens_per_sec": round(med_tokens / med_latency if med_latency > 0 else 0, 2),
        "median_ttft_s": round(median(ttft_values), 4),
        "median_peak_memory_mb": round(median(peak_mems), 2),
        "num_runs": benchmark_runs,
        "all_latencies": latencies,
        "all_ttft": ttft_values,
        "all_peak_memories": peak_mems,
    }


def safe_benchmark(model, tokenizer, seq_len, batch_size, max_new_tokens,
                   warmup_runs, benchmark_runs, max_pos, label, gen_kwargs=None):
    if max_pos and (seq_len + max_new_tokens) > max_pos:
        print(f"    SKIP — seq({seq_len})+gen({max_new_tokens}) > max_pos({max_pos})")
        return None
    try:
        result = benchmark_single(
            model, tokenizer, seq_len, batch_size, max_new_tokens,
            warmup_runs, benchmark_runs, gen_kwargs,
        )
        print(
            f"    latency={result['median_latency_s']:.4f}s  "
            f"tok/s={result['median_tokens_per_sec']:.2f}  "
            f"ttft={result['median_ttft_s']:.4f}s  "
            f"mem={result['median_peak_memory_mb']:.1f}MB"
        )
        return result
    except torch.cuda.OutOfMemoryError:
        print(f"    OOM — skipping")
        torch.cuda.empty_cache()
        gc.collect()
        return None
    except Exception as e:
        print(f"    ERROR — {e}")
        return None


# ---------------------------------------------------------------------------
# CSV output
# ---------------------------------------------------------------------------
def write_experiment_csv(results: list[dict], filepath: Path, extra_fields: list[str] = None):
    filepath.parent.mkdir(parents=True, exist_ok=True)
    base_fields = [
        "implementation", "seq_len", "actual_input_len", "batch_size",
        "max_new_tokens", "median_latency_s", "std_latency_s",
        "median_tokens_per_sec", "median_ttft_s", "median_peak_memory_mb", "num_runs",
    ]
    if extra_fields:
        base_fields = extra_fields + base_fields
    with open(filepath, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=base_fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(results)
    print(f"  -> Saved: {filepath}")


def write_raw_csv(results: list[dict], filepath: Path, extra_fields: list[str] = None):
    filepath.parent.mkdir(parents=True, exist_ok=True)
    base_fields = ["implementation", "seq_len", "batch_size", "max_new_tokens",
                    "run", "latency_s", "ttft_s", "peak_memory_mb"]
    if extra_fields:
        base_fields = extra_fields + base_fields
    with open(filepath, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=base_fields, extrasaction="ignore")
        writer.writeheader()
        for r in results:
            for i in range(r["num_runs"]):
                row = {
                    "implementation": r.get("implementation", ""),
                    "seq_len": r["seq_len"],
                    "batch_size": r["batch_size"],
                    "max_new_tokens": r["max_new_tokens"],
                    "run": i + 1,
                    "latency_s": round(r["all_latencies"][i], 4),
                    "ttft_s": round(r["all_ttft"][i], 4),
                    "peak_memory_mb": round(r["all_peak_memories"][i], 2),
                }
                # Copy extra fields
                if extra_fields:
                    for ef in extra_fields:
                        row[ef] = r.get(ef, "")
                writer.writerow(row)
    print(f"  -> Saved: {filepath}")


def print_table(results: list[dict], title: str, extra_cols: list[str] = None):
    extra_cols = extra_cols or []
    extra_hdr = "".join(f"{c:>12}" for c in extra_cols)
    header = (
        f"{'Impl':<10} {extra_hdr}{'SeqLen':>6} {'Batch':>5} {'GenTok':>6} "
        f"{'Lat(s)':>8} {'Tok/s':>8} {'TTFT(s)':>8} {'Mem(MB)':>9}"
    )
    print(f"\n{'=' * len(header)}")
    print(title)
    print(f"{'=' * len(header)}")
    print(header)
    print("-" * len(header))
    for r in results:
        extra_vals = "".join(f"{str(r.get(c, '')):>12}" for c in extra_cols)
        print(
            f"{r.get('implementation', 'N/A'):<10} {extra_vals}"
            f"{r['seq_len']:>6} {r['batch_size']:>5} {r['max_new_tokens']:>6} "
            f"{r['median_latency_s']:>8.4f} {r['median_tokens_per_sec']:>8.2f} "
            f"{r['median_ttft_s']:>8.4f} {r['median_peak_memory_mb']:>9.2f}"
        )
    print("=" * len(header))


# ---------------------------------------------------------------------------
# Experiment runners
# ---------------------------------------------------------------------------
def run_exp1_core_sweep(args, tokenizer, results_dir):
    """Exp 1: implementations x seq_lengths x batch_sizes."""
    print("\n" + "#" * 60)
    print("# EXPERIMENT 1: Core Sweep (impl x seq_len x batch)")
    print("#" * 60)

    all_results = []
    total = len(args.implementations) * len(args.seq_lengths) * len(args.batch_sizes)
    idx = 0

    for attn_impl in args.implementations:
        try:
            model = load_model(args.model, attn_impl)
        except Exception as e:
            print(f"  SKIPPING '{attn_impl}': {e}")
            idx += len(args.seq_lengths) * len(args.batch_sizes)
            continue

        max_pos = getattr(model.config, "max_position_embeddings", None)
        if max_pos:
            print(f"  max_position_embeddings: {max_pos}")

        for seq_len in args.seq_lengths:
            for batch_size in args.batch_sizes:
                idx += 1
                print(f"\n  [{idx}/{total}] impl={attn_impl}, seq={seq_len}, batch={batch_size}")
                result = safe_benchmark(
                    model, tokenizer, seq_len, batch_size, args.max_new_tokens,
                    args.warmup_runs, args.benchmark_runs, max_pos,
                    label=f"exp1_{attn_impl}_{seq_len}_{batch_size}",
                )
                if result:
                    result["implementation"] = attn_impl
                    all_results.append(result)

        del model
        torch.cuda.empty_cache()
        gc.collect()

    if all_results:
        print_table(all_results, "EXP 1: Core Sweep Results")
        write_experiment_csv(all_results, results_dir / "exp1_core_sweep.csv")
        write_raw_csv(all_results, results_dir / "exp1_core_sweep_raw.csv")
    return all_results


def run_exp2_gen_length(args, tokenizer, results_dir):
    """Exp 2: vary max_new_tokens with fixed seq_len=512, batch=1."""
    print("\n" + "#" * 60)
    print("# EXPERIMENT 2: Generation Length Scaling")
    print("#" * 60)

    all_results = []
    seq_len = 512
    batch_size = 1

    for attn_impl in args.implementations:
        try:
            model = load_model(args.model, attn_impl)
        except Exception as e:
            print(f"  SKIPPING '{attn_impl}': {e}")
            continue

        max_pos = getattr(model.config, "max_position_embeddings", None)

        for gen_tokens in GEN_LENGTH_TOKENS:
            print(f"\n  impl={attn_impl}, seq={seq_len}, gen={gen_tokens}")
            result = safe_benchmark(
                model, tokenizer, seq_len, batch_size, gen_tokens,
                args.warmup_runs, args.benchmark_runs, max_pos,
                label=f"exp2_{attn_impl}_{gen_tokens}",
            )
            if result:
                result["implementation"] = attn_impl
                all_results.append(result)

        del model
        torch.cuda.empty_cache()
        gc.collect()

    if all_results:
        print_table(all_results, "EXP 2: Generation Length Scaling")
        write_experiment_csv(all_results, results_dir / "exp2_gen_length.csv")
        write_raw_csv(all_results, results_dir / "exp2_gen_length_raw.csv")
    return all_results


def run_exp3_io_ratio(args, tokenizer, results_dir):
    """Exp 3: vary input/output length ratio with fixed batch=1."""
    print("\n" + "#" * 60)
    print("# EXPERIMENT 3: Input/Output Length Ratio")
    print("#" * 60)

    all_results = []
    batch_size = 1

    for attn_impl in args.implementations:
        try:
            model = load_model(args.model, attn_impl)
        except Exception as e:
            print(f"  SKIPPING '{attn_impl}': {e}")
            continue

        max_pos = getattr(model.config, "max_position_embeddings", None)

        for input_len, gen_len in IO_RATIO_CONFIGS:
            print(f"\n  impl={attn_impl}, input={input_len}, gen={gen_len}")
            result = safe_benchmark(
                model, tokenizer, input_len, batch_size, gen_len,
                args.warmup_runs, args.benchmark_runs, max_pos,
                label=f"exp3_{attn_impl}_{input_len}_{gen_len}",
            )
            if result:
                result["implementation"] = attn_impl
                result["io_ratio"] = f"{input_len}:{gen_len}"
                all_results.append(result)

        del model
        torch.cuda.empty_cache()
        gc.collect()

    if all_results:
        print_table(all_results, "EXP 3: Input/Output Ratio", extra_cols=["io_ratio"])
        write_experiment_csv(all_results, results_dir / "exp3_io_ratio.csv",
                             extra_fields=["io_ratio"])
        write_raw_csv(all_results, results_dir / "exp3_io_ratio_raw.csv",
                       extra_fields=["io_ratio"])
    return all_results


def run_exp4_dtype(args, tokenizer, results_dir):
    """Exp 4: vary dtype with fixed eager impl, batch=1."""
    print("\n" + "#" * 60)
    print("# EXPERIMENT 4: Precision / dtype Comparison")
    print("#" * 60)

    all_results = []
    attn_impl = "eager"
    batch_size = 1

    for dtype_name, dtype_val in DTYPES.items():
        try:
            model = load_model(args.model, attn_impl, dtype=dtype_val)
        except Exception as e:
            print(f"  SKIPPING dtype={dtype_name}: {e}")
            continue

        max_pos = getattr(model.config, "max_position_embeddings", None)

        for seq_len in DTYPE_SEQ_LENGTHS:
            print(f"\n  dtype={dtype_name}, seq={seq_len}")
            result = safe_benchmark(
                model, tokenizer, seq_len, batch_size, args.max_new_tokens,
                args.warmup_runs, args.benchmark_runs, max_pos,
                label=f"exp4_{dtype_name}_{seq_len}",
            )
            if result:
                result["implementation"] = attn_impl
                result["dtype"] = dtype_name
                all_results.append(result)

        del model
        torch.cuda.empty_cache()
        gc.collect()

    if all_results:
        print_table(all_results, "EXP 4: Precision (dtype) Comparison", extra_cols=["dtype"])
        write_experiment_csv(all_results, results_dir / "exp4_dtype.csv",
                             extra_fields=["dtype"])
        write_raw_csv(all_results, results_dir / "exp4_dtype_raw.csv",
                       extra_fields=["dtype"])
    return all_results


def run_exp5_decode_strategy(args, tokenizer, results_dir):
    """Exp 5: greedy vs sampling strategies, fixed seq=512, batch=1."""
    print("\n" + "#" * 60)
    print("# EXPERIMENT 5: Decode Strategy Comparison")
    print("#" * 60)

    all_results = []
    seq_len = 512
    batch_size = 1

    for attn_impl in args.implementations:
        try:
            model = load_model(args.model, attn_impl)
        except Exception as e:
            print(f"  SKIPPING '{attn_impl}': {e}")
            continue

        max_pos = getattr(model.config, "max_position_embeddings", None)

        for strat_name, strat_kwargs in DECODE_STRATEGIES.items():
            print(f"\n  impl={attn_impl}, strategy={strat_name}")
            result = safe_benchmark(
                model, tokenizer, seq_len, batch_size, args.max_new_tokens,
                args.warmup_runs, args.benchmark_runs, max_pos,
                label=f"exp5_{attn_impl}_{strat_name}",
                gen_kwargs=strat_kwargs,
            )
            if result:
                result["implementation"] = attn_impl
                result["decode_strategy"] = strat_name
                all_results.append(result)

        del model
        torch.cuda.empty_cache()
        gc.collect()

    if all_results:
        print_table(all_results, "EXP 5: Decode Strategy Comparison",
                    extra_cols=["decode_strategy"])
        write_experiment_csv(all_results, results_dir / "exp5_decode_strategy.csv",
                             extra_fields=["decode_strategy"])
        write_raw_csv(all_results, results_dir / "exp5_decode_strategy_raw.csv",
                       extra_fields=["decode_strategy"])
    return all_results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Baseline inference benchmark suite.")
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--implementations", nargs="+", default=DEFAULT_IMPLEMENTATIONS,
                    choices=["eager", "sdpa", "flash_attention_2"])
    p.add_argument("--seq_lengths", nargs="+", type=int, default=DEFAULT_SEQ_LENGTHS)
    p.add_argument("--batch_sizes", nargs="+", type=int, default=DEFAULT_BATCH_SIZES)
    p.add_argument("--max_new_tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS)
    p.add_argument("--warmup_runs", type=int, default=DEFAULT_WARMUP_RUNS)
    p.add_argument("--benchmark_runs", type=int, default=DEFAULT_BENCHMARK_RUNS)
    p.add_argument("--results_dir", type=str, default=None)
    p.add_argument("--experiments", nargs="+", type=int, default=[1, 2, 3, 4, 5],
                    choices=[1, 2, 3, 4, 5],
                    help="Which experiments to run (1-5, default: all)")
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
        1: ("Core Sweep", run_exp1_core_sweep),
        2: ("Generation Length Scaling", run_exp2_gen_length),
        3: ("Input/Output Ratio", run_exp3_io_ratio),
        4: ("Precision (dtype)", run_exp4_dtype),
        5: ("Decode Strategy", run_exp5_decode_strategy),
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
    print("ALL EXPERIMENTS COMPLETE")
    print("=" * 60)
    for exp_num in sorted(args.experiments):
        name, _ = exp_map[exp_num]
        count = len(total_results.get(exp_num, []))
        print(f"  Exp {exp_num} ({name}): {count} configs benchmarked")
    print(f"\nResults directory: {results_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
