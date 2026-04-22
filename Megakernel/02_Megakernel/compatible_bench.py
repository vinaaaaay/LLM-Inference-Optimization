import gc
import os
import time
import statistics
import warnings

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from model import Decoder, MODEL_NAME

# -----------------------------------------------------------------------------
# RTX 3050 / benchmark settings
# -----------------------------------------------------------------------------
os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "8.6")  # RTX 3050 = SM 8.6

DEVICE = "cuda"
DTYPE = torch.float16

CONTEXT_LENGTHS = [10, 50, 100, 200]
WARMUP = 5
RUNS = 10

# Use deterministic decode
DO_SAMPLE = False

# -----------------------------------------------------------------------------
# Global CUDA settings
# -----------------------------------------------------------------------------
torch.set_grad_enabled(False)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

warnings.filterwarnings("ignore", message="`do_sample` is set to `False`")
warnings.filterwarnings("ignore", message="The attention mask is not set")


def cuda_sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def mean_std(values):
    mean = statistics.mean(values)
    std = statistics.stdev(values) if len(values) > 1 else 0.0
    return mean, std


def benchmark_megakernel(decoder, num_tokens):
    # Warmup
    for _ in range(WARMUP):
        decoder.reset()
        tok = decoder.step(1)
        for _ in range(num_tokens - 1):
            tok = decoder.step(tok)
    cuda_sync()

    times = []
    for _ in range(RUNS):
        decoder.reset()
        cuda_sync()
        t0 = time.perf_counter()

        tok = decoder.step(1)
        for _ in range(num_tokens - 1):
            tok = decoder.step(tok)

        cuda_sync()
        times.append(time.perf_counter() - t0)

    mean_t, std_t = mean_std(times)
    tok_per_s = num_tokens / mean_t
    tok_per_s_std = (num_tokens / mean_t**2) * std_t if mean_t > 0 else 0.0
    return tok_per_s, tok_per_s_std, times


def benchmark_huggingface(model, input_ids, attention_mask, num_tokens):
    gen_kwargs = dict(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=num_tokens,
        do_sample=DO_SAMPLE,
        use_cache=True,
        pad_token_id=model.config.pad_token_id,
        eos_token_id=model.config.eos_token_id,
    )

    # Warmup
    for _ in range(WARMUP):
        with torch.inference_mode():
            _ = model.generate(**gen_kwargs)
    cuda_sync()

    times = []
    for _ in range(RUNS):
        cuda_sync()
        t0 = time.perf_counter()

        with torch.inference_mode():
            _ = model.generate(**gen_kwargs)

        cuda_sync()
        times.append(time.perf_counter() - t0)

    mean_t, std_t = mean_std(times)
    tok_per_s = num_tokens / mean_t
    tok_per_s_std = (num_tokens / mean_t**2) * std_t if mean_t > 0 else 0.0
    return tok_per_s, tok_per_s_std, times


def build_single_token_prompt(tokenizer, model):
    """
    Make the HF baseline as close as possible to the megakernel decode benchmark:
    start from a single token, then generate N new tokens autoregressively.
    """
    bos = model.config.bos_token_id
    if bos is None:
        bos = tokenizer.bos_token_id
    if bos is None:
        bos = 1  # matches the megakernel benchmark convention

    input_ids = torch.tensor([[bos]], device=DEVICE, dtype=torch.long)
    attention_mask = torch.ones_like(input_ids, device=DEVICE)
    return input_ids, attention_mask


print("Loading megakernel...")
decoder = Decoder()

print("Loading HuggingFace model/tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

if tokenizer.pad_token_id is None:
    tokenizer.pad_token = tokenizer.eos_token

hf_model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=DTYPE,
    low_cpu_mem_usage=True,
).to(DEVICE)
hf_model.eval()

if hf_model.config.pad_token_id is None:
    hf_model.config.pad_token_id = tokenizer.pad_token_id
if hf_model.config.eos_token_id is None:
    hf_model.config.eos_token_id = tokenizer.eos_token_id

input_ids, attention_mask = build_single_token_prompt(tokenizer, hf_model)

# Optional: one dry run to trigger lazy init / kernel compilation effects
with torch.inference_mode():
    _ = hf_model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=1,
        do_sample=False,
        use_cache=True,
        pad_token_id=hf_model.config.pad_token_id,
        eos_token_id=hf_model.config.eos_token_id,
    )
cuda_sync()

header = (
    f"{'Tokens':<8}"
    f"{'Megakernel (tok/s)':<24}"
    f"{'HF (tok/s)':<24}"
    f"{'Speedup':<12}"
)
print(f"\n{header}")
print("-" * len(header))

for n in CONTEXT_LENGTHS:
    mega_tps, mega_std, _ = benchmark_megakernel(decoder, n)
    hf_tps, hf_std, _ = benchmark_huggingface(hf_model, input_ids, attention_mask, n)
    speedup = mega_tps / hf_tps

    mega_str = f"{mega_tps:.1f} ± {mega_std:.1f}"
    hf_str = f"{hf_tps:.1f} ± {hf_std:.1f}"

    print(f"{n:<8}{mega_str:<24}{hf_str:<24}{speedup:<12.2f}x")

del hf_model
gc.collect()
torch.cuda.empty_cache()