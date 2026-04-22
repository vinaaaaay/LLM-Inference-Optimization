import gc
import time

import torch
import torch

torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_math_sdp(False)
from transformers import AutoModelForCausalLM, AutoTokenizer

from model import Decoder, MODEL_NAME

CONTEXT_LENGTHS = [10, 50, 100, 200]
WARMUP = 3
RUNS = 5


def benchmark_megakernel(decoder, num_tokens):
    for _ in range(WARMUP):
        decoder.reset()
        tok = decoder.step(1)
        for _ in range(num_tokens - 1):
            tok = decoder.step(tok)

    times = []
    for _ in range(RUNS):
        decoder.reset()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        tok = decoder.step(1)
        for _ in range(num_tokens - 1):
            tok = decoder.step(tok)
        torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)

    return num_tokens / (sum(times) / len(times))


def benchmark_huggingface(tokenizer, model, num_tokens):
    input_ids = tokenizer("Hello", return_tensors="pt").input_ids.to("cuda")

    # Prefill
    with torch.no_grad():
        outputs = model(input_ids, use_cache=True)
        past_key_values = outputs.past_key_values

    for _ in range(WARMUP):
        tok = input_ids[:, -1]
        pkv = past_key_values

        for _ in range(num_tokens):
            with torch.no_grad():
                out = model(input_ids=tok.unsqueeze(0), past_key_values=pkv, use_cache=True)
                tok = out.logits[:, -1].argmax(dim=-1)
                pkv = out.past_key_values

    times = []
    for _ in range(RUNS):
        tok = input_ids[:, -1]
        pkv = past_key_values

        torch.cuda.synchronize()
        t0 = time.perf_counter()

        for _ in range(num_tokens):
            with torch.no_grad():
                out = model(input_ids=tok.unsqueeze(0), past_key_values=pkv, use_cache=True)
                tok = out.logits[:, -1].argmax(dim=-1)
                pkv = out.past_key_values

        torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)

    return num_tokens / (sum(times) / len(times))


print("Loading megakernel...")
decoder = Decoder()

print("Loading HuggingFace model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
hf_model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME, torch_dtype=torch.bfloat16, device_map="cuda",attn_implementation="eager"
)
print(hf_model.config._attn_implementation)
hf_model.eval()

header = f"{'Tokens':<10} {'Megakernel (tok/s)':<22} {'HuggingFace (tok/s)':<22} {'Speedup':<10}"
print(f"\n{header}")
print("-" * len(header))

for n in CONTEXT_LENGTHS:
    mega_tps = benchmark_megakernel(decoder, n)
    hf_tps = benchmark_huggingface(tokenizer, hf_model, n)
    speedup = mega_tps / hf_tps
    print(f"{n:<10} {mega_tps:<22.1f} {hf_tps:<22.1f} {speedup:<10.2f}x")

del hf_model
gc.collect()
torch.cuda.empty_cache()