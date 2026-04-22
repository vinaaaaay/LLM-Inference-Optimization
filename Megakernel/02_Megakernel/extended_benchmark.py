import gc
import time

import torch
torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_math_sdp(False)

from transformers import AutoModelForCausalLM, AutoTokenizer
from model_batched import Decoder, BatchedDecoder, MODEL_NAME

CONTEXT_LENGTHS = [10, 50, 100, 200]
BATCH_SIZES     = [1, 2, 4, 8]
WARMUP = 3
RUNS   = 5


# ── Megakernel batched benchmark ──────────────────────────────────────────────
def benchmark_megakernel_batched(batched_decoder: BatchedDecoder,
                                  num_tokens: int) -> float:
    B           = batched_decoder.batch_size
    total_tokens = num_tokens * B
    start_ids   = [1] * B

    def _run():
        batched_decoder.reset()
        toks = batched_decoder.step(start_ids)
        for _ in range(num_tokens - 1):
            toks = batched_decoder.step(toks)

    for _ in range(WARMUP):
        _run()

    times = []
    for _ in range(RUNS):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        _run()
        torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)

    return total_tokens / (sum(times) / len(times))


# ── HuggingFace batched benchmark (unchanged) ─────────────────────────────────
def benchmark_huggingface_batch(tokenizer, model, num_tokens, batch_size):
    total_tokens = num_tokens * batch_size
    prompts      = ["Hello"] * batch_size
    enc = tokenizer(prompts, return_tensors="pt", padding=True).to("cuda")
    input_ids, attention_mask = enc["input_ids"], enc["attention_mask"]

    with torch.no_grad():
        outputs         = model(input_ids, attention_mask=attention_mask, use_cache=True)
        past_key_values = outputs.past_key_values
        last_token_ids  = input_ids[:, -1]

    def _run():
        tok      = last_token_ids
        pkv      = past_key_values
        cur_mask = attention_mask
        for _ in range(num_tokens):
            with torch.no_grad():
                out = model(input_ids=tok.unsqueeze(1),
                            past_key_values=pkv,
                            attention_mask=cur_mask,
                            use_cache=True)
                tok      = out.logits[:, -1].argmax(dim=-1)
                pkv      = out.past_key_values
                cur_mask = torch.cat(
                    [cur_mask, torch.ones(batch_size, 1, device="cuda",
                                         dtype=cur_mask.dtype)], dim=1)

    for _ in range(WARMUP):
        _run()

    times = []
    for _ in range(RUNS):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        _run()
        torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)

    return total_tokens / (sum(times) / len(times))


# ── Load models ───────────────────────────────────────────────────────────────
print("Loading single-sequence megakernel (for weight sharing)...")
base_decoder = Decoder()

print("Building batched decoders (weights shared, no extra VRAM)...")
batched_decoders = {
    b: BatchedDecoder(b, shared_weights_from=base_decoder)
    for b in BATCH_SIZES
}

print("Loading HuggingFace model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

hf_model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME, torch_dtype=torch.bfloat16, device_map="cuda",
    attn_implementation="eager")
hf_model.eval()

# ── Results table ─────────────────────────────────────────────────────────────
COL = [10, 8, 26, 26, 10]
header = (f"{'Tokens':<{COL[0]}} {'Batch':<{COL[1]}} "
          f"{'Megakernel batched':<{COL[2]}} "
          f"{'HuggingFace batch':<{COL[3]}} "
          f"{'Speedup':<{COL[4]}}")
print(f"\n{header}")
print("-" * len(header))

results = {}

for n in CONTEXT_LENGTHS:
    for b in BATCH_SIZES:
        mega_tps = benchmark_megakernel_batched(batched_decoders[b], n)
        hf_tps   = benchmark_huggingface_batch(tokenizer, hf_model, n, b)
        speedup  = mega_tps / hf_tps
        results[(n, b)] = (mega_tps, hf_tps, speedup)
        print(f"{n:<{COL[0]}} {b:<{COL[1]}} "
              f"{mega_tps:<{COL[2]}.1f} "
              f"{hf_tps:<{COL[3]}.1f} "
              f"{speedup:<{COL[4]}.2f}x")
    print()

# ── Scaling efficiency ────────────────────────────────────────────────────────
print("\n── Batch-scaling efficiency (relative to batch=1) ──────────────────")
eff_header = (f"{'Tokens':<{COL[0]}} {'Batch':<{COL[1]}} "
              f"{'Mega scale':<{COL[2]}} {'HF scale':<{COL[3]}}")
print(eff_header)
print("-" * len(eff_header))

for n in CONTEXT_LENGTHS:
    base_mega, base_hf, _ = results[(n, 1)]
    for b in BATCH_SIZES:
        m, h, _ = results[(n, b)]
        print(f"{n:<{COL[0]}} {b:<{COL[1]}} "
              f"{m/base_mega:<{COL[2]}.2f}x "
              f"{h/base_hf:<{COL[3]}.2f}x")
    print()

del hf_model
gc.collect()
torch.cuda.empty_cache()