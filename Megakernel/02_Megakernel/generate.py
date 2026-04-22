import sys
import time

import torch

from model import Decoder

prompt = sys.argv[1] if len(sys.argv) > 1 else "The capital of France is"

print("Loading model and compiling kernel...")
decoder = Decoder()

print("Warming up...")
decoder.generate("Hello", max_new_tokens=5)

decoder.reset()
input_ids = decoder.tokenizer.encode(prompt)

for token_id in input_ids[:-1]:
    decoder.step(token_id)

print(f"\n{prompt}", end="", flush=True)

torch.cuda.synchronize()
start = time.perf_counter()

next_token = decoder.step(input_ids[-1])
count = 0

for _ in range(100):
    text = decoder.tokenizer.decode([next_token])
    print(text, end="", flush=True)
    count += 1
    if next_token == decoder.tokenizer.eos_token_id:
        break
    next_token = decoder.step(next_token)

torch.cuda.synchronize()
elapsed = time.perf_counter() - start

print(f"\n\n{count} tokens in {elapsed:.2f}s = {count / elapsed:.1f} tok/s")