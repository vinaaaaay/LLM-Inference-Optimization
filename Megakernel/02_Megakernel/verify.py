import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from model import Decoder, MODEL_NAME

PROMPT = "The capital of France is"
NUM_TOKENS = 20

print("Loading megakernel...")
decoder = Decoder()

print("Loading HuggingFace model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
hf_model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME, torch_dtype=torch.bfloat16, device_map="cuda"
)
hf_model.eval()

input_ids = tokenizer.encode(PROMPT)

decoder.reset()
for token_id in input_ids[:-1]:
    decoder.step(token_id)

mega_tokens = []
next_token = decoder.step(input_ids[-1])
for _ in range(NUM_TOKENS):
    mega_tokens.append(next_token)
    if next_token == tokenizer.eos_token_id:
        break
    next_token = decoder.step(next_token)

input_tensor = torch.tensor([input_ids], device="cuda")
with torch.no_grad():
    hf_output = hf_model.generate(
        input_tensor,
        max_new_tokens=NUM_TOKENS,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
    )
hf_tokens = hf_output[0][len(input_ids):].tolist()

mega_text = tokenizer.decode(mega_tokens, skip_special_tokens=True)
hf_text = tokenizer.decode(hf_tokens, skip_special_tokens=True)

print(f"\nPrompt: {PROMPT}")
print(f"Megakernel: {mega_text}")
print(f"HuggingFace: {hf_text}")

first_match = len(mega_tokens) > 0 and len(hf_tokens) > 0 and mega_tokens[0] == hf_tokens[0]
print(f"\nFirst token: mega={mega_tokens[0] if mega_tokens else None}, hf={hf_tokens[0] if hf_tokens else None}")
print(f"Result: {'PASS' if first_match else 'FAIL'}")