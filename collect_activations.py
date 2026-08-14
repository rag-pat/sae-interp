from transformers import GPT2Tokenizer, GPT2Model

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2Model.from_pretrained("gpt2")

print("Loaded GPT-2 small.")
print("Number of layers:", model.config.n_layer)
print("Activation vector size per token:", model.config.n_embd)

total_params = sum(p.numel() for p in model.parameters())
print("Total weights (the 'pile of numbers'):", total_params)

print("Tokens for 'unhappy':", tokenizer.tokenize("unhappy"))
print("Vocabulary size (total distinct tokens):", tokenizer.vocab_size)

print(model)

# --- Forward hook: attach a "camera" to layer 6 ---
captured = {}

def hook_fn(module, input, output):
    captured["activations"] = output[0] if isinstance(output, tuple) else output

layer_index = 6  # 0-indexed, so this is the 7th block of 12
model.h[layer_index].register_forward_hook(hook_fn)

# --- Test with one fake sentence, no real dataset yet ---
test_text = "The quick brown fox jumps over the lazy dog."
inputs = tokenizer(test_text, return_tensors="pt")
model(**inputs)

print("\n--- Hook test ---")
print("Captured activations shape:", captured["activations"].shape)
print("First token's first 5 numbers:", captured["activations"][0, 0, :5])
