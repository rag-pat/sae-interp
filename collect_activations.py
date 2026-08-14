import torch
from transformers import GPT2Tokenizer, GPT2Model
from datasets import load_dataset

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token  # GPT-2 has no pad token by default; reuse the end-of-text token
model = GPT2Model.from_pretrained("gpt2")
model.eval()  # inference only, no training happening

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

# --- Load a real text dataset (small, diverse Wikipedia sentences) ---
print("\n--- Loading text dataset ---")
dataset = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="train")
sentences = [row["text"].strip() for row in dataset if row["text"].strip()]
print("Number of non-empty lines loaded:", len(sentences))
print("Example line:", sentences[0])
print("Example line:", sentences[10])

# --- Run many sentences through the model, in batches, and collect activations ---
print("\n--- Collecting activations ---")

num_sentences_to_use = 2000  # a few thousand, per the plan
batch_size = 16
text_subset = sentences[:num_sentences_to_use]

all_activations = []  # will hold one (num_tokens, 768) tensor per batch

with torch.no_grad():  # we're not training, so skip tracking gradients (saves memory/time)
    for start in range(0, len(text_subset), batch_size):
        batch = text_subset[start:start + batch_size]
        inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=64)
        model(**inputs)

        batch_activations = captured["activations"]        # (batch, seq_len, 768)
        real_token_mask = inputs["attention_mask"].bool()   # True = real token, False = padding

        for i in range(batch_activations.shape[0]):
            real_vectors = batch_activations[i][real_token_mask[i]]  # (this sentence's num_tokens, 768)
            all_activations.append(real_vectors)

        if start % (batch_size * 20) == 0:
            print(f"  processed {start}/{len(text_subset)} sentences")

all_activations = torch.cat(all_activations, dim=0)
print("Total token-level activation vectors collected:", all_activations.shape)

# --- Save to disk: this file is Stage 1's actual output ---
save_path = "data/activations.pt"
torch.save(all_activations, save_path)
print(f"Saved to {save_path}")
