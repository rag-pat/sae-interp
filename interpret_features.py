import torch
from transformers import GPT2Tokenizer, GPT2Model
from datasets import load_dataset

# --- Load GPT-2 small + tokenizer (same setup as Stage 1) ---
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token
model = GPT2Model.from_pretrained("gpt2")
model.eval()

# --- Attach the same forward hook to layer 6 ---
captured = {}

def hook_fn(module, input, output):
    captured["activations"] = output[0] if isinstance(output, tuple) else output

layer_index = 6
model.h[layer_index].register_forward_hook(hook_fn)

# --- Load the same text dataset as Stage 1 ---
dataset = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="train")
sentences = [row["text"].strip() for row in dataset if row["text"].strip()]

# --- Run sentences through the model again, but this time keep the actual words too ---
print("--- Collecting activations + the words they belong to ---")

num_sentences_to_use = 2000
batch_size = 16
text_subset = sentences[:num_sentences_to_use]

all_activations = []
all_tokens = []            # the actual word/piece for each activation vector, same order
all_sentence_indices = []  # which sentence each token came from, so we can show context later

with torch.no_grad():
    for start in range(0, len(text_subset), batch_size):
        batch = text_subset[start:start + batch_size]
        inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=64)
        model(**inputs)

        batch_activations = captured["activations"]
        real_token_mask = inputs["attention_mask"].bool()

        for i in range(batch_activations.shape[0]):
            sentence_index = start + i
            real_vectors = batch_activations[i][real_token_mask[i]]
            real_ids = inputs["input_ids"][i][real_token_mask[i]]
            real_tokens = tokenizer.convert_ids_to_tokens(real_ids.tolist())

            all_activations.append(real_vectors)
            all_tokens.extend(real_tokens)
            all_sentence_indices.extend([sentence_index] * len(real_tokens))

        if start % (batch_size * 20) == 0:
            print(f"  processed {start}/{len(text_subset)} sentences")

all_activations = torch.cat(all_activations, dim=0)
print("Total token-level activation vectors:", all_activations.shape)
print("Total tokens (words) tracked:", len(all_tokens))
print("Example token:", repr(all_tokens[100]), "- from sentence:", text_subset[all_sentence_indices[100]][:80])

# --- Save activations AND the words/sentences they came from, together this time ---
torch.save({
    "activations": all_activations,
    "tokens": all_tokens,
    "sentence_indices": all_sentence_indices,
    "sentences": text_subset,
}, "data/activations_with_tokens.pt")
print("Saved to data/activations_with_tokens.pt")

# --- Run every token's activation through the trained autoencoder to see feature firing strengths ---
from train_autoencoder import SparseAutoencoder

print("\n--- Getting each feature's firing strength on every token ---")

autoencoder = SparseAutoencoder()
autoencoder.load_state_dict(torch.load("data/sparse_autoencoder.pt"))
autoencoder.eval()

# Note: we don't save this matrix to disk - it would be ~4.9GB (81718 tokens x 16000 features),
# mostly zeros since the model is sparse. We compute it fresh here and only keep what we need next.
batch_size = 512
all_features = []

with torch.no_grad():
    for start in range(0, all_activations.shape[0], batch_size):
        batch = all_activations[start:start + batch_size]
        _, features = autoencoder(batch)  # only need the feature slots, not the reconstruction
        all_features.append(features)

all_features = torch.cat(all_features, dim=0)
print("Feature firing strengths shape:", all_features.shape)
print("Overall fraction of features active:", (all_features > 0).float().mean().item())

example_index = 100
top5 = torch.topk(all_features[example_index], k=5)
print(f"\nToken {all_tokens[example_index]!r} - top 5 firing feature slots:", top5.indices.tolist())
print("Their firing values:", [round(v, 3) for v in top5.values.tolist()])
