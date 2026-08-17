import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from train_autoencoder import SparseAutoencoder

feature_to_ablate = 458  # the apostrophe detector - cleanest feature we found

# --- Find real sentences where this feature actually fires strongly ---
data = torch.load("data/activations_with_tokens.pt")
all_activations = data["activations"]
all_sentence_indices = data["sentence_indices"]
text_subset = data["sentences"]

autoencoder = SparseAutoencoder()
autoencoder.load_state_dict(torch.load("data/sparse_autoencoder.pt"))
autoencoder.eval()

batch_size = 512
chunks = []
with torch.no_grad():
    for start in range(0, all_activations.shape[0], batch_size):
        batch = all_activations[start:start + batch_size]
        _, features = autoencoder(batch)
        chunks.append(features)
all_features = torch.cat(chunks, dim=0)

firing_values = all_features[:, feature_to_ablate]
top = torch.topk(firing_values, k=5)
example_sentences = [text_subset[all_sentence_indices[idx]] for idx in top.indices.tolist()]

print("--- Real sentences where the apostrophe feature fires strongly ---")
for s in example_sentences:
    print(" -", s[:100])

# --- Set up GPT-2 with its language modeling head, so we get real next-token predictions ---
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token
lm_model = GPT2LMHeadModel.from_pretrained("gpt2")
lm_model.eval()

current_mode = {"mode": "normal"}  # "normal", "reconstruct", or "ablate"


def intervention_hook(module, input, output):
    if current_mode["mode"] == "normal":
        return None  # no change - real GPT-2 behavior

    real_output = output[0] if isinstance(output, tuple) else output
    with torch.no_grad():
        reconstruction, features = autoencoder(real_output)
        if current_mode["mode"] == "ablate":
            features = features.clone()
            features[..., feature_to_ablate] = 0.0  # force this feature off
            reconstruction = autoencoder.decoder(features)

    if isinstance(output, tuple):
        return (reconstruction,) + output[1:]
    return reconstruction


layer_index = 6
lm_model.transformer.h[layer_index].register_forward_hook(intervention_hook)


def predict_after_apostrophe(sentence):
    inputs = tokenizer(sentence, return_tensors="pt")
    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0].tolist())
    apostrophe_positions = [i for i, t in enumerate(tokens) if "'" in t]
    if not apostrophe_positions:
        return None
    pos = apostrophe_positions[0]

    with torch.no_grad():
        outputs = lm_model(**inputs)
    logits = outputs.logits[0, pos]
    probs = torch.softmax(logits, dim=-1)
    top5 = torch.topk(probs, k=5)
    predicted = list(zip(
        tokenizer.convert_ids_to_tokens(top5.indices.tolist()),
        [round(v, 4) for v in top5.values.tolist()],
    ))
    return predicted, tokens, pos


print("\n--- Comparing predictions: normal vs. reconstructed vs. ablated ---")
for sentence in example_sentences[:3]:
    print(f"\nSentence: {sentence[:100]}")
    for mode in ["normal", "reconstruct", "ablate"]:
        current_mode["mode"] = mode
        result = predict_after_apostrophe(sentence)
        if result is None:
            print(f"  [{mode:11s}] no apostrophe token found")
            continue
        predicted, tokens, pos = result
        print(f"  [{mode:11s}] top predictions after {tokens[pos]!r}: {predicted}")
