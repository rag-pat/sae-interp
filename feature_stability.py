import torch
from train_autoencoder import SparseAutoencoder

data = torch.load("data/activations_with_tokens.pt")
all_tokens = data["tokens"]
all_activations = data["activations"]

# --- Step 1: the defining token(s) for each of our 5 named features ---
# "Ġ" marks "a space comes before this piece" - wikitext puts spaces before most punctuation,
# so colon/apostrophe show up as "Ġ:" and "Ġ'" rather than bare ":" and "'"
concept_tokens = {
    "colon": ["Ġ:"],
    "apostrophe": ["Ġ'"],
    "new": ["ĠNew", "Ġnew"],
    "perfect_tense_aux": ["Ġhave", "Ġhad", "Ġbeen"],
    "comparative": ["Ġthan", "Ġas", "Ġbefore"],
}

# --- Step 2: find every position in our dataset matching each concept ---
concept_positions = {}
for concept, tokens in concept_tokens.items():
    positions = [i for i, tok in enumerate(all_tokens) if tok in tokens]
    concept_positions[concept] = positions
    print(f"{concept:20s} ({tokens}): {len(positions)} matching positions in the dataset")


def get_model_features(state_dict_path):
    """Run all activations through a trained autoencoder, return its (81718, 16000) feature matrix."""
    autoencoder = SparseAutoencoder()
    autoencoder.load_state_dict(torch.load(state_dict_path))
    autoencoder.eval()

    batch_size = 512
    chunks = []
    with torch.no_grad():
        for start in range(0, all_activations.shape[0], batch_size):
            batch = all_activations[start:start + batch_size]
            _, features = autoencoder(batch)
            chunks.append(features)
    return torch.cat(chunks, dim=0)


def find_best_matching_slot(positions, features):
    """Which slot fires strongest, on average, at this concept's positions?"""
    concept_activations = features[positions]      # (num_positions, 16000)
    avg_firing = concept_activations.mean(dim=0)    # (16000,) - one avg per slot
    best_slot = torch.argmax(avg_firing).item()
    return best_slot, avg_firing[best_slot].item()


# --- Try it on one other lambda model first: 0.0001 ---
lambda_value = 0.0001
print(f"\n--- Best matching slots in the lambda={lambda_value} model ---")
model_features = get_model_features(f"data/sparse_autoencoder_lambda_{lambda_value}.pt")

for concept, positions in concept_positions.items():
    best_slot, avg_value = find_best_matching_slot(positions, model_features)
    print(f"{concept:20s}: best match = slot {best_slot:5d} (avg firing {avg_value:.3f})")


def top_words_for_slot(features, feature_index, k=10):
    firing_values = features[:, feature_index]
    top = torch.topk(firing_values, k=k)
    return [all_tokens[idx].replace("Ġ", " ").strip() for idx in top.indices.tolist()]


# --- Repeat for the remaining lambda models, and deep-read each best match ---
import json

lambda_values_to_check = [0.0001, 0.0003, 0.003, 0.01, 0.03]
stability_results = {concept: {} for concept in concept_positions}

for lam in lambda_values_to_check:
    print(f"\n=== Lambda = {lam} ===")
    features_for_lambda = get_model_features(f"data/sparse_autoencoder_lambda_{lam}.pt")

    for concept, positions in concept_positions.items():
        best_slot, avg_value = find_best_matching_slot(positions, features_for_lambda)
        top_words = top_words_for_slot(features_for_lambda, best_slot, k=10)
        print(f"  {concept:20s} -> slot {best_slot:5d} (avg {avg_value:.2f}): {top_words}")
        stability_results[concept][lam] = {
            "slot": best_slot,
            "avg_firing": avg_value,
            "top_10_words": top_words,
        }

with open("data/feature_stability_results.json", "w") as f:
    json.dump(stability_results, f, indent=2)
print("\nSaved to data/feature_stability_results.json")
