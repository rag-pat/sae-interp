import torch

data = torch.load("data/activations_with_tokens.pt")
all_tokens = data["tokens"]

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
