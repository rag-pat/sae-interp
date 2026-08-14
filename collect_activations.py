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
