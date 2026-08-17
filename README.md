# Sparse Autoencoders for GPT-2 Interpretability

This project trains a sparse autoencoder to untangle GPT-2 small's internal activations into individual, human-readable features, then verifies those features are real — first by checking which ones stay consistent across different training settings, and then by forcing one off during a live forward pass to see if the model's behavior actually breaks without it.
