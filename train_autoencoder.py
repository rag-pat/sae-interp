import torch
import torch.nn as nn


class SparseAutoencoder(nn.Module):
    def __init__(self, input_size=768, feature_size=16000):
        super().__init__()
        self.encoder = nn.Linear(input_size, feature_size)
        self.relu = nn.ReLU()  # zeroes out negative values -> makes sparsity possible
        self.decoder = nn.Linear(feature_size, input_size)

    def forward(self, x):
        features = self.relu(self.encoder(x))       # (batch, 16000) - mostly zeros
        reconstruction = self.decoder(features)      # (batch, 768) - rebuilt activation
        return reconstruction, features


if __name__ == "__main__":
    # quick sanity test with fake random data, no real activations yet
    model = SparseAutoencoder()
    fake_batch = torch.randn(4, 768)  # pretend we have 4 activation vectors

    reconstruction, features = model(fake_batch)

    print("Input shape:", fake_batch.shape)
    print("Feature layer shape:", features.shape)
    print("Reconstruction shape:", reconstruction.shape)
    print("Fraction of features that fired (nonzero):", (features > 0).float().mean().item())
