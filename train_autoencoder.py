import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader


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


def compute_loss(original, reconstruction, features, lambda_sparsity):
    reconstruction_loss = torch.mean((reconstruction - original) ** 2)  # how bad the rebuild is
    sparsity_loss = torch.mean(torch.sum(features, dim=1))              # how much is switched on
    total_loss = reconstruction_loss + lambda_sparsity * sparsity_loss
    return total_loss, reconstruction_loss, sparsity_loss


if __name__ == "__main__":
    # quick sanity test with fake random data, no real activations yet
    model = SparseAutoencoder()
    fake_batch = torch.randn(4, 768)  # pretend we have 4 activation vectors

    reconstruction, features = model(fake_batch)

    print("Input shape:", fake_batch.shape)
    print("Feature layer shape:", features.shape)
    print("Reconstruction shape:", reconstruction.shape)
    print("Fraction of features that fired (nonzero):", (features > 0).float().mean().item())

    # --- Load the real activations we collected in Stage 1, and set up batching ---
    activations = torch.load("data/activations.pt")
    print("\nLoaded activations shape:", activations.shape)

    dataset = TensorDataset(activations)
    dataloader = DataLoader(dataset, batch_size=256, shuffle=True)

    first_batch = next(iter(dataloader))[0]
    print("One batch handed out by the DataLoader:", first_batch.shape)

    # --- Try the loss on that real batch (model is still untrained, just checking it runs) ---
    lambda_sparsity = 0.001  # placeholder, we'll tune this later
    reconstruction, features = model(first_batch)
    total, recon_loss, sparsity_loss = compute_loss(first_batch, reconstruction, features, lambda_sparsity)

    print("\nReconstruction loss:", recon_loss.item())
    print("Sparsity loss:", sparsity_loss.item())
    print("Total loss:", total.item())

    # --- Set up the optimizer: the tool that will actually apply weight updates ---
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    print("\nOptimizer ready:", optimizer)
