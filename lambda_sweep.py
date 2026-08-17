import json
import torch
from torch.utils.data import TensorDataset, DataLoader
from train_autoencoder import SparseAutoencoder, compute_loss

# --- Load the same activations used in Stage 2 ---
activations = torch.load("data/activations.pt")
dataset = TensorDataset(activations)
dataloader = DataLoader(dataset, batch_size=256, shuffle=True)

# --- Lambda values to sweep, log-scale around our known-good 0.001 from Stage 2 ---
lambda_values = [0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03]
num_epochs = 20

# Reuse the result we already have for lambda = 0.001 instead of retraining it
known_results = {
    0.001: {"reconstruction_loss": 0.2411, "active_fraction": 0.0345},
}

results = []

for lam in lambda_values:
    if lam in known_results:
        print(f"\nLambda {lam}: reusing existing Stage 2 result (no retrain needed)")
        results.append({"lambda": lam, **known_results[lam]})
        continue

    print(f"\n--- Training with lambda = {lam} ---")
    model = SparseAutoencoder()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    avg_recon = None
    avg_active = None

    for epoch in range(num_epochs):
        total_recon_loss = 0.0
        total_active_fraction = 0.0
        num_batches = 0

        for (batch,) in dataloader:
            reconstruction, features = model(batch)
            loss, recon_loss, sparsity_loss = compute_loss(batch, reconstruction, features, lam)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_recon_loss += recon_loss.item()
            total_active_fraction += (features > 0).float().mean().item()
            num_batches += 1

        avg_recon = total_recon_loss / num_batches
        avg_active = total_active_fraction / num_batches
        print(f"  epoch {epoch + 1}/{num_epochs} - reconstruction: {avg_recon:.4f} - active: {avg_active:.4f}")

    results.append({"lambda": lam, "reconstruction_loss": avg_recon, "active_fraction": avg_active})

    save_path = f"data/sparse_autoencoder_lambda_{lam}.pt"
    torch.save(model.state_dict(), save_path)
    print(f"Saved to {save_path}")

print("\n=== Lambda sweep results ===")
for r in results:
    print(r)

with open("data/lambda_sweep_results.json", "w") as f:
    json.dump(results, f, indent=2)
print("\nSaved summary to data/lambda_sweep_results.json")
