import json
import matplotlib.pyplot as plt

with open("data/lambda_sweep_results.json") as f:
    results = json.load(f)

lambdas = [r["lambda"] for r in results]
active_fractions = [r["active_fraction"] for r in results]
reconstruction_losses = [r["reconstruction_loss"] for r in results]

fig, ax = plt.subplots(figsize=(7, 5))
ax.plot(active_fractions, reconstruction_losses, marker="o", color="#2563eb")

# label each point with its lambda value
for lam, x, y in zip(lambdas, active_fractions, reconstruction_losses):
    ax.annotate(f"λ={lam}", (x, y), textcoords="offset points", xytext=(6, 6), fontsize=8)

ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("Fraction of features active (log scale)")
ax.set_ylabel("Reconstruction loss (log scale)")
ax.set_title("Sparsity vs. reconstruction tradeoff across lambda values")
ax.grid(True, which="both", linestyle="--", alpha=0.4)

fig.tight_layout()
fig.savefig("data/lambda_sweep_chart.png", dpi=150)
print("Saved chart to data/lambda_sweep_chart.png")
