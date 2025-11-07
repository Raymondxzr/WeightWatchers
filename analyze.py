#!/usr/bin/env python3
import os, csv, torch
import matplotlib.pyplot as plt

# ======== HARD-CODED CONFIG ========
OUT_DIR = "Alignment/hooked_llama/neuron_activation"
BASE = "Meta-Llama-3-8B-Instruct"
FT   = "Meta-Llama-3-8B-Instruct-TARharden"
DATA = "harmbench"
SUFFIX = "tar_completion"
TOPK = 5000
# ===================================

# ---- 1. Load single .pt file ----
pt_path = os.path.join(OUT_DIR, f"{BASE}_vs_{FT}_on_{DATA}_{SUFFIX}.pt")
if not os.path.exists(pt_path):
    raise FileNotFoundError(f"{pt_path} not found.")
print(f"Loaded file: {pt_path}")

cs, ranks, f_mu, f_std, s_mu, s_std = torch.load(pt_path, map_location="cpu")
q = torch.quantile(cs.flatten(), torch.tensor([0, .5, .9, .99]))
print("  change_scores shape:", tuple(cs.shape))
print("  ranks shape:", tuple(ranks.shape))
print("  NaNs?", bool(cs.isnan().any()), bool(f_mu.isnan().any()), bool(s_mu.isnan().any()))
print("  change_scores percentiles:", [round(v.item(), 8) for v in q])

# ---- 2. Export top-K neurons ----
K = TOPK
L, H = cs.shape
flat = cs.flatten()
topv, topi = torch.topk(flat, K)
layers = (topi // H).tolist()
neurons = (topi % H).tolist()
csv_path = os.path.join(OUT_DIR, f"top{K}_neurons.csv")
with open(csv_path, "w", newline="") as f:
    wcsv = csv.writer(f)
    wcsv.writerow(["layer", "neuron", "change_score"])
    for l, n, v in zip(layers, neurons, topv.tolist()):
        wcsv.writerow([l, n, v])
print(f"Wrote {csv_path}")

# ---- 3. Plots ----
layer_mean = cs.mean(dim=1).numpy()
layer_p95  = torch.quantile(cs, 0.95, dim=1).numpy()

plt.figure()
plt.plot(range(len(layer_mean)), layer_mean, label="mean")
plt.plot(range(len(layer_p95)), layer_p95, label="p95")
plt.xlabel("layer"); plt.ylabel("change score")
plt.title("Layer-wise change scores")
plt.legend(); plt.tight_layout()
layer_png = os.path.join(OUT_DIR, "layer_summary.png")
plt.savefig(layer_png, dpi=200)
print(f"Wrote {layer_png}")

plt.figure()
clip = torch.quantile(cs, 0.99).item()
arr = torch.clamp(cs, max=clip).numpy()[:, ::8]  # downsample hidden dim by 8
plt.imshow(arr, aspect="auto", origin="lower")
plt.colorbar(label="change score (clipped p99)")
plt.xlabel("hidden dim (every 8th)"); plt.ylabel("layer")
plt.title("Change score heatmap")
plt.tight_layout()
heat_png = os.path.join(OUT_DIR, "heatmap.png")
plt.savefig(heat_png, dpi=200)
print(f"Wrote {heat_png}")

print("\n✅ Done. Outputs:")
print("  -", csv_path)
print("  -", layer_png)
print("  -", heat_png)
