#!/usr/bin/env python3
import os
import csv
import argparse
import torch
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(
        description="Analyze change-score .pt file (top-K neurons + plots)."
    )
    parser.add_argument(
        "--score_file",
        type=str,
        required=True,
        help="Path to change-score .pt file (e.g., .../completion.pt)",
    )
    parser.add_argument(
        "--topk",
        type=int,
        default=5000,
        help="Number of top neurons to export (default: 5000)",
    )
    args = parser.parse_args()

    pt_path = args.score_file
    K = args.topk

    if not os.path.exists(pt_path):
        raise FileNotFoundError(f"{pt_path} not found.")

    out_dir = os.path.dirname(pt_path)
    model = os.path.basename(out_dir)
    dataset = os.path.basename(os.path.dirname(out_dir))

    print("=" * 60)
    print(f"[ANALYZE] dataset={dataset}, model={model}")
    print(f"  Loading score file: {pt_path}")

    # ---- 1. Load single .pt file ----
    cs, ranks, f_mu, f_std, s_mu, s_std = torch.load(pt_path, map_location="cpu")
    q = torch.quantile(cs.flatten(), torch.tensor([0.0, 0.5, 0.9, 0.99]))
    print("  change_scores shape:", tuple(cs.shape))
    print("  ranks shape:", tuple(ranks.shape))
    print(
        "  NaNs?",
        bool(cs.isnan().any()),
        bool(f_mu.isnan().any()),
        bool(s_mu.isnan().any()),
    )
    print("  change_scores percentiles:", [round(v.item(), 8) for v in q])

    # ---- 2. Export top-K neurons ----
    L, H = cs.shape
    flat = cs.flatten()
    topv, topi = torch.topk(flat, K)
    layers = (topi // H).tolist()
    neurons = (topi % H).tolist()

    csv_path = os.path.join(out_dir, f"top{K}_neurons.csv")
    with open(csv_path, "w", newline="") as f:
        wcsv = csv.writer(f)
        wcsv.writerow(["layer", "neuron", "change_score"])
        for l, n, v in zip(layers, neurons, topv.tolist()):
            wcsv.writerow([l, n, v])
    print(f"  Wrote CSV: {csv_path}")

    # ---- 3. Plots ----
    layer_mean = cs.mean(dim=1).numpy()
    layer_p95 = torch.quantile(cs, 0.95, dim=1).numpy()

    # Layer summary plot
    plt.figure()
    plt.plot(range(len(layer_mean)), layer_mean, label="mean")
    plt.plot(range(len(layer_p95)), layer_p95, label="p95")
    plt.xlabel("layer")
    plt.ylabel("change score")
    plt.title(f"Layer-wise change scores ({dataset}, {model})")
    plt.legend()
    plt.tight_layout()
    layer_png = os.path.join(out_dir, "layer_summary.png")
    plt.savefig(layer_png, dpi=200)
    plt.close()
    print(f"  Wrote PNG: {layer_png}")

    # Heatmap
    plt.figure()
    clip = torch.quantile(cs, 0.99).item()
    arr = torch.clamp(cs, max=clip).numpy()[:, ::8]  # downsample hidden dim by 8
    plt.imshow(arr, aspect="auto", origin="lower")
    plt.colorbar(label="change score (clipped p99)")
    plt.xlabel("hidden dim (every 8th)")
    plt.ylabel("layer")
    plt.title(f"Change score heatmap ({dataset}, {model})")
    plt.tight_layout()
    heat_png = os.path.join(out_dir, "heatmap.png")
    plt.savefig(heat_png, dpi=200)
    plt.close()
    print(f"  Wrote PNG: {heat_png}")

    print("  ✅ Done.")
    print("  Outputs:")
    print("    -", csv_path)
    print("    -", layer_png)
    print("    -", heat_png)


if __name__ == "__main__":
    main()
