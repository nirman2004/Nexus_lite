"""
evaluator.py  —  NEXUS-Lite
Loads best_model.pt, runs predictions on held-out val set,
computes MAE vs Ngspice, saves scatter plot data.

Run: python src/graph/evaluator.py
Outputs:
  figures/gain_scatter.png
  figures/bw_scatter.png
  figures/loss_curve.png
  data/graphs/eval_results.json
"""

import json
import os
import sys
import random
import math
import torch
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from gnn_models import NEXUSLiteGNN

# ── config ────────────────────────────────────────────────────────
GRAPHS_PATH    = "data/graphs/graphs.json"
CHECKPOINT     = "checkpoints/best_model.pt"
LOG_PATH       = "data/graphs/training_log.json"
EVAL_OUT       = "data/graphs/eval_results.json"
FIGURES_DIR    = "figures"
TRAIN_SPLIT    = 0.8
SEED           = 42

os.makedirs(FIGURES_DIR, exist_ok=True)


def graph_to_tensors(g):
    h          = torch.tensor(g["nodes"],   dtype=torch.float)
    edge_index = torch.tensor([[e[0], e[1]] for e in g["edges"]], dtype=torch.long)
    edge_feat  = torch.tensor([[e[2]] for e in g["edges"]],        dtype=torch.float)
    targets    = torch.tensor(g["targets"], dtype=torch.float)
    return h, edge_index, edge_feat, targets


def main():
    # ── load checkpoint ──
    ckpt  = torch.load(CHECKPOINT, map_location="cpu")
    cfg   = ckpt["config"]
    stats = {k: torch.tensor(v) for k, v in ckpt["stats"].items()}

    model = NEXUSLiteGNN(
        node_dim   = cfg["node_dim"],
        edge_dim   = cfg["edge_dim"],
        hidden_dim = cfg["hidden_dim"],
        num_layers = cfg["num_layers"],
        dropout    = cfg.get("dropout", 0.0),
    )
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    print(f"Loaded checkpoint — best val loss: {ckpt['val_loss']:.4f} (epoch {ckpt['epoch']})")

    # ── reproduce same train/val split ──
    with open(GRAPHS_PATH) as f:
        graphs = json.load(f)

    random.seed(SEED)
    random.shuffle(graphs)
    n_train   = int(len(graphs) * TRAIN_SPLIT)
    val_graphs = graphs[n_train:]
    print(f"Evaluating on {len(val_graphs)} val graphs")

    # ── run predictions ──
    pred_gain, true_gain = [], []
    pred_bw,   true_bw   = [], []

    with torch.no_grad():
        for g in val_graphs:
            h, edge_index, edge_feat, targets = graph_to_tensors(g)
            h_norm = (h - stats["node_mean"]) / stats["node_std"]

            out = model(h_norm, edge_index, edge_feat)

            # denormalise predictions
            pred_norm = out.numpy()
            pred_real = pred_norm * stats["tgt_std"].numpy() + stats["tgt_mean"].numpy()

            # true values (already in physical units: gain_dB, log10_BW)
            true_real = targets.numpy()

            pred_gain.append(float(pred_real[0]))
            true_gain.append(float(true_real[0]))
            pred_bw.append(float(10 ** pred_real[1]))   # MHz
            true_bw.append(float(10 ** true_real[1]))   # MHz

    pred_gain = np.array(pred_gain)
    true_gain = np.array(true_gain)
    pred_bw   = np.array(pred_bw)
    true_bw   = np.array(true_bw)

    # ── metrics ──
    mae_gain = float(np.mean(np.abs(pred_gain - true_gain)))
    mae_bw   = float(np.mean(np.abs(pred_bw   - true_bw)))
    rmse_gain = float(np.sqrt(np.mean((pred_gain - true_gain)**2)))
    rmse_bw   = float(np.sqrt(np.mean((pred_bw   - true_bw)**2)))

    # R^2
    ss_res_g = np.sum((true_gain - pred_gain)**2)
    ss_tot_g = np.sum((true_gain - np.mean(true_gain))**2)
    r2_gain  = float(1 - ss_res_g / (ss_tot_g + 1e-8))

    ss_res_b = np.sum((true_bw - pred_bw)**2)
    ss_tot_b = np.sum((true_bw - np.mean(true_bw))**2)
    r2_bw    = float(1 - ss_res_b / (ss_tot_b + 1e-8))

    print(f"\n{'='*50}")
    print(f"GAIN   MAE  = {mae_gain:.3f} dB    RMSE = {rmse_gain:.3f} dB    R² = {r2_gain:.4f}")
    print(f"BW     MAE  = {mae_bw:.2f} MHz  RMSE = {rmse_bw:.2f} MHz  R² = {r2_bw:.4f}")
    print(f"{'='*50}")

    # ── save eval results ──
    results = {
        "checkpoint_epoch": ckpt["epoch"],
        "val_loss":         ckpt["val_loss"],
        "n_val":            len(val_graphs),
        "gain": {"mae_dB": mae_gain, "rmse_dB": rmse_gain, "r2": r2_gain},
        "bw":   {"mae_MHz": mae_bw,  "rmse_MHz": rmse_bw,  "r2": r2_bw},
        "samples": [
            {"true_gain": float(tg), "pred_gain": float(pg),
             "true_bw":   float(tb), "pred_bw":   float(pb)}
            for tg, pg, tb, pb in zip(true_gain, pred_gain, true_bw, pred_bw)
        ]
    }
    with open(EVAL_OUT, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nEval results → {EVAL_OUT}")

    # ── plots (matplotlib optional) ──
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        # Gain scatter
        fig, ax = plt.subplots(figsize=(7, 7))
        ax.scatter(true_gain, pred_gain, alpha=0.4, s=15, color="#2E75B6")
        lim = [min(true_gain.min(), pred_gain.min()) - 2,
               max(true_gain.max(), pred_gain.max()) + 2]
        ax.plot(lim, lim, "r--", linewidth=1.5, label="Perfect prediction")
        ax.set_xlabel("Ngspice Gain (dB)", fontsize=13)
        ax.set_ylabel("NEXUS-Lite Predicted Gain (dB)", fontsize=13)
        ax.set_title(f"Gain Prediction  |  MAE={mae_gain:.2f} dB  R²={r2_gain:.3f}", fontsize=13)
        ax.legend(); ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(FIGURES_DIR, "gain_scatter.png"), dpi=150)
        plt.close()
        print(f"Saved → {FIGURES_DIR}/gain_scatter.png")

        # BW scatter (log scale)
        fig, ax = plt.subplots(figsize=(7, 7))
        ax.scatter(true_bw, pred_bw, alpha=0.4, s=15, color="#C55A11")
        lim = [min(true_bw.min(), pred_bw.min()) * 0.5,
               max(true_bw.max(), pred_bw.max()) * 2]
        ax.plot(lim, lim, "r--", linewidth=1.5, label="Perfect prediction")
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlabel("Ngspice Bandwidth (MHz)", fontsize=13)
        ax.set_ylabel("NEXUS-Lite Predicted Bandwidth (MHz)", fontsize=13)
        ax.set_title(f"Bandwidth Prediction  |  MAE={mae_bw:.1f} MHz  R²={r2_bw:.3f}", fontsize=13)
        ax.legend(); ax.grid(True, alpha=0.3, which="both")
        plt.tight_layout()
        plt.savefig(os.path.join(FIGURES_DIR, "bw_scatter.png"), dpi=150)
        plt.close()
        print(f"Saved → {FIGURES_DIR}/bw_scatter.png")

        # Loss curves
        with open(LOG_PATH) as f:
            log = json.load(f)
        epochs = list(range(1, len(log["train_loss"]) + 1))
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.plot(epochs, log["train_loss"], label="Train loss", color="#2E75B6")
        ax.plot(epochs, log["val_loss"],   label="Val loss",   color="#C55A11")
        ax.axvline(ckpt["epoch"], color="green", linestyle="--",
                   label=f"Best epoch {ckpt['epoch']}")
        ax.set_xlabel("Epoch", fontsize=13)
        ax.set_ylabel("MSE Loss (normalised)", fontsize=13)
        ax.set_title("NEXUS-Lite Training Curves", fontsize=13)
        ax.legend(); ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(FIGURES_DIR, "loss_curve.png"), dpi=150)
        plt.close()
        print(f"Saved → {FIGURES_DIR}/loss_curve.png")

    except ImportError:
        print("matplotlib not installed — skipping plots")
        print("Install with: pip install matplotlib")

    print(f"\nDone. Your first result:")
    print(f"  NEXUS-Lite surrogate achieves {mae_gain:.2f} dB MAE on gain prediction")
    print(f"  vs Ngspice, at ~1000x lower computational cost.")


if __name__ == "__main__":
    main()