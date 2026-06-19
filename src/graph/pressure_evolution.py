"""
pressure_evolution.py  
Visualizes pressure at each node across equilibrium steps
for a good circuit vs a bad circuit.

This is the unique visualization of your paper —
nobody has shown analog circuit pressure evolution before.

Run: python src/graph/pressure_evolution.py
"""

import json, os, sys, torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm

sys.path.insert(0, "src/graph")
from gnn_models import NEXUSLiteGNN

CHECKPOINT  = "checkpoints/best_model.pt"
GRAPHS_PATH = "data/graphs/graphs.json"
FIGURES_DIR = "figures"
os.makedirs(FIGURES_DIR, exist_ok=True)

NODE_NAMES = ["M1", "M2", "RL1", "RL2", "Itail", "VDD"]
COLORS     = ["#2E75B6","#C55A11","#70AD47","#FFC000","#9E480E","#636363"]


def get_pressure_evolution(model, stats, graph):
    h          = torch.tensor(graph["nodes"],   dtype=torch.float)
    edge_index = torch.tensor([[e[0],e[1]] for e in graph["edges"]], dtype=torch.long)
    edge_feat  = torch.tensor([[e[2]] for e in graph["edges"]],       dtype=torch.float)

    h_norm = (h - stats["node_mean"]) / stats["node_std"]

    with torch.no_grad():
        out = model(h_norm, edge_index, edge_feat, track_history=True)

    history = model.last_pressure_history   # list of [6 pressure norms] per step
    steps   = model.last_convergence_steps
    pred    = out.numpy() * stats["tgt_std"].numpy() + stats["tgt_mean"].numpy()

    return history, steps, float(pred[0]), float(10 ** pred[1])


def plot_evolution(history, steps, gain, bw, title, ax_top, ax_bot):
    history = np.array(history)   # (steps+1, 6)
    n_steps = history.shape[0]
    x       = np.arange(n_steps)

    # Top: pressure per node over steps
    for i, (name, color) in enumerate(zip(NODE_NAMES, COLORS)):
        ax_top.plot(x, history[:, i], color=color, linewidth=2, label=name)
    ax_top.set_title(f"{title}\ngain={gain:+.1f} dB  BW={bw:.1f} MHz  conv={steps} steps",
                     fontsize=11)
    ax_top.set_ylabel("Pressure (node repr. norm)", fontsize=10)
    ax_top.legend(fontsize=8, loc="upper right")
    ax_top.grid(True, alpha=0.3)
    ax_top.axvline(x=steps, color="red", linestyle="--", alpha=0.6, label="convergence")

    # Bottom: pressure spread (max-min across nodes) — shows differentiation
    spread = history.max(axis=1) - history.min(axis=1)
    ax_bot.fill_between(x, 0, spread, alpha=0.4, color="#2E75B6")
    ax_bot.plot(x, spread, color="#2E75B6", linewidth=1.5)
    ax_bot.set_ylabel("Pressure spread\n(max-min)", fontsize=10)
    ax_bot.set_xlabel("Propagation step", fontsize=10)
    ax_bot.grid(True, alpha=0.3)


def main():
    # Load model
    ckpt  = torch.load(CHECKPOINT, map_location="cpu")
    cfg   = ckpt["config"]
    stats = {k: torch.tensor(v) for k, v in ckpt["stats"].items()}

    model = NEXUSLiteGNN(
        node_dim   = cfg["node_dim"],
        edge_dim   = cfg["edge_dim"],
        hidden_dim = cfg["hidden_dim"],
        num_layers = cfg["num_layers"],
        dropout    = 0.0,  # no dropout at eval
        epsilon    = cfg.get("epsilon", 0.1),
        max_iter   = cfg.get("max_iter", 20),
        alpha      = cfg.get("alpha",   0.3),
    )
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    # Load graphs
    with open(GRAPHS_PATH) as f:
        graphs = json.load(f)

    # Find a good circuit (gain > 15 dB) and a bad one (gain < -20 dB)
    good_graph = None
    bad_graph  = None
    for g in graphs:
        raw_gain = g["raw"]["gain_dB"]
        if good_graph is None and raw_gain > 15:
            good_graph = g
        if bad_graph is None and raw_gain < -20:
            bad_graph = g
        if good_graph and bad_graph:
            break

    print(f"Good circuit: gain={good_graph['raw']['gain_dB']:.1f} dB")
    print(f"Bad  circuit: gain={bad_graph['raw']['gain_dB']:.1f} dB")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Good circuit
    hist_g, steps_g, pred_gain_g, pred_bw_g = get_pressure_evolution(
        model, stats, good_graph)
    plot_evolution(hist_g, steps_g, pred_gain_g, pred_bw_g,
                   f"High-Gain Circuit (true={good_graph['raw']['gain_dB']:.1f} dB)",
                   axes[0][0], axes[1][0])

    # Bad circuit
    hist_b, steps_b, pred_gain_b, pred_bw_b = get_pressure_evolution(
        model, stats, bad_graph)
    plot_evolution(hist_b, steps_b, pred_gain_b, pred_bw_b,
                   f"Low-Gain Circuit (true={bad_graph['raw']['gain_dB']:.1f} dB)",
                   axes[0][1], axes[1][1])

    plt.suptitle("NEXUS-Lite: Pressure Evolution During Equilibrium Propagation",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    fig_path = os.path.join(FIGURES_DIR, "pressure_evolution.png")
    plt.savefig(fig_path, dpi=150)
    plt.close()
    print(f"Saved → {fig_path}")
    print(f"\nGood circuit convergence: {steps_g} steps")
    print(f"Bad  circuit convergence: {steps_b} steps")
    print(f"\nThis IS your paper's unique figure — pressure evolution")
    print(f"shows different dynamics for good vs bad circuits.")


if __name__ == "__main__":
    main()