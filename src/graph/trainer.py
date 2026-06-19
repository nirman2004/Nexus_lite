"""
trainer.py — NEXUS-Lite updated training script

Changes:
- hidden_dim = 256
- max_iter = 40
- epsilon = 0.05
- edge-aware message projection in the model
- mean + max + std pooling in readout
"""

import json
import os
import random
import sys

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

sys.path.insert(0, os.path.dirname(__file__))
from gnn_models import NEXUSLiteGNN

GRAPHS_PATH = "data/graphs/graphs.json"
CHECKPOINT_DIR = "checkpoints"
LOG_PATH = "data/graphs/training_log.json"

HIDDEN_DIM = 256
NUM_LAYERS = 4
NODE_DIM = 10
LR = 3e-4
WEIGHT_DECAY = 1e-3
EPOCHS = 300
TRAIN_SPLIT = 0.8
PATIENCE = 30
SEED = 42

os.makedirs(CHECKPOINT_DIR, exist_ok=True)


def graph_to_tensors(g):
    h = torch.tensor(g["nodes"], dtype=torch.float)
    edge_index = torch.tensor([[e[0], e[1]] for e in g["edges"]], dtype=torch.long)
    edge_feat = torch.tensor([[e[2]] for e in g["edges"]], dtype=torch.float)
    targets = torch.tensor(g["targets"], dtype=torch.float)
    return h, edge_index, edge_feat, targets


def normalize_features(graphs):
    all_nodes = torch.cat([torch.tensor(g["nodes"], dtype=torch.float) for g in graphs], dim=0)
    all_targets = torch.stack([torch.tensor(g["targets"], dtype=torch.float) for g in graphs], dim=0)
    return {
        "node_mean": all_nodes.mean(0),
        "node_std": all_nodes.std(0).clamp(min=1e-6),
        "tgt_mean": all_targets.mean(0),
        "tgt_std": all_targets.std(0).clamp(min=1e-6),
    }


def train():
    random.seed(SEED)
    torch.manual_seed(SEED)

    with open(GRAPHS_PATH) as f:
        graphs = json.load(f)
    print(f"Loaded {len(graphs)} graphs")

    stats = normalize_features(graphs)
    random.shuffle(graphs)

    n_train = int(len(graphs) * TRAIN_SPLIT)
    train_graphs = graphs[:n_train]
    val_graphs = graphs[n_train:]

    print(f"Train: {len(train_graphs)}  Val: {len(val_graphs)}")

    model = NEXUSLiteGNN(
        node_dim=NODE_DIM,
        edge_dim=1,
        hidden_dim=HIDDEN_DIM,
        num_layers=NUM_LAYERS,
        dropout=0.2,
        epsilon=0.05,
        max_iter=40,
        alpha=0.3,
    )
    total = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {total:,}  (equilibrium GNN, updated)")

    optimizer = Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=15, min_lr=1e-6
    )
    criterion = nn.MSELoss()

    best_val_loss = float("inf")
    patience_count = 0
    log = {"train_loss": [], "val_loss": [], "avg_conv_steps": []}

    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_loss = 0.0
        conv_steps = 0

        for g in train_graphs:
            h, edge_index, edge_feat, targets = graph_to_tensors(g)
            h_norm = (h - stats["node_mean"]) / stats["node_std"]
            tgt_norm = (targets - stats["tgt_mean"]) / stats["tgt_std"]

            optimizer.zero_grad()
            pred = model(h_norm, edge_index, edge_feat)
            loss = criterion(pred, tgt_norm)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_loss += loss.item()
            conv_steps += model.last_convergence_steps

        train_loss /= len(train_graphs)
        avg_steps = conv_steps / len(train_graphs)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for g in val_graphs:
                h, edge_index, edge_feat, targets = graph_to_tensors(g)
                h_norm = (h - stats["node_mean"]) / stats["node_std"]
                tgt_norm = (targets - stats["tgt_mean"]) / stats["tgt_std"]

                pred = model(h_norm, edge_index, edge_feat)
                val_loss += criterion(pred, tgt_norm).item()

        val_loss /= len(val_graphs)

        scheduler.step(val_loss)
        log["train_loss"].append(round(train_loss, 6))
        log["val_loss"].append(round(val_loss, 6))
        log["avg_conv_steps"].append(round(avg_steps, 2))

        if epoch % 20 == 0 or epoch == 1:
            lr_now = optimizer.param_groups[0]["lr"]
            print(
                f"Epoch {epoch:3d}/{EPOCHS}  "
                f"train={train_loss:.4f}  val={val_loss:.4f}  "
                f"conv_steps={avg_steps:.1f}  lr={lr_now:.2e}"
            )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_count = 0
            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "stats": {k: v.tolist() for k, v in stats.items()},
                "val_loss": val_loss,
                "config": {
                    "hidden_dim": HIDDEN_DIM,
                    "num_layers": NUM_LAYERS,
                    "node_dim": NODE_DIM,
                    "edge_dim": 1,
                    "dropout": 0.2,
                    "epsilon": 0.05,
                    "max_iter": 40,
                    "alpha": 0.3,
                }
            }, os.path.join(CHECKPOINT_DIR, "best_model.pt"))
        else:
            patience_count += 1
            if patience_count >= PATIENCE:
                print(f"\nEarly stopping at epoch {epoch}")
                break

    print(f"\nBest val loss : {best_val_loss:.4f}")
    print(f"Checkpoint    → {CHECKPOINT_DIR}/best_model.pt")
    with open(LOG_PATH, "w") as f:
        json.dump(log, f, indent=2)
    print(f"Training log  → {LOG_PATH}")
    print("\nNote: avg_conv_steps should DECREASE over training")
    print("as weights learn stronger contractive dynamics.")


if __name__ == "__main__":
    train()