"""
gnn_models.py — NEXUS-Lite updated for stronger surrogate performance

Changes:
- hidden_dim default increased to 256
- max_iter default increased to 40
- epsilon default tightened to 0.05
- edge features now also enter message content, not only the gate
- readout uses mean + max + std pooling
"""

import torch
import torch.nn as nn


class EdgeGatedConv(nn.Module):
    def __init__(self, hidden_dim: int, edge_dim: int, dropout: float = 0.0):
        super().__init__()
        self.W_msg = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.edge_proj = nn.Linear(edge_dim, hidden_dim, bias=False)

        # gate input = h_src || h_dst || edge_feat
        self.W_gate = nn.Linear(hidden_dim * 2 + edge_dim, 1)

        self.W_upd = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, h, edge_index, edge_feat):
        src_idx = edge_index[:, 0]
        dst_idx = edge_index[:, 1]

        h_src = h[src_idx]
        h_dst = h[dst_idx]

        gate = torch.sigmoid(
            self.W_gate(torch.cat([h_src, h_dst, edge_feat], dim=-1))
        )

        msg = gate * (self.W_msg(h_src) + self.edge_proj(edge_feat))

        agg = torch.zeros_like(h)
        count = torch.zeros(h.size(0), 1, device=h.device)

        agg.scatter_add_(0, dst_idx.unsqueeze(1).expand_as(msg), msg)
        count.scatter_add_(
            0,
            dst_idx.unsqueeze(1),
            torch.ones(msg.size(0), 1, device=h.device),
        )

        agg = agg / count.clamp(min=1.0)

        h_new = self.W_upd(torch.cat([h, agg], dim=-1))
        return self.norm(h + h_new)


class NEXUSLiteGNN(nn.Module):
    def __init__(
        self,
        node_dim: int = 10,
        edge_dim: int = 1,
        hidden_dim: int = 256,
        num_layers: int = 4,   # kept for config compatibility
        out_dim: int = 2,
        dropout: float = 0.2,
        epsilon: float = 0.05,
        max_iter: int = 40,
        alpha: float = 0.3,
    ):
        super().__init__()
        self.epsilon = epsilon
        self.max_iter = max_iter
        self.alpha = alpha

        self.node_enc = nn.Sequential(
            nn.Linear(node_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.edge_enc = nn.Linear(edge_dim, edge_dim)
        self.pressure_conv = EdgeGatedConv(hidden_dim, edge_dim, dropout)

        self.readout = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )

        self.last_convergence_steps = 0
        self.last_pressure_history = []

    def forward(self, h, edge_index, edge_feat, track_history=False):
        h = self.node_enc(h)
        e = self.edge_enc(edge_feat)

        if track_history:
            self.last_pressure_history = [h.detach().norm(dim=1).tolist()]

        for t in range(self.max_iter):
            h_prop = self.pressure_conv(h, edge_index, e)
            h_new = (1 - self.alpha) * h + self.alpha * h_prop
            delta = (h_new - h).norm() / (h.size(0) ** 0.5)
            h = h_new

            if track_history:
                self.last_pressure_history.append(h.detach().norm(dim=1).tolist())

            if delta.item() < self.epsilon:
                self.last_convergence_steps = t + 1
                break
        else:
            self.last_convergence_steps = self.max_iter

        h_mean = h.mean(dim=0)
        h_max = h.max(dim=0).values
        h_std = h.std(dim=0)

        return self.readout(torch.cat([h_mean, h_max, h_std], dim=-1))


if __name__ == "__main__":
    torch.manual_seed(42)

    model = NEXUSLiteGNN(
        node_dim=10,
        edge_dim=1,
        hidden_dim=256,
        num_layers=4,
        dropout=0.2,
        epsilon=0.05,
        max_iter=40,
        alpha=0.3,
    )

    ckpt = torch.load("checkpoints/best_model.pt", map_location="cpu")
    if "model_state" in ckpt:
        try:
            model.load_state_dict(ckpt["model_state"], strict=False)
            print("Checkpoint loaded successfully.")
        except Exception as e:
            print("Checkpoint load skipped:", e)

    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    h = torch.randn(6, 10)
    ei = torch.tensor(
        [
            [0, 2], [2, 0], [1, 3], [3, 1],
            [0, 4], [4, 0], [1, 4], [4, 1],
            [2, 5], [5, 2], [3, 5], [5, 3]
        ],
        dtype=torch.long,
    )
    ef = torch.tensor([[0], [0], [0], [0], [1], [1], [1], [1], [2], [2], [2], [2]], dtype=torch.float)

    out = model(h, ei, ef)
    print(f"Output: {out.shape}  conv_steps={model.last_convergence_steps}")
    print("Smoke test passed.")