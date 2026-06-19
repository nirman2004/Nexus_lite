import torch
import torch.nn as nn
import torch.nn.functional as F


class GATLayer(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()

        self.W = nn.Linear(hidden_dim, hidden_dim, bias=False)

        self.attn = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, 1)
        )

        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, h, edge_index):

        src_idx = edge_index[:, 0]
        dst_idx = edge_index[:, 1]

        h_src = self.W(h[src_idx])
        h_dst = self.W(h[dst_idx])

        scores = self.attn(
            torch.cat([h_src, h_dst], dim=-1)
        ).squeeze(-1)

        alpha = torch.zeros_like(scores)

        for node in torch.unique(dst_idx):
            mask = dst_idx == node
            alpha[mask] = torch.softmax(scores[mask], dim=0)

        msg = alpha.unsqueeze(-1) * h_src

        agg = torch.zeros_like(h)

        agg.scatter_add_(
            0,
            dst_idx.unsqueeze(1).expand_as(msg),
            msg
        )

        return self.norm(h + agg)


class GATBaseline(nn.Module):

    def __init__(
        self,
        node_dim=10,
        hidden_dim=128,
        num_layers=4,
        out_dim=2,
        dropout=0.2
    ):
        super().__init__()

        self.node_enc = nn.Sequential(
            nn.Linear(node_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim)
        )

        self.layers = nn.ModuleList([
            GATLayer(hidden_dim)
            for _ in range(num_layers)
        ])

        self.readout = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim)
        )

    def forward(
        self,
        h,
        edge_index,
        edge_feat=None
    ):

        h = self.node_enc(h)

        for layer in self.layers:
            h = layer(h, edge_index)

        h_mean = h.mean(dim=0)
        h_max = h.max(dim=0).values

        graph_emb = torch.cat(
            [h_mean, h_max],
            dim=-1
        )

        return self.readout(graph_emb)