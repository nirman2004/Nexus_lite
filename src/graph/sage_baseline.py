import torch
import torch.nn as nn
from torch_geometric.nn import SAGEConv


class GraphSAGERegressor(nn.Module):
    def __init__(self, node_dim=10, hidden_dim=128, num_layers=4, out_dim=2, dropout=0.2):
        super().__init__()

        self.node_enc = nn.Sequential(
            nn.Linear(node_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.convs = nn.ModuleList([
            SAGEConv(hidden_dim, hidden_dim) for _ in range(num_layers)
        ])

        self.dropout = nn.Dropout(dropout)

        self.readout = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x, edge_index, edge_feat=None):
        x = self.node_enc(x)

        for conv in self.convs:
            x = conv(x, edge_index)
            x = torch.relu(x)
            x = self.dropout(x)

        h_mean = x.mean(dim=0)
        h_max = x.max(dim=0).values
        graph_emb = torch.cat([h_mean, h_max], dim=-1)

        return self.readout(graph_emb)


if __name__ == "__main__":
    torch.manual_seed(42)
    model = GraphSAGERegressor(node_dim=10, hidden_dim=128, num_layers=4, out_dim=2, dropout=0.2)

    h = torch.randn(6, 10)
    ei = torch.tensor([
        [0, 2], [2, 0], [1, 3], [3, 1],
        [0, 4], [4, 0], [1, 4], [4, 1],
        [2, 5], [5, 2], [3, 5], [5, 3]
    ], dtype=torch.long).t().contiguous()

    out = model(h, ei)
    print("Output shape:", out.shape)
    print("Smoke test passed.")