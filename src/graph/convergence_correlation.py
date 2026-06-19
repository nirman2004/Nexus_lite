"""
convergence_correlation.py
Tests whether T* (equilibrium convergence steps) correlates
with circuit stability proxies extracted from AC data.
Stability proxy: frequency at which gain first drops 3dB from peak
(= dominant pole frequency = 1 / (2*pi*tau) ~ circuit stability).
Circuits that settle quickly (low tau) should converge in fewer steps.
"""
import json, os, sys, glob
import torch, numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, pearsonr

sys.path.insert(0, 'src/graph')
from gnn_models import NEXUSLiteGNN

RESULT_DIR  = 'data/results'
GRAPHS_PATH = 'data/graphs/graphs.json'
CHECKPOINT  = 'checkpoints/best_model.pt'
FIGURES_DIR = 'figures'
os.makedirs(FIGURES_DIR, exist_ok=True)

# ── Load model ────────────────────────────────────────────────────
ckpt  = torch.load(CHECKPOINT, map_location='cpu')
cfg   = ckpt['config']
stats = {k: torch.tensor(v) for k, v in ckpt['stats'].items()}
model = NEXUSLiteGNN(
    node_dim=cfg['node_dim'], edge_dim=cfg['edge_dim'],
    hidden_dim=cfg['hidden_dim'], num_layers=cfg['num_layers'],
    dropout=0.0, epsilon=cfg.get('epsilon',0.1),
    max_iter=cfg.get('max_iter',20), alpha=cfg.get('alpha',0.3)
)
model.load_state_dict(ckpt['model_state'])
model.eval()

# ── Load graphs ───────────────────────────────────────────────────
with open(GRAPHS_PATH) as f:
    graphs = json.load(f)

# ── For each graph: get T* and bandwidth (stability proxy) ────────
conv_steps_list = []
bw_list         = []
gain_list       = []

with torch.no_grad():
    for g in graphs[:2000]:
        h  = torch.tensor(g['nodes'],   dtype=torch.float)
        ei = torch.tensor([[e[0],e[1]] for e in g['edges']], dtype=torch.long)
        ef = torch.tensor([[e[2]] for e in g['edges']],       dtype=torch.float)
        h_norm = (h - stats['node_mean']) / stats['node_std']
        model(h_norm, ei, ef)
        conv_steps_list.append(model.last_convergence_steps)
        bw_list.append(g['raw']['bandwidth_MHz'])
        gain_list.append(g['raw']['gain_dB'])

T_star = np.array(conv_steps_list)
BW     = np.array(bw_list)
Gain   = np.array(gain_list)

# Filter out ceiling hits and non-physical values
mask = (BW < 990) & (Gain > -40) & (Gain < 25)
T_f  = T_star[mask]
BW_f = np.log10(BW[mask] + 0.1)
G_f  = Gain[mask]

rho_bw,  p_bw  = spearmanr(T_f, BW_f)
rho_g,   p_g   = spearmanr(T_f, G_f)
r_bw,    _     = pearsonr(T_f,  BW_f)

print(f"\n{'='*50}")
print(f"CONVERGENCE CORRELATION ANALYSIS")
print(f"{'='*50}")
print(f"Samples analyzed   : {mask.sum()}")
print(f"T* vs log10(BW)  Spearman ρ = {rho_bw:.4f}  p={p_bw:.4e}")
print(f"T* vs Gain       Spearman ρ = {rho_g:.4f}   p={p_g:.4e}")
print(f"T* vs log10(BW)  Pearson  r = {r_bw:.4f}")
print(f"{'='*50}")

# ── Plot ──────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

ax = axes[0]
ax.scatter(BW_f, T_f, alpha=0.3, s=12, color='#2E75B6')
ax.set_xlabel('log10(Bandwidth) [MHz]', fontsize=12)
ax.set_ylabel('Convergence steps T*', fontsize=12)
ax.set_title(f'T* vs Bandwidth  (ρ={rho_bw:.3f})', fontsize=12)
ax.grid(True, alpha=0.3)

ax = axes[1]
ax.scatter(G_f, T_f, alpha=0.3, s=12, color='#C55A11')
ax.set_xlabel('Gain (dB)', fontsize=12)
ax.set_ylabel('Convergence steps T*', fontsize=12)
ax.set_title(f'T* vs Gain  (ρ={rho_g:.3f})', fontsize=12)
ax.grid(True, alpha=0.3)

plt.suptitle('Pressure Equilibrium Convergence vs. Circuit Performance', fontsize=13)
plt.tight_layout()
fig_path = os.path.join(FIGURES_DIR, 'convergence_correlation.png')
plt.savefig(fig_path, dpi=150)
plt.close()
print(f"Plot → {fig_path}")

results = {
    'n_samples': int(mask.sum()),
    'spearman_T_BW': round(rho_bw, 4),
    'spearman_T_Gain': round(rho_g, 4),
    'pearson_T_BW': round(r_bw, 4),
    'p_bw': float(p_bw), 'p_gain': float(p_g)
}
with open('data/monarch/convergence_correlation.json', 'w') as f:
    json.dump(results, f, indent=2)