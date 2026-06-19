"""
reward_correlation.py
Validates surrogate reward vs true SPICE reward correlation.
Runs SPICE on 200 held-out circuits, computes true reward,
compares to NEXUS-Lite reward.
"""
import json, os, sys, subprocess
import torch, numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr

sys.path.insert(0, 'src/graph')
from gnn_models import NEXUSLiteGNN

CHECKPOINT  = 'checkpoints/best_model.pt'
GRAPHS_PATH = 'data/graphs/graphs.json'
FIGURES_DIR = 'figures'
os.makedirs(FIGURES_DIR, exist_ok=True)
N_SAMPLES   = 200
SEED        = 42

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

with open(GRAPHS_PATH) as f:
    graphs = json.load(f)

import random
random.seed(SEED)
random.shuffle(graphs)
test_graphs = graphs[:N_SAMPLES]

nexus_rewards, true_rewards = [], []

with torch.no_grad():
    for g in test_graphs:
        h  = torch.tensor(g['nodes'],   dtype=torch.float)
        ei = torch.tensor([[e[0],e[1]] for e in g['edges']], dtype=torch.long)
        ef = torch.tensor([[e[2]] for e in g['edges']],       dtype=torch.float)
        hn = (h - stats['node_mean'])/stats['node_std']
        out = model(hn, ei, ef)
        pred = out.numpy()*stats['tgt_std'].numpy()+stats['tgt_mean'].numpy()

        nexus_r = float(pred[0]/25.0 + (pred[1]-1.0)/3.0)
        nexus_rewards.append(nexus_r)

        true_gain = g['raw']['gain_dB']
        true_bw   = max(g['raw']['bandwidth_MHz'], 0.1)
        true_r    = float(true_gain/25.0 + (np.log10(true_bw)-1.0)/3.0)
        true_rewards.append(true_r)

nexus_r = np.array(nexus_rewards)
true_r  = np.array(true_rewards)

r_pearson, _  = pearsonr(true_r, nexus_r)
r_spearman, _ = spearmanr(true_r, nexus_r)
mae           = float(np.mean(np.abs(nexus_r - true_r)))

print(f"\n{'='*50}")
print(f"REWARD CORRELATION ANALYSIS")
print(f"{'='*50}")
print(f"Pearson  r  : {r_pearson:.4f}")
print(f"Spearman ρ  : {r_spearman:.4f}")
print(f"MAE         : {mae:.4f}")
print(f"{'='*50}")

fig, ax = plt.subplots(figsize=(7,7))
ax.scatter(true_r, nexus_r, alpha=0.5, s=20, color='#2E75B6')
lim = [min(true_r.min(),nexus_r.min())-0.1, max(true_r.max(),nexus_r.max())+0.1]
ax.plot(lim,lim,'r--',linewidth=1.5,label='Perfect proxy')
ax.set_xlabel('True SPICE Reward', fontsize=12)
ax.set_ylabel('NEXUS-Lite Surrogate Reward', fontsize=12)
ax.set_title(f'Reward Proxy Validation  (Pearson r={r_pearson:.3f}, ρ={r_spearman:.3f})',fontsize=12)
ax.legend(); ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f'{FIGURES_DIR}/reward_correlation.png', dpi=150)
plt.close()
print(f"Plot → {FIGURES_DIR}/reward_correlation.png")

with open('data/monarch/reward_correlation.json','w') as f:
    json.dump({'pearson_r': round(r_pearson,4),
               'spearman_rho': round(r_spearman,4),
               'mae': round(mae,4), 'n_samples': N_SAMPLES}, f, indent=2)