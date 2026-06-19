"""
timing_benchmark_v2.py 
"""
import time, subprocess, torch, json, sys, os
import numpy as np

sys.path.insert(0, 'src/graph')
from gnn_models import NEXUSLiteGNN

# ── 1. SPICE timing ───────────────────────────────────────────────
N_WARMUP = 3
N_SPICE  = 30
netlists = [f'data/netlists/{i}.sp' for i in range(N_SPICE + N_WARMUP)]

# warmup
for nl in netlists[:N_WARMUP]:
    subprocess.run(['ngspice', '-b', nl], capture_output=True)

spice_times = []
for nl in netlists[N_WARMUP:N_WARMUP+N_SPICE]:
    t0 = time.perf_counter()
    subprocess.run(['ngspice', '-b', nl], capture_output=True)
    spice_times.append(time.perf_counter() - t0)

spice_mean = np.mean(spice_times) * 1000
spice_std  = np.std(spice_times)  * 1000

# ── 2. NEXUS-Lite timing (with warmup) ───────────────────────────
ckpt  = torch.load('checkpoints/best_model.pt', map_location='cpu')
cfg   = ckpt['config']
stats = {k: torch.tensor(v) for k, v in ckpt['stats'].items()}

model = NEXUSLiteGNN(
    node_dim=cfg['node_dim'], edge_dim=cfg['edge_dim'],
    hidden_dim=cfg['hidden_dim'], num_layers=cfg['num_layers'],
    dropout=0.0, epsilon=cfg.get('epsilon',0.1),
    max_iter=cfg.get('max_iter',20), alpha=cfg.get('alpha',0.3),
)
model.load_state_dict(ckpt['model_state'])
model.eval()

import json as json_mod
with open('data/graphs/graphs.json') as f:
    graphs = json_mod.load(f)

def g2t(g):
    h  = torch.tensor(g['nodes'],   dtype=torch.float)
    ei = torch.tensor([[e[0],e[1]] for e in g['edges']], dtype=torch.long)
    ef = torch.tensor([[e[2]] for e in g['edges']],       dtype=torch.float)
    return h, ei, ef

# warmup
with torch.no_grad():
    for g in graphs[:10]:
        h, ei, ef = g2t(g)
        hn = (h - stats['node_mean']) / stats['node_std']
        _ = model(hn, ei, ef)

N_INF = 500
nexus_times = []
with torch.no_grad():
    for g in graphs[:N_INF]:
        h, ei, ef = g2t(g)
        hn = (h - stats['node_mean']) / stats['node_std']
        t0 = time.perf_counter()
        _ = model(hn, ei, ef)
        nexus_times.append(time.perf_counter() - t0)

nexus_mean = np.mean(nexus_times) * 1000
nexus_std  = np.std(nexus_times)  * 1000
speedup    = spice_mean / nexus_mean

# median is more robust than mean for this
spice_med  = np.median(spice_times) * 1000
nexus_med  = np.median(nexus_times) * 1000
speedup_med = spice_med / nexus_med

print(f"\n{'='*55}")
print(f"WALL-CLOCK BENCHMARK (with warmup)")
print(f"{'='*55}")
print(f"Ngspice   mean: {spice_mean:.1f} ± {spice_std:.1f} ms  (n={N_SPICE})")
print(f"Ngspice median: {spice_med:.1f} ms")
print(f"NEXUS-Lite mean: {nexus_mean:.4f} ± {nexus_std:.4f} ms  (n={N_INF})")
print(f"NEXUS-Lite median: {nexus_med:.4f} ms")
print(f"Speedup (mean)  : {speedup:.0f}x")
print(f"Speedup (median): {speedup_med:.0f}x")
print(f"{'='*55}")

results = {
    'spice_mean_ms':   round(spice_mean, 2),
    'spice_median_ms': round(spice_med,  2),
    'spice_std_ms':    round(spice_std,  2),
    'nexus_mean_ms':   round(nexus_mean, 5),
    'nexus_median_ms': round(nexus_med,  5),
    'nexus_std_ms':    round(nexus_std,  5),
    'speedup_mean':    round(speedup,    1),
    'speedup_median':  round(speedup_med,1),
}
os.makedirs('data/monarch', exist_ok=True)
with open('data/monarch/timing_benchmark.json', 'w') as f:
    json.dump(results, f, indent=2)
print(f"Saved → data/monarch/timing_benchmark.json")