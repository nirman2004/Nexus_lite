"""
current_mirror_v2.py 
Uses sequential simulation instead of multiprocessing Pool.
"""
import json, os, random, subprocess, glob
import numpy as np
import torch, torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
import math, sys

sys.path.insert(0, 'src/graph')
from gnn_models import NEXUSLiteGNN

MIRROR_DIR  = 'data/mirror'
NETLIST_DIR = 'data/mirror/netlists'
RESULT_DIR  = 'data/mirror/results'
GRAPHS_PATH = 'data/mirror/graphs.json'
CKPT_PATH   = 'checkpoints/mirror_model.pt'
FIGURES_DIR = 'figures'
MODEL_FILE  = 'models/nmos.lib'
ABS_RESULT  = os.path.abspath(RESULT_DIR)
N_SAMPLES   = 1000   # reduced for speed
SEED        = 42

for d in [MIRROR_DIR, NETLIST_DIR, RESULT_DIR, FIGURES_DIR]:
    os.makedirs(d, exist_ok=True)

random.seed(SEED)


def generate_mirror_netlist(sid, p):
    ac_out = os.path.join(ABS_RESULT, f'{sid}_ac.txt')
    return f"""* Current Mirror
.include {MODEL_FILE}

VDD vdd 0 {p['VDD']}
Iref vdd ref_node {p['Iref']}
Rload out_node vdd {p['Rload']}

Mref ref_node ref_node 0 0 nmos W={p['W']} L={p['L']}
Mout out_node ref_node 0 0 nmos W={p['W_out']} L={p['L']}

.control
set wr_singlescale
set wr_vecnames
ac dec 50 1k 1g
wrdata {ac_out} v(out_node)
quit
.endc

.end
"""


def parse_mirror_ac(ac_file):
    try:
        data = np.loadtxt(ac_file, skiprows=1)
        if data.ndim < 2 or data.shape[0] < 5:
            return None
        vout = data[:, 0]
        iout = np.abs(data[:, 1])
        # Iout mean in saturation (mid-range)
        mid = len(vout) // 2
        iout_val = float(np.mean(iout[mid-2:mid+2]))
        # Rout from full sweep slope
        dv = np.diff(vout)
        di = np.diff(iout)
        valid = np.abs(di) > 1e-15
        if valid.sum() < 2:
            rout = 1e6  # very high Rout if current is flat
        else:
            rout = float(np.mean(np.abs(dv[valid] / di[valid])))
        if iout_val <= 0:
            return None
        return {'vout_low': iout_val, 'rout': rout}
    except:
        return None


def build_mirror_graph(sample, m):
    p = sample['parameters']
    W, W_out = p['W'], p['W_out']
    L, Iref  = p['L'], p['Iref']
    VDD, Rload = p['VDD'], p['Rload']
    mirror_ratio = W_out / W

    nodes = [
        [W,     L,    Iref, VDD, 0,     0, 0, W/L,       Iref/W,  0],
        [W_out, L,    0,    VDD, Rload, 0, 0, W_out/L,   0,       0],
        [0,     0,    0,    VDD, Rload, 0, 1, 0,          0,       0],
        [0,     0,    0,    VDD, 0,     0, 3, 0,          0,       0],
    ]
    edges = [
        [0,1,1],[1,0,1],
        [1,2,0],[2,1,0],
        [2,3,2],[3,2,2],
        [0,3,2],[3,0,2],
    ]
    target_val = math.log10(max(m['vout_low'], 1e-9))
    return {
        'sample_id': sample['sample_id'],
        'nodes': nodes, 'edges': edges,
        'targets': [mirror_ratio, target_val],
        'raw': m
    }


# ── Step 1: Generate ──────────────────────────────────────────────
print(f"Generating {N_SAMPLES} current mirror netlists...")
samples = []
for i in range(N_SAMPLES):
    p = {
        'W':      random.uniform(4e-6,  12e-6),
        'W_out':  random.uniform(4e-6,  12e-6),
        'L':      random.uniform(180e-9, 360e-9),
        'Iref':   random.uniform(50e-6,  200e-6),
        'VDD':    random.uniform(1.5,    1.8),
        'Rload':  random.uniform(5e3,   20e3),
    }
    samples.append({'sample_id': i, 'topology': 'current_mirror', 'parameters': p})
    with open(os.path.join(NETLIST_DIR, f'{i}.sp'), 'w') as f:
        abs_result = os.path.abspath(RESULT_DIR)
        dc_out = os.path.join(abs_result, f'{i}_ac.txt')
        vdd = p['VDD']
        netlist = (
            f"* Current Mirror {i}\n"
            f".include models/nmos.lib\n"
            f"VDD vdd 0 {vdd}\n"
            f"VOUT vout 0 DC 0.5\n"
            f"IREF vdd ref_node DC {p['Iref']}\n"
            f"MREF ref_node ref_node 0 0 nmos W={p['W']} L={p['L']}\n"
            f"MOUT vout ref_node 0 0 nmos W={p['W_out']} L={p['L']}\n"
            f".control\nset wr_singlescale\nset wr_vecnames\n"
            f"dc VOUT 0.05 {vdd*0.9:.3f} 0.05\n"
            f"wrdata {dc_out} i(VOUT)\nquit\n.endc\n.end\n"
        )
        f.write(netlist)

with open(os.path.join(MIRROR_DIR, 'samples.json'), 'w') as f:
    json.dump(samples, f, indent=2)
print(f"Done. Netlists → {NETLIST_DIR}")

# ── Step 2: Simulate (sequential) ────────────────────────────────
print(f"Simulating {N_SAMPLES} netlists (sequential)...")
metrics = {}
ok, fail = 0, 0
for i, sample in enumerate(samples):
    sp = os.path.join(NETLIST_DIR, f'{i}.sp')
    subprocess.run(['ngspice', '-b', sp], capture_output=True, text=True)
    ac = os.path.join(RESULT_DIR, f'{i}_ac.txt')
    m  = parse_mirror_ac(ac) if os.path.exists(ac) else None
    if m:
        metrics[str(i)] = m
        ok += 1
    else:
        fail += 1
    if (i+1) % 100 == 0:
        print(f"  {i+1}/{N_SAMPLES}  ok={ok}  fail={fail}")

print(f"Done: {ok} ok, {fail} failed")

# ── Step 3: Build graphs ──────────────────────────────────────────
graphs = []
for sample in samples:
    sid = str(sample['sample_id'])
    if sid in metrics:
        graphs.append(build_mirror_graph(sample, metrics[sid]))

with open(GRAPHS_PATH, 'w') as f:
    json.dump(graphs, f, indent=2)
print(f"Built {len(graphs)} mirror graphs")

# ── Step 4: Train ─────────────────────────────────────────────────
random.seed(SEED); torch.manual_seed(SEED)
random.shuffle(graphs)
n_train   = int(len(graphs)*0.8)
train_g, val_g = graphs[:n_train], graphs[n_train:]

all_nodes   = torch.cat([torch.tensor(g['nodes'], dtype=torch.float) for g in graphs])
all_targets = torch.stack([torch.tensor(g['targets'], dtype=torch.float) for g in graphs])
stats = {
    'node_mean': all_nodes.mean(0),   'node_std':  all_nodes.std(0).clamp(min=1e-6),
    'tgt_mean':  all_targets.mean(0), 'tgt_std':   all_targets.std(0).clamp(min=1e-6),
}

model = NEXUSLiteGNN(node_dim=10, edge_dim=1, hidden_dim=64,
                     num_layers=3, dropout=0.2, epsilon=0.1,
                     max_iter=20, alpha=0.3, out_dim=2)
optimizer = Adam(model.parameters(), lr=3e-4, weight_decay=1e-3)
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)
criterion = nn.MSELoss()

def g2t(g):
    h  = torch.tensor(g['nodes'],  dtype=torch.float)
    ei = torch.tensor([[e[0],e[1]] for e in g['edges']], dtype=torch.long)
    ef = torch.tensor([[e[2]] for e in g['edges']],       dtype=torch.float)
    t  = torch.tensor(g['targets'], dtype=torch.float)
    return h, ei, ef, t

print(f"\nTraining mirror surrogate (train={len(train_g)}, val={len(val_g)})...")
best_val, patience_count = float('inf'), 0
for epoch in range(1, 151):
    model.train()
    tl = 0.0
    for g in train_g:
        h,ei,ef,t = g2t(g)
        hn = (h-stats['node_mean'])/stats['node_std']
        tn = (t-stats['tgt_mean'])/stats['tgt_std']
        optimizer.zero_grad()
        loss = criterion(model(hn,ei,ef), tn)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        tl += loss.item()
    tl /= len(train_g)

    model.eval()
    vl = 0.0
    with torch.no_grad():
        for g in val_g:
            h,ei,ef,t = g2t(g)
            hn = (h-stats['node_mean'])/stats['node_std']
            tn = (t-stats['tgt_mean'])/stats['tgt_std']
            vl += criterion(model(hn,ei,ef), tn).item()
    vl /= len(val_g)
    scheduler.step(vl)

    if epoch % 25 == 0 or epoch == 1:
        print(f"  Epoch {epoch:3d}: train={tl:.4f}  val={vl:.4f}")

    if vl < best_val:
        best_val = vl
        patience_count = 0
        torch.save({
            'model_state': model.state_dict(),
            'stats': {k:v.tolist() for k,v in stats.items()},
            'val_loss': vl, 'epoch': epoch,
            'config': {'node_dim':10,'edge_dim':1,'hidden_dim':64,
                       'num_layers':3,'dropout':0.2,'epsilon':0.1,
                       'max_iter':20,'alpha':0.3}
        }, CKPT_PATH)
    else:
        patience_count += 1
        if patience_count >= 20:
            print(f"  Early stop at epoch {epoch}")
            break

print(f"Best val loss: {best_val:.4f}")

# ── Step 5: Evaluate ──────────────────────────────────────────────
ckpt = torch.load(CKPT_PATH, map_location='cpu')
model.load_state_dict(ckpt['model_state'])
model.eval()

pred_ratio, true_ratio = [], []
with torch.no_grad():
    for g in val_g:
        h,ei,ef,t = g2t(g)
        hn = (h-stats['node_mean'])/stats['node_std']
        out = model(hn,ei,ef)
        pred_ratio.append(float(out[0]*stats['tgt_std'][0]+stats['tgt_mean'][0]))
        true_ratio.append(float(t[0]))

rho, _ = spearmanr(true_ratio, pred_ratio)
mae    = float(np.mean(np.abs(np.array(pred_ratio)-np.array(true_ratio))))

print(f"\n{'='*50}")
print(f"CURRENT MIRROR RESULTS")
print(f"  Mirror ratio  MAE : {mae:.4f}")
print(f"  Mirror ratio  ρ   : {rho:.4f}")
print(f"  Val loss          : {best_val:.4f}")
print(f"{'='*50}")

# Plot
fig, ax = plt.subplots(figsize=(6,6))
ax.scatter(true_ratio, pred_ratio, alpha=0.4, s=15, color='#70AD47')
lim = [min(min(true_ratio),min(pred_ratio))-0.1,
       max(max(true_ratio),max(pred_ratio))+0.1]
ax.plot(lim,lim,'r--',linewidth=1.5)
ax.set_xlabel('True W_out/W ratio', fontsize=12)
ax.set_ylabel('Predicted W_out/W ratio', fontsize=12)
ax.set_title(f'Current Mirror  MAE={mae:.3f}  ρ={rho:.3f}', fontsize=12)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f'{FIGURES_DIR}/mirror_scatter.png', dpi=150)
plt.close()
print(f"Plot → {FIGURES_DIR}/mirror_scatter.png")

with open('data/mirror/eval_results.json','w') as f:
    json.dump({'mae': round(mae,4), 'spearman_rho': round(rho,4),
               'val_loss': round(best_val,4), 'n_val': len(val_g)}, f, indent=2)
print(f"Results → data/mirror/eval_results.json")