"""
multiseed_rl.py  — 5-seed RL experiment
Runs NEXUS-MONARCH with 5 different seeds.
Reports mean ± std for final gain and DRC rate.
"""
import os, sys, json
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback

sys.path.insert(0, 'src/graph')
sys.path.insert(0, 'src/monarch')
from gnn_models import NEXUSLiteGNN
from graph_builder import build_graph
from environment import DiffPairEnv, params_to_dict

CHECKPOINT_DIR = 'checkpoints/multiseed'
LOG_DIR        = 'data/monarch'
FIGURES_DIR    = 'figures'
NEXUS_CKPT     = 'checkpoints/best_model.pt'
TOTAL_STEPS    = 100_000
EVAL_FREQ      = 20_000
N_EVAL_EPS     = 10
SEEDS          = [42, 7, 13, 99, 2024]

os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(LOG_DIR,        exist_ok=True)
os.makedirs(FIGURES_DIR,    exist_ok=True)


def load_nexus():
    ckpt  = torch.load(NEXUS_CKPT, map_location='cpu')
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
    return {'model': model, 'stats': stats}


def nexus_reward(params_arr, nexus):
    sample = {'sample_id': -1, 'parameters': params_to_dict(params_arr)}
    dummy  = {'gain_dB': 0.0, 'bandwidth_MHz': 10.0}
    g = build_graph(sample, dummy)
    h  = torch.tensor(g['nodes'],   dtype=torch.float)
    ei = torch.tensor([[e[0],e[1]] for e in g['edges']], dtype=torch.long)
    ef = torch.tensor([[e[2]] for e in g['edges']],       dtype=torch.float)
    stats  = nexus['stats']
    h_norm = (h - stats['node_mean']) / stats['node_std']
    with torch.no_grad():
        out = nexus['model'](h_norm, ei, ef)
    pred    = out.numpy() * stats['tgt_std'].numpy() + stats['tgt_mean'].numpy()
    return float(pred[0]/25.0 + (pred[1]-1.0)/3.0), float(pred[0]), float(10**pred[1])


class NexusEnv(DiffPairEnv):
    def __init__(self, nexus, **kwargs):
        super().__init__(reward_mode='random', **kwargs)
        self._nexus    = nexus
        self._prev_raw = 0.0
        self.last_gain = 0.0
        self.last_bw   = 0.0

    def reset(self, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)
        r, g, b = nexus_reward(self.params, self._nexus)
        self._prev_raw, self.last_gain, self.last_bw = r, g, b
        return obs, info

    def step(self, action):
        obs, _, done, trunc, info = super().step(action)
        raw, gain, bw = nexus_reward(self.params, self._nexus)
        penalty = info['n_violations'] * 0.5
        reward  = (raw - self._prev_raw) - penalty
        self._prev_raw, self.last_gain, self.last_bw = raw, gain, bw
        info['nexus_gain'] = gain
        info['nexus_bw']   = bw
        return obs, float(reward), done, trunc, info


class QuickCallback(BaseCallback):
    def __init__(self, eval_env, eval_freq, n_eps):
        super().__init__(verbose=0)
        self.eval_env  = eval_env
        self.eval_freq = eval_freq
        self.n_eps     = n_eps
        self.gains, self.drcs, self.steps_log = [], [], []

    def _on_step(self):
        if self.n_calls % self.eval_freq == 0:
            gains, drcs = [], []
            for _ in range(self.n_eps):
                obs, _ = self.eval_env.reset()
                ep_gain, ep_ok, ep_steps, done = 0.0, 0, 0, False
                while not done:
                    a, _ = self.model.predict(obs, deterministic=True)
                    obs, _, term, trunc, info = self.eval_env.step(a)
                    done = term or trunc
                    ep_gain += info.get('nexus_gain', 0)
                    ep_steps += 1
                    if info['drc_ok']: ep_ok += 1
                gains.append(ep_gain / max(ep_steps,1))
                drcs.append(ep_ok   / max(ep_steps,1))
            self.gains.append(float(np.mean(gains)))
            self.drcs.append(float(np.mean(drcs)))
            self.steps_log.append(self.n_calls)
        return True


def run_one_seed(seed, nexus):
    def make_env():
        return Monitor(NexusEnv(nexus=nexus, max_steps=50, action_scale=0.05))

    train_env = make_vec_env(make_env, n_envs=4, seed=seed)
    eval_env  = NexusEnv(nexus=nexus, max_steps=50, action_scale=0.05)

    model = PPO('MlpPolicy', train_env, learning_rate=3e-4,
                n_steps=512, batch_size=64, n_epochs=10,
                gamma=0.99, gae_lambda=0.95, clip_range=0.2,
                ent_coef=0.01, vf_coef=0.5, max_grad_norm=0.5,
                policy_kwargs=dict(net_arch=[128,128]),
                verbose=0, seed=seed)

    cb = QuickCallback(eval_env, EVAL_FREQ, N_EVAL_EPS)
    model.learn(total_timesteps=TOTAL_STEPS, callback=cb)

    final_gain = cb.gains[-1] if cb.gains else 0.0
    final_drc  = cb.drcs[-1]  if cb.drcs  else 0.0
    print(f"  Seed {seed:5d}: final_gain={final_gain:+.2f} dB  DRC={final_drc*100:.1f}%")
    return {'seed': seed, 'gains': cb.gains, 'drcs': cb.drcs,
            'steps': cb.steps_log,
            'final_gain': final_gain, 'final_drc': final_drc}


def main():
    print('='*55)
    print('MULTI-SEED RL EXPERIMENT (5 seeds)')
    print('='*55)

    nexus   = load_nexus()
    results = []
    for seed in SEEDS:
        print(f"\nRunning seed {seed}...")
        r = run_one_seed(seed, nexus)
        results.append(r)

    final_gains = [r['final_gain'] for r in results]
    final_drcs  = [r['final_drc']  for r in results]

    print(f"\n{'='*55}")
    print(f"SUMMARY ({len(SEEDS)} seeds)")
    print(f"{'='*55}")
    print(f"Final gain: {np.mean(final_gains):+.2f} ± {np.std(final_gains):.2f} dB")
    print(f"Final DRC : {np.mean(final_drcs)*100:.1f} ± {np.std(final_drcs)*100:.1f} %")
    print(f"{'='*55}")

    # Save
    summary = {
        'seeds': SEEDS,
        'final_gains': final_gains,
        'final_drcs':  final_drcs,
        'mean_gain': round(float(np.mean(final_gains)), 4),
        'std_gain':  round(float(np.std(final_gains)),  4),
        'mean_drc':  round(float(np.mean(final_drcs)),  4),
        'std_drc':   round(float(np.std(final_drcs)),   4),
        'per_seed':  results
    }
    with open(f'{LOG_DIR}/multiseed_results.json', 'w') as f:
        json.dump(summary, f, indent=2)

    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    colors = ['#2E75B6','#C55A11','#70AD47','#FFC000','#9E480E']
    for i, r in enumerate(results):
        ax1.plot(r['steps'], r['gains'], color=colors[i],
                 linewidth=1.5, alpha=0.8, label=f"seed {r['seed']}")
        ax2.plot(r['steps'], [d*100 for d in r['drcs']],
                 color=colors[i], linewidth=1.5, alpha=0.8)

    # Mean band
    min_len = min(len(r['gains']) for r in results)
    gain_arr = np.array([r['gains'][:min_len] for r in results])
    steps_arr = results[0]['steps'][:min_len]
    ax1.fill_between(steps_arr,
                     gain_arr.mean(0)-gain_arr.std(0),
                     gain_arr.mean(0)+gain_arr.std(0),
                     alpha=0.2, color='#2E75B6', label='mean±std')
    ax1.plot(steps_arr, gain_arr.mean(0), 'k--', linewidth=2, label='mean')

    ax1.set_xlabel('Training steps', fontsize=12)
    ax1.set_ylabel('Mean predicted gain (dB)', fontsize=12)
    ax1.set_title(f'Gain: {np.mean(final_gains):+.2f}±{np.std(final_gains):.2f} dB', fontsize=12)
    ax1.legend(fontsize=9); ax1.grid(True, alpha=0.3)

    ax2.set_xlabel('Training steps', fontsize=12)
    ax2.set_ylabel('DRC-clean steps (%)', fontsize=12)
    ax2.set_title(f'DRC: {np.mean(final_drcs)*100:.1f}±{np.std(final_drcs)*100:.1f}%', fontsize=12)
    ax2.set_ylim(0, 105); ax2.grid(True, alpha=0.3)

    plt.suptitle('NEXUS-MONARCH: 5-Seed RL Results', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{FIGURES_DIR}/multiseed_rl.png', dpi=150)
    plt.close()
    print(f"Plot → {FIGURES_DIR}/multiseed_rl.png")


if __name__ == '__main__':
    main()