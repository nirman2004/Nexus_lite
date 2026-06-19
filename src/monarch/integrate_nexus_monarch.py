"""
integrate_nexus_monarch.py  
NEXUS-MONARCH Integration: NEXUS-Lite pressure as RL reward.

This is the core contribution of the paper:
  reward = NEXUS-Lite equilibrium pressure (gain + BW channels)

Run: python src/monarch/integrate_nexus_monarch.py
"""

import os, sys, json
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback

sys.path.insert(0, "src/graph")
sys.path.insert(0, "src/monarch")

from gnn_models import NEXUSLiteGNN
from graph_builder import build_graph
from environment import DiffPairEnv, params_to_dict

# ── config ────────────────────────────────────────────────────────
CHECKPOINT_DIR = "checkpoints/monarch_nexus"
LOG_DIR        = "data/monarch"
FIGURES_DIR    = "figures"
NEXUS_CKPT     = "checkpoints/best_model.pt"
TOTAL_STEPS    = 200_000
EVAL_FREQ      = 10_000
N_EVAL_EPS     = 20
SEED           = 42

os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(LOG_DIR,        exist_ok=True)
os.makedirs(FIGURES_DIR,    exist_ok=True)


def load_nexus():
    ckpt  = torch.load(NEXUS_CKPT, map_location="cpu")
    cfg   = ckpt["config"]
    stats = {k: torch.tensor(v) for k, v in ckpt["stats"].items()}
    model = NEXUSLiteGNN(
        node_dim   = cfg["node_dim"],
        edge_dim   = cfg["edge_dim"],
        hidden_dim = cfg["hidden_dim"],
        num_layers = cfg["num_layers"],
        dropout    = cfg.get("dropout", 0.0),
        epsilon    = cfg.get("epsilon", 0.1),
        max_iter   = cfg.get("max_iter", 20),
        alpha      = cfg.get("alpha",   0.3),
    )
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    print(f"NEXUS-Lite loaded — val_loss={ckpt['val_loss']:.4f}  epoch={ckpt['epoch']}")
    return {"model": model, "stats": stats}


def nexus_reward(params_arr, nexus):
    """Compute NEXUS-Lite pressure reward for a parameter vector."""
    sample = {
        "sample_id": -1,
        "parameters": params_to_dict(params_arr)
    }
    dummy = {"gain_dB": 0.0, "bandwidth_MHz": 10.0}
    g = build_graph(sample, dummy)

    h          = torch.tensor(g["nodes"],   dtype=torch.float)
    edge_index = torch.tensor([[e[0], e[1]] for e in g["edges"]], dtype=torch.long)
    edge_feat  = torch.tensor([[e[2]] for e in g["edges"]],        dtype=torch.float)

    stats  = nexus["stats"]
    h_norm = (h - stats["node_mean"]) / stats["node_std"]

    with torch.no_grad():
        out = nexus["model"](h_norm, edge_index, edge_feat)

    pred    = out.numpy() * stats["tgt_std"].numpy() + stats["tgt_mean"].numpy()
    gain_dB = float(pred[0])
    bw_log  = float(pred[1])

    # Combined reward — gain dominates, BW secondary
    reward = gain_dB / 25.0 + (bw_log - 1.0) / 3.0
    return reward, gain_dB, float(10 ** bw_log)


# ── patched environment with NEXUS reward ────────────────────────
class NexusEnv(DiffPairEnv):
    """DiffPairEnv with NEXUS-Lite reward injected directly."""

    def __init__(self, nexus, **kwargs):
        super().__init__(reward_mode="random", **kwargs)
        self._nexus      = nexus
        self._prev_raw   = 0.0
        self.last_gain   = 0.0
        self.last_bw     = 0.0

    def reset(self, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)
        r, g, b = nexus_reward(self.params, self._nexus)
        self._prev_raw = r
        self.last_gain = g
        self.last_bw   = b
        return obs, info

    def step(self, action):
        obs, _, done, trunc, info = super().step(action)

        # Override reward with NEXUS pressure
        raw, gain, bw = nexus_reward(self.params, self._nexus)

        # DRC penalty
        drc_penalty = info["n_violations"] * 0.5

        # Improvement-based reward
        reward = (raw - self._prev_raw) - drc_penalty
        self._prev_raw = raw
        self.last_gain = gain
        self.last_bw   = bw

        info["nexus_gain"] = gain
        info["nexus_bw"]   = bw
        info["raw_reward"] = raw

        return obs, float(reward), done, trunc, info


# ── callback ─────────────────────────────────────────────────────
class NexusCallback(BaseCallback):

    def __init__(self, eval_env, eval_freq, n_eval_eps, verbose=1):
        super().__init__(verbose)
        self.eval_env   = eval_env
        self.eval_freq  = eval_freq
        self.n_eval_eps = n_eval_eps
        self.log = {
            "steps": [], "mean_reward": [],
            "drc_ok_rate": [], "mean_gain": [], "mean_bw": []
        }
        self.best_reward = -np.inf

    def _on_step(self):
        if self.n_calls % self.eval_freq == 0:
            self._evaluate()
        return True

    def _evaluate(self):
        rewards, drc_oks, gains, bws = [], [], [], []

        for _ in range(self.n_eval_eps):
            obs, _ = self.eval_env.reset()
            ep_reward, ep_ok, ep_steps = 0.0, 0, 0
            ep_gains, ep_bws = [], []
            done = False

            while not done:
                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, term, trunc, info = self.eval_env.step(action)
                done = term or trunc
                ep_reward += reward
                ep_steps  += 1
                if info["drc_ok"]:
                    ep_ok += 1
                ep_gains.append(info.get("nexus_gain", 0.0))
                ep_bws.append(info.get("nexus_bw", 0.0))

            rewards.append(ep_reward)
            drc_oks.append(ep_ok / max(ep_steps, 1))
            gains.append(np.mean(ep_gains))
            bws.append(np.mean(ep_bws))

        mean_reward = float(np.mean(rewards))
        drc_ok_rate = float(np.mean(drc_oks))
        mean_gain   = float(np.mean(gains))
        mean_bw     = float(np.mean(bws))

        self.log["steps"].append(self.n_calls)
        self.log["mean_reward"].append(round(mean_reward, 4))
        self.log["drc_ok_rate"].append(round(drc_ok_rate, 4))
        self.log["mean_gain"].append(round(mean_gain, 4))
        self.log["mean_bw"].append(round(mean_bw, 4))

        if self.verbose:
            print(f"  Step {self.n_calls:>8,}  "
                  f"reward={mean_reward:+.3f}  "
                  f"DRC-ok={drc_ok_rate*100:.1f}%  "
                  f"gain={mean_gain:+.2f}dB  "
                  f"BW={mean_bw:.1f}MHz")

        if mean_reward > self.best_reward:
            self.best_reward = mean_reward
            self.model.save(os.path.join(CHECKPOINT_DIR, "best_model"))
            if self.verbose:
                print(f"  ✓ Best reward={mean_reward:.4f} → saved")


# ── main ─────────────────────────────────────────────────────────
def train():
    print("=" * 60)
    print("NEXUS-MONARCH  Month 4 — NEXUS-Lite Pressure Reward")
    print("=" * 60)

    nexus = load_nexus()

    def make_env():
        env = NexusEnv(nexus=nexus, max_steps=50, action_scale=0.05)
        return Monitor(env)

    train_env = make_vec_env(make_env, n_envs=4, seed=SEED)
    eval_env  = NexusEnv(nexus=nexus, max_steps=50, action_scale=0.05)

    model = PPO(
        policy        = "MlpPolicy",
        env           = train_env,
        learning_rate = 3e-4,
        n_steps       = 512,
        batch_size    = 64,
        n_epochs      = 10,
        gamma         = 0.99,
        gae_lambda    = 0.95,
        clip_range    = 0.2,
        ent_coef      = 0.01,
        vf_coef       = 0.5,
        max_grad_norm = 0.5,
        policy_kwargs = dict(net_arch=[128, 128]),
        verbose       = 0,
        seed          = SEED,
    )

    print(f"PPO parameters : {sum(p.numel() for p in model.policy.parameters()):,}")
    print(f"Training steps : {TOTAL_STEPS:,}  across 4 envs\n")

    callback = NexusCallback(
        eval_env=eval_env, eval_freq=EVAL_FREQ,
        n_eval_eps=N_EVAL_EPS, verbose=1
    )

    model.learn(total_timesteps=TOTAL_STEPS, callback=callback)
    model.save(os.path.join(CHECKPOINT_DIR, "final_model"))

    # ── save log ──
    log_path = os.path.join(LOG_DIR, "nexus_monarch_log.json")
    with open(log_path, "w") as f:
        json.dump(callback.log, f, indent=2)
    print(f"\nLog → {log_path}")

    # ── plots ──
    steps = callback.log["steps"]
    fig, axes = plt.subplots(3, 1, figsize=(10, 12), sharex=True)

    axes[0].plot(steps, callback.log["mean_reward"], color="#2E75B6", linewidth=2)
    axes[0].set_ylabel("Mean Episode Reward", fontsize=11)
    axes[0].set_title("NEXUS-MONARCH Integration — Month 4", fontsize=13)
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(steps, callback.log["mean_gain"], color="#C55A11", linewidth=2)
    axes[1].axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    axes[1].set_ylabel("Mean Predicted Gain (dB)", fontsize=11)
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(steps, [r*100 for r in callback.log["drc_ok_rate"]],
                 color="#70AD47", linewidth=2)
    axes[2].axhline(y=80, color="green", linestyle="--", alpha=0.7, label="80% target")
    axes[2].set_ylabel("DRC-Clean Steps (%)", fontsize=11)
    axes[2].set_xlabel("Training Steps", fontsize=11)
    axes[2].set_ylim(0, 105)
    axes[2].legend(); axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    fig_path = os.path.join(FIGURES_DIR, "nexus_monarch_training.png")
    plt.savefig(fig_path, dpi=150)
    plt.close()
    print(f"Plot → {fig_path}")

    # ── final summary ──
    final_gain = callback.log["mean_gain"][-1]
    best_gain  = max(callback.log["mean_gain"])
    final_drc  = callback.log["drc_ok_rate"][-1] * 100

    print(f"\n{'='*60}")
    print(f"Initial mean gain : {callback.log['mean_gain'][0]:+.2f} dB")
    print(f"Final  mean gain  : {final_gain:+.2f} dB")
    print(f"Best   mean gain  : {best_gain:+.2f} dB")
    print(f"Final  DRC-ok     : {final_drc:.1f}%")
    gain_improved = final_gain > callback.log["mean_gain"][0]
    print(f"\nCore claim {'VALIDATED ✓' if gain_improved else 'needs more training'}")
    print(f"NEXUS-Lite pressure drives MONARCH toward higher-gain circuits")
    print(f"{'='*60}")


if __name__ == "__main__":
    train()