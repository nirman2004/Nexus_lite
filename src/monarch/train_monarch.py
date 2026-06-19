"""
train_monarch.py  —  MONARCH-Lite
Trains a PPO agent on DiffPairEnv with random reward .
Validates that the agent learns DRC constraint satisfaction.

Run: python src/monarch/train_monarch.py

Outputs:
  checkpoints/monarch_ppo/   — SB3 model checkpoints
  figures/monarch_training.png — DRC violation rate over training
  data/monarch/training_log.json
"""

import os
import sys
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor

sys.path.insert(0, os.path.dirname(__file__))
from environment import DiffPairEnv

# ── config ────────────────────────────────────────────────────────
CHECKPOINT_DIR = "checkpoints/monarch_ppo"
LOG_DIR        = "data/monarch"
FIGURES_DIR    = "figures"
TOTAL_STEPS    = 200_000     # enough to see DRC learning on CPU
EVAL_FREQ      = 10_000      # evaluate every N steps
N_EVAL_EPS     = 20          # episodes per evaluation
SEED           = 42

os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(LOG_DIR,        exist_ok=True)
os.makedirs(FIGURES_DIR,    exist_ok=True)


# ── custom callback to track DRC violations ──────────────────────
class DRCCallback(BaseCallback):
    """
    Tracks DRC violation rate and mean reward over training.
    Saves checkpoint at best DRC satisfaction rate.
    """

    def __init__(self, eval_env, eval_freq, n_eval_eps, verbose=1):
        super().__init__(verbose)
        self.eval_env    = eval_env
        self.eval_freq   = eval_freq
        self.n_eval_eps  = n_eval_eps
        self.log = {
            "steps":         [],
            "mean_reward":   [],
            "drc_viol_rate": [],   # fraction of steps with violations
            "drc_ok_rate":   [],   # fraction of steps fully DRC clean
        }
        self.best_drc_ok = -1.0

    def _on_step(self) -> bool:
        if self.n_calls % self.eval_freq == 0:
            self._evaluate()
        return True

    def _evaluate(self):
        rewards, drc_ok_rates = [], []

        for _ in range(self.n_eval_eps):
            obs, _ = self.eval_env.reset()
            ep_reward = 0.0
            ep_steps  = 0
            ep_ok     = 0

            done = False
            while not done:
                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = self.eval_env.step(action)
                done = terminated or truncated
                ep_reward += reward
                ep_steps  += 1
                if info["drc_ok"]:
                    ep_ok += 1

            rewards.append(ep_reward)
            drc_ok_rates.append(ep_ok / max(ep_steps, 1))

        mean_reward  = float(np.mean(rewards))
        drc_ok_rate  = float(np.mean(drc_ok_rates))
        drc_viol_rate = 1.0 - drc_ok_rate

        self.log["steps"].append(self.n_calls)
        self.log["mean_reward"].append(round(mean_reward, 4))
        self.log["drc_ok_rate"].append(round(drc_ok_rate, 4))
        self.log["drc_viol_rate"].append(round(drc_viol_rate, 4))

        if self.verbose:
            print(f"  Step {self.n_calls:>8,}  "
                  f"reward={mean_reward:+.3f}  "
                  f"DRC-ok={drc_ok_rate*100:.1f}%  "
                  f"DRC-viol={drc_viol_rate*100:.1f}%")

        # save best checkpoint by DRC ok rate
        if drc_ok_rate > self.best_drc_ok:
            self.best_drc_ok = drc_ok_rate
            self.model.save(os.path.join(CHECKPOINT_DIR, "best_model"))
            if self.verbose:
                print(f"  ✓ New best DRC-ok rate: {drc_ok_rate*100:.1f}%  → checkpoint saved")


def train():
    print("=" * 55)
    print("MONARCH-Lite PPO Training  (Month 3 — random reward)")
    print("=" * 55)

    # ── environments ──
    # Training env (vectorized for SB3)
    def make_env():
        env = DiffPairEnv(max_steps=50, action_scale=0.05, reward_mode="random")
        return Monitor(env)

    train_env = make_vec_env(make_env, n_envs=4, seed=SEED)
    eval_env  = DiffPairEnv(max_steps=50, action_scale=0.05, reward_mode="random")

    # ── PPO agent ──
    model = PPO(
        policy          = "MlpPolicy",
        env             = train_env,
        learning_rate   = 3e-4,
        n_steps         = 512,         # steps per env before update
        batch_size      = 64,
        n_epochs        = 10,
        gamma           = 0.99,
        gae_lambda      = 0.95,
        clip_range      = 0.2,
        ent_coef        = 0.01,        # entropy bonus encourages exploration
        vf_coef         = 0.5,
        max_grad_norm   = 0.5,
        policy_kwargs   = dict(net_arch=[128, 128]),
        verbose         = 0,
        seed            = SEED,
    )

    total_params = sum(p.numel() for p in model.policy.parameters())
    print(f"PPO policy parameters: {total_params:,}")
    print(f"Training for {TOTAL_STEPS:,} steps across 4 parallel envs")
    print(f"Evaluating every {EVAL_FREQ:,} steps on {N_EVAL_EPS} episodes\n")

    # ── callback ──
    callback = DRCCallback(
        eval_env  = eval_env,
        eval_freq = EVAL_FREQ,
        n_eval_eps = N_EVAL_EPS,
        verbose   = 1,
    )

    # ── train ──
    model.learn(
        total_timesteps = TOTAL_STEPS,
        callback        = callback,
        progress_bar    = False,
    )

    # ── save final model ──
    model.save(os.path.join(CHECKPOINT_DIR, "final_model"))
    print(f"\nFinal model → {CHECKPOINT_DIR}/final_model")

    # ── save log ──
    log_path = os.path.join(LOG_DIR, "training_log.json")
    with open(log_path, "w") as f:
        json.dump(callback.log, f, indent=2)
    print(f"Training log → {log_path}")

    # ── plot ──
    steps        = callback.log["steps"]
    drc_ok       = callback.log["drc_ok_rate"]
    mean_rewards = callback.log["mean_reward"]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    ax1.plot(steps, [r*100 for r in drc_ok], color="#2E75B6", linewidth=2)
    ax1.axhline(y=80, color="green", linestyle="--", alpha=0.7, label="80% target")
    ax1.set_ylabel("DRC-Clean Steps (%)", fontsize=12)
    ax1.set_title("MONARCH-Lite PPO Training (Month 3 — Random Reward)", fontsize=13)
    ax1.legend(); ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 105)

    ax2.plot(steps, mean_rewards, color="#C55A11", linewidth=2)
    ax2.set_xlabel("Training Steps", fontsize=12)
    ax2.set_ylabel("Mean Episode Reward", fontsize=12)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    fig_path = os.path.join(FIGURES_DIR, "monarch_training.png")
    plt.savefig(fig_path, dpi=150)
    plt.close()
    print(f"Training plot → {fig_path}")

    # ── final summary ──
    final_drc = callback.log["drc_ok_rate"][-1] * 100
    best_drc  = callback.best_drc_ok * 100
    print(f"\n{'='*55}")
    print(f"Final DRC-clean rate : {final_drc:.1f}%")
    print(f"Best  DRC-clean rate : {best_drc:.1f}%")
    print(f"Month 3 validation: agent {'PASSED' if best_drc > 50 else 'NEEDS MORE TRAINING'}")
    print(f"{'='*55}")
    print(f"\nNext: Month 4 — connect NEXUS-Lite reward")
    print(f"  Change reward_mode='random' → reward_mode='nexus'")
    print(f"  in src/monarch/environment.py")


if __name__ == "__main__":
    train()