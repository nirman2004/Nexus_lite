"""
environment.py  —  MONARCH-Lite
Circuit sizing as a Markov Decision Process.

State  : normalized parameter vector (7 params)
Action : continuous delta on each parameter (7 dims)
Reward : placeholder (random) 
         → replaced with NEXUS-Lite pressure in Month 4

Episode: agent adjusts parameters step by step,
         reward given at each step based on improvement.

DRC Constraints (hard):
  W/L ratio    ≥ 5        (minimum for saturation)
  W            ∈ [1µm, 20µm]
  L            ∈ [180nm, 2µm]
  Ibias        ∈ [10µA, 1mA]
  VDD          ∈ [1.2V, 1.8V]
  RL           ∈ [1kΩ, 50kΩ]
  CL           ∈ [0.1pF, 10pF]
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces


# ── Parameter bounds ─────────────────────────────────────────────
PARAM_BOUNDS = {
    "W":      (1e-6,   20e-6),
    "L":      (180e-9,  2e-6),
    "Ibias":  (10e-6,   1e-3),
    "VDD":    (1.2,     1.8),
    "RL":     (1e3,    50e3),
    "CL":     (0.1e-12,10e-12),
    "Vin_cm": (0.5,     1.0),
}
PARAM_NAMES = list(PARAM_BOUNDS.keys())
N_PARAMS    = len(PARAM_NAMES)

# DRC constraints
WL_RATIO_MIN = 5.0      # W/L ≥ 5 for saturation

# Normalise params to [0, 1]
BOUNDS_LOW  = np.array([PARAM_BOUNDS[k][0] for k in PARAM_NAMES], dtype=np.float32)
BOUNDS_HIGH = np.array([PARAM_BOUNDS[k][1] for k in PARAM_NAMES], dtype=np.float32)


def normalize(params: np.ndarray) -> np.ndarray:
    return (params - BOUNDS_LOW) / (BOUNDS_HIGH - BOUNDS_LOW + 1e-12)

def denormalize(norm: np.ndarray) -> np.ndarray:
    return norm * (BOUNDS_HIGH - BOUNDS_LOW) + BOUNDS_LOW

def params_to_dict(params: np.ndarray) -> dict:
    return {k: float(v) for k, v in zip(PARAM_NAMES, params)}

def check_drc(params: np.ndarray) -> tuple:
    """
    Returns (is_valid, violations_list)
    violations_list: list of (constraint_name, value, limit)
    """
    p = params_to_dict(params)
    violations = []

    for k, (lo, hi) in PARAM_BOUNDS.items():
        v = p[k]
        if v < lo:
            violations.append((k, v, lo))
        if v > hi:
            violations.append((k, v, hi))

    wl = p["W"] / p["L"]
    if wl < WL_RATIO_MIN:
        violations.append(("W/L", wl, WL_RATIO_MIN))

    return len(violations) == 0, violations

def clip_to_bounds(params: np.ndarray) -> np.ndarray:
    return np.clip(params, BOUNDS_LOW, BOUNDS_HIGH)


class DiffPairEnv(gym.Env):
    """
    Differential pair sizing environment.

    Observation: normalized parameter vector (7,)
    Action:      normalized delta vector (7,)  ∈ [-1, 1]
                 scaled by action_scale before applying
    Reward:      placeholder random reward (Month 3)
                 → NEXUS-Lite pressure (Month 4)
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        max_steps:    int   = 50,
        action_scale: float = 0.05,   # max 5% change per step
        reward_mode:  str   = "random",  # "random" | "nexus"
        nexus_model=None,              # injected in Month 4
    ):
        super().__init__()

        self.max_steps    = max_steps
        self.action_scale = action_scale
        self.reward_mode  = reward_mode
        self.nexus_model  = nexus_model
        self.step_count   = 0

        # Observation: normalized params (7,)
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(N_PARAMS,), dtype=np.float32
        )

        # Action: delta in [-1, 1] per param, scaled by action_scale
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(N_PARAMS,), dtype=np.float32
        )

        self.params = None   # current params (physical units)
        self._prev_reward = 0.0

    # ── reset ────────────────────────────────────────────────────
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.step_count = 0

        # Random init within bounds
        self.params = BOUNDS_LOW + np.random.rand(N_PARAMS).astype(np.float32) * (BOUNDS_HIGH - BOUNDS_LOW)
        self._prev_reward = self._compute_reward(self.params)

        obs = normalize(self.params).astype(np.float32)
        return obs, {}

    # ── step ─────────────────────────────────────────────────────
    def step(self, action: np.ndarray):
        self.step_count += 1

        # Apply action: delta scaled by action_scale
        delta = action * self.action_scale * (BOUNDS_HIGH - BOUNDS_LOW)
        new_params = self.params + delta

        # Clip to bounds (hard constraint enforcement)
        new_params = clip_to_bounds(new_params)

        # Check DRC
        drc_ok, violations = check_drc(new_params)

        # Compute reward
        reward = self._compute_reward(new_params)

        # DRC penalty: subtract per violation
        drc_penalty = len(violations) * 0.5
        reward -= drc_penalty

        # Improvement bonus
        improvement = reward - self._prev_reward
        reward = improvement  # reward = delta improvement

        self.params = new_params
        self._prev_reward = self._compute_reward(new_params)

        obs  = normalize(self.params).astype(np.float32)
        done = self.step_count >= self.max_steps

        info = {
            "drc_ok":       drc_ok,
            "n_violations": len(violations),
            "violations":   violations,
            "params":       params_to_dict(self.params),
            "raw_reward":   float(self._compute_reward(self.params)),
        }

        return obs, float(reward), done, False, info

    # ── reward ───────────────────────────────────────────────────
    def _compute_reward(self, params: np.ndarray) -> float:
        if self.reward_mode == "random":
            # Month 3 placeholder — random reward
            # Agent learns DRC satisfaction only
            return float(np.random.uniform(-1, 1))

        elif self.reward_mode == "nexus":
            # Month 4 — NEXUS-Lite pressure surrogate
            # nexus_model injected at construction time
            if self.nexus_model is None:
                raise ValueError("nexus_model must be provided for reward_mode='nexus'")
            return self._nexus_reward(params)

        else:
            raise ValueError(f"Unknown reward_mode: {self.reward_mode}")

    def _nexus_reward(self, params: np.ndarray) -> float:
        """Month 4: call NEXUS-Lite surrogate for reward."""
        import torch, sys, os
        sys.path.insert(0, "src/graph")
        from graph_builder import build_graph
        import math

        sample = {
            "sample_id": -1,
            "parameters": params_to_dict(params)
        }
        # dummy metrics for graph construction (not used in forward pass)
        dummy_metrics = {"gain_dB": 0.0, "bandwidth_MHz": 10.0}
        g = build_graph(sample, dummy_metrics)

        h          = torch.tensor(g["nodes"],   dtype=torch.float)
        edge_index = torch.tensor([[e[0], e[1]] for e in g["edges"]], dtype=torch.long)
        edge_feat  = torch.tensor([[e[2]] for e in g["edges"]],        dtype=torch.float)

        stats = self.nexus_model["stats"]
        model = self.nexus_model["model"]

        h_norm = (h - stats["node_mean"]) / stats["node_std"]

        with torch.no_grad():
            out = model(h_norm, edge_index, edge_feat)

        pred = out.numpy() * stats["tgt_std"].numpy() + stats["tgt_mean"].numpy()
        gain_dB = float(pred[0])   # gain pressure channel
        bw_log  = float(pred[1])   # bandwidth pressure channel

        # Combined reward: gain + bandwidth (both normalized to ~same scale)
        reward = gain_dB / 25.0 + (bw_log - 1.0) / 3.0
        return float(reward)

    # ── render ───────────────────────────────────────────────────
    def render(self):
        p = params_to_dict(self.params)
        drc_ok, viols = check_drc(self.params)
        print(f"Step {self.step_count}/{self.max_steps}  DRC={'OK' if drc_ok else 'FAIL'}")
        print(f"  W={p['W']*1e6:.2f}µm  L={p['L']*1e9:.0f}nm  "
              f"Ibias={p['Ibias']*1e6:.1f}µA  RL={p['RL']/1e3:.1f}kΩ")
        if viols:
            for v in viols:
                print(f"  VIOLATION: {v[0]}={v[1]:.3e} (limit {v[2]:.3e})")


# ── smoke test ───────────────────────────────────────────────────
if __name__ == "__main__":
    import gymnasium as gym

    env = DiffPairEnv(max_steps=10, reward_mode="random")
    obs, _ = env.reset(seed=42)

    print(f"Observation space : {env.observation_space}")
    print(f"Action space      : {env.action_space}")
    print(f"Initial obs shape : {obs.shape}")
    print(f"Initial obs       : {obs.round(3)}")
    print()

    total_reward = 0
    drc_violations = 0

    for step in range(10):
        action = env.action_space.sample()
        obs, reward, done, _, info = env.step(action)
        total_reward += reward
        drc_violations += info["n_violations"]
        env.render()

    print(f"\nEpisode done")
    print(f"Total reward     : {total_reward:.4f}")
    print(f"Total DRC viols  : {drc_violations}")
    print(f"\nSmoke test passed.")