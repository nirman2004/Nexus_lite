"""
graph_builder.py v2  —  NEXUS-Lite
Enriched node features: adds derived physical quantities
  RL*CL  → dominant pole time constant (directly sets BW)
  W/L    → aspect ratio (sets gm, saturation behavior)
  Ibias/W → current density (sets operating point)

Node feature vector (10 features):
  [W, L, Ibias, VDD, RL, CL, device_type, RL_CL, W_over_L, Id_over_W]
"""

import json
import os
import math

SAMPLES_PATH = "data/param_samples/samples.json"
METRICS_PATH = "data/results/all_metrics.json"
OUTPUT_PATH  = "data/graphs/graphs.json"

GAIN_MIN = -40.0
GAIN_MAX =  25.0

DEVICE_NMOS     = 0
DEVICE_RESISTOR = 1
DEVICE_ISOURCE  = 2
DEVICE_SUPPLY   = 3


def build_graph(sample: dict, metrics: dict) -> dict:
    p     = sample["parameters"]
    W     = p["W"]
    L     = p["L"]
    Ibias = p["Ibias"]
    VDD   = p["VDD"]
    RL    = p["RL"]
    CL    = p["CL"]

    # ── derived features ──────────────────────────────────────────
    RL_CL     = RL * CL          # dominant pole time constant (seconds)
    W_over_L  = W / L            # aspect ratio
    Id_over_W = Ibias / W        # current density (A/m)

    # Node features: 10 per node
    # [W, L, Ibias, VDD, RL, CL, device_type, RL_CL, W_over_L, Id_over_W]
    nodes = [
        [W,   L,   Ibias, VDD, RL,  CL,  DEVICE_NMOS,     RL_CL, W_over_L, Id_over_W],  # M1
        [W,   L,   Ibias, VDD, RL,  CL,  DEVICE_NMOS,     RL_CL, W_over_L, Id_over_W],  # M2
        [0,   0,   0,     VDD, RL,  0,   DEVICE_RESISTOR, RL_CL, 0,        0],           # RL1
        [0,   0,   0,     VDD, RL,  0,   DEVICE_RESISTOR, RL_CL, 0,        0],           # RL2
        [0,   0,   Ibias, 0,   0,   0,   DEVICE_ISOURCE,  0,     0,        0],           # Itail
        [0,   0,   0,     VDD, 0,   0,   DEVICE_SUPPLY,   0,     0,        0],           # VDD
    ]

    raw_edges = [
        (0, 2, 0),  # M1  → RL1   drain-load
        (1, 3, 0),  # M2  → RL2   drain-load
        (0, 4, 1),  # M1  → Itail source-tail
        (1, 4, 1),  # M2  → Itail source-tail
        (2, 5, 2),  # RL1 → VDD   load-supply
        (3, 5, 2),  # RL2 → VDD   load-supply
    ]

    edges = []
    for src, dst, etype in raw_edges:
        edges.append([src, dst, etype])
        edges.append([dst, src, etype])

    gain_raw = metrics["gain_dB"]
    bw_raw   = metrics["bandwidth_MHz"]

    gain_clipped = max(GAIN_MIN, min(GAIN_MAX, gain_raw))
    bw_log       = math.log10(max(bw_raw, 1e-3))

    return {
        "sample_id": sample["sample_id"],
        "nodes":     nodes,
        "edges":     edges,
        "targets":   [gain_clipped, bw_log],
        "raw": {
            "gain_dB":       gain_raw,
            "bandwidth_MHz": bw_raw
        }
    }


def main():
    with open(SAMPLES_PATH) as f:
        samples = json.load(f)

    with open(METRICS_PATH) as f:
        metrics = json.load(f)

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

    graphs  = []
    skipped = 0

    for sample in samples:
        sid = str(sample["sample_id"])
        if sid not in metrics:
            skipped += 1
            continue
        g = build_graph(sample, metrics[sid])
        graphs.append(g)

    with open(OUTPUT_PATH, "w") as f:
        json.dump(graphs, f, indent=2)

    print(f"Built {len(graphs)} graphs  ({skipped} skipped)")
    print(f"Saved → {OUTPUT_PATH}")

    g0 = graphs[0]
    print(f"\nSample graph[0]:")
    print(f"  nodes  : {len(g0['nodes'])} x {len(g0['nodes'][0])} features  (was 7, now 10)")
    print(f"  edges  : {len(g0['edges'])} (bidirectional)")
    print(f"  targets: gain={g0['targets'][0]:.3f} dB  log10(BW)={g0['targets'][1]:.3f}")


if __name__ == "__main__":
    main()