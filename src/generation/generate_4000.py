"""
Run this from ~/NEXUS-Lite to:
1. Append 4000 new samples to samples.json
2. Generate their netlists with correct wrdata paths
"""
import json
import os
import random

SAMPLES_FILE = "data/param_samples/samples.json"
NETLIST_DIR  = "data/netlists"
MODEL_FILE   = "models/nmos.lib"
ABS_RESULT   = os.path.abspath("data/results")

os.makedirs(NETLIST_DIR, exist_ok=True)
os.makedirs(ABS_RESULT,  exist_ok=True)

# ── 1. append 4000 new samples ────────────────────────────────────
with open(SAMPLES_FILE) as f:
    existing = json.load(f)

start_id = max(s["sample_id"] for s in existing) + 1
print(f"Existing: {len(existing)} samples. Generating from id {start_id}...")

new_samples = []
for i in range(4000):
    sid = start_id + i
    new_samples.append({
        "sample_id": sid,
        "topology":  "differential_pair",
        "parameters": {
            "W":      random.uniform(1e-6,   20e-6),
            "L":      random.uniform(180e-9,  2e-6),
            "Ibias":  random.uniform(10e-6,   1e-3),
            "VDD":    random.uniform(1.2,     1.8),
            "RL":     random.uniform(1e3,    50e3),
            "CL":     random.uniform(0.1e-12,10e-12),
            "Vin_cm": random.uniform(0.5,    1.0),
        }
    })

all_samples = existing + new_samples
with open(SAMPLES_FILE, "w") as f:
    json.dump(all_samples, f, indent=4)
print(f"Saved {len(all_samples)} total samples → {SAMPLES_FILE}")

# ── 2. generate netlists for new samples only ─────────────────────
generated = 0
for sample in new_samples:
    sid    = sample["sample_id"]
    p      = sample["parameters"]
    ac_out = os.path.join(ABS_RESULT, f"{sid}_ac.txt")

    netlist = f"""* Differential Pair
.include {MODEL_FILE}

VDD vdd 0 {p['VDD']}
VINP inp 0 DC {p['Vin_cm']} AC 1
VINN inn 0 DC {p['Vin_cm']}
I1 tail 0 DC {p['Ibias']}
RL1 outp vdd {p['RL']}
RL2 outn vdd {p['RL']}
CL1 outp 0 {p['CL']}
CL2 outn 0 {p['CL']}
M1 outp inp tail 0 nmos W={p['W']} L={p['L']}
M2 outn inn tail 0 nmos W={p['W']} L={p['L']}

.control
set wr_singlescale
set wr_vecnames
ac dec 100 1 1e9
wrdata {ac_out} v(outp) v(outn)
quit
.endc

.end
"""
    with open(os.path.join(NETLIST_DIR, f"{sid}.sp"), "w") as f:
        f.write(netlist)
    generated += 1

print(f"Generated {generated} netlists → {NETLIST_DIR}/")
print("\nNext: python src/simulation/simulation_runner.py")