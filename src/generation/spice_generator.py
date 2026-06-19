import json
import os


SAMPLES_FILE = "data/param_samples/samples.json"

OUTPUT_DIR = "data/netlists"

MODEL_FILE = "models/nmos.lib"


def generate_netlist(sample):

    params = sample["parameters"]

    sample_id = sample["sample_id"]

    W = params["W"]
    L = params["L"]
    Ibias = params["Ibias"]
    VDD = params["VDD"]
    RL = params["RL"]
    CL = params["CL"]
    Vin_cm = params["Vin_cm"]

    netlist = f"""
* Differential Pair

.include {MODEL_FILE}

VDD vdd 0 {VDD}

VINP inp 0 DC {Vin_cm} AC 1
VINN inn 0 DC {Vin_cm}

I1 tail 0 DC {Ibias}

RL1 outp vdd {RL}
RL2 outn vdd {RL}

CL1 outp 0 {CL}
CL2 outn 0 {CL}

M1 outp inp tail 0 nmos W={W} L={L}
M2 outn inn tail 0 nmos W={W} L={L}

.op

.ac dec 100 1 1e9

.control

run

wrdata data/results/{sample_id}_ac.txt v(outp)

quit

.endc

.end
"""

    return netlist


def main():

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    with open(SAMPLES_FILE, "r") as f:

        samples = json.load(f)

    for sample in samples:

        sample_id = sample["sample_id"]

        netlist = generate_netlist(sample)

        output_file = os.path.join(
            OUTPUT_DIR,
            f"{sample_id}.sp"
        )

        with open(output_file, "w") as f:

            f.write(netlist)

    print("Netlists generated")


if __name__ == "__main__":
    main()