import json
import os
import random


OUTPUT_FILE = "data/param_samples/samples.json"

NUM_SAMPLES = 1000


def generate_sample(sample_id):

    sample = {

        "sample_id": sample_id,

        "topology": "differential_pair",

        "parameters": {

            "W": random.uniform(1e-6, 20e-6),

            "L": random.uniform(180e-9, 2e-6),

            "Ibias": random.uniform(10e-6, 1e-3),

            "VDD": random.uniform(1.2, 1.8),

            "RL": random.uniform(1e3, 50e3),

            "CL": random.uniform(0.1e-12, 10e-12),

            "Vin_cm": random.uniform(0.5, 1.0)
        }
    }

    return sample


def main():

    os.makedirs("data/param_samples", exist_ok=True)

    samples = []

    for i in range(NUM_SAMPLES):

        sample = generate_sample(i)

        samples.append(sample)

    with open(OUTPUT_FILE, "w") as f:

        json.dump(samples, f, indent=4)

    print(f"Generated {NUM_SAMPLES} samples")


if __name__ == "__main__":
    main()