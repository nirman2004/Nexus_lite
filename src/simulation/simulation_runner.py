import os
import subprocess
import glob
import numpy as np
import json
from multiprocessing import Pool

NETLIST_DIR = "data/netlists"
RESULT_DIR  = "data/results"
os.makedirs(RESULT_DIR, exist_ok=True)


def parse_ac_results(ac_file):
    try:
        # skip header row written by wr_vecnames
        data = np.loadtxt(ac_file, skiprows=1)

        if data.ndim == 1:
            data = data.reshape(1, -1)
        if data.shape[0] < 10:
            return None

        freq      = data[:, 0]
        re_outp   = data[:, 1]
        im_outp   = data[:, 2]
        magnitude = np.sqrt(re_outp**2 + im_outp**2)

        gain_dB_arr = 20 * np.log10(magnitude + 1e-12)
        gain_dB_max = float(np.max(gain_dB_arr))
        peak_idx    = np.argmax(gain_dB_arr)
        target      = gain_dB_max - 3.0

        bw_hz = float(freq[-1])
        for i in range(peak_idx, len(gain_dB_arr)):
            if gain_dB_arr[i] < target:
                bw_hz = float(freq[i])
                break

        return {
            "gain_dB":       round(gain_dB_max, 4),
            "bandwidth_MHz": round(bw_hz / 1e6,  4),
        }

    except Exception as e:
        return None


def run(netlist):
    name     = os.path.basename(netlist).replace(".sp", "")
    ac_file  = os.path.join(RESULT_DIR, f"{name}_ac.txt")
    log_file = os.path.join(RESULT_DIR, f"{name}_log.txt")

    proc = subprocess.run(
        ["ngspice", "-b", netlist],
        capture_output=True,
        text=True
    )

    with open(log_file, "w") as f:
        f.write(proc.stdout or "")
        f.write("\n--- STDERR ---\n")
        f.write(proc.stderr or "")

    metrics = None
    if os.path.exists(ac_file):
        metrics = parse_ac_results(ac_file)

    if metrics:
        print(f"  OK   {name}: gain={metrics['gain_dB']} dB  BW={metrics['bandwidth_MHz']} MHz")
    else:
        print(f"  FAIL {name}: check {log_file}")

    return {"name": name, "metrics": metrics}


def main():
    netlists = sorted(glob.glob(f"{NETLIST_DIR}/*.sp"))
    print(f"Found {len(netlists)} netlists — running with {os.cpu_count()} cores\n")

    with Pool(processes=os.cpu_count()) as pool:
        all_results = pool.map(run, netlists)

    output = {r["name"]: r["metrics"] for r in all_results if r["metrics"]}
    out_path = os.path.join(RESULT_DIR, "all_metrics.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)

    failed = [r["name"] for r in all_results if r["metrics"] is None]
    print(f"\n{'='*50}")
    print(f"Done: {len(all_results) - len(failed)}/{len(all_results)} succeeded")
    print(f"Metrics saved → {out_path}")
    if failed:
        print(f"Failed ({len(failed)}): {failed[:5]} ...")


if __name__ == "__main__":
    main()