import glob
import os
import re

NETLIST_DIR = "data/netlists"
ABS_RESULT_DIR = os.path.abspath("data/results")
os.makedirs(ABS_RESULT_DIR, exist_ok=True)

files = sorted(glob.glob(f"{NETLIST_DIR}/*.sp"))
print(f"Found {len(files)} netlists")

patched = 0
failed  = 0

for fpath in files:
    name = os.path.basename(fpath).replace(".sp", "")
    ac_path = os.path.join(ABS_RESULT_DIR, f"{name}_ac.txt")

    with open(fpath, "r") as f:
        content = f.read()

    # Remove .op so only AC runs, force wrdata after ac analysis
    new_control = f""".control
set wr_singlescale
set wr_vecnames
ac dec 100 1 1e9
wrdata {ac_path} v(outp) v(outn)
quit
.endc"""

    # Replace existing .control...endc block
    new_content = re.sub(
        r'\.control.*?\.endc',
        new_control,
        content,
        flags=re.DOTALL | re.IGNORECASE
    )

    # Also remove standalone .op and .ac lines since control block handles it
    new_content = re.sub(r'^\s*\.op\s*$', '', new_content, flags=re.MULTILINE)
    new_content = re.sub(r'^\s*\.ac\s.*$', '', new_content, flags=re.MULTILINE)

    if new_content == content:
        print(f"  WARN: no .control block found in {name}.sp")
        failed += 1
        continue

    with open(fpath, "w") as f:
        f.write(new_content)

    patched += 1

print(f"\nDone: {patched} patched, {failed} skipped")