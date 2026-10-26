"""Checks that the text sent to the planner is IDENTICAL on every run. The reply cache only works if it is.
Builds the dataset profile and the planner prompt in several fresh Python processes with different hash
seeds and compares them. No model calls, no key needed.   Run:  python scripts/check_prompt_stability.py
Exit code 0 = stable, 1 = the prompt changes between runs (a diff of the first difference is printed)."""
import difflib
import hashlib
import os
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def build_text() -> str:
    from agents.planner import INSTRUCTION
    from tools.cleaner import clean_dataframe
    from tools.loader import load_csv_bytes
    from tools.profiler import profile_dataframe, profile_to_text
    df, _ = load_csv_bytes((ROOT / "sample_data" / "sales_sample.csv").read_bytes())
    df, _ = clean_dataframe(df)
    return INSTRUCTION.replace("{profile_text}", profile_to_text(profile_dataframe(df))).replace("{feedback}", "")


if len(sys.argv) > 2 and sys.argv[1] == "--child":
    Path(sys.argv[2]).write_text(build_text())
    sys.exit(0)

SEEDS = ("1", "2", "3")
texts = {}
with tempfile.TemporaryDirectory() as tmp:
    for seed in SEEDS:
        out = Path(tmp) / f"prompt_{seed}.txt"
        r = subprocess.run([sys.executable, __file__, "--child", str(out)], capture_output=True, text=True,
                           env={**os.environ, "PYTHONHASHSEED": seed}, cwd=ROOT, timeout=180)
        if r.returncode != 0:
            sys.exit(f"The child process failed (seed {seed}):\n{r.stderr[-1500:]}")
        texts[seed] = out.read_text()

hashes = {s: hashlib.sha256(t.encode()).hexdigest()[:12] for s, t in texts.items()}
for s in SEEDS:
    print(f"  run with hash seed {s}: {len(texts[s])} characters, fingerprint {hashes[s]}")
if len(set(hashes.values())) == 1:
    print("\nSTABLE: the planner prompt is identical on every run.")
    sys.exit(0)

other = next(s for s in SEEDS[1:] if hashes[s] != hashes[SEEDS[0]])
print(f"\nUNSTABLE: the planner prompt differs between runs. First differences (seed {SEEDS[0]} vs seed {other}):\n")
diff = difflib.unified_diff(texts[SEEDS[0]].splitlines(), texts[other].splitlines(),
                            f"seed {SEEDS[0]}", f"seed {other}", lineterm="", n=0)
for i, line in enumerate(diff):
    if i >= 30:
        print("   ... (more differences)")
        break
    print("  ", line[:220])
sys.exit(1)
