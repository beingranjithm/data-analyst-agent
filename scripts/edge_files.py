"""Day 19: odd CSV files through the whole no-AI pipeline (no Gemini requests, no key needed). Run from the project root:
    python scripts/edge_files.py
Each of about 20 awkward files goes through load, clean, profile, the chat summary, starter questions, filters, the 30
evaluation questions, the rule-based charts and one real sandbox calculation. A file may be REFUSED with a clear message
at the first step (fine); anything else that raises is a CRASH (a bug).
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from evals.edge_files import CRASH, describe, make_files, run_sweep

files = make_files()
print(f"Pushing {len(files)} odd files through the pipeline (no AI involved) ...\n", flush=True)
results = run_sweep(files, progress=lambda o: print(f"  {o.name:<42} {describe(o)}", flush=True))
bad = [o for o in results if o.crashes]
refused = [o for o in results if o.refused_at and not o.crashes and not o.ugly_at]
ugly = [o for o in results if o.ugly_at and not o.crashes]
print(f"\nSUMMARY: {len(results) - len(bad) - len(refused) - len(ugly)} loaded and passed every step, {len(refused)} refused "
      f"with a clear message, {len(ugly)} refused with a raw library message, {len(bad)} CRASHED")
for o in bad:
    for stage, detail in o.crashes:
        print(f"  BUG: {o.name} -> {stage}: {detail}")
for o in ugly:
    print(f"  UGLY MESSAGE: {o.name} -> {o.ugly_at}: {o.stages[o.ugly_at][1]}")
