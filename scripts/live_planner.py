"""LIVE test: sends the sample dataset PROFILE (not data) to Gemini and prints the plan.
Run from the project root:  python scripts/live_planner.py
"""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv

load_dotenv(ROOT / "hello_agent" / ".env")
if not (os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")):
    sys.exit("No API key found in hello_agent/.env (expected GOOGLE_API_KEY=...). Nothing was sent.")

from agents.planner import plan_dashboard
from tools.charts import suggest_charts
from tools.cleaner import clean_dataframe
from tools.loader import load_csv_bytes
from tools.profiler import profile_dataframe

df, _ = load_csv_bytes((ROOT / "sample_data" / "sales_sample.csv").read_bytes())
df, _ = clean_dataframe(df)
profile = profile_dataframe(df)

res = plan_dashboard(profile)
print(f"ok={res.ok}  attempts={res.attempts}")
if not res.ok:
    print("PROBLEMS:", *res.problems, sep="\n  - ")
    print("RAW (first 500 chars):", res.raw[:500])
    sys.exit(1)

p = res.plan
print(f"\n{p.title}\n{p.summary}\n")
print("KPIs:", [(k.label, k.agg, k.measure) for k in p.kpis])
print("\nAI PLAN")
for i, c in enumerate(p.charts, 1):
    print(f" {i}. [{c.type}] {c.title}  {c.to_spec()}\n      why: {c.why}")
print("\nRULE-BASED GALLERY")
for i, s in enumerate(suggest_charts(profile, max_charts=8), 1):
    print(f" {i}. {s}")
