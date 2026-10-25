"""LIVE test: Gemini plans the dashboard, then writes code for the first N charts, which run in your sandbox.
Run from the project root:  python scripts/live_chart_agent.py [N]      (default N = 3)
Sends only the text profile and chart specs to Gemini, never rows. Writes HTML files to /tmp.
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

from agents.chart_agent import make_chart
from agents.planner import plan_dashboard
from tools.cleaner import clean_dataframe
from tools.loader import load_csv_bytes
from tools.profiler import profile_dataframe

N = int(sys.argv[1]) if len(sys.argv) > 1 else 3

df, _ = load_csv_bytes((ROOT / "sample_data" / "sales_sample.csv").read_bytes())
df, _ = clean_dataframe(df)
profile = profile_dataframe(df)

print("Planning (1 Gemini call) ...")
plan = plan_dashboard(profile)
if not plan.ok:
    print("Planner failed:", *plan.problems, sep="\n  - ")
    sys.exit(1)

charts = plan.plan.charts[:N]
passed = 0
for i, chart in enumerate(charts, 1):
    print(f"\n=== Chart {i}/{len(charts)}: [{chart.type}] {chart.title}")
    res = make_chart(df, profile, chart)
    print("RESULT:", "OK" if res.ok else "FAILED", "-", res.message)
    if res.code:
        print("--- code the model wrote ---")
        print(res.code)
        print("----------------------------")
    if res.ok:
        passed += 1
        out = Path(f"/tmp/chart_day9_{i}.html")
        res.figure.write_html(out)
        print("saved:", out)
    elif not res.code:
        print("RAW REPLY (first 400 chars):", res.raw[:400])
print(f"\nFirst-try success: {passed}/{len(charts)}")
