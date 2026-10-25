"""LIVE test of the self-healing loop. Run from the project root:
    python scripts/live_healing.py [N]        (N = how many of the 6 test charts to run, default 6)
Each chart uses 1 to 3 Gemini calls. Only the text profile, the chart spec and SCRUBBED error
messages are sent to Gemini, never rows. Figures are saved as /tmp/chart_day10_<n>.html.
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

from agents.chart_loop import make_chart_healing
from tools.cleaner import clean_dataframe
from tools.loader import load_csv_bytes
from tools.plan_schema import ChartSpec
from tools.profiler import profile_dataframe

N = int(sys.argv[1]) if len(sys.argv) > 1 else 6

df, _ = load_csv_bytes((ROOT / "sample_data" / "sales_sample.csv").read_bytes())
df, _ = clean_dataframe(df)
profile = profile_dataframe(df)

CHARTS = [
    ("TYPO TEST (column 'revnue' does not exist)",
     ChartSpec(type="bar", title="Revenue by region", dimension="region", measure="revnue", agg="sum")),
    ("MISSING-ROWS TEST (region has missing values)",
     ChartSpec(type="bar", title="Revenue by region", dimension="region", measure="revenue", agg="sum")),
    ("histogram",
     ChartSpec(type="histogram", title="Distribution of discount", measure="discount")),
    ("box",
     ChartSpec(type="box", title="Revenue spread by category", dimension="category", measure="revenue")),
    ("scatter",
     ChartSpec(type="scatter", title="Units vs revenue", x="units", y="revenue")),
    ("heatmap",
     ChartSpec(type="heatmap", title="Correlation of numbers", measures=["units", "unit_price", "discount", "revenue"])),
]

summary = []
for i, (label, chart) in enumerate(CHARTS[:N], 1):
    print(f"\n=== {i}. {label}  [{chart.type}] {chart.title}")
    res = make_chart_healing(df, profile, chart)
    for a in res.history:
        print(f"  attempt {a.number}: {'OK' if a.ok else 'FAILED'} - {a.message}")
    print("  RESULT:", "OK" if res.ok else "FAILED", "-", res.message)
    if res.ok:
        out = Path(f"/tmp/chart_day10_{i}.html")
        res.figure.write_html(out)
        notes = [a.text for a in (res.figure.layout.annotations or [])]
        print("  saved:", out, "| annotations:", notes if notes else "none")
    summary.append((label, res.ok, res.attempts))

print("\nSUMMARY")
for label, ok, attempts in summary:
    print(f"  {'OK    ' if ok else 'FAILED'} attempts={attempts}  {label}")
