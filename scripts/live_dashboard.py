"""LIVE test: builds a whole dashboard from the sample file. Run from the project root:
    python scripts/live_dashboard.py            (AI plan + AI charts: about 5 to 20 Gemini calls)
    python scripts/live_dashboard.py --no-ai    (rule-based only: NO Gemini calls, no key needed)
    python scripts/live_dashboard.py --max-charts 3   (AI, but only 3 charts: saves free-tier quota)
Only the text profile, chart specs and scrubbed error messages go to Gemini, never rows.
Writes /tmp/dashboard_day11.html (open it with: xdg-open /tmp/dashboard_day11.html).
"""
import html
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

USE_AI = "--no-ai" not in sys.argv
MAX_CHARTS = int(sys.argv[sys.argv.index("--max-charts") + 1]) if "--max-charts" in sys.argv else None

if USE_AI:
    from dotenv import load_dotenv

    load_dotenv(ROOT / "hello_agent" / ".env")
    if not (os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")):
        sys.exit("No API key found in hello_agent/.env (expected GOOGLE_API_KEY=...). Nothing was sent.")

from agents.dashboard import build_dashboard
from agents.resilience import quiet_adk_logs
from tools.cleaner import clean_dataframe
from tools.loader import load_csv_bytes
from tools.profiler import profile_dataframe

quiet_adk_logs()
df, _ = load_csv_bytes((ROOT / "sample_data" / "sales_sample.csv").read_bytes())
df, _ = clean_dataframe(df)
profile = profile_dataframe(df)


def progress(stage, done, total, message):
    print(f"  [{stage}] {message}", flush=True)


t0 = time.time()
d = build_dashboard(df, profile, use_ai=USE_AI, max_charts=MAX_CHARTS, progress=progress)
print(f"\nDone in {time.time() - t0:.0f}s.  mode={d.mode}")
print(f"TITLE: {d.title}\nSUMMARY: {d.summary}")
for n in d.notes:
    print("NOTE:", n)

print("\nKPIs")
for k in d.kpis:
    print(f"  {k.label}: {k.value if k.value is None else format(k.value, ',.2f')}  ({k.agg} of {k.measure}) {k.note}")

print("\nCHARTS")
for i, c in enumerate(d.charts, 1):
    status = f"OK via {c.source}" if c.ok else "FAILED"
    print(f"  {i}. [{c.spec.type}] {c.spec.title}: {status}, AI attempts={c.attempts}")
    if c.ai_message:
        print(f"       AI route failed because: {c.ai_message[:300]}")
    if not c.ok:
        print(f"       {c.message[:300]}")
    for w in c.warnings:
        print(f"       warning: {w}")

ai = sum(1 for c in d.ok_charts if c.source == "ai")
print(f"\nSUMMARY: {len(d.ok_charts)}/{len(d.charts)} charts shown ({ai} by AI, {len(d.ok_charts) - ai} rule-based), "
      f"{len(d.failed_charts)} failed")

parts = [f"<h1>{html.escape(d.title)}</h1><p>{html.escape(d.summary)}</p><p><i>mode: {html.escape(d.mode)}, {d.rows} rows</i></p>",
         "<div style='display:flex;gap:16px;flex-wrap:wrap'>" + "".join(
             f"<div style='border:1px solid #ccc;border-radius:8px;padding:12px 18px'><div style='color:#666'>{html.escape(k.label)}</div>"
             f"<div style='font-size:26px'>{'n/a' if k.value is None else format(k.value, ',.2f')}</div></div>" for k in d.kpis) + "</div>"]
first = True
for c in d.ok_charts:
    parts.append(c.figure.to_html(full_html=False, include_plotlyjs=True if first else False))
    first = False
for c in d.failed_charts:
    parts.append(f"<p style='color:#b00'>Chart skipped: {html.escape(c.spec.title)}: {html.escape(c.message[:300])}</p>")
Path("/tmp/dashboard_day11.html").write_text("<html><body style='font-family:sans-serif;max-width:1000px;margin:auto'>" + "".join(parts) + "</body></html>")
print("saved: /tmp/dashboard_day11.html")
