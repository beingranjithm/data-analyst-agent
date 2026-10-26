"""LIVE test of the approval workflow: the planner proposes a dashboard, YOU decide, then it is built.
Run from the project root:
    python scripts/live_approval.py                  (real models; asks which charts to drop)
    python scripts/live_approval.py --auto           (approve everything without asking)
    python scripts/live_approval.py --replay         (re-use saved model replies: no requests, no key needed)
    python scripts/live_approval.py --no-ai          (rule-based charts only: no model at all)
    python scripts/live_approval.py --no-cache       (ignore saved replies and ask the models again)
    python scripts/live_approval.py --max-charts 3   (propose at most 3 charts)
Writes /tmp/dashboard_day12.html (open it with: xdg-open /tmp/dashboard_day12.html)."""
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

USE_AI = "--no-ai" not in sys.argv
AUTO = "--auto" in sys.argv
MODE = "replay" if "--replay" in sys.argv else ("off" if "--no-cache" in sys.argv else "use")
MAX_CHARTS = int(sys.argv[sys.argv.index("--max-charts") + 1]) if "--max-charts" in sys.argv else None

if USE_AI and MODE != "replay":
    from dotenv import load_dotenv

    load_dotenv(ROOT / "hello_agent" / ".env")
    if not (os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")):
        sys.exit("No API key found in hello_agent/.env (expected GOOGLE_API_KEY=...). Nothing was sent.")

from agents.approval_workflow import ApprovalSession
from agents.llm_factory import make_llm
from agents.resilience import quiet_adk_logs
from tools.approval_cli import HELP, parse_answer
from tools.cleaner import clean_dataframe
from tools.dashboard_html import write_dashboard_html
from tools.loader import load_csv_bytes
from tools.profiler import profile_dataframe

quiet_adk_logs()
df, _ = load_csv_bytes((ROOT / "sample_data" / "sales_sample.csv").read_bytes())
df, _ = clean_dataframe(df)
profile = profile_dataframe(df)

planner_llm = make_llm("planner", mode=MODE) if USE_AI else None
chart_llm = make_llm("chart", mode=MODE) if USE_AI else None


def progress(stage, done, total, message):
    print(f"  [{stage}] {message}", flush=True)


session = ApprovalSession(df, profile, use_ai=USE_AI, max_charts=MAX_CHARTS, progress=progress, auto_approve=AUTO,
                          planner_model=planner_llm, chart_model=chart_llm)
t0 = time.time()
proposal = session.start()
plan = proposal.plan

print(f"\nPROPOSED DASHBOARD: {plan.title}   (mode: {proposal.mode})")
if plan.summary:
    print(plan.summary)
for n in proposal.notes:
    print("NOTE:", n)
print("KPIs:", "; ".join(f"{k.label} ({k.agg} of {k.measure})" for k in plan.kpis) or "none")
print("CHARTS")
for i, c in enumerate(plan.charts, 1):
    print(f"  {i}. [{c.type}] {c.title}" + (f"  -  {c.why}" if c.why else ""))

if AUTO:
    decision = None
    print("\n--auto: everything is approved.")
else:
    while True:
        answer = input(f"\nDrop which charts?  {HELP}\n> ")
        try:
            decision = parse_answer(answer)
            if session.state == "awaiting":
                dashboard = session.decide(decision)
            break
        except ValueError as e:
            print("  ->", e)
dashboard = session.dashboard if AUTO else dashboard

if dashboard is None:
    print("\nCancelled. Nothing was built.")
else:
    print(f"\nDone in {time.time() - t0:.0f}s (waiting for you included).  mode={dashboard.mode}")
    for n in dashboard.notes:
        print("NOTE:", n)
    for k in dashboard.kpis:
        print(f"  KPI {k.label}: {'n/a' if k.value is None else format(k.value, ',.2f')} {k.note}")
    for i, c in enumerate(dashboard.charts, 1):
        status = f"OK via {c.source}" if c.ok else "FAILED"
        print(f"  {i}. [{c.spec.type}] {c.spec.title}: {status}, AI attempts={c.attempts}")
        if c.ai_message:
            print(f"       AI route failed because: {c.ai_message[:300]}")
        if not c.ok:
            print(f"       {c.message[:300]}")
    ai = sum(1 for c in dashboard.ok_charts if c.source == "ai")
    print(f"\nSUMMARY: {len(dashboard.ok_charts)}/{len(dashboard.charts)} charts shown ({ai} by AI, "
          f"{len(dashboard.ok_charts) - ai} rule-based), {len(dashboard.failed_charts)} failed")
    print("saved:", write_dashboard_html(dashboard, "/tmp/dashboard_day12.html"))
for name, llm in (("planner", planner_llm), ("chart", chart_llm)):
    if llm is not None:
        print(f"{name} models -> {llm.summary()}")
