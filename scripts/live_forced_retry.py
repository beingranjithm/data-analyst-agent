"""LIVE proof that the retry loop works with real Gemini. Run from the project root:
    python scripts/live_forced_retry.py
Attempt 1 is a deliberately broken reply made by us; attempt 2 is REAL Gemini answering the
retry prompt. 2 Gemini calls in total (1 per scenario). Rows are never sent; the second
scenario also checks that a data value in the error message is hidden from Gemini.
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

from google.adk.models import Gemini

from agents.chart_loop import make_chart_healing
from agents.testing import ForcedFirstFailure
from tools.cleaner import clean_dataframe
from tools.loader import load_csv_bytes
from tools.plan_schema import ChartSpec
from tools.profiler import profile_dataframe

MODEL = "gemini-3.5-flash"
df, _ = load_csv_bytes((ROOT / "sample_data" / "sales_sample.csv").read_bytes())
df, _ = clean_dataframe(df)
profile = profile_dataframe(df)
chart = ChartSpec(type="bar", title="Revenue by region", dimension="region", measure="revenue", agg="sum")
secret = str(df["region"].dropna().iloc[0])      # a real data value from the sample file


def block(code):
    return f"```python\n{code}\n```"


def retry_prompt(llm):
    p = llm.prompts[1]
    i = p.find("YOUR PREVIOUS ATTEMPT FAILED")
    return p[i:p.find("ENVIRONMENT", i)].strip() if i >= 0 else "(no retry prompt was sent)"


SCENARIOS = [
    ("A. forced column error",
     block("fig = px.bar(df, x='regoin', y='revenue')"), False),
    ("B. forced error that contains a real data value",
     block("raise ValueError(\"bad value '\" + str(df['region'].dropna().iloc[0]) + \"'\")"), True),
]

results = []
for label, broken, check_leak in SCENARIOS:
    print(f"\n=== {label}")
    llm = ForcedFirstFailure(model=MODEL, inner=Gemini(model=MODEL), first_reply=broken, prompts=[])
    res = make_chart_healing(df, profile, chart, model=llm)
    for a in res.history:
        print(f"  attempt {a.number}: {'OK' if a.ok else 'FAILED'} - {a.message}")
    print("  RESULT:", "OK" if res.ok else "FAILED", "-", res.message)
    if "503" in res.message or "429" in res.message:
        print("  (Google's server was busy or rate-limited. Not a bug in our code: wait a minute and run again.)")
    print("  --- what Gemini was told on the retry ---")
    print("  " + retry_prompt(llm).replace("\n", "\n  "))
    print("  -----------------------------------------")
    leak = None
    if check_leak:
        first_msg = res.history[0].message
        if secret not in first_msg:
            print("  PRIVACY CHECK: INCONCLUSIVE (the error message did not contain the data value; "
                  "your code guard probably blocked the code first)")
        else:
            leak = secret in retry_prompt(llm)      # only the part that came from the error message
            in_profile = secret in llm.prompts[0]   # the 1st prompt contained no error feedback at all
            print("  the value also appears in the normal profile part of the prompt:", "yes" if in_profile else "no")
            print("  PRIVACY CHECK:", "FAILED - the data value reached Gemini through the error message!"
                  if leak else "PASSED - the data value was hidden in the error message sent to Gemini")
    if res.ok:
        out = Path(f"/tmp/chart_day10_forced_{label[0]}.html")
        res.figure.write_html(out)
        print("  saved:", out)
    results.append((label, res.ok, res.attempts, leak))

print("\nSUMMARY")
for label, ok, attempts, leak in results:
    extra = "" if leak is None else ("  PRIVACY LEAK!" if leak else "")
    print(f"  {'OK    ' if ok else 'FAILED'} attempts={attempts}  {label}{extra}")
