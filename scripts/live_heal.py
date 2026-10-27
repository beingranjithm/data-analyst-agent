"""LIVE test of the SELF-HEALING chat: questions that tend to go wrong. Run from the project root:
    python scripts/live_heal.py              (10 questions = about 20 to 30 Gemini calls; the first 8 replay for free if nothing changed)
    python scripts/live_heal.py --replay     (re-use saved replies: NO Gemini requests, no key needed)
    python scripts/live_heal.py --no-cache   (ignore saved replies and ask the model again)
The AI sees the text profile and, for calculations, only a small result (at most 20 rows), never the table.
Each answer is checked: numbers against pandas, and 'trap' questions (a region or a column that does not exist)
must be answered with 'there is no such thing', never with an invented number. The traps use simple wording
checks, so a WRONG there deserves a look at the answer before you worry.
Two questions are traps for the first try (a region spelled in lowercase that the summary does not list, and a
statistics fit that tempts the AI to import a library the sandbox blocks). Real Gemini often gets these right first
time, so two more questions FORCE a failure: the first calculation is replaced by a simulated sandbox error (or a
simulated empty result), and the real AI has to repair its own code. Those two prove the self-healing.
"""
import os
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

MODE = "replay" if "--replay" in sys.argv else ("off" if "--no-cache" in sys.argv else "use")

if MODE != "replay":
    from dotenv import load_dotenv

    load_dotenv(ROOT / "hello_agent" / ".env")
    if not (os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")):
        sys.exit("No API key found in hello_agent/.env (expected GOOGLE_API_KEY=...). Nothing was sent.")

from agents.chat_agent import MAX_CODE_RUNS, MAX_RESULT_ROWS, ChatSession, default_runner
from agents.llm_factory import make_llm
from agents.resilience import quiet_adk_logs
from tools.cleaner import clean_dataframe
from tools.loader import load_csv_bytes
from tools.profiler import profile_dataframe
from ui.dashboard_tab import usage_of

quiet_adk_logs()
df, _ = load_csv_bytes((ROOT / "sample_data" / "sales_sample.csv").read_bytes())
df, _ = clean_dataframe(df)
profile = profile_dataframe(df)
model = make_llm("chat", mode=MODE)
chat = ChatSession(profile, model, tables=df, max_code_runs=MAX_CODE_RUNS,
                   on_wait=lambda n, seconds, why: print(f"  (server busy, waiting {seconds:.0f}s, try {n}: {why[:80]})", flush=True))

_NUM = re.compile(r"(?<![\w.])-?(?:\d{1,3}(?:,\d{3})+(?!\d)(?:\.\d+)?|\d+(?:\.\d+)?)")
NOT_FOUND = ("no ", "not ", "n't", "none", "nothing", "cannot", "unable", "does not", "do not")


def has_number(text, truth):
    """True when the text contains the true number, as written or rounded (1,708,451 for 1708450.5)."""
    for tok in _NUM.findall(text):
        tok = tok.replace(",", "")
        decimals = len(tok.split(".")[1]) if "." in tok else 0
        if abs(float(tok) - float(truth)) <= 0.5 * 10 ** -decimals + 1e-9:
            return True
    return False


def says_not_found(text):
    return any(w in text.lower() for w in NOT_FOUND)


def claims_zero_total(text):
    """True when the answer states a total of 0 (a full stop after the 0 is fine: 'the total is 0.')."""
    return bool(re.search(r"(revenue|total)\s+(is|was|of|equals)\s+\$?0(?:\.0+)?(?!\d|\.\d)", text.lower()))


month_totals = df.groupby(df["order_date"].dt.month)["revenue"].sum()
# a region the summary does NOT list (it shows only the 3 most common), asked in lowercase: the first try may miss
_regions = df["region"].value_counts()
HIDDEN = _regions.index[3] if len(_regions) > 3 else _regions.index[-1]
_ok = df[["units", "revenue"]].dropna()
SLOPE = float(np.polyfit(_ok["units"], _ok["revenue"], 1)[0])
top_product = df.groupby("product")["units"].sum().sort_values(ascending=False)

# (question, check(reply) -> bool or None, what a good answer looks like)
CHECKS = [
    (f"What is the total revenue for the {HIDDEN.lower()} region?",
     lambda r: has_number(r.text, df[df["region"] == HIDDEN]["revenue"].sum()),
     f"the {HIDDEN} total (the AI may have to fix the lowercase spelling)"),
    ("What is the total revenue for the Atlantis region?",
     lambda r: ("minor" if claims_zero_total(r.text) else True) if says_not_found(r.text) and not r.ungrounded else False,
     "says nothing matched and quotes no total (a quoted zero total is only a minor slip)"),
    ("What is the average customer age?",
     lambda r: says_not_found(r.text) and not r.ungrounded,
     "says there is no age column, no invented number"),
    ("Which product sold the most units, and how many units did it sell?",
     lambda r: top_product.index[0].lower() in r.text.lower() and has_number(r.text, top_product.iloc[0]),
     "top product AND its units"),
    ("Show me all the orders in the data, every row.",
     lambda r: all(c.result_text.count("\n") + 1 <= MAX_RESULT_ROWS + 3 for c in r.calculations),
     "never more than 20 rows go to the AI"),
    ("Fit a straight line of revenue against units and tell me the slope.",
     lambda r: has_number(r.text, SLOPE),
     "the slope from a simple linear fit (libraries like scipy are not allowed, numpy is)"),
    ("What is the total revenue for each day?",
     lambda r: all(c.result_text.count("\n") + 1 <= MAX_RESULT_ROWS + 3 for c in r.calculations),
     "a long table: never more than 20 rows go to the AI"),
    ("What was the revenue growth from January to December, as a percentage?",
     lambda r: has_number(r.text, 100 * (month_totals[12] - month_totals[1]) / month_totals[1]),
     "(December total - January total) / January total x 100"),
]


def run(question, check, expect):
    print(f"\nYOU: {question}\n  (a good answer: {expect})")
    reply = chat.ask(question)
    if not reply.ok:
        print("ERROR:", reply.error)
        return None, 0
    for n, c in enumerate(reply.calculations, 1):
        state = "FAILED" if not c.ok else "found nothing" if c.empty else "ran OK"
        print(f"  --- try {n}: {state} ---")
        print("  " + c.code.replace("\n", "\n  "))
        print("  >> " + c.result_text[:300].replace("\n", "\n  >> "))
        if "only the first" in c.result_text:
            print(f"  (size limit worked: the AI was shown {c.result_text.count(chr(10)) + 1} lines of a longer table)")
    print("AI:", reply.text)
    if reply.ungrounded:
        print("  ⚠ numbers in neither the summary nor a result (double-check):", ", ".join(reply.ungrounded))
    verdict = check(reply)
    print("  CHECK:", "GOOD" if verdict is True else "OK, with a minor slip (it quotes a zero total, but says nothing matched)"
          if verdict == "minor" else "WRONG (read the answer above)")
    return verdict, len(reply.calculations)


class SimulatedRun:
    """What the sandbox would hand back after a failure ("error") or when nothing was found ("empty")."""

    def __init__(self, kind):
        self.ok = kind != "error"
        self.status = "ok" if self.ok else "error"
        self.stdout, self.value, self.problems, self.error = "", None, [], None
        self.table = pd.DataFrame({"result": []}) if kind == "empty" else None
        self.table_total_rows = 0

    def message(self):
        return "ValueError: could not convert string to float: 'n/a'"


class ForceFirst:
    """A runner that sabotages the FIRST calculation and runs every later one for real in the sandbox."""

    def __init__(self, kind):
        self.kind, self.fired = kind, False

    def __call__(self, code, tables):
        if not self.fired:
            self.fired = True
            return SimulatedRun(self.kind)
        return default_runner(code, tables)


def run_forced(kind, label):
    forced = ChatSession(profile, model, tables=df, max_code_runs=MAX_CODE_RUNS, runner=ForceFirst(kind),
                         on_wait=lambda n, seconds, why: print(f"  (server busy, waiting {seconds:.0f}s)", flush=True))
    east = df[df["region"] == "East"]["revenue"].sum()
    question = "What is the total revenue for the East region?"
    print(f"\nYOU: {question}\n  (FORCED FAILURE: {label}; the AI must repair its code and still answer {east:,.2f})")
    reply = forced.ask(question)
    if not reply.ok:
        print("ERROR:", reply.error)
        return None, 0
    for n, c in enumerate(reply.calculations, 1):
        state = "FAILED (simulated)" if not c.ok else "found nothing (simulated)" if c.empty else "ran OK (real sandbox)"
        print(f"  --- try {n}: {state} ---")
        print("  " + c.code.replace("\n", "\n  "))
        print("  >> " + c.result_text[:200].replace("\n", "\n  >> "))
    print("AI:", reply.text)
    healed = (len(reply.calculations) >= 2 and (not reply.calculations[0].ok or reply.calculations[0].empty)
              and reply.calculations[-1].ok and not reply.calculations[-1].empty)
    right = has_number(reply.text, east)
    print("  CHECK:", "GOOD (it healed itself and the number matches pandas)" if healed and right else
          "WRONG (it did not repair its code, or the number is off)")
    return (healed and right), len(reply.calculations)


outcomes = []
for question, check, expect in CHECKS:
    chat.reset()                                  # every question starts fresh, so tries are not shared
    outcomes.append(run(question, check, expect))
outcomes.append(run_forced("error", "the first run is replaced by a simulated sandbox error"))
outcomes.append(run_forced("empty", "the first run is replaced by a simulated empty result"))
total = len(CHECKS) + 2
good = sum(1 for v, _ in outcomes if v is True)
minor = sum(1 for v, _ in outcomes if v == "minor")
healed = sum(1 for v, n in outcomes if v is True and n > 1)
print(f"\nSUMMARY: {good}/{total} good, {minor} with a minor slip; {healed} needed more than one try (self-healed); "
      f"{sum(1 for v, _ in outcomes if v is None)} could not be answered")
real, replayed = usage_of(model)
print(f"chat models -> {real} real request(s), {replayed} replayed from saved replies")
