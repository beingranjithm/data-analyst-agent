"""LIVE test of the chat WITH calculations on the sample file. Run from the project root:
    python scripts/live_calc.py                  (5 questions = about 10 Gemini calls; each answer is checked)
    python scripts/live_calc.py --ask "Which product sold the most units?"   (one question of your own)
    python scripts/live_calc.py --interactive    (type questions until you press Enter on an empty line)
    python scripts/live_calc.py --replay         (re-use saved replies: NO Gemini requests, no key needed)
    python scripts/live_calc.py --no-cache       (ignore saved replies and ask the model again)
The AI sees the text profile and, for calculations, only a small result (at most 20 rows), never the table.
The code it writes runs in the sandbox on this computer. For the 5 built-in questions the right answer is also
computed directly with pandas, so each answer is marked CORRECT or WRONG.
"""
import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

MODE = "replay" if "--replay" in sys.argv else ("off" if "--no-cache" in sys.argv else "use")
ASK = sys.argv[sys.argv.index("--ask") + 1] if "--ask" in sys.argv and len(sys.argv) > sys.argv.index("--ask") + 1 else None

if MODE != "replay":
    from dotenv import load_dotenv

    load_dotenv(ROOT / "hello_agent" / ".env")
    if not (os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")):
        sys.exit("No API key found in hello_agent/.env (expected GOOGLE_API_KEY=...). Nothing was sent.")

from agents.chat_agent import MAX_RESULT_ROWS, ChatSession
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
chat = ChatSession(profile, model, tables=df,
                   on_wait=lambda n, seconds, why: print(f"  (server busy, waiting {seconds:.0f}s, try {n}: {why[:80]})", flush=True))

_NUM = re.compile(r"(?<![\w.])-?\d[\d,]*(?:\.\d+)?")


def has_number(text, truth):
    """True when the text contains the true number, as written or rounded (1,708,451 for 1708450.5)."""
    for tok in _NUM.findall(text):
        tok = tok.replace(",", "")
        decimals = len(tok.split(".")[1]) if "." in tok else 0
        if abs(float(tok) - float(truth)) <= 0.5 * 10 ** -decimals + 1e-9:
            return True
    return False


def top_category():
    totals = df.groupby("category")["revenue"].sum().sort_values(ascending=False)
    return totals.index[0], totals.iloc[0]


# (question, how to check the answer text against the truth computed straight from the data)
CHECKS = [
    ("What was the total revenue of the East region in March?",
     lambda a: has_number(a, df[(df["region"] == "East") & (df["order_date"].dt.month == 3)]["revenue"].sum())),
    ("What is the average discount?",
     lambda a: has_number(a, df["discount"].mean())),
    ("Which category has the highest total revenue, and what is that total?",
     lambda a: top_category()[0].lower() in a.lower() and has_number(a, top_category()[1])),
    ("How many orders were placed in the fourth quarter (October to December)?",
     lambda a: has_number(a, (df["order_date"].dt.month >= 10).sum())),
    ("What share of total revenue comes from Electronics, as a percentage?",
     lambda a: has_number(a, 100 * df[df["category"] == "Electronics"]["revenue"].sum() / df["revenue"].sum())),
]


def ask(question, check=None):
    print(f"\nYOU: {question}")
    reply = chat.ask(question)
    if not reply.ok:
        print("ERROR:", reply.error)
        return None
    for c in reply.calculations:
        print(f"  --- code the AI wrote ({'ran OK' if c.ok else 'FAILED'}) ---")
        print("  " + c.code.replace("\n", "\n  "))
        print(f"  --- what the AI was shown ({len(c.result_text)} characters) ---")
        print("  " + c.result_text.replace("\n", "\n  "))
    print("AI:", reply.text)
    if reply.ungrounded:
        print("  ⚠ numbers in neither the summary nor a result (double-check):", ", ".join(reply.ungrounded))
    if check is not None:
        verdict = check(reply.text)
        print("  CHECK:", "CORRECT (matches pandas)" if verdict else "WRONG (does not match pandas)")
        return verdict
    return True


if ASK:
    ask(ASK)
elif "--interactive" in sys.argv:
    while True:
        q = input("\nYour question (empty line to stop): ").strip()
        if not q:
            break
        ask(q)
else:
    results = [ask(q, check) for q, check in CHECKS]
    # privacy check: a request for the whole table must still show the AI at most MAX_RESULT_ROWS rows
    chat.reset()
    print("\n--- size check: asking for the whole table ---")
    reply = chat.ask("Show me all the orders in the data, every row.")
    sizes = [c.result_text.count("\n") + 1 for c in reply.calculations] if reply.ok else []
    big = [n for n in sizes if n > MAX_RESULT_ROWS + 3]
    print("AI was shown", sizes or "no table", "lines;", "TOO MANY" if big else "within the limit")
    done = [r for r in results if r is not None]
    print(f"\nSUMMARY: {sum(1 for r in done if r)}/{len(CHECKS)} answers correct, "
          f"{len(CHECKS) - len(done)} could not be answered, size limit {'FAILED' if big else 'ok'}")

real, replayed = usage_of(model)
print(f"\nchat models -> {real} real request(s), {replayed} replayed from saved replies")
