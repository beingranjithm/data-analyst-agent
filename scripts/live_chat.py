"""LIVE test of the chat agent on the sample file. Run from the project root:
    python scripts/live_chat.py                  (4 questions = about 4 Gemini calls)
    python scripts/live_chat.py --ask "What is in this data?"   (one question of your own)
    python scripts/live_chat.py --interactive    (type questions until you press Enter on an empty line)
    python scripts/live_chat.py --replay         (re-use saved replies: NO requests, no key needed)
    python scripts/live_chat.py --no-cache       (ignore saved replies and ask the model again)
The model sees only the text profile (column names, types, statistics, category names), never rows.
"""
import os
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

from agents.chat_agent import ChatSession, starter_questions
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
chat = ChatSession(profile, model, on_wait=lambda n, seconds, why: print(f"  (server busy, waiting {seconds:.0f}s, try {n}: {why[:80]})", flush=True))


def ask(question):
    print(f"\nYOU: {question}")
    reply = chat.ask(question)
    if not reply.ok:
        print("ERROR:", reply.error)
        return False
    print("AI:", reply.text)
    if reply.ungrounded:
        print("  ⚠ numbers not in the data summary (double-check):", ", ".join(reply.ungrounded))
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
    questions = starter_questions(profile)[:3] + ["What was the total revenue of the East region in March?"]
    for q in questions:
        if not ask(q):
            break
    else:
        print("\n(The last question needs a calculation over the rows, so the right answer is to say it cannot be "
              "answered from the summary. Calculations arrive on Day 15.)")

real, replayed = usage_of(model)
print(f"\nchat models -> {real} real request(s), {replayed} replayed from saved replies")
