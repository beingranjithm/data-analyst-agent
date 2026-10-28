"""Day 18: how accurate is the chat? 30 questions, true answers from pandas, an automatic score and a report.
Run from the project root:
    python scripts/eval_chat.py                   (first run: about 55 to 75 Gemini requests; answers are saved)
    python scripts/eval_chat.py --replay          (re-use saved replies: NO Gemini requests, no key needed)
    python scripts/eval_chat.py --no-cache        (ignore saved replies and ask the model again)
    python scripts/eval_chat.py --only 1,7,13     (just these questions)     --category traps    (just one group)
    python scripts/eval_chat.py --max-requests 80 (stop before more than 80 real requests; default 150)
It stops cleanly when the free daily quota is used up. Answered questions are saved, so running it again later
(tomorrow) carries on and only asks the questions that are left.
The AI sees only the text profile and small calculation results (at most 20 rows), never the table.
The report is written to eval_results/chat_eval.md and .json (a smaller run never overwrites a bigger one).
"""
import argparse
import datetime as dt
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--replay", action="store_true")
ap.add_argument("--no-cache", action="store_true")
ap.add_argument("--only", default="")
ap.add_argument("--category", default="")
ap.add_argument("--max-requests", type=int, default=150)
args = ap.parse_args()
MODE = "replay" if args.replay else ("off" if args.no_cache else "use")

if MODE != "replay":
    from dotenv import load_dotenv

    load_dotenv(ROOT / "hello_agent" / ".env")
    if not (os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")):
        sys.exit("No API key found in hello_agent/.env (expected GOOGLE_API_KEY=...). Nothing was sent.")

from agents.chat_agent import MAX_CODE_RUNS, ChatSession
from agents.llm_factory import ROLE_MODELS, make_llm
from agents.resilience import quiet_adk_logs
from evals.chat_eval import FAIL, MINOR, NO_ANSWER, NOT_RUN, PASS, build_questions, run_eval, save_report, summarize
from tools.cleaner import clean_dataframe
from tools.loader import load_csv_bytes
from tools.profiler import profile_dataframe
from ui.dashboard_tab import usage_of

quiet_adk_logs()
df, _ = load_csv_bytes((ROOT / "sample_data" / "sales_sample.csv").read_bytes())
df, _ = clean_dataframe(df)
profile = profile_dataframe(df)
questions, skipped = build_questions(df)
if args.only:
    wanted = {int(x) for x in args.only.split(",") if x.strip().isdigit()}
    questions = [q for q in questions if q.id in wanted]
if args.category:
    questions = [q for q in questions if q.category == args.category]
if not questions:
    sys.exit("No questions match. Nothing was sent.")

model = make_llm("chat", mode=MODE)
chat = ChatSession(profile, model, tables=df, max_code_runs=MAX_CODE_RUNS,
                   on_wait=lambda n, seconds, why: print(f"  (server busy, waiting {seconds:.0f}s, try {n}: {why[:80]})", flush=True))
ICON = {PASS: "PASS ", MINOR: "MINOR", FAIL: "WRONG", NO_ANSWER: "NONE ", NOT_RUN: "SKIP "}


def ask(text):
    chat.reset()                                     # every question starts fresh
    return chat.ask(text)


def show(r):
    print(f"{r.id:>3} [{r.category}] {ICON[r.status]} {r.question}", flush=True)
    if r.status in (FAIL, MINOR):
        print(f"      AI said: {r.answer[:200]!r}")
        for why in r.reasons:
            print(f"      - {why}")
    elif r.status in (NO_ANSWER, NOT_RUN):
        print(f"      {r.error[:160]}")


print(f"Asking {len(questions)} question(s) about {len(df):,} rows (mode: {MODE}; request budget {args.max_requests})."
      + (f" {len(skipped)} question(s) skipped: this table cannot answer them." if skipped else ""), flush=True)
results = run_eval(questions, ask, requests_used=lambda: usage_of(model)[0], max_requests=args.max_requests, progress=show)
real, replayed = usage_of(model)
s = summarize(results)
print(f"\nSUMMARY: {s['passed']} of {s['answered']} answered questions fully correct"
      + (f" ({s['strict'] * 100:.0f}%)" if s["strict"] is not None else "")
      + f"; {s['minor']} minor slip(s); {s['failed']} wrong; {s['no_answer']} no answer; {s['not_run']} not run")
print(f"app number check: flagged {s['wrong_flagged']} of {s['failed']} wrong answers; {s['false_alarms']} false alarm(s)")
# Day 18b: how often the number-copy guard rewrote an answer
print(f"number-copy guard: {s['corrected']} answer(s) rewritten, {s['corrected_ok']} correct afterwards")
print(f"self-healed: {s['healed']}. chat models -> {real} real request(s), {replayed} replayed from saved replies")
meta = {"date": dt.date.today().isoformat(), "mode": MODE, "rows": f"{len(df):,}", "chat models": ", ".join(ROLE_MODELS["chat"]),
        "questions": f"{len(questions)} asked, {len(skipped)} skipped", "real requests": real, "replayed": replayed}
path, replaced = save_report(results, skipped, meta, ROOT / "eval_results")
print(f"report: {path.relative_to(ROOT)}" + ("" if replaced else "  (smaller than the saved report, so the saved one was kept)"))
if s["not_run"] or s["no_answer"]:
    print("Some questions were not answered. Run the same command later: answered ones replay for free.")
