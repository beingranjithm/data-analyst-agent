"""Day 22: how accurate is the chat across SEVERAL tables? Questions about the Formula 1 files, true answers from pandas.
Run from the project root (the 14 F1 csv files in one folder):
    python scripts/eval_f1.py ~/Downloads/f1                  (first run: about 40 to 70 Gemini requests; answers are saved)
    python scripts/eval_f1.py ~/Downloads/f1 --replay         (re-use saved replies: NO Gemini requests, no key needed)
    python scripts/eval_f1.py --demo                          (record the PUBLIC DEMO's answers on the small demo set)
    python scripts/eval_f1.py ~/Downloads/f1 --focus results  (analyse this table instead of the default one)
    also: --no-cache  --only 1,7,13  --category traps  --max-requests 80
The AI sees only the text summary of the tables and small calculation results (at most 20 rows), never the tables.
The report is written to eval_results/f1_eval.md and .json.  Use PUBLIC data only (the F1 files are public).
"""
import argparse
import datetime as dt
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("folder", nargs="?", default="", help="folder with the F1 csv files (not needed with --demo)")
ap.add_argument("--demo", action="store_true",
                help="the small demo set (sample_data/f1_demo): asks the demo questions and saves the answers to deploy/demo_cache.json")
ap.add_argument("--focus", default="")
ap.add_argument("--replay", action="store_true")
ap.add_argument("--no-cache", action="store_true")
ap.add_argument("--only", default="")
ap.add_argument("--category", default="")
ap.add_argument("--max-requests", type=int, default=150)
args = ap.parse_args()
MODE = "replay" if args.replay else ("off" if args.no_cache else "use")
if args.demo:
    folder = ROOT / "sample_data" / "f1_demo"
    os.environ.setdefault("DAA_CACHE_PATH", str(ROOT / "deploy" / "demo_cache.json"))    # the file the public demo replays
elif args.folder:
    folder = Path(args.folder).expanduser()
else:
    sys.exit("Give the folder with the F1 files, or use --demo. Nothing was sent.")
files = sorted(folder.glob("*.csv"))
if len(files) < 2:
    sys.exit(f"Expected several .csv files in {folder}. Nothing was sent.")

if MODE != "replay":
    from dotenv import load_dotenv

    load_dotenv(ROOT / "hello_agent" / ".env")
    if not (os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")):
        sys.exit("No API key found in hello_agent/.env (expected GOOGLE_API_KEY=...). Nothing was sent.")

from agents.llm_factory import ROLE_MODELS, make_llm
from agents.multi_chat import chat_session_for, prepare_chat
from agents.resilience import quiet_adk_logs
from evals.chat_eval import FAIL, MINOR, NO_ANSWER, NOT_RUN, PASS, run_eval, save_report, summarize
from evals.f1_eval import build_demo_questions, build_f1_questions, read_raw
from ui.dashboard_tab import usage_of

quiet_adk_logs()
print(f"Reading {len(files)} files ...", flush=True)
try:
    prep = prepare_chat([(f.name, f.read_bytes()) for f in files], args.focus)
except KeyError as e:
    sys.exit(str(e.args[0]))
for filename, message in prep["errors"]:
    print(f"  left out {filename}: {message}")
table, multi, frames, links, active = prep["table"], prep["multi"], prep["frames"], prep["links"], prep["active"]
print(f"{len(links)} links; analysing {active} ({len(table['df']):,} rows, {table['df'].shape[1]} columns)", flush=True)

questions, skipped = (build_demo_questions if args.demo else build_f1_questions)(read_raw(folder))
if args.only:
    wanted = {int(x) for x in args.only.split(",") if x.strip().isdigit()}
    questions = [q for q in questions if q.id in wanted]
if args.category:
    questions = [q for q in questions if q.category == args.category]
if not questions:
    sys.exit("No questions match. Nothing was sent.")

model = make_llm("chat", mode=MODE)
chat = chat_session_for(prep, model,
                   on_wait=lambda n, seconds, why: print(f"  (server busy, waiting {seconds:.0f}s, try {n}: {why[:80]})", flush=True))
ICON = {PASS: "PASS ", MINOR: "MINOR", FAIL: "WRONG", NO_ANSWER: "NONE ", NOT_RUN: "SKIP "}


LAST = {}


def ask(text):
    chat.reset()
    reply = chat.ask(text)
    LAST[text] = list(getattr(reply, "calculations", None) or [])
    return reply


def show(r):
    print(f"{r.id:>3} [{r.category}] {ICON[r.status]} {r.question}", flush=True)
    if r.status in (FAIL, MINOR):
        print(f"      AI said: {r.answer[:200]!r}")
        for why in r.reasons:
            print(f"      - {why}")
        for n, c in enumerate(LAST.get(r.question, []), 1):          # what the AI actually calculated (code only, no data)
            print(f"      code {n} ({'ok' if c.ok else 'failed'}): " + c.code.strip().replace("\n", "\n          "))
            print(f"      result {n}: " + c.result_text[:300].replace("\n", " | "))
    elif r.status in (NO_ANSWER, NOT_RUN):
        print(f"      {r.error[:160]}")


print(f"Asking {len(questions)} question(s) (mode: {MODE}; request budget {args.max_requests})."
      + (f" {len(skipped)} skipped." if skipped else ""), flush=True)
results = run_eval(questions, ask, requests_used=lambda: usage_of(model)[0], max_requests=args.max_requests, progress=show)
real, replayed = usage_of(model)
s = summarize(results)
print(f"\nSUMMARY: {s['passed']} of {s['answered']} answered questions fully correct"
      + (f" ({s['strict'] * 100:.0f}%)" if s["strict"] is not None else "")
      + f"; {s['minor']} minor slip(s); {s['failed']} wrong; {s['no_answer']} no answer; {s['not_run']} not run")
print(f"number-copy guard: {s['corrected']} answer(s) rewritten, {s['corrected_ok']} correct afterwards; self-healed: {s['healed']}")
print(f"chat models -> {real} real request(s), {replayed} replayed from saved replies")
meta = {"date": dt.date.today().isoformat(), "mode": MODE, "tables": len(frames), "analysed": active,
        "chat models": ", ".join(ROLE_MODELS["chat"]), "questions": f"{len(questions)} asked, {len(skipped)} skipped",
        "real requests": real, "replayed": replayed}
path, replaced = save_report(results, skipped, meta, ROOT / "eval_results", name="f1_demo_eval" if args.demo else "f1_eval",
                             title="Chat accuracy across several tables (Formula 1" + (", demo set)" if args.demo else ")"))
print(f"report: {path.relative_to(ROOT)}" + ("" if replaced else "  (smaller than the saved report, so the saved one was kept)"))
