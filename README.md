# Agentic Data Analyst

Upload one or more CSV files and get an **AI-planned dashboard** you approve chart by chart, plus a **chat** that answers
questions about your data and shows the calculation behind every number.

**Live demo (free, no sign-up): https://analystiq-agent.streamlit.app**
It opens on a small Olist e-commerce data set (Brazilian orders, Oct-Nov 2017). The saved demo questions and the demo
dashboard work without any key.
On the free host, a visitor's own free [Gemini API key](https://aistudio.google.com) lets the chat answer questions about an
uploaded file from its data summary. Live AI dashboards and live calculations on your own data need a host that supports the
sandbox's file-system lock, so for those run the app yourself (see "Run it on your own computer").

Built with Google ADK 2.x (agents), Gemini, pandas, Plotly and Streamlit.

---

## What it does

- **AI dashboard with a human in the loop.** A planner agent proposes charts and KPIs from a text summary of your data.
  You drop what you don't want and approve. A chart agent then writes the code for each chart. If the code fails, the agent
  gets a cleaned-up error message and fixes it itself (up to 3 attempts), then a rule-based chart builder takes over, so a
  dashboard always comes out. The result can be downloaded as a single web page.
- **Chat that shows its work.** The chat agent answers from the data summary. For questions that need real numbers it writes
  pandas code, runs it in a sandbox and answers from the result. Every answer has a "Show the calculation" box with the code and
  the result. Numbers in an answer that appear in neither the summary nor a calculation get a visible warning.
- **Several CSV files at once.** Relationships between files are found without AI (which column of one file points to
  which row of another). Linked lookup tables are joined onto the main table with left joins, so the number of rows never
  changes. The chat can combine tables, and a guard blocks the classic mistake of merging two large event tables and then summing.
- **Everything else you would expect:** automatic data cleaning and profiling, filters in the sidebar that apply to the
  dashboard and the chat, a chart gallery, a "Build a chart" tab, and a Code sandbox tab (local runs only).

## How it works

```mermaid
flowchart LR
    U[Upload CSV files] --> L[Loader and cleaner]
    L --> P[Profiler: text summary of the data]
    L --> R[Relationship finder - no AI]
    P --> PL[Planner agent proposes a dashboard]
    PL --> A[You approve the plan]
    A --> CA[Chart agent writes code]
    CA --> S[Sandbox runs the code on the data]
    S -- "error, cleaned of data values" --> CA
    S --> D[Dashboard and HTML download]
    P --> C[Chat agent]
    R --> C
    C -- "pandas code" --> S
    S -- "small result, at most 20 rows" --> C
```

The agents see a **text summary** of the data (column types, ranges, averages, categories), not the table. Code written by
the AI runs on the real data in a sandbox, and the AI only ever sees a small, size-capped result of that run.

**Built for free-tier limits.** Requests are paced, replies are cached so the same request is never sent twice, and a chain
of three Gemini models takes over when one model's daily quota is used up. A "Replay only" mode answers entirely from saved
replies with no requests at all, which is how the public demo works without a key.

## Evaluation (with the limits stated)

Accuracy is measured against answers computed independently with plain pandas on the same data.

| Test | Questions | Result |
|---|---|---|
| Olist e-commerce demo set (Oct-Nov 2017), several tables per question, answered live by Gemini | 14 | **13/14** on the first real run |

How to read these numbers honestly:
- **The one miss is a real finding, not noise.** For "which customer state placed the most orders", the AI answered
  from the flattened order-items table, which only has a row for orders that have at least one item — it undercounted
  the states that way. The raw `orders` table it also had access to would have given the right answer; it chose the
  flattened one anyway. The wrong cached reply was removed rather than patched, so that one question has no saved
  answer on the public demo right now and needs a Gemini key to answer it live until it is re-recorded.
- **The checker is automatic.** It compares numbers and names in the answer with the expected ones, so it has limits.
  It can mark a correct answer wrong if it is phrased unusually, and it cannot judge explanations.
- **The demo set is small and its questions are the ones the demo shows.** 13/14 is one real run against one small,
  single-month dataset, not a broad accuracy claim.

Run them yourself (needs your Gemini key, see below):
```
python scripts/eval_olist.py "<folder with the Olist csv files>"          # the 14-question test on the full dataset
python scripts/eval_olist.py --demo                                       # the same 14 questions on the small demo set
python scripts/eval_chat.py                                               # a 30-question test on a single table
```
Reports are written to `eval_results/`.

## Safety and privacy

- **Your data stays in memory for the session.** The hosted demo does not store uploads. It says so in a banner and asks
  you not to upload confidential data. On the free Gemini tier Google may use prompts to improve its products.
- **Bring your own key.** On the public site your key lives only in your browser session. It is not saved, logged, shared
  with other visitors, or kept in the environment, and "Forget my key" erases it.
- **AI-written code runs in a sandbox:** a separate process, a cleaned environment (no keys), no network, no `open()`, an
  allow-list of imports, a static check of the code before it runs, time and memory limits, and a file-system lock
  (Linux Landlock) where the host supports it.
- **This is process-level isolation, not a container.** That is strong enough for a demo with fail-closed checks, but it is
  not the same as running each program in a fresh container.
- **The free host has no file-system lock.** On Streamlit Community Cloud the lock is not available, so the app refuses to
  run any code the AI writes there. The only code that runs on the hosted demo is the saved demo code that was recorded and
  checked on my machine, word for word. With your own key on the hosted site the chat answers from the data summary only, and
  a live "Use AI" dashboard falls back to fixed rules without sending any AI request. For live AI dashboards and live
  calculations, run the app yourself (below).

## Run it on your own computer

You need Python 3.12 or newer and a free Gemini API key from [Google AI Studio](https://aistudio.google.com).

```bash
git clone https://github.com/beingranjithm/data-analyst-agent.git
cd data-analyst-agent
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -r requirements.txt
mkdir -p hello_agent
echo "GOOGLE_API_KEY=your-key-here" > hello_agent/.env      # keep this file out of git
streamlit run app/streamlit_app.py
```

Run the tests (no real AI calls, fake models only):
```bash
pip install pytest
python -m pytest -q
```

To put your own copy online for free, follow [DEPLOY.md](DEPLOY.md).

## Project layout

```
app/streamlit_app.py     the web app (tabs, sidebar, hosted mode)
agents/                  planner, chart agent, self-healing loop, chat agent, multi-table chat,
                         model cache, pacing and the three-model fallback
tools/                   loading, cleaning, profiling, relationship finder, filters, sandbox, hosting rules
ui/                      the dashboard, chat and key panels
evals/ and scripts/      the accuracy tests and helper scripts
tests/                   automated tests
deploy/demo_cache.json   the saved demo answers
sample_data/             a small sales file and the Olist e-commerce demo set
```

## How it was tested

- **About 780 automated tests** run with fake models and a fake sandbox, so they are fast, free and repeatable.
- **Mutation checks:** for new features I deliberately broke the code in small ways and confirmed the tests noticed, then
  added tests for the ones that slipped through.
- **Real Gemini runs happen only in the evaluations** above and on my own computer, never in the automated tests.
- The sandbox's real behaviour on the free host was checked by deploying and trying it, not by a unit test.

## Known limits

- The 13/14 score is one run, not blind, and not a broad accuracy claim (see above).
- The "do not merge two event tables" guard is a rule of thumb on the code text. It stops the common mistake, not every mistake.
- Very wide tables get fewer filters, and big files are slow or refused on the free host (15 MB per file, 30 MB in total).
- The free host goes to sleep when nobody uses it, so the first visit can take a short while.
- Excel files with several sheets are not supported yet. CSV only.

## Data credit

The Olist demo set is a subset (orders placed in October-November 2017, plus the rows they point to) of the
[Brazilian E-Commerce Public Dataset by Olist](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce) on
Kaggle, listed there as **CC BY-NC-SA 4.0** (attribution, non-commercial, share-alike). See
[DATA_LICENSE.md](DATA_LICENSE.md). This project is free and non-commercial; the data's licence does not allow
commercial use.
