# Putting the app online for free (Streamlit Community Cloud)

This makes a public web address where anyone can upload CSV files and use the dashboard and the chat. The visitor
pastes **their own** Gemini key, so your key and your quota are never used.

## What "public demo mode" changes (it switches on with the setting `DAA_HOSTED = "1"`)
- The visitor's key lives only in their own browser session: not in a file, not in the environment, not in logs.
- The disk cache of AI replies is off (it would keep visitors' questions on the server). "Replay only" and "Ask the AI" remain.
- Smaller limits: 15 MB per file, 30 MB in total, 15 files, and a 512 MB cap for the code sandbox.
- The "Code sandbox" tab (a box for typing any Python) is hidden.
- The AI's code only runs if the sandbox's file-system lock (Linux Landlock) works on the host. The app checks this once.
  If it does not work, the chat answers from the data summary only, and says so. It never runs the code unprotected.

## Steps
1. Make sure the repository is on GitHub and contains no `.env` file or key (`git status`, and search for `AIza`).
2. In the project folder, with the virtual environment on, run `python scripts/make_requirements.py`.
   It writes `requirements.txt` with the packages and versions you tested with. Commit and push it.
3. Go to https://share.streamlit.io and sign in with GitHub. Choose "Create app" (wording may differ slightly).
4. Pick the repository and branch, and set the main file to `app/streamlit_app.py`.
5. Open "Advanced settings". Choose a Python version (the newest offered that is at least 3.12). In **Secrets** add exactly:
   `DAA_HOSTED = "1"`
6. Deploy. The first start takes a few minutes. Open the address, paste your own Gemini key in the sidebar and try the sample data.
7. If the build fails, copy the last lines of the log and send them to Claude.

## Things to check after it is live (these could not be tested before deploying)
- The sidebar shows "Your Gemini key" and the blue "Public demo" banner.
- Result of the first real test on Streamlit Community Cloud: the host does NOT allow the file-system lock, so the Chat tab says
  "Live calculations are switched off on this host". The saved demo answers still work (recorded code runs); a visitor's own key gets summary-only chat
  and fixed-rule dashboards. A host with the lock would allow live calculations.
- Uploading the Olist files stays inside the limits (or shows the friendly size message).
- Nothing about your own key appears anywhere. Never put your key in the Secrets box.

## Honest limits
- The code sandbox is a locked-down process, not a container. Strong enough for a demo with fail-closed checks, but a determined
  attacker is a risk you accept by making this public.
- Free hosts have little memory, so big files can be slow or fail. Free apps may go to sleep when nobody uses them.
- Visitors' free Gemini keys may let Google use their prompts, which is why the banner says: no confidential data.

## Taking it down
In the Streamlit dashboard, open the app's menu and delete the app. Making the GitHub repository private also stops new deploys.

## Recording the demo answers (once, and again whenever the AI's prompts or the data change)
The public demo opens on a small Olist e-commerce set (orders from November 2017) and answers the listed
questions from saved replies, so visitors need no key. The saved replies are a file in the repository:
`deploy/demo_cache.json`.
1. Build the demo set once: `python scripts/build_demo_olist.py "/path/to/Olist Data Set"` (writes `sample_data/olist_demo/`).
   Check the dataset's licence first and keep its attribution in the README — Olist's licence (CC BY-NC-SA 4.0) is
   non-commercial and requires attribution, unlike a public-domain dataset.
2. Record the chat answers (needs your key in `hello_agent/.env`):
   `python scripts/eval_olist.py --demo`. It also scores them against answers computed with plain pandas. Fix or re-run until you are happy.
3. Record one dashboard: `DAA_CACHE_PATH=deploy/demo_cache.json streamlit run app/streamlit_app.py`, choose "Olist e-commerce
   demo set (Nov 2017)", open the AI dashboard tab, leave all settings as they are, press Plan, then approve. (Do not change "Maximum charts".)
4. Commit `sample_data/olist_demo/` and `deploy/demo_cache.json`. A reply is only found again when the request is exactly the same, so
   after any change to the prompts, the code that builds them, or the data, record again.
5. Check: `python scripts/eval_olist.py --demo --replay` must answer every question (all PASS) with 0 real requests.

## Re-recording the demo video
The 2-minute video on the project page (`docs/assets/demo.mp4`) is recorded from the app itself, replaying the saved
demo answers, so it needs no key. Re-record it after any visible change to the app or the demo data:
1. Start the app in public-demo mode: `DAA_HOSTED=1 streamlit run app/streamlit_app.py --server.headless true`
2. In a second terminal (once: `pip install playwright` and `playwright install chromium`; ffmpeg must be installed):
   `python scripts/record_demo.py`
3. Watch `docs/assets/demo.mp4`, then commit it. The captions and timings are in the script.
