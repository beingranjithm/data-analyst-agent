"""Day 26: record the 2-minute demo video (docs/assets/demo.mp4) from the real app.

The app must be running in public-demo mode, so the video replays the saved answers (no key, no AI requests):
    DAA_HOSTED=1 streamlit run app/streamlit_app.py --server.headless true
Then, in a second terminal (needs: pip install playwright && playwright install chromium; and ffmpeg):
    python scripts/record_demo.py                      # writes docs/assets/demo.mp4 (about 2 minutes)
Re-record after any visible change to the app, the demo data or the saved replies.

Storyboard (captions are drawn on the page, so they are part of the recording):
  0  title card
  1  the demo set opens: 8 files linked without AI, rows unchanged
  2  Overview: the text summary is all the AI ever reads
  3  AI dashboard: replay mode -> plan -> review/approve -> build
  4  the finished dashboard: KPI changes, insight titles, notes on left-out rows
  5  Chat: a question that needs two tables -> answer -> "Show the calculation"
  6  a trap question about data that doesn't exist
  7  end card
"""
import argparse
import base64
import os
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parent.parent
URL = os.getenv("DEMO_URL", "http://localhost:8501")
OUT = Path(tempfile.mkdtemp(prefix="daa_video_"))
CHROME = os.getenv("CHROME_PATH")            # optional: a system Chrome instead of Playwright's Chromium
W, H = 1280, 720

OVERLAY_JS = """
() => {
  if (document.getElementById('demo-cap')) return;
  const st = document.createElement('style');
  st.textContent = `
    [data-testid="stSidebar"], [data-testid="stToolbar"], [data-testid="stHeader"] { display: none !important; }
    #demo-cap { position: fixed; left: 50%; bottom: 26px; transform: translateX(-50%); z-index: 99999;
      max-width: 1040px; padding: 12px 22px; border-radius: 10px; background: rgba(18,22,28,.88);
      color: #fff; font: 500 21px/1.35 "Source Sans Pro", system-ui, sans-serif; text-align: center;
      box-shadow: 0 10px 30px rgba(0,0,0,.25); transition: opacity .35s ease; opacity: 0; pointer-events: none; }
    #demo-cap b { color: #ffb4a3; font-weight: 600; }
    #demo-cur { position: fixed; z-index: 100000; width: 22px; height: 22px; margin: -11px 0 0 -11px;
      border-radius: 50%; background: rgba(194,59,42,.30); border: 2px solid #c23b2a; pointer-events: none;
      transition: transform .12s ease; left: -50px; top: -50px; }
    #demo-cur.down { transform: scale(.7); background: rgba(194,59,42,.6); }`;
  document.head.appendChild(st);
  const cap = document.createElement('div'); cap.id = 'demo-cap'; document.body.appendChild(cap);
  const cur = document.createElement('div'); cur.id = 'demo-cur'; document.body.appendChild(cur);
  document.addEventListener('mousemove', e => { cur.style.left = e.clientX + 'px'; cur.style.top = e.clientY + 'px'; }, true);
  document.addEventListener('mousedown', () => cur.classList.add('down'), true);
  document.addEventListener('mouseup', () => cur.classList.remove('down'), true);
}
"""

CARD_JS = """(body) => {
  let c = document.getElementById('demo-card');
  if (!c) {
    const st = document.createElement('style');
    st.textContent = `#demo-card{position:fixed;inset:0;z-index:100001;background:#12161c;color:#e8eae3;
      font-family:"Source Sans Pro",system-ui,sans-serif;display:flex;flex-direction:column;justify-content:center;
      padding:0 110px;transition:opacity .6s ease;opacity:0}
      #demo-card .eyebrow{color:#f3875f;font-size:22px;margin:0 0 18px;font-weight:600}
      #demo-card h1{font-size:58px;line-height:1.08;margin:0 0 22px;font-weight:700;letter-spacing:-.02em;max-width:18ch;color:#fff}
      #demo-card p{font-size:24px;color:#aab3b8;margin:6px 0;max-width:46ch;line-height:1.45}
      #demo-card .mono{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;color:#dce5de;font-size:21px}
      #demo-card .row{margin-top:30px;display:flex;gap:46px}
      #demo-card .n{font-family:ui-monospace,monospace;font-size:40px;color:#fff}
      #demo-card .l{font-size:17px;color:#99a2a7;max-width:15ch}`;
    document.head.appendChild(st);
    c = document.createElement('div'); c.id = 'demo-card'; document.body.appendChild(c);
  }
  if (body === null) { c.style.opacity = 0; setTimeout(() => c.style.display = 'none', 650); return; }
  c.innerHTML = body; c.style.display = 'flex'; requestAnimationFrame(() => c.style.opacity = 1);
}"""

TITLE_BODY = """
  <p class="eyebrow">Built with Google ADK and Gemini</p>
  <h1>Agentic Data Analyst: a 2-minute tour</h1>
  <p>Upload CSV files, approve an AI-planned dashboard, and ask questions. Every number comes with the code behind it.</p>
  <p class="mono" style="margin-top:26px">Demo data: Olist e-commerce orders, November 2017</p>"""

END_BODY = """
  <p class="eyebrow">Try it yourself: free, no sign-up</p>
  <h1>Ask your data a question. See exactly how it answered.</h1>
  <div class="row">
    <div><div class="n">13/14</div><div class="l">questions right on the first real eval run</div></div>
    <div><div class="n">8</div><div class="l">CSV tables linked with no AI</div></div>
    <div><div class="n">889</div><div class="l">automated tests</div></div>
  </div>
  <p class="mono" style="margin-top:34px">analystiq-agent.streamlit.app</p>
  <p class="mono">github.com/beingranjithm/data-analyst-agent</p>"""


class Demo:
    def __init__(self, page):
        self.page = page
        self.t0 = time.time()
        self.log = []

    def mark(self, label):
        self.log.append((round(time.time() - self.t0, 1), label))

    def wait(self, ms):
        self.page.wait_for_timeout(ms)

    def idle(self, ms=800):
        self.wait(400)
        try:
            self.page.wait_for_selector('[data-testid="stStatusWidget"]', state="detached", timeout=180000)
        except Exception:
            pass
        self.wait(ms)

    def overlay(self):
        self.page.evaluate(OVERLAY_JS)

    def caption(self, html, hold=0):
        self.page.evaluate("""(h) => { const c = document.getElementById('demo-cap');
            if (!h) { c.style.opacity = 0; return; } c.innerHTML = h; c.style.opacity = 1; }""", html)
        self.mark(f"caption: {html}")
        if hold:
            self.wait(hold)

    def move_to(self, locator, steps=25):
        box = locator.bounding_box()
        if box:
            self.page.mouse.move(box["x"] + box["width"] / 2, box["y"] + box["height"] / 2, steps=steps)
            self.wait(250)

    def click(self, locator):
        locator.scroll_into_view_if_needed()
        self.wait(200)
        self.move_to(locator)
        locator.click()

    def scroll_to(self, locator, offset=60, duration=1400):
        """Smoothly scroll Streamlit's main scroller so `locator` lands `offset` px from the top."""
        locator.evaluate("""(el, a) => { const [off, dur] = a; let s = el.parentElement;
            while (s && !(s.scrollHeight > s.clientHeight && getComputedStyle(s).overflowY !== 'visible')) s = s.parentElement;
            s = s || document.scrollingElement;
            const target = el.getBoundingClientRect().top - s.getBoundingClientRect().top + s.scrollTop - off;
            const start = s.scrollTop, t0 = performance.now();
            const step = now => { const k = Math.min(1, (now - t0) / dur), e = k < .5 ? 2*k*k : 1 - Math.pow(-2*k+2, 2)/2;
              s.scrollTop = start + (target - start) * e; if (k < 1) requestAnimationFrame(step); };
            requestAnimationFrame(step); }""", [offset, duration])
        self.wait(duration + 250)


def record():
    frames_dir = OUT / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    for f in frames_dir.glob("*.jpg"):
        f.unlink()
    frames = []
    with sync_playwright() as p:
        b = p.chromium.launch(executable_path=CHROME, args=["--no-sandbox"]) if CHROME else p.chromium.launch()
        ctx = b.new_context(viewport={"width": W, "height": H})
        page = ctx.new_page()
        cdp = ctx.new_cdp_session(page)

        def on_frame(ev):                      # Chrome sends a frame whenever the screen changes
            path = frames_dir / f"{len(frames):06d}.jpg"
            path.write_bytes(base64.b64decode(ev["data"]))
            frames.append((str(path), ev["metadata"]["timestamp"]))
            cdp.send("Page.screencastFrameAck", {"sessionId": ev["sessionId"]})
        cdp.on("Page.screencastFrame", on_frame)
        d = Demo(page)

        # load the app BEFORE recording, so the video never shows the loading screen
        page.goto(URL)
        page.wait_for_selector("text=tables and", timeout=120000)
        d.overlay()
        d.idle(800)
        links = page.get_by_text("tables and", exact=False).first
        d.scroll_to(links, 20, 300)
        page.evaluate(CARD_JS, TITLE_BODY)
        d.wait(900)

        # 0 ---- title card (recording starts here)
        cdp.send("Page.startScreencast", {"format": "jpeg", "quality": 90, "maxWidth": W, "maxHeight": H})
        d.t0 = time.time()
        d.mark("title")
        d.wait(4600)
        page.evaluate(CARD_JS, None)
        d.wait(700)

        # 1 ---- the demo set
        d.caption("The demo opens on <b>8 CSV files</b> of real e-commerce orders (Olist, November 2017).", 3800)
        d.caption("The app finds how the files link together by <b>comparing values, with no AI</b>.", 1600)
        d.scroll_to(page.get_by_text("order_items.order_id", exact=False).first, 220, 1500)
        d.wait(1500)
        d.scroll_to(page.get_by_text("rows (unchanged)", exact=False).first, 260, 1300)
        d.caption("Lookup columns are joined on, and the row count is checked: <b>8,665 rows, unchanged</b>.", 3800)
        d.caption("")

        # 2 ---- Overview: what the AI reads
        tab = page.get_by_role("tab", name="Overview")
        d.click(tab)
        d.idle(600)
        exp = page.get_by_text("What the AI agents will read", exact=False).first
        d.scroll_to(exp, 160, 1600)
        d.click(exp)
        d.idle(600)
        d.caption("Each table is cleaned and profiled. <b>The AI only ever reads this text summary</b>, never the rows.", 1000)
        d.scroll_to(exp, 40, 1600)
        d.wait(3000)

        # 3 ---- AI dashboard: plan -> approve -> build
        d.caption("")
        tabs = page.get_by_role("tab", name="AI dashboard")
        d.scroll_to(tabs, 120, 1200)
        d.click(tabs)
        d.idle(600)
        sel = page.get_by_role("tabpanel", name="AI dashboard").get_by_test_id("stSelectbox").first
        d.caption("This public demo <b>replays saved AI answers</b>, so no API key is needed.", 600)
        d.click(sel)
        d.wait(700)
        opt = page.get_by_role("option", name="Replay only: no AI requests at all")
        d.click(opt)
        d.idle(900)
        plan = page.get_by_role("button", name="Plan a dashboard")
        d.caption("")
        d.click(plan)
        page.wait_for_selector("text=Review the plan", timeout=180000)
        d.idle(500)
        d.scroll_to(page.get_by_text("Review the plan").first, 30, 1200)
        d.caption("A <b>planner agent</b> proposes KPIs and charts, each with a reason.", 3400)
        d.scroll_to(page.get_by_text("Charts to build", exact=False).first, 40, 1500)
        d.caption("<b>You approve it first</b>: untick any chart you don't want, then build.", 3000)
        build = page.get_by_role("button", name="Build the dashboard")
        d.click(build)
        d.caption("A <b>chart agent</b> writes Plotly code for each chart, and a locked-down sandbox runs it.", 0)
        page.wait_for_selector("text=Download as a web page", timeout=300000)
        d.idle(1200)

        # 4 ---- the finished dashboard
        title = page.get_by_text("E-Commerce Sales and Logistics Dashboard").first
        d.scroll_to(title, 30, 1400)
        d.caption("The finished dashboard. <b>KPI cards compare the last 14 days with the 14 before</b>.", 4000)
        d.scroll_to(page.get_by_text("Total price peaked", exact=False).first, 70, 1600)
        d.caption("Chart titles are <b>facts computed from the data</b>: here, the Black Friday peak.", 4000)
        d.scroll_to(page.get_by_text("accounts for 98% of rows", exact=False).first, 90, 1800)
        d.caption("Every chart says why it's there, and <b>how it was made</b> is one click away.", 3200)
        d.scroll_to(page.get_by_text("What changed", exact=False).first, 240, 1800)
        d.caption("<b>What changed</b> cards sum up the biggest moves.", 2800)

        # 5 ---- Chat with the calculation
        d.caption("")
        chat_tab = page.get_by_role("tab", name="Chat")
        d.scroll_to(chat_tab, 120, 1400)
        d.click(chat_tab)
        d.idle(800)
        q = page.get_by_role("button", name="credit card", exact=False).first
        d.scroll_to(q, 260, 1200)
        d.caption("Now ask a question in plain English. This one needs <b>two tables</b>.", 1600)
        d.click(q)
        d.idle(1400)
        calc = page.locator('[data-testid="stExpander"] summary').filter(has_text="Show the calculation").last
        d.scroll_to(page.get_by_text("How many orders were paid by credit card", exact=False).last, 60, 1400)
        d.caption("The answer: <b>919 orders</b>.", 2200)
        d.click(calc)
        d.idle(700)
        d.scroll_to(page.get_by_text("There are 919 orders", exact=False).last, 20, 900)
        d.caption("Every number comes with <b>the pandas code that produced it</b>, and the result the AI was shown.", 4600)

        # 6 ---- a trap question
        reset = page.get_by_role("button", name="Start a new conversation")
        d.caption("")
        d.click(reset)
        d.idle(1000)
        trap = page.get_by_role("button", name="average customer age", exact=False).first
        d.scroll_to(trap, 260, 1200)
        d.caption("And when the data <b>can't</b> answer?", 1400)
        d.click(trap)
        d.idle(1400)
        d.scroll_to(page.get_by_text("What is the average customer age?", exact=False).last, 60, 1200)
        d.caption("It says so, <b>instead of inventing a number</b>.", 4000)

        # 7 ---- end card
        d.caption("")
        page.evaluate(CARD_JS, END_BODY)
        d.mark("end card")
        d.wait(5600)
        d.mark("end")

        end_ts = time.time()
        cdp.send("Page.stopScreencast")
        ctx.close()
        b.close()
    # each frame lasts until the next one arrives; the last one until the end of the recording
    lines = []
    for i, (path, ts) in enumerate(frames):
        nxt = frames[i + 1][1] if i + 1 < len(frames) else end_ts
        lines += [f"file '{path}'", f"duration {max(0.001, nxt - ts):.4f}"]
    lines.append(f"file '{frames[-1][0]}'")
    concat = OUT / "frames.txt"
    concat.write_text("\n".join(lines) + "\n")
    raw = OUT / "raw.mp4"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "concat", "-safe", "0", "-i", str(concat),
                    "-vf", f"fps=30,scale={W}:{H}:force_original_aspect_ratio=decrease,pad={W}:{H}:(ow-iw)/2:(oh-ih)/2",
                    "-c:v", "libx264", "-preset", "veryfast", "-crf", "18", "-pix_fmt", "yuv420p", str(raw)], check=True)
    return raw, d.log, frames[0][1]


def encode(raw: Path, out: Path):
    """The small web version: H.264, CRF 26, no audio, starts playing before it has fully downloaded."""
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(raw), "-c:v", "libx264", "-preset", "veryfast", "-crf", "26",
                    "-pix_fmt", "yuv420p", "-movflags", "+faststart", "-an", str(out)], check=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(ROOT / "docs" / "assets" / "demo.mp4"))
    args = ap.parse_args()
    raw, log, _ = record()
    encode(raw, Path(args.out))
    shutil.rmtree(OUT, ignore_errors=True)
    for t, label in log:
        print(f"{t:6.1f}s  {label}")
    print("written:", args.out)
