"""The "AI dashboard" tab (Day 13): plan with AI -> you review and approve -> build -> view/download.

A small state machine kept in st.session_state (Streamlit re-runs the whole script on every click):
    idle -> review -> done | cancelled        and  error (from any step)
The ApprovalSession (agents/approval_workflow.py) lives in st.session_state; progress messages from the
planner and the chart builder are streamed into a st.status box while the blocking calls run.
Do not click other widgets while "Planning" or "Building" is running: Streamlit would restart the script.
"""
import os
from pathlib import Path
from typing import Callable, Optional

import streamlit as st

from agents.approval_workflow import ApprovalDecision
from tools.dashboard_html import render_dashboard_html

ROOT = Path(__file__).resolve().parent.parent
CACHE_MODES = {
    "Smart cache: re-use saved AI replies": "use",
    "Replay only: no AI requests at all": "replay",
    "Ask the AI again: ignore saved replies": "off",
}
# Streamlit forgets a widget's state while the widget is not drawn, so the settings are ALSO kept in these
# ordinary keys (and survive both "Start over" and a new dataset)
SETTING_KEYS = ("ai_saved_use_ai", "ai_saved_mode", "ai_saved_max")
PRIVACY_NOTE = ("🔒 The AI reads a summary of your data (column names, types, statistics and category names), "
                "never your rows. Google's free tier may use prompts to improve its products, so use sample or "
                "public data only.")


# ---------------------------------------------------------------- small helpers (unit tested)
def format_kpi(value) -> str:
    if value is None:
        return "n/a"
    if abs(value) >= 1000 or float(value).is_integer():
        return f"{value:,.0f}"
    return f"{value:,.2f}"


def usage_of(llm) -> tuple[int, int]:
    """(real requests, replayed from cache) for a CachingLlm, a FallbackLlm chain, or None."""
    if llm is None:
        return 0, 0
    members = getattr(llm, "chain", None) or [llm]
    return (sum(getattr(m, "real_calls", 0) for m in members), sum(getattr(m, "hits", 0) for m in members))


def _safe(text) -> str:
    from ui.hosted_ui import safe_text
    return safe_text(text)


def cache_modes() -> dict:
    """The reply settings the visitor may choose. On the public host the disk cache ("use") is off: it would keep
    visitors' questions and data summaries on the server."""
    from tools.hosting import is_hosted
    return {k: v for k, v in CACHE_MODES.items() if v != "use"} if is_hosted() else dict(CACHE_MODES)


def api_key_available() -> bool:
    from tools.hosting import is_hosted
    if is_hosted():                                   # public host: only the visitor's own key counts, never the server's
        from ui.hosted_ui import user_key
        return bool(user_key())
    try:
        from dotenv import load_dotenv
        load_dotenv(ROOT / "hello_agent" / ".env")
    except ImportError:   # pragma: no cover
        pass
    return bool(os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY"))


def default_session_factory(df, profile, settings, progress):
    """Builds the real models (planner / chart chains with cache and pacing) and an ApprovalSession."""
    from agents.approval_workflow import ApprovalSession
    from agents.llm_factory import make_llm
    from ui.hosted_ui import user_key
    use_ai, mode = settings["use_ai"], settings["cache_mode"]
    key = user_key() or None                          # the visitor's own key (public host); None = normal local setup
    planner_llm = make_llm("planner", mode=mode, api_key=key) if use_ai else None
    chart_llm = make_llm("chart", mode=mode, api_key=key) if use_ai else None
    session = ApprovalSession(df, profile, use_ai=use_ai, max_charts=settings["max_charts"], progress=progress,
                              planner_model=planner_llm, chart_model=chart_llm)
    session.models = (planner_llm, chart_llm)
    return session


def _status_writer(status, bar=None):
    def write(stage, done, total, message):
        status.update(label=message)
        status.write(message)
        if bar is not None and stage in ("chart", "done") and total:
            bar.progress(min(1.0, done / total), text=message)
    return write


def _clear_run():
    ss = st.session_state
    for key in ("ai_session", "ai_sink", "ai_dashboard", "ai_error"):
        ss.pop(key, None)
    ss["ai_stage"] = "idle"


# ---------------------------------------------------------------- the stages
def _idle(df, profile, session_factory):
    ss = st.session_state
    st.markdown("The AI **proposes** a dashboard. You **approve it** (drop any chart you do not want) "
                "before anything is built.")
    c1, c2, c3 = st.columns([1, 3, 1])
    labels = list(cache_modes())
    use_ai = c1.checkbox("Use AI", value=ss.get("ai_saved_use_ai", True), key="ai_use_ai",
                         help="Off = a dashboard made by fixed rules, with no AI and no requests.")
    mode_label = c2.selectbox("AI replies", labels, index=labels.index(ss.get("ai_saved_mode", labels[0])),
                              key="ai_cache_mode", disabled=not use_ai)
    max_charts = c3.number_input("Maximum charts", 1, 10, int(ss.get("ai_saved_max", 6)), key="ai_max_charts")
    ss["ai_saved_use_ai"], ss["ai_saved_mode"], ss["ai_saved_max"] = use_ai, mode_label, int(max_charts)
    mode = cache_modes()[mode_label]
    from tools import hosting                                  # Day 23c: looked up here so a test can replace it
    no_lock = hosting.is_hosted() and use_ai and mode != "replay" and not hosting.isolation_ok()[0]
    if no_lock:                                                # AI chart code cannot run here: do not waste the visitor's requests
        st.info("The AI's chart code cannot run on this host (the sandbox's file-system lock is not available), so a "
                "live AI run is switched off here and the dashboard is made by fixed rules. 'Replay only' shows the "
                "saved demo dashboard.")
        use_ai = False
    needs_key = use_ai and mode != "replay"
    if needs_key:
        st.caption(PRIVACY_NOTE)
    blocked = False
    if session_factory is None and needs_key and not api_key_available():
        from tools.hosting import is_hosted
        st.error("Paste your own Gemini key in the sidebar, or choose 'Replay only' or untick 'Use AI'." if is_hosted() else
                 "No Gemini API key found. Put GOOGLE_API_KEY=... in hello_agent/.env, "
                 "or choose 'Replay only' or untick 'Use AI'.")
        blocked = True

    if st.button("✨ Plan a dashboard", type="primary", key="ai_plan", disabled=blocked):
        settings = {"use_ai": use_ai, "cache_mode": mode, "max_charts": int(max_charts)}
        ss["ai_run_id"] = ss.get("ai_run_id", 0) + 1
        sink = {"fn": None}
        progress = lambda stage, done, total, message: sink["fn"] and sink["fn"](stage, done, total, message)  # noqa: E731
        error, session = None, None
        try:
            session = (session_factory or default_session_factory)(df, profile, settings, progress)
            with st.status("Planning the dashboard ...", expanded=True) as status:
                sink["fn"] = _status_writer(status)
                try:
                    session.start()
                finally:
                    sink["fn"] = None
                status.update(label="Plan ready", state="complete")
        except Exception as e:
            error = f"Could not plan the dashboard: {type(e).__name__}: {e}"
        if error:
            ss["ai_error"], ss["ai_stage"] = error, "error"
        else:
            ss["ai_session"], ss["ai_sink"], ss["ai_stage"] = session, sink, "review"
        st.rerun()


def _review():
    ss = st.session_state
    session, sink, run = ss.get("ai_session"), ss.get("ai_sink"), ss.get("ai_run_id", 0)
    if session is None or session.proposal is None:
        _clear_run()
        st.rerun()
    proposal = session.proposal
    st.subheader("Review the plan")
    st.caption(f"How it was planned: {proposal.mode}")
    for note in proposal.notes:
        st.info(note)
    if proposal.plan.summary:
        st.write(proposal.plan.summary)
    if proposal.plan.kpis:
        st.write("**Numbers shown at the top:** " + "; ".join(
            f"{k.label} ({k.agg} of {k.measure})" for k in proposal.plan.kpis))
    title = st.text_input("Dashboard title", value=proposal.plan.title, key=f"ai_title_{run}")
    st.markdown("**Charts to build** (untick the ones you do not want)")
    keep = []
    for i, c in enumerate(proposal.plan.charts, 1):
        keep.append(st.checkbox(f"{i}. [{c.type}] {c.title}", value=True, key=f"ai_keep_{run}_{i}"))
        if c.why:
            st.caption(c.why)
    left, right = st.columns(2)
    build = left.button("✅ Build the dashboard", type="primary", key="ai_build")
    cancel = right.button("Cancel", key="ai_cancel")

    result, error = None, None
    if cancel:
        try:
            session.decide(ApprovalDecision(approve=False))
        except Exception as e:
            error = f"{type(e).__name__}: {e}"
        if error:
            ss["ai_error"], ss["ai_stage"] = error, "error"
        else:
            ss["ai_stage"] = "cancelled"
        st.rerun()
    if build:
        if not any(keep):
            st.error("Select at least one chart, or press Cancel.")
            return
        new_title = title.strip()
        decision = ApprovalDecision(drop=[i for i, k in enumerate(keep, 1) if not k],
                                    title=None if new_title == proposal.plan.title else new_title)
        try:
            with st.status("Building the dashboard ...", expanded=True) as status:
                bar = st.progress(0.0)
                sink["fn"] = _status_writer(status, bar)
                try:
                    result = session.decide(decision)
                finally:
                    sink["fn"] = None
                status.update(label="Dashboard ready", state="complete")
        except ValueError as e:                      # a decision that does not fit: show it, keep waiting
            st.error(_safe(str(e)))
            return
        except Exception as e:
            error = f"Could not build the dashboard: {type(e).__name__}: {e}"
        if error:
            ss["ai_error"], ss["ai_stage"] = error, "error"
        else:
            ss["ai_dashboard"], ss["ai_stage"] = result, "done"
        st.rerun()


def _done():
    ss = st.session_state
    d, session = ss.get("ai_dashboard"), ss.get("ai_session")
    if d is None:
        _clear_run()
        st.rerun()
    st.subheader(d.title)
    if d.summary:
        st.write(d.summary)
    st.caption(f"{d.rows:,} rows · planned: {d.mode}")
    for note in d.notes:
        st.info(note)
    if d.kpis:
        for col, k in zip(st.columns(len(d.kpis)), d.kpis):
            exact = "" if k.value is None else f"Exact: {k.value:,.2f}"
            col.metric(k.label, format_kpi(k.value), help=" · ".join(x for x in (exact, k.note) if x) or None)
    left, right = st.columns(2)
    for i, c in enumerate(d.ok_charts):
        with (left if i % 2 == 0 else right):
            st.plotly_chart(c.figure, key=f"ai_chart_{i}")
            badge = "🤖 AI-written" if c.source == "ai" else "📐 rule-based"
            st.caption(f"{badge} · {c.spec.why}" if c.spec.why else badge)
            for w in c.warnings:
                st.warning(w)
            with st.expander("How this chart was made"):
                st.write(f"Built by {'the AI' if c.source == 'ai' else 'the rule-based chart builder'} "
                         f"(AI attempts: {c.attempts})")
                if c.ai_message:
                    st.caption(f"The AI route failed: {c.ai_message[:300]}")
                st.json(c.spec.to_spec())
                if c.code:
                    st.code(c.code, language="python")
    for c in d.failed_charts:
        st.warning(f"Skipped chart '{c.spec.title}': {_safe(c.message[:300])}")
    models = getattr(session, "models", None)
    if models:
        real = sum(usage_of(m)[0] for m in models)
        hits = sum(usage_of(m)[1] for m in models)
        st.caption(f"AI requests in this run: {real} sent to Google, {hits} replayed from saved replies.")
    a, b = st.columns(2)
    a.download_button("⬇ Download as a web page (HTML)", render_dashboard_html(d), file_name="dashboard.html",
                      mime="text/html", key="ai_download")
    if b.button("Start over", key="ai_restart"):
        _clear_run()
        st.rerun()


def _simple(message, kind, button_label, button_key):
    getattr(st, kind)(message)
    if st.button(button_label, key=button_key):
        _clear_run()
        st.rerun()


# ---------------------------------------------------------------- entry point
def render_ai_dashboard(df, profile, data_key: str, *, session_factory: Optional[Callable] = None,
                        scope: str = ""):
    """Draws the tab. `data_key` identifies the dataset: a new key starts over (settings are kept).
    `session_factory(df, profile, settings, progress)` can replace the real models (tests do)."""
    ss = st.session_state
    if ss.get("ai_data_key") != data_key:
        for key in [k for k in list(ss.keys()) if k.startswith("ai_") and k not in SETTING_KEYS]:
            del ss[key]
        ss["ai_data_key"] = data_key
    stage = ss.setdefault("ai_stage", "idle")
    if stage == "idle":                        # Day 17: remember which rows the dashboard is built for
        ss["ai_scope"] = scope
        if scope:
            st.info(f"🔎 The dashboard will be built only for the filtered rows: {scope}")
    else:
        built = ss.get("ai_scope", "")
        if built != scope:
            st.warning(f"The filter changed after this dashboard was started. It covers "
                       f"{built or 'all rows'}, but the filter is now {scope or 'all rows'}. "
                       "Start over (or cancel) to build it for the current rows.")
        elif built and stage == "done":
            st.caption(f"🔎 Built only for the filtered rows: {built}")
    if stage == "review":
        _review()
    elif stage == "done":
        _done()
    elif stage == "cancelled":
        _simple("Cancelled. Nothing was built.", "info", "Start over", "ai_restart")
    elif stage == "error":
        _simple(_safe(ss.get("ai_error", "Something went wrong.")), "error", "Try again", "ai_restart")
    else:
        _idle(df, profile, session_factory)
