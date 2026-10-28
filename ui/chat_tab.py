"""The "Chat" tab (Days 14-15): ask questions about the dataset in plain language.

The chat agent sees the data SUMMARY (the profile), never the table. For questions that need numbers it writes
pandas code, which runs in the sandbox on your computer; the AI then sees only a small, size-capped result.
The conversation lives in st.session_state as a ChatSession. Numbers in an answer that are in neither the
summary nor a calculation result get a visible warning, and every calculation can be inspected.
Do not click other widgets while an answer is being written: Streamlit would restart the script.
"""
from typing import Callable, Optional

import streamlit as st

from ui.hosted_ui import safe_text
from ui.dashboard_tab import CACHE_MODES, PRIVACY_NOTE, api_key_available, cache_modes, usage_of  # noqa: F401

SETTING_KEYS = ("chat_saved_mode",)          # kept when the dataset changes (ordinary keys: see dashboard_tab)
LIMIT_NOTE = ("ℹ️ I answer from a summary of the data (column types, ranges, averages, categories). I cannot "
              "calculate new numbers from the rows; when a question needs that, I will say so.")
CALC_NOTE = ("🧮 For questions that need numbers, the AI writes pandas code that runs in a locked-down sandbox on "
             "your computer. The AI sees only a small result (at most 20 rows), never the whole table. "
             "Open \"Show the calculation\" under an answer to check the code and the result.")


def default_session_factory(profile, settings):
    """Builds the real chat models (cache + pacing + fallback chain) and a ChatSession."""
    from agents.chat_agent import MAX_CODE_RUNS, ChatSession
    from agents.llm_factory import make_llm
    from ui.hosted_ui import user_key
    model = make_llm("chat", mode=settings["cache_mode"], api_key=user_key() or None)
    multi = settings.get("multi")                     # Day 22: {"tables", "note", "runner"} when several files were uploaded
    if multi:
        return ChatSession(profile, model, tables=multi["tables"], max_code_runs=MAX_CODE_RUNS,
                           scope=settings.get("scope", ""), extra_note=multi["note"], runner=multi["runner"])
    return ChatSession(profile, model, tables=settings.get("df"), max_code_runs=MAX_CODE_RUNS,
                       scope=settings.get("scope", ""))


def _clear_chat():
    ss = st.session_state
    for key in [k for k in list(ss.keys()) if k.startswith("chat_") and k not in SETTING_KEYS]:
        del ss[key]


def _show_calculations(calcs):
    failed = [c for c in calcs if not c.ok]
    label = "⚠ The calculation failed" if failed and len(failed) == len(calcs) else "🧮 Show the calculation"
    with st.expander(label):
        for n, c in enumerate(calcs, 1):
            if len(calcs) > 1:
                st.caption(f"Calculation {n}" + (" (failed)" if not c.ok else " (found nothing)" if c.empty else ""))
            st.code(c.code, language="python")
            st.text(("Result shown to the AI:\n" if c.ok else "Error:\n") + c.result_text)


def _show_history(session, notes, calcs, fixed=None):
    for i, (role, text) in enumerate(session.transcript):
        with st.chat_message("user" if role == "user" else "assistant"):
            st.markdown(text)
            if calcs.get(i):
                _show_calculations(calcs[i])
            if (fixed or {}).get(i):                       # Day 18b
                st.caption("✓ The first draft of this answer did not match the calculation, so it "  # Day 19
                           "was rewritten once. Open \"Show the calculation\" to check it.")
                draft = (fixed or {}).get(i)
                if isinstance(draft, str) and draft:       # Day 20: show what was replaced
                    with st.expander("See the first draft (before the check)"):
                        st.write(draft)
            numbers = notes.get(i)
            if numbers:
                st.warning("These numbers are not in the data summary, so please double-check them: "
                           + ", ".join(numbers))


def _starters(profile, can_calculate):
    """Starter questions; with data to calculate on, one of them needs a real calculation."""
    from agents.chat_agent import starter_questions
    qs = starter_questions(profile)
    measures, dims = profile.get("measures"), profile.get("dimensions")
    if can_calculate and measures and dims:
        return qs[:3] + [f"Which {dims[0]} has the highest total {measures[0]}?"]
    return qs


def render_chat(profile, data_key: str, *, df=None, session_factory: Optional[Callable] = None, scope: str = "",
                multi=None, starters=None):
    """Draws the tab. `data_key` identifies the dataset: a new key starts a new conversation.
    `df` is the data the AI may calculate on (None = answer from the summary only).
    `session_factory(profile, settings)` can replace the real models (tests do); settings holds the data as "df"."""
    ss = st.session_state
    if ss.get("chat_data_key") != data_key:
        _clear_chat()
        ss["chat_data_key"] = data_key

    labels = list(cache_modes())
    mode_label = st.selectbox("AI replies", labels, index=labels.index(ss.get("chat_saved_mode", labels[0])),
                              key="chat_cache_mode")
    ss["chat_saved_mode"] = mode_label
    mode = cache_modes()[mode_label]
    needs_key = mode != "replay"
    blocked = False
    if needs_key:
        st.caption(PRIVACY_NOTE)
    if session_factory is None and needs_key and not api_key_available():
        from tools.hosting import is_hosted
        st.error("Paste your own Gemini key in the sidebar, or choose 'Replay only'." if is_hosted() else
                 "No Gemini API key found. Put GOOGLE_API_KEY=... in hello_agent/.env, or choose 'Replay only'.")
        blocked = True
    st.caption(CALC_NOTE if df is not None else LIMIT_NOTE)
    if multi:
        st.caption(f"🔗 The chat works on '{next(iter(multi['tables']))}' and can also use your other uploaded tables, "
                   "with the links found between them. Open \"Show the calculation\" to see what it used.")
    if scope:
        st.info(f"🔎 Answering only for the filtered rows: {scope}. Changing the filter starts a new conversation.")

    if ss.get("chat_mode") != mode:                 # a different reply setting means a different model object
        ss.pop("chat_session", None)
        ss["chat_mode"] = mode
    session = ss.get("chat_session")
    transcript_kept = session.transcript if session is not None else ss.get("chat_kept", [])
    notes = ss.setdefault("chat_notes", {})
    calcs = ss.setdefault("chat_calcs", {})
    fixed = ss.setdefault("chat_fixed", {})

    if session is None and not blocked:
        try:
            session = (session_factory or default_session_factory)(profile, {"cache_mode": mode, "df": df, "scope": scope,
                                                                                     "multi": multi})
        except Exception as e:
            st.error(safe_text(f"Could not start the chat: {type(e).__name__}: {e}"))
            return
        session.transcript = list(transcript_kept)  # a changed setting keeps what was already said
        ss["chat_session"] = session
    if session is None:
        return

    _show_history(session, notes, calcs, fixed)

    question = None
    if not session.transcript:
        st.write("**Try one of these:**")
        if starters:
            st.caption("These questions have saved answers, so they also work with 'Replay only' and no key.")
        for i, q in enumerate(starters or _starters(profile, df is not None)):
            if st.button(q, key=f"chat_starter_{i}"):
                question = q
    typed = st.chat_input("Ask a question about this dataset", key="chat_input")
    question = typed or question

    error = ss.pop("chat_error", None)
    if error:
        st.error(safe_text(error))

    if question:
        with st.chat_message("user"):
            st.markdown(question)
        with st.chat_message("assistant"):
            with st.spinner("Thinking ..."):
                reply = session.ask(question)
        if reply.ok:
            if reply.ungrounded:
                notes[len(session.transcript) - 1] = reply.ungrounded
            if reply.calculations:
                calcs[len(session.transcript) - 1] = reply.calculations
            if getattr(reply, "corrected", False):
                fixed[len(session.transcript) - 1] = getattr(reply, "first_draft", "") or True
        else:
            ss["chat_error"] = reply.error
        ss["chat_kept"] = list(session.transcript)
        st.rerun()

    if session.transcript:
        a, b = st.columns([1, 3])
        if a.button("Start a new conversation", key="chat_reset"):
            session.reset()
            ss["chat_notes"], ss["chat_calcs"], ss["chat_kept"], ss["chat_fixed"] = {}, {}, [], {}
            st.rerun()
        real, replayed = usage_of(getattr(session, "model", None))
        b.caption(f"AI requests in this chat: {real} real, {replayed} replayed from saved replies.")
