"""The "Chat" tab (Day 14): ask questions about the dataset in plain language.

The chat agent sees only the data SUMMARY (the profile), never your rows, so it can describe the dataset but
cannot yet calculate new numbers (that arrives with Day 15). The conversation lives in st.session_state as a
ChatSession; numbers in an answer that are not in the summary get a visible warning.
Do not click other widgets while an answer is being written: Streamlit would restart the script.
"""
from typing import Callable, Optional

import streamlit as st

from ui.dashboard_tab import CACHE_MODES, PRIVACY_NOTE, api_key_available, usage_of

SETTING_KEYS = ("chat_saved_mode",)          # kept when the dataset changes (ordinary keys: see dashboard_tab)
LIMIT_NOTE = ("ℹ️ I answer from a summary of the data (column types, ranges, averages, categories). I cannot "
              "calculate new numbers from the rows yet; when a question needs that, I will say so.")


def default_session_factory(profile, settings):
    """Builds the real chat models (cache + pacing + fallback chain) and a ChatSession."""
    from agents.chat_agent import ChatSession
    from agents.llm_factory import make_llm
    model = make_llm("chat", mode=settings["cache_mode"])
    return ChatSession(profile, model)


def _clear_chat():
    ss = st.session_state
    for key in [k for k in list(ss.keys()) if k.startswith("chat_") and k not in SETTING_KEYS]:
        del ss[key]


def _show_history(session, notes):
    for i, (role, text) in enumerate(session.transcript):
        with st.chat_message("user" if role == "user" else "assistant"):
            st.markdown(text)
            numbers = notes.get(i)
            if numbers:
                st.warning("These numbers are not in the data summary, so please double-check them: "
                           + ", ".join(numbers))


def render_chat(profile, data_key: str, *, session_factory: Optional[Callable] = None):
    """Draws the tab. `data_key` identifies the dataset: a new key starts a new conversation.
    `session_factory(profile, settings)` can replace the real models (tests do)."""
    from agents.chat_agent import starter_questions
    ss = st.session_state
    if ss.get("chat_data_key") != data_key:
        _clear_chat()
        ss["chat_data_key"] = data_key

    labels = list(CACHE_MODES)
    mode_label = st.selectbox("AI replies", labels, index=labels.index(ss.get("chat_saved_mode", labels[0])),
                              key="chat_cache_mode")
    ss["chat_saved_mode"] = mode_label
    mode = CACHE_MODES[mode_label]
    needs_key = mode != "replay"
    blocked = False
    if needs_key:
        st.caption(PRIVACY_NOTE)
    if session_factory is None and needs_key and not api_key_available():
        st.error("No Gemini API key found. Put GOOGLE_API_KEY=... in hello_agent/.env, or choose 'Replay only'.")
        blocked = True
    st.caption(LIMIT_NOTE)

    if ss.get("chat_mode") != mode:                 # a different reply setting means a different model object
        ss.pop("chat_session", None)
        ss["chat_mode"] = mode
    session = ss.get("chat_session")
    transcript_kept = session.transcript if session is not None else ss.get("chat_kept", [])
    notes = ss.setdefault("chat_notes", {})

    if session is None and not blocked:
        try:
            session = (session_factory or default_session_factory)(profile, {"cache_mode": mode})
        except Exception as e:
            st.error(f"Could not start the chat: {type(e).__name__}: {e}")
            return
        session.transcript = list(transcript_kept)  # a changed setting keeps what was already said
        ss["chat_session"] = session
    if session is None:
        return

    _show_history(session, notes)

    question = None
    if not session.transcript:
        st.write("**Try one of these:**")
        for i, q in enumerate(starter_questions(profile)):
            if st.button(q, key=f"chat_starter_{i}"):
                question = q
    typed = st.chat_input("Ask a question about this dataset", key="chat_input")
    question = typed or question

    error = ss.pop("chat_error", None)
    if error:
        st.error(error)

    if question:
        with st.chat_message("user"):
            st.markdown(question)
        with st.chat_message("assistant"):
            with st.spinner("Thinking ..."):
                reply = session.ask(question)
        if reply.ok:
            if reply.ungrounded:
                notes[len(session.transcript) - 1] = reply.ungrounded
        else:
            ss["chat_error"] = reply.error
        ss["chat_kept"] = list(session.transcript)
        st.rerun()

    if session.transcript:
        a, b = st.columns([1, 3])
        if a.button("Start a new conversation", key="chat_reset"):
            session.reset()
            ss["chat_notes"], ss["chat_kept"] = {}, []
            st.rerun()
        real, replayed = usage_of(getattr(session, "model", None))
        b.caption(f"AI requests in this chat: {real} real, {replayed} replayed from saved replies.")
