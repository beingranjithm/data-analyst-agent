"""Day 23a: the visitor-facing parts of the public hosted version (see tools/hosting.py)."""
import streamlit as st

from tools.hosting import is_hosted, redact

KEY_STATE = "user_api_key"          # lives in THIS visitor's session only
BANNER = ("🌐 **Public demo.** Do not upload private or confidential data. Your files are processed in memory for "
          "this session only. The AI answers with **your own Gemini key** (free at aistudio.google.com); Google's "
          "free tier may use prompts to improve its products. Your key stays in this browser session: it is not "
          "saved, logged or shared.")


def user_key() -> str:
    return st.session_state.get(KEY_STATE, "") if is_hosted() else ""


def safe_text(text) -> str:
    """Text for the screen with the visitor's key taken out (error messages can echo what was sent)."""
    return redact(text, user_key())


def forget_key() -> None:
    """Erases the key AND everything that holds a model connection made with it (the chat, the dashboard run), so
    nothing in this session can still use it."""
    ss = st.session_state
    for k in (KEY_STATE, "key_box"):
        ss.pop(k, None)
    from ui.chat_tab import _clear_chat
    from ui.dashboard_tab import _clear_run
    _clear_chat()
    _clear_run()


def render_hosted_sidebar() -> None:
    """Banner + key box. Does nothing on a normal (local) run."""
    if not is_hosted():
        return
    st.info(BANNER)
    with st.sidebar:
        st.markdown("### Your Gemini key")
        typed = st.text_input("Paste your key", type="password", key="key_box", placeholder="AIza...", autocomplete="off",
                              help="Needed for AI answers. Not needed for 'Replay only'.")
        if typed:
            st.session_state[KEY_STATE] = typed.strip()
        if st.session_state.get(KEY_STATE):
            st.caption("✅ A key is set for this session.")
            if st.button("Forget my key", key="forget_key"):
                forget_key()
                st.rerun()
