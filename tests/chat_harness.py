"""A tiny Streamlit script that draws ONLY the Chat tab with a scripted fake model (used by AppTest)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from agents.chat_agent import ChatSession  # noqa: E402
from tools.profiler import profile_dataframe  # noqa: E402
from ui.chat_tab import render_chat  # noqa: E402
from ui_fakes import ScriptedLlm, _no_sleep  # noqa: E402

df = pd.DataFrame({"region": ["North", "South", "East", "West"] * 10,
                   "revenue": [round(10 + (i * 7.3) % 90, 2) for i in range(40)]})
profile = profile_dataframe(df)


def factory(profile, settings):
    if st.session_state.get("scenario") == "factory_error":
        raise RuntimeError("boom")
    llm = ScriptedLlm(replies=list(st.session_state.get("replies", [])))
    st.session_state["llm"] = llm                      # so the test can see what the model was asked
    return ChatSession(profile, llm, sleep=_no_sleep)


render_chat(profile, st.session_state.get("data_key", "k1"), session_factory=factory)
