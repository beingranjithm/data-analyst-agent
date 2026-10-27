"""A tiny Streamlit script that draws ONLY the Chat tab, WITH data to calculate on, a scripted fake model and a
fake runner in place of the sandbox (used by AppTest)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from agents.chat_agent import ChatSession  # noqa: E402
from calc_fakes import local_runner  # noqa: E402
from tools.profiler import profile_dataframe  # noqa: E402
from ui.chat_tab import render_chat  # noqa: E402
from ui_fakes import ScriptedLlm, _no_sleep  # noqa: E402

df = pd.DataFrame({"region": ["North", "South", "East", "West"] * 10,
                   "revenue": [round(10 + (i * 7.3) % 90, 2) for i in range(40)]})
profile = profile_dataframe(df)


def factory(profile, settings):
    assert settings["df"] is df, "the tab must hand the data to the session"
    llm = ScriptedLlm(replies=list(st.session_state.get("replies", [])))
    st.session_state["llm"] = llm
    return ChatSession(profile, llm, tables=settings["df"], runner=local_runner, sleep=_no_sleep,
                       max_code_runs=int(st.session_state.get("max_code_runs", 1)))


render_chat(profile, st.session_state.get("data_key", "k1"), df=df, session_factory=factory)
