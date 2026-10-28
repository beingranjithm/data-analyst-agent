"""Like ui_harness.py (the AI dashboard tab with fake models) but with a filter label from session_state["scope"]."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

import ui_fakes  # noqa: E402
from tools.profiler import profile_dataframe  # noqa: E402
from ui.dashboard_tab import render_ai_dashboard  # noqa: E402

df = pd.DataFrame({"region": ["North", "South", "East", "West"] * 10,
                   "revenue": [round(10 + (i * 7.3) % 90, 2) for i in range(40)]})
profile = profile_dataframe(df)
render_ai_dashboard(df, profile, st.session_state.get("data_key", "k1"),
                    session_factory=ui_fakes.make_factory(st.session_state.get("scenario", "ok")),
                    scope=st.session_state.get("scope", ""))
