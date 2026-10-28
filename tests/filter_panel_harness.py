"""A tiny Streamlit script that draws ONLY the sidebar filter panel and reports what it returns (used by AppTest)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from tools.profiler import profile_dataframe  # noqa: E402
from ui.filter_panel import render_filter_panel  # noqa: E402

days = pd.date_range("2025-01-01", periods=40, freq="D")
df = pd.DataFrame({"order_date": days, "region": ["East", "West", "North", "South"] * 10,
                   "revenue": [float(i) for i in range(40)]})
profile = profile_dataframe(df)
profile["time_column"] = "order_date"
for c in profile["columns"]:
    if c["name"] == "order_date":
        c["role"] = "datetime"

out, label, sig = render_filter_panel(df, profile, st.session_state.get("data_key", "k1"))
st.write(f"ROWS={len(out)}")
st.write(f"LABEL={label}")
st.write(f"SIG={sig}")
