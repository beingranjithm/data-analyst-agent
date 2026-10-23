import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

# Let the app find our own "tools" folder
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tools.profiler import describe_column, profile_dataframe, profile_to_text  # noqa: E402

st.set_page_config(page_title="Agentic Data Analyst", page_icon="📊", layout="wide")

SAMPLE_PATH = Path(__file__).parent.parent / "sample_data" / "sales_sample.csv"
MAX_MB = 50
ROLE_LABELS = {
    "numeric": "🔢 numeric", "datetime": "📅 date/time", "categorical": "🏷️ category",
    "boolean": "✅ yes/no", "id": "🔑 identifier", "text": "📝 free text",
    "high_cardinality": "🏷️ many categories", "empty": "⚪ empty",
}


def load_csv(file):
    """Read a CSV, trying a few common text encodings."""
    for encoding in ("utf-8", "utf-8-sig", "latin-1"):
        try:
            file.seek(0)
            return pd.read_csv(file, encoding=encoding)
        except UnicodeDecodeError:
            continue
    raise ValueError("Could not read this file. Is it a valid CSV?")


st.title("📊 Agentic Data Analyst")
st.caption("Upload a CSV and get an instant overview. Dashboards and chat come next.")

use_sample = st.sidebar.checkbox("Use built-in sample sales data")
uploaded = st.file_uploader("Upload a CSV file", type=["csv"])

df = None
try:
    if use_sample:
        df = pd.read_csv(SAMPLE_PATH)
    elif uploaded is not None:
        if uploaded.size > MAX_MB * 1024 * 1024:
            st.error(f"File is larger than {MAX_MB} MB. Please upload a smaller file.")
            st.stop()
        df = load_csv(uploaded)
except pd.errors.EmptyDataError:
    st.error("This file is empty. Please upload a CSV that contains data.")
    st.stop()
except Exception as e:
    st.error(f"Something went wrong while reading the file: {e}")
    st.stop()

if df is None:
    st.info("Upload a CSV above, or tick 'Use built-in sample sales data' in the sidebar.")
    st.stop()

profile = profile_dataframe(df)
st.session_state["df"] = df
st.session_state["profile"] = profile
d = profile["dataset"]

c1, c2, c3, c4 = st.columns(4)
c1.metric("Rows", f"{d['rows']:,}")
c2.metric("Columns", f"{d['columns']:,}")
c3.metric("Missing cells", f"{d['missing_cells']:,}")
c4.metric("Duplicate rows", f"{d['duplicate_rows']:,}")

st.subheader("Data preview")
st.dataframe(df.head(50))

st.subheader("What the profiler found")
st.write(f"**Time column:** {profile['time_column'] or 'none detected'}")
st.write(f"**Measures (numbers to analyse):** {', '.join(profile['measures']) or 'none'}")
st.write(f"**Dimensions (ways to group):** {', '.join(profile['dimensions']) or 'none'}")
st.write(f"**Identifier columns:** {', '.join(profile['id_columns']) or 'none'}")

if profile["warnings"]:
    st.subheader("Data quality warnings")
    for w in profile["warnings"]:
        st.warning(w)
else:
    st.success("No data quality problems found.")

st.subheader("Column profile")
rows = [{
    "Column": c["name"],
    "Detected as": ROLE_LABELS.get(c["role"], c["role"]),
    "Missing %": c["missing_pct"],
    "Unique values": c["unique"],
    "Details": describe_column(c),
} for c in profile["columns"]]
st.dataframe(pd.DataFrame(rows))

if profile["correlations"]:
    st.subheader("Strong relationships between numeric columns")
    st.dataframe(pd.DataFrame(profile["correlations"]).rename(
        columns={"a": "Column A", "b": "Column B", "r": "Correlation"}))

with st.expander("What the AI agents will read (compact summary)"):
    st.text(profile_to_text(profile))

st.download_button(
    "Download full profile (JSON)",
    json.dumps(profile, indent=2),
    file_name="profile.json",
    mime="application/json",
)
