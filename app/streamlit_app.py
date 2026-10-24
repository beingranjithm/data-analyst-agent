import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

# Let the app find our own "tools" folder
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tools.cleaner import clean_dataframe  # noqa: E402
from tools.loader import load_csv_bytes  # noqa: E402
from tools.profiler import describe_column, profile_dataframe, profile_to_text  # noqa: E402

st.set_page_config(page_title="Agentic Data Analyst", page_icon="📊", layout="wide")

ROOT = Path(__file__).resolve().parent.parent
SAMPLE_PATH = ROOT / "sample_data" / "sales_sample.csv"
MESSY_DIR = ROOT / "sample_data" / "messy"
MAX_MB = 50
ROLE_LABELS = {
    "numeric": "🔢 numeric", "datetime": "📅 date/time", "categorical": "🏷️ category",
    "boolean": "✅ yes/no", "id": "🔑 identifier", "text": "📝 free text",
    "high_cardinality": "🏷️ many categories", "empty": "⚪ empty",
}


def build_table(name, data: bytes):
    """Load -> clean -> profile. Returns a dict describing one table."""
    df, load_notes = load_csv_bytes(data)
    df, clean_log = clean_dataframe(df)
    return {
        "name": name,
        "df": df,
        "profile": profile_dataframe(df),
        "cleaning_log": load_notes + clean_log,
    }


st.title("📊 Agentic Data Analyst")
st.caption("Upload a CSV and get an instant overview. Dashboards and chat come next.")

# ---- choose the data source ----
source = st.sidebar.radio("Data source", ["Upload my own CSV", "Built-in sample sales data", "Messy test file"])
name, data = None, None

if source == "Built-in sample sales data":
    name, data = "sales_sample", SAMPLE_PATH.read_bytes()
elif source == "Messy test file":
    files = sorted(MESSY_DIR.glob("*.csv"))
    if not files:
        st.warning("No messy files found. Run: python sample_data/make_messy_files.py")
        st.stop()
    pick = st.sidebar.selectbox("Pick a file", [f.name for f in files])
    name, data = Path(pick).stem, (MESSY_DIR / pick).read_bytes()
else:
    uploaded = st.file_uploader("Upload a CSV file", type=["csv"])
    if uploaded is not None:
        if uploaded.size > MAX_MB * 1024 * 1024:
            st.error(f"File is larger than {MAX_MB} MB. Please upload a smaller file.")
            st.stop()
        name, data = Path(uploaded.name).stem, uploaded.getvalue()

if data is None:
    st.info("Upload a CSV above, or choose a built-in file in the sidebar.")
    st.stop()

try:
    table = build_table(name, data)
except ValueError as e:
    st.error(str(e))
    st.stop()
except Exception as e:
    st.error(f"Something unexpected went wrong while reading the file: {e}")
    st.stop()

# Data is kept as a NAMED COLLECTION of tables (only one for now).
# This lets us support several files and sheets later without a rewrite.
st.session_state["tables"] = {table["name"]: table}
st.session_state["active_table"] = table["name"]

df, profile = table["df"], table["profile"]
d = profile["dataset"]
st.subheader(f"Table: {table['name']}")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Rows", f"{d['rows']:,}")
c2.metric("Columns", f"{d['columns']:,}")
c3.metric("Missing cells", f"{d['missing_cells']:,}")
c4.metric("Duplicate rows", f"{d['duplicate_rows']:,}")

if table["cleaning_log"]:
    with st.expander(f"🧹 Cleaned automatically ({len(table['cleaning_log'])} changes)", expanded=True):
        for line in table["cleaning_log"]:
            st.write(f"- {line}")

st.subheader("Data preview")
st.dataframe(df.head(50))

st.subheader("What the profiler found")
st.write(f"**Time column:** {profile['time_column'] or 'none detected'}")
st.write(f"**Measures (numbers to analyse):** {', '.join(profile['measures'][:30]) or 'none'}")
st.write(f"**Dimensions (ways to group):** {', '.join(profile['dimensions'][:30]) or 'none'}")
st.write(f"**Identifier columns:** {', '.join(profile['id_columns']) or 'none'}")

if profile["warnings"]:
    st.subheader("Data quality warnings")
    for w in profile["warnings"][:25]:
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
