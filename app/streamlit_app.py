from pathlib import Path

import pandas as pd
import streamlit as st

st.set_page_config(page_title="Agentic Data Analyst", page_icon="📊", layout="wide")

SAMPLE_PATH = Path(__file__).parent.parent / "sample_data" / "sales_sample.csv"
MAX_MB = 50


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

st.session_state["df"] = df

c1, c2, c3 = st.columns(3)
c1.metric("Rows", f"{len(df):,}")
c2.metric("Columns", f"{df.shape[1]:,}")
c3.metric("Missing cells", f"{int(df.isna().sum().sum()):,}")

st.subheader("Data preview")
st.dataframe(df.head(50))

st.subheader("Column overview")
overview = pd.DataFrame({
    "Column": df.columns,
    "Type": df.dtypes.astype(str).values,
    "Missing": df.isna().sum().values,
    "Unique values": df.nunique().values,
})
st.dataframe(overview)
