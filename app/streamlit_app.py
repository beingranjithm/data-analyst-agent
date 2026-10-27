import hashlib
import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

# Let the app find our own "tools" folder
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tools.charts import (AGG_FUNCS, ChartError, build_chart, compute_kpis,  # noqa: E402
                          spec_title, suggest_charts)
from tools.cleaner import clean_dataframe  # noqa: E402
from tools.loader import load_csv_bytes  # noqa: E402
from tools.profiler import describe_column, profile_dataframe, profile_to_text  # noqa: E402
from tools.sandbox import run_code  # noqa: E402
from ui.chat_tab import render_chat  # noqa: E402
from ui.dashboard_tab import render_ai_dashboard  # noqa: E402

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
CHART_TYPES = {
    "Trend over time (line)": "line",
    "Compare categories (bar)": "bar",
    "Share of total (donut)": "donut",
    "Distribution of a number (histogram)": "histogram",
    "Spread by category (box)": "box",
    "Relationship between two numbers (scatter)": "scatter",
    "Correlation overview (heatmap)": "heatmap",
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


def show_chart(df, spec, key):
    """Draw one chart, or a friendly message if it cannot be built."""
    try:
        st.plotly_chart(build_chart(df, spec), key=key)
    except ChartError as e:
        st.warning(f"Could not build this chart: {e}")


st.title("📊 Agentic Data Analyst")
st.caption("Upload a CSV and get an instant overview. The '✨ AI dashboard' tab plans and builds a dashboard with AI after you approve the plan. Chat comes next.")

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
st.session_state["tables"] = {table["name"]: table}
st.session_state["active_table"] = table["name"]

df, profile = table["df"], table["profile"]
d = profile["dataset"]
st.subheader(f"Table: {table['name']}")

tab_overview, tab_ai, tab_chat, tab_gallery, tab_builder, tab_sandbox = st.tabs(
    ["Overview", "✨ AI dashboard", "💬 Chat", "Chart gallery", "Build a chart", "Code sandbox"])

# ============ TAB 1: OVERVIEW ============
with tab_overview:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows", f"{d['rows']:,}")
    c2.metric("Columns", f"{d['columns']:,}")
    c3.metric("Missing cells", f"{d['missing_cells']:,}")
    c4.metric("Duplicate rows", f"{d['duplicate_rows']:,}")

    if table["cleaning_log"]:
        with st.expander(f"🧹 Cleaned automatically ({len(table['cleaning_log'])} changes)", expanded=False):
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

    st.download_button("Download full profile (JSON)", json.dumps(profile, indent=2),
                       file_name="profile.json", mime="application/json")

# ============ TAB 2: CHART GALLERY (rule-based) ============
with tab_gallery:
    st.caption("A dashboard made by simple fixed rules. On Day 8 an AI planner will do this job instead, "
               "and we can compare the two.")
    kpis = compute_kpis(df, profile)
    for col, k in zip(st.columns(len(kpis)), kpis):
        col.metric(k["label"], k["value"])

    specs = suggest_charts(profile)
    if not specs:
        st.info("This table has no columns that suit automatic charts "
                "(it needs numbers and/or categories). Try the 'Build a chart' tab.")
    left, right = st.columns(2)
    for i, spec in enumerate(specs):
        with (left if i % 2 == 0 else right):
            show_chart(df, spec, key=f"gallery_{i}")
            st.caption(f"💡 {spec['why']}")
            with st.expander("Chart spec"):
                st.json(spec)

# ============ TAB 3: BUILD A CHART BY HAND ============
with tab_builder:
    st.caption("Pick a chart type and columns yourself. The 'spec' shown below is the same format "
               "the AI agents will produce later.")
    by_role = lambda *roles: [c["name"] for c in profile["columns"] if c["role"] in roles]  # noqa: E731
    time_cols = by_role("datetime")
    measure_cols = profile["measures"]
    dim_cols = by_role("categorical", "boolean", "high_cardinality")

    label = st.selectbox("Chart type", list(CHART_TYPES))
    kind = CHART_TYPES[label]
    agg_choice = st.selectbox("How to combine values", ["auto"] + list(AGG_FUNCS)) if kind in ("line", "bar", "donut") else "auto"
    spec, problem = {"type": kind}, None
    if agg_choice != "auto":
        spec["agg"] = agg_choice

    if kind == "line":
        if not time_cols or not measure_cols:
            problem = "This chart needs a date column and a numeric column."
        else:
            spec["time"] = st.selectbox("Date column", time_cols)
            spec["measure"] = st.selectbox("Measure", measure_cols)
    elif kind in ("bar", "donut", "box"):
        if not dim_cols or not measure_cols:
            problem = "This chart needs a category column and a numeric column."
        else:
            spec["dimension"] = st.selectbox("Category column", dim_cols)
            spec["measure"] = st.selectbox("Measure", measure_cols)
    elif kind == "histogram":
        if not measure_cols:
            problem = "This chart needs a numeric column."
        else:
            spec["measure"] = st.selectbox("Numeric column", measure_cols)
    elif kind == "scatter":
        if len(measure_cols) < 2:
            problem = "This chart needs at least two numeric columns."
        else:
            spec["x"] = st.selectbox("X axis", measure_cols)
            spec["y"] = st.selectbox("Y axis", measure_cols, index=1)
            color = st.selectbox("Colour by (optional)", ["(none)"] + dim_cols)
            if color != "(none)":
                spec["color"] = color
    elif kind == "heatmap":
        if len(measure_cols) < 2:
            problem = "This chart needs at least two numeric columns."
        else:
            spec["measures"] = st.multiselect("Numeric columns", measure_cols, default=measure_cols[:8])

    if problem:
        st.info(problem)
    else:
        show_chart(df, spec, key="builder_chart")
        with st.expander("Chart spec", expanded=True):
            st.json(spec)

# ============ TAB 4: CODE SANDBOX (try the safety checks yourself) ============
SANDBOX_EXAMPLES = {
    "✅ Revenue by region (table)": "result = df.groupby('region')['revenue'].sum().reset_index()",
    "✅ A chart": ("fig = px.bar(df.groupby('category', as_index=False)['revenue'].sum(),\n"
                  "             x='category', y='revenue', title='Revenue by category')\n"
                  "result = 'chart made'"),
    "✅ Print and answer": "print('Rows:', len(df))\nresult = df['revenue'].sum()",
    "❌ Wrong column name": "result = df['revenu'].sum()",
    "❌ Read a secret file": "result = open('.env').read()",
    "❌ Import os": "import os\nresult = os.listdir('/')",
    "❌ Run a command": "import subprocess\nresult = subprocess.run(['ls'])",
    "❌ Sneaky: via pandas internals": "result = pd.io.common.os.listdir('/home')",
    "❌ Sneaky: DataSource": "result = np.lib.npyio.DataSource().open('/etc/passwd').read()",
    "❌ Infinite loop": "while True:\n    pass",
    "❌ Memory bomb": "x = [0] * (10 ** 10)",
    "❌ Output flood": "for i in range(10 ** 6):\n    print('x' * 100)",
}

with tab_sandbox:
    st.caption("This is where AI-written code will run. Try the examples, especially the red ones, "
               "and watch which safety layer stops each attack.")
    example = st.selectbox("Example code", list(SANDBOX_EXAMPLES))
    code = st.text_area("Python code (use df, pd, np, px, go; put the answer in `result` or a Plotly figure in `fig`)",
                        value=SANDBOX_EXAMPLES[example], height=180, key=f"sandbox_code_{example}")
    cols = st.columns(3)
    timeout_s = cols[0].number_input("Time limit (seconds)", 2, 60, 5)
    memory = cols[1].number_input("Memory limit (MB)", 128, 4096, 1024, step=128)
    skip_check = cols[2].checkbox("Skip the static check (tests the inner locks)")

    if st.button("▶ Run in sandbox", type="primary"):
        tables_for_run = {n: t["df"] for n, t in st.session_state["tables"].items()}
        with st.spinner("Running..."):
            res = run_code(code, tables_for_run, active=st.session_state["active_table"],
                           timeout=int(timeout_s), memory_mb=int(memory), check=not skip_check)
        if res.ok:
            st.success(f"Ran successfully in {res.seconds}s")
        elif res.status == "blocked":
            st.error("🛑 Blocked before running (static check)")
        else:
            st.error(f"❌ {res.status.upper()}: {res.message()}")
        if res.status == "blocked":
            for p in res.problems:
                st.write(f"- {p}")
        if res.isolation == "none":
            st.warning("The operating-system file lock (Landlock) is NOT active on this computer. "
                       "Only the other layers protect you.")
        elif res.isolation.startswith("landlock"):
            st.caption(f"File-system lock: active ({res.isolation}) | time: {res.seconds}s")
        if res.stdout:
            st.text("Printed output:\n" + res.stdout[:3000])
        if res.table is not None:
            st.write(f"Result table ({res.table_total_rows} rows total):")
            st.dataframe(res.table)
        elif res.value is not None:
            st.write("Result:", res.value)
        if res.figure is not None:
            st.plotly_chart(res.figure, key="sandbox_fig")

# ============ TAB 5: AI DASHBOARD (plan -> you approve -> build) ============
with tab_ai:
    render_ai_dashboard(df, profile, data_key=f"{table['name']}:{hashlib.sha1(data).hexdigest()[:12]}")

# ============ TAB: CHAT (questions about the data summary) ============
with tab_chat:
    render_chat(profile, data_key=f"{table['name']}:{hashlib.sha1(data).hexdigest()[:12]}", df=df)
