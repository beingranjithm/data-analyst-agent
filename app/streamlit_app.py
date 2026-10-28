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
from ui.filter_panel import render_filter_panel  # noqa: E402
from ui.kpi_card import show_kpi  # noqa: E402  (Day 20)
from tools.multi import build_table  # noqa: E402  (Day 21: shared with the several-files path)
from ui.multi_panel import render_multi  # noqa: E402  (Day 21)
from tools.hosting import file_limit_mb, is_hosted, isolation_ok  # noqa: E402  (Day 23a: public host)
from ui.hosted_ui import render_hosted_sidebar  # noqa: E402

st.set_page_config(page_title="Agentic Data Analyst", page_icon="📊", layout="wide")
render_hosted_sidebar()                                   # Day 23a: banner + the visitor's own key (public host only)

ROOT = Path(__file__).resolve().parent.parent
SAMPLE_PATH = ROOT / "sample_data" / "sales_sample.csv"
MESSY_DIR = ROOT / "sample_data" / "messy"
DEMO_DIR = ROOT / "sample_data" / "f1_demo"                 # Day 23b: the small Formula 1 demo set
DEMO_LABEL = "Formula 1 demo set (2021-2022)"
MAX_MB = file_limit_mb(50)
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


def show_chart(df, spec, key):
    """Draw one chart, or a friendly message if it cannot be built."""
    try:
        st.plotly_chart(build_chart(df, spec), key=key)
    except ChartError as e:
        st.warning(f"Could not build this chart: {e}")


st.title("📊 Agentic Data Analyst")
st.caption("Upload a CSV, or pick a built-in file. The '✨ AI dashboard' tab plans a dashboard with AI and builds it after "
           "you approve the plan. The '💬 Chat' tab answers questions about your data and shows the calculation behind "
           "every number. Filters are in the sidebar.")  # Day 20

# ---- choose the data source ----
SOURCES = ["Upload my own CSV", "Built-in sample sales data", DEMO_LABEL] + ([] if is_hosted() else ["Messy test file"])
source = st.sidebar.radio("Data source", SOURCES, index=SOURCES.index(DEMO_LABEL) if is_hosted() else 0,
                              key="data_source")
name, data, batch = None, None, None

if source == DEMO_LABEL:                                    # Day 23b: several linked tables to try for free
    demo_files = sorted(DEMO_DIR.glob("*.csv"))
    if not demo_files:
        st.warning("The demo files are missing. Run: python scripts/build_demo_f1.py <folder with the 14 F1 files>")
        st.stop()
    batch = [(p.name, p.read_bytes()) for p in demo_files]
elif source == "Built-in sample sales data":
    name, data = "sales_sample", SAMPLE_PATH.read_bytes()
elif source == "Messy test file":
    files = sorted(MESSY_DIR.glob("*.csv"))
    if not files:
        st.warning("No messy files found. Run: python sample_data/make_messy_files.py")
        st.stop()
    pick = st.sidebar.selectbox("Pick a file", [f.name for f in files])
    name, data = Path(pick).stem, (MESSY_DIR / pick).read_bytes()
else:
    uploaded = st.file_uploader("Upload one CSV file, or several files that belong together", type=["csv"],
                                accept_multiple_files=True)          # Day 21: several files = several linked tables
    for up in uploaded or []:
        if up.size > MAX_MB * 1024 * 1024:
            st.error(f"{up.name} is larger than {MAX_MB} MB. Please upload a smaller file.")
            st.stop()
    if uploaded and len(uploaded) == 1:
        name, data = Path(uploaded[0].name).stem, uploaded[0].getvalue()
    elif uploaded:
        batch = [(up.name, up.getvalue()) for up in uploaded]

if data is None and batch is None:
    st.info("Upload a CSV above (or several that belong together), or choose a built-in file in the sidebar.")
    st.stop()

flat = None
if batch is not None:                                  # Day 21: several files
    table, multi_key, all_tables, flat = render_multi(batch)
    data = multi_key.encode()
else:
    try:
        table = build_table(name, data)
    except ValueError as e:
        st.error(str(e))
        st.stop()
    except Exception as e:
        st.error(f"Something unexpected went wrong while reading the file: {e}")
        st.stop()
    all_tables = {table["name"]: table}

# Data is kept as a NAMED COLLECTION of tables. With several files: every table, plus the analysed one with its
# linked columns added (named <table>_flat), which is the active table.
if flat is not None and flat.joined:
    st.session_state["tables"] = {**all_tables, f"{table['name']}_flat": table}
    st.session_state["active_table"] = f"{table['name']}_flat"
else:
    st.session_state["tables"] = all_tables
    st.session_state["active_table"] = table["name"]

df, profile = table["df"], table["profile"]
d = profile["dataset"]
st.subheader(f"Table: {table['name']}" + (f" (with columns from {', '.join(l.parent for l, _ in flat.joined)})"
                                          if flat is not None and flat.joined else ""))

# Day 17: sidebar filters. The Overview and the Code sandbox always use all rows; the other tabs use the slice.
base_key = f"{table['name']}:{hashlib.sha1(data).hexdigest()[:12]}"
fdf, filter_label, filter_sig = render_filter_panel(df, profile, base_key)
if len(fdf) == 0:
    st.warning("No rows match the current filters. Widen them (or press 'Clear filters') in the sidebar.")
    st.stop()
fprofile = profile_dataframe(fdf) if filter_label else profile
full_profile = profile

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

# From here on, df and profile mean the FILTERED rows (the Overview above used all rows).
df, profile = fdf, fprofile

# ============ TAB 2: CHART GALLERY (rule-based) ============
with tab_gallery:
    st.caption("A dashboard made by simple fixed rules, with no AI. Compare it with the '✨ AI dashboard' tab. "
               "It uses the rows chosen in the sidebar filters.")
    kpis = compute_kpis(df, profile)
    for col, k in zip(st.columns(len(kpis)), kpis):
        show_kpi(col, k)

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
               "the AI planner produces.")
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
    if is_hosted():                                        # Day 23a: no box where visitors type their own Python
        st.info("The code sandbox tab is switched off on the public demo. The chat and the AI dashboard still run "
                "their AI-written code in the same locked-down sandbox.")
    else:
        st.caption("This is where AI-written code runs (the chat's calculations and the AI dashboard's charts). "
                   "Try the examples, especially the red ones, "
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
    render_ai_dashboard(df, full_profile, data_key=base_key, scope=filter_label)

# ============ TAB: CHAT (questions about the data summary) ============
with tab_chat:
    multi = None
    if len(all_tables) > 1:                                  # Day 22: several files -> the chat may combine them
        from agents.multi_chat import build_multi_chat
        active = st.session_state["active_table"]
        multi = build_multi_chat(active, df, {n: t["df"] for n, t in all_tables.items()},
                                 list(st.session_state.get("links_enabled", [])),
                                 origin={active: table["name"]} if active != table["name"] else {})
    chat_df = df
    calc_live = True
    if is_hosted():                                          # Day 23a: AI code runs here only if the file-system lock works
        _ok, _why = isolation_ok()
        if not _ok:                                          # Day 23c: saved demo calculations still run; new ones do not
            calc_live = False
            st.warning(f"Live calculations are switched off on this host ({_why}). The saved demo answers still work "
                       "with 'Replay only'. With your own key the chat answers from the data summary only.")
    from evals.f1_eval import DEMO_QUESTIONS
    demo_starters = DEMO_QUESTIONS if (source == DEMO_LABEL and chat_df is not None and not filter_label) else None
    render_chat(profile, data_key=base_key + ("|" + filter_sig if filter_sig else ""), df=chat_df, scope=filter_label,
                multi=multi, starters=demo_starters, calc_live=calc_live)
