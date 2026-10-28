"""Day 21b: the panel where the person checks which tables were found, how they link, and which one to analyse."""
import pandas as pd
import streamlit as st

from tools.multi import table_order


def render_links_panel(frames: dict, links: list, sig: str):
    """frames = {name: DataFrame}. Returns (links to use, table to analyse). Widget keys contain `sig`, so a new
    set of files starts with every link on and the default table."""
    order = table_order(frames, links)
    joinable = [l for l in links if l.kind == "many-to-one"]
    others = [l for l in links if l.kind != "many-to-one"]
    with st.expander(f"🔗 {len(frames)} tables and {len(joinable)} links found", expanded=True):
        st.caption("The app compared the columns of your files. A link means: every value in the first column also "
                   "exists in the second, and the second has no repeats (for example results.driverId → drivers.driverId). "
                   "Untick a link that is wrong. Rows are never added or removed.")
        st.dataframe(pd.DataFrame({"table": order, "rows": [len(frames[t]) for t in order],
                                   "columns": [frames[t].shape[1] for t in order]}),
                     hide_index=True, width="stretch")
        enabled = []
        cols = st.columns(2)
        for i, l in enumerate(joinable):
            label = f"{l.text()}  ({l.containment:.0%} of values found)"
            if cols[i % 2].checkbox(label, value=True, key=f"lnk|{sig}|{l.text()}"):
                enabled.append(l)
        if others:
            st.caption("Also found, but not used for joining (both sides have no repeats): "
                       + "; ".join(l.text() for l in others))
        focus = st.selectbox("Analyse this table", order, key=f"focus|{sig}",       # the first one is the most useful
                             format_func=lambda t: f"{t} ({len(frames[t]):,} rows)",
                             help="Its rows stay exactly as they are. The columns of the tables it links to are added.")
    return enabled, focus
