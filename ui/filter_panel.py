"""The sidebar filter panel (Day 17). One place to choose a slice; the dashboard, gallery, chart builder and
chat all use the same slice. Widget keys contain the data key, so a new file starts with no filters."""
import streamlit as st

from tools.filters import apply_filters, describe_filters, filter_options, is_active, signature

PREFIX = "flt_"


def render_filter_panel(df, profile, data_key: str):
    """Draws the panel. Returns (filtered_df, label, sig). label/sig are "" when nothing is filtered."""
    options = filter_options(df, profile)
    if not options["values"] and not options["date"]:
        return df, "", ""
    chosen = {"values": {}, "date": None}
    with st.sidebar.expander("🔎 Filters", expanded=False):
        st.caption("Applies to the AI dashboard, chart gallery, chart builder and chat. "
                   "The Overview and Code sandbox tabs always use all rows.")
        for col, opts in options["values"].items():
            chosen["values"][col] = st.multiselect(col, opts, default=opts, key=f"{PREFIX}{data_key}:{col}")
        if options["date"]:
            col, first, last = options["date"]
            picked = st.date_input(col, value=(first, last), min_value=first, max_value=last,
                                   key=f"{PREFIX}{data_key}:{col}:dates")
            if isinstance(picked, (tuple, list)) and len(picked) == 2:       # one date = still choosing
                chosen["date"] = (picked[0], picked[1])
        if st.button("Clear filters", key=f"{PREFIX}clear"):
            for k in [k for k in st.session_state.keys() if k.startswith(PREFIX) and k != f"{PREFIX}clear"]:
                del st.session_state[k]
            st.rerun()
    out = apply_filters(df, options, chosen)
    label = describe_filters(options, chosen, len(out), len(df))
    if label:
        st.sidebar.info("🔎 Filter on: " + label)
    return out, label, signature(options, chosen)
