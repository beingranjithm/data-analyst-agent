"""Day 21b: several uploaded CSV files treated as one data set. Returns the table the rest of the app works on."""
import streamlit as st

from tools.multi import build_flat, build_table, check_limits, files_signature, links_signature, load_tables
from tools.profiler import profile_dataframe
from tools.relations import find_links
from ui.links_panel import render_links_panel
from ui.memo import memo


def render_multi(batch: list):
    """batch = [(filename, bytes)], two or more files. Returns (table dict, key text, {name: table dict}, Flat or None).
    Stops the page (st.stop) when nothing usable was uploaded."""
    problem = check_limits(batch)
    if problem:
        st.error(problem)
        st.stop()
    sig = files_signature(batch)
    with st.spinner(f"Reading {len(batch)} files ... (big files take a few seconds, but only once)"):
        tables, errors = memo(("files", sig), lambda: load_tables(batch, build_table))
    for filename, message in errors:
        st.error(f"**{filename}** was left out: {message}")
    if not tables:
        st.stop()
    frames = {n: t["df"] for n, t in tables.items()}
    if len(tables) == 1:                                             # only one file could be read: the normal single-table view
        (name, table), = tables.items()
        return table, f"{name}:{sig}", tables, None
    links = memo(("links", sig), lambda: find_links(frames))
    enabled, focus = render_links_panel(frames, links, sig)
    key = f"{sig}|{focus}|{links_signature(enabled)}"
    with st.spinner(f"Preparing {focus} ..."):
        table, flat = memo(("flat", key), lambda: build_flat(frames, focus, enabled, profile_dataframe, raw_tables=tables))
    if flat.joined:
        st.success(f"{focus}: {flat.rows_after:,} rows (unchanged), {flat.df.shape[1]} columns after adding columns from "
                   + ", ".join(l.parent for l, _ in flat.joined) + ".")
    for t, why in flat.skipped:
        st.warning(f"Did not add **{t}**: {why}")
    return table, key, tables, flat
