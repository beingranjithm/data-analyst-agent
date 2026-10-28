"""Day 20: one KPI card. st.metric cuts long values ('2025-01-01 to 2025-12-31' became '2025-01-01 t...'), so long
values are written out in full instead. The text is escaped: a value can come from the data."""
from __future__ import annotations

import html

import streamlit as st

MAX_METRIC_CHARS = 12


def kpi_html(label, value) -> str:
    return (f"<div style='font-size:0.85rem;opacity:0.7'>{html.escape(str(label))}</div>"
            f"<div style='font-size:1.4rem;font-weight:600;line-height:1.3'>{html.escape(str(value))}</div>")


def show_kpi(col, k: dict) -> None:
    """Short values use the normal st.metric; long ones are shown in full."""
    if len(str(k["value"])) <= MAX_METRIC_CHARS:
        col.metric(k["label"], k["value"])
    else:
        col.markdown(kpi_html(k["label"], k["value"]), unsafe_allow_html=True)
