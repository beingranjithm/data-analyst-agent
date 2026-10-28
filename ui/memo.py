"""Day 21b: a small per-session memory. Streamlit runs the whole page again on every click, and reading, cleaning and
profiling a big file takes seconds. `memo` keeps the result of such a step in the person's session (no copying), keyed
by what went in. Only the most recent few results are kept, so memory stays bounded."""
import streamlit as st

MAX_ENTRIES = 12


def memo(key, make):
    store = st.session_state.setdefault("_memo", {})
    if key in store:
        store[key] = store.pop(key)                       # most recently used goes last
        return store[key]
    value = make()
    store[key] = value
    while len(store) > MAX_ENTRIES:
        store.pop(next(iter(store)))                      # drop the oldest
    return value
