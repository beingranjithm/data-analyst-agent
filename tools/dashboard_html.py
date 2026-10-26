"""Turns a finished Dashboard into one self-contained HTML page (KPI cards, charts, skipped-chart notes)."""
import html
from pathlib import Path


def _number(value) -> str:
    return "n/a" if value is None else format(value, ",.2f")


def render_dashboard_html(d) -> str:
    esc = html.escape
    parts = [f"<h1>{esc(d.title)}</h1>"]
    if d.summary:
        parts.append(f"<p>{esc(d.summary)}</p>")
    parts.append(f"<p><i>mode: {esc(d.mode)}, {d.rows} rows</i></p>")
    for note in d.notes:
        parts.append(f"<p style='color:#8a6d00'>Note: {esc(note)}</p>")
    cards = "".join(
        f"<div style='border:1px solid #ccc;border-radius:8px;padding:12px 18px' title='{esc(k.note)}'>"
        f"<div style='color:#666'>{esc(k.label)}</div><div style='font-size:26px'>{_number(k.value)}</div></div>"
        for k in d.kpis)
    parts.append(f"<div style='display:flex;gap:16px;flex-wrap:wrap'>{cards}</div>")
    first = True
    for c in d.ok_charts:
        parts.append(c.figure.to_html(full_html=False, include_plotlyjs=True if first else False))
        first = False
        for w in c.warnings:
            parts.append(f"<p style='color:#8a6d00'>Warning on '{esc(c.spec.title)}': {esc(w)}</p>")
    for c in d.failed_charts:
        parts.append(f"<p style='color:#b00'>Chart skipped: {esc(c.spec.title)}: {esc(c.message[:300])}</p>")
    return ("<html><head><meta charset='utf-8'></head><body style='font-family:sans-serif;max-width:1000px;margin:auto'>"
            + "".join(parts) + "</body></html>")


def write_dashboard_html(d, path) -> Path:
    path = Path(path)
    path.write_text(render_dashboard_html(d), encoding="utf-8")
    return path
