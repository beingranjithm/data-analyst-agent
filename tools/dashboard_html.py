"""Turns a finished Dashboard into one self-contained HTML page (KPI cards, charts, skipped-chart notes)."""
import html
from pathlib import Path


def _number(value) -> str:
    return "n/a" if value is None else format(value, ",.2f")


def _delta_badge(k, comparison_label: str) -> str:
    """A small colored "+12.3% vs previous 30 days" badge, or "" when there's no usable delta."""
    if k.delta_pct is None or k.delta_direction is None:
        return ""
    text = f"{k.delta_pct:+.1f}% {comparison_label}".strip()
    if k.delta_direction == "flat" or k.delta_good is None:
        color = "#666"
    else:
        color = "#1a7f37" if k.delta_good else "#b00"
    return f"<div style='color:{color};font-size:13px;margin-top:2px'>{html.escape(text)}</div>"


_CARD_COLORS = {"Win": "#1a7f37", "Risk": "#b42318", "Watch": "#b54708"}
_CARD_ICONS = {"Win": "&#8599;&#65039;", "Risk": "&#9888;&#65039;", "Watch": "&#128065;&#65039;"}


def _change_card_column(card) -> str:
    color = _CARD_COLORS.get(card.tag, "#555")
    icon = _CARD_ICONS.get(card.tag, "&#8226;")
    return (f"<div style='padding:4px 14px'>"
            f"<div style='color:{color};font-weight:600;margin-bottom:4px'>{icon} {html.escape(card.tag)}</div>"
            f"<div style='color:#333;font-size:14px'>{html.escape(card.text)}</div></div>")


def _change_cards_section(cards) -> str:
    """One bordered card holding every "what changed" entry side by side, one column each."""
    cols = "".join(_change_card_column(c) for c in cards)
    return (f"<div style='border:1px solid #ddd;border-radius:10px;padding:14px 4px;margin-bottom:20px'>"
            f"<div style='padding:0 14px;font-weight:600;margin-bottom:8px'>&#128161; What changed</div>"
            f"<div style='display:grid;grid-template-columns:repeat({len(cards)}, 1fr)'>{cols}</div></div>")


def _kpi_card(k, comparison_label: str, is_star: bool) -> str:
    esc = html.escape
    label = ("⭐ " if is_star else "") + esc(k.label)
    return (f"<div style='border:1px solid #ccc;border-radius:8px;padding:12px 18px' title='{esc(k.note)}'>"
            f"<div style='color:#666'>{label}</div>"
            f"<div style='font-size:26px'>{_number(k.value)}</div>{_delta_badge(k, comparison_label)}</div>")


def _kpi_section(kpis, comparison_label: str, north_star) -> str:
    """Every KPI in one row of equal, bordered tiles (a CSS grid, so it never wraps mid-row)."""
    cards = "".join(_kpi_card(k, comparison_label, bool(north_star) and k.label == north_star) for k in kpis)
    return (f"<div style='display:grid;grid-template-columns:repeat({len(kpis)}, 1fr);"
            f"gap:16px;margin-bottom:20px'>{cards}</div>")


def render_dashboard_html(d) -> str:
    esc = html.escape
    parts = [f"<h1>{esc(d.title)}</h1>"]
    if getattr(d, "coverage", ""):
        parts.append(f"<p style='color:#666'>{esc(d.coverage)}</p>")
    if d.summary:
        parts.append(f"<p>{esc(d.summary)}</p>")
    if getattr(d, "data_health", ""):
        parts.append(f"<p style='color:#055160;background:#e8f4f8;padding:8px 12px;"
                      f"border-radius:6px'>{esc(d.data_health)}</p>")
    parts.append(f"<p><i>mode: {esc(d.mode)}, {d.rows} rows</i></p>")
    for note in d.notes:
        parts.append(f"<p style='color:#8a6d00'>Note: {esc(note)}</p>")
    comparison_label = getattr(d, "comparison_label", "")
    north_star = getattr(d, "north_star", None)
    parts.append(_kpi_section(d.kpis, comparison_label, north_star))
    first = True
    for c in d.ok_charts:
        chart_html = c.figure.to_html(full_html=False, include_plotlyjs=True if first else False)
        first = False
        warnings = "".join(f"<p style='color:#8a6d00'>Warning on '{esc(c.spec.title)}': {esc(w)}</p>"
                           for w in c.warnings)
        parts.append(f"<div style='border:1px solid #ccc;border-radius:8px;padding:12px;"
                      f"margin-bottom:16px'>{chart_html}{warnings}</div>")
    for c in d.failed_charts:
        parts.append(f"<p style='color:#b00'>Chart skipped: {esc(c.spec.title)}: {esc(c.message[:300])}</p>")
    if getattr(d, "change_cards", None):                      # "what changed" sits below the charts
        parts.append(_change_cards_section(d.change_cards))
    return ("<html><head><meta charset='utf-8'></head><body style='font-family:sans-serif;max-width:1000px;margin:auto'>"
            + "".join(parts) + "</body></html>")


def write_dashboard_html(d, path) -> Path:
    path = Path(path)
    path.write_text(render_dashboard_html(d), encoding="utf-8")
    return path
