"""One consistent look for every dashboard chart (Day 26).

AI-written chart code sometimes styles a chart its own way: a dark background (#111111, plotly_dark) on an
otherwise white dashboard, or a "rows excluded" note placed at a fixed spot under the plot, where it lands on
top of the x-axis title. These fixes run on the finished figure, after the AI's code has run, so they never
change what is sent to the model (saved demo replies keep matching) and they apply the same way to AI and
rule-based charts.
"""
from __future__ import annotations

import re
from typing import Optional

LIGHT_TEMPLATE = "plotly_white"
NOTE_GAP_WITH_AXIS_TITLE = 64      # px below the plot area: past tick labels AND the axis title
NOTE_GAP_WITHOUT_AXIS_TITLE = 40   # px below the plot area: past tick labels only
NOTE_LINE = 18                     # px between stacked notes
NOTE_COLOR = "#5c6570"

_HEX = re.compile(r"^#([0-9a-f]{3}|[0-9a-f]{6})$", re.I)
_RGB = re.compile(r"^rgba?\(\s*([\d.]+)\s*,\s*([\d.]+)\s*,\s*([\d.]+)", re.I)
_NAMED = {"black": (0, 0, 0), "white": (255, 255, 255)}


def luminance(color) -> Optional[float]:
    """Perceived brightness 0 (black) .. 1 (white) of a CSS colour, or None if it can't be read."""
    if not isinstance(color, str):
        return None
    c = color.strip().lower()
    rgb = None
    if c in _NAMED:
        rgb = _NAMED[c]
    elif _HEX.match(c):
        h = c[1:]
        if len(h) == 3:
            h = "".join(ch * 2 for ch in h)
        rgb = tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))
    else:
        m = _RGB.match(c)
        if m:
            rgb = tuple(float(v) for v in m.groups())
    if rgb is None:
        return None
    r, g, b = rgb
    return (0.299 * r + 0.587 * g + 0.114 * b) / 255


def _is_dark(color) -> bool:
    lum = luminance(color)
    return lum is not None and lum < 0.5


def _is_light(color) -> bool:
    lum = luminance(color)
    return lum is not None and lum > 0.7


def has_dark_background(fig) -> bool:
    lay = fig.layout
    if _is_dark(lay.paper_bgcolor) or _is_dark(lay.plot_bgcolor):
        return True
    tpl = lay.template
    try:
        return _is_dark(tpl.layout.paper_bgcolor) or _is_dark(tpl.layout.plot_bgcolor)
    except AttributeError:
        return False


def _axes(lay):
    for name in dir(lay):
        if re.fullmatch(r"[xy]axis\d*", name):
            ax = getattr(lay, name, None)
            if ax is not None:
                yield ax


def make_light(fig) -> bool:
    """Turn a dark-styled chart into the dashboard's light style. Returns True if anything changed.
    Background, template and any light-on-dark text/grid colours are reset; trace colours are kept."""
    if not has_dark_background(fig):
        return False
    lay = fig.layout
    lay.template = LIGHT_TEMPLATE
    lay.paper_bgcolor = None
    lay.plot_bgcolor = None
    if _is_light(lay.font.color):
        lay.font.color = None
    if lay.title and lay.title.font and _is_light(lay.title.font.color):
        lay.title.font.color = None
    if lay.legend and lay.legend.font and _is_light(lay.legend.font.color):
        lay.legend.font.color = None
    for ax in _axes(lay):
        for prop in ("color", "gridcolor", "linecolor", "zerolinecolor"):
            if getattr(ax, prop, None) is not None:
                setattr(ax, prop, None)
        if ax.tickfont and _is_light(ax.tickfont.color):
            ax.tickfont.color = None
        if ax.title and ax.title.font and _is_light(ax.title.font.color):
            ax.title.font.color = None
    for a in lay.annotations or []:
        if a.font and _is_light(a.font.color):
            a.font.color = None
    return True


def _x_axis_has_title(lay) -> bool:
    try:
        return bool(lay.xaxis.title and lay.xaxis.title.text)
    except AttributeError:
        return False


def place_notes(fig) -> int:
    """Move notes written under the plot (paper-referenced annotations with y < 0) to a spot below the x-axis
    title, stack several notes, and make room for them in the bottom margin. Returns how many were moved."""
    lay = fig.layout
    notes = [a for a in (lay.annotations or [])
             if a.yref == "paper" and isinstance(a.y, (int, float)) and a.y < 0]
    if not notes:
        return 0
    gap = NOTE_GAP_WITH_AXIS_TITLE if _x_axis_has_title(lay) else NOTE_GAP_WITHOUT_AXIS_TITLE
    for i, a in enumerate(sorted(notes, key=lambda n: -n.y)):     # the one nearest the plot stays on top
        a.y = 0
        a.yanchor = "top"
        a.yshift = -(gap + i * NOTE_LINE)
        if a.xref == "paper":
            a.x, a.xanchor = 0, "left"
        if a.font is None or a.font.size is None:
            a.font = dict(size=11, color=NOTE_COLOR)
        a.showarrow = False
    needed = gap + len(notes) * NOTE_LINE + 12
    current = lay.margin.b if lay.margin and lay.margin.b is not None else 0
    if current < needed:
        lay.margin.b = needed
    return len(notes)


def tidy_pie_labels(fig) -> int:
    """Pie/donut slice labels drawn outside the circle pile up on top of the title when slices are tiny.
    Keep labels inside the slices and hide those that don't fit; the legend still names every slice."""
    changed = 0
    for t in fig.data:
        if t.type == "pie" and t.textposition not in ("inside", "none"):
            t.textposition = "inside"
            changed += 1
    if changed:
        fig.layout.uniformtext = dict(minsize=11, mode="hide")
    return changed


def tidy_log_axes(fig) -> int:
    """On a log axis Plotly may label in-between ticks with bare digits ("2, 5, 1000, 5, 2, 100"), which reads
    like nonsense. Label whole powers of ten only (10, 100, 1000) unless the code chose its own ticks."""
    changed = 0
    for ax in _axes(fig.layout):
        if ax.type == "log" and ax.dtick is None and ax.tickvals is None:
            ax.dtick = 1
            changed += 1
    return changed


def same_title_font(fig) -> bool:
    """Some AI code sets its own title font (e.g. Arial): drop it so every chart title looks the same."""
    t = fig.layout.title
    if t and t.font and t.font.family:
        t.font.family = None
        return True
    return False


TITLE_WRAP = 52                     # characters per line before a title is wrapped (half-width dashboard card)


def wrap_title(fig, width: int = TITLE_WRAP) -> bool:
    """Plotly never wraps a title, so a long one is cut off at the card's edge. Break it into lines at spaces."""
    t = fig.layout.title
    text = t.text if t else None
    if not text or "<br" in text or len(text) <= width:
        return False
    lines, line = [], ""
    for word in text.split():
        if line and len(line) + 1 + len(word) > width:
            lines.append(line)
            line = word
        else:
            line = f"{line} {word}".strip()
    lines.append(line)
    t.text = "<br>".join(lines)
    return True


def tidy_figure(fig):
    """Apply every style fix. Never raises: a style fix must never cost a chart."""
    for fix in (make_light, place_notes, tidy_pie_labels, tidy_log_axes, same_title_font):
        try:
            fix(fig)
        except Exception:
            pass
    return fig
