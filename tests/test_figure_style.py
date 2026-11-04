"""Day 26: one light style for every dashboard chart, and notes that don't sit on the axis title."""
import plotly.express as px
import plotly.graph_objects as go
import pytest

from tools.figure_style import (NOTE_GAP_WITH_AXIS_TITLE, NOTE_GAP_WITHOUT_AXIS_TITLE, has_dark_background,
                                luminance, make_light, place_notes, same_title_font, tidy_figure, tidy_log_axes,
                                tidy_pie_labels, wrap_title)


def _line():
    return px.line(x=[1, 2, 3], y=[3, 1, 2])


@pytest.mark.parametrize("color, expected", [("#000000", 0.0), ("#fff", 1.0), ("white", 1.0),
                                             ("rgb(255, 255, 255)", 1.0), ("rgba(0,0,0,0.5)", 0.0)])
def test_luminance_reads_common_css_colours(color, expected):
    assert luminance(color) == pytest.approx(expected)


def test_luminance_of_something_unreadable_is_none():
    assert luminance("steelblue") is None and luminance(None) is None


def test_ai_dark_background_is_made_light():
    # the Olist demo's AI line chart set paper/plot backgrounds to #111111 with light text
    fig = _line()
    fig.update_layout(paper_bgcolor="#111111", plot_bgcolor="#111111", font_color="#eeeeee",
                      title=dict(text="t", font=dict(color="white")))
    fig.update_xaxes(gridcolor="#333333", tickfont=dict(color="#dddddd"))
    fig.update_traces(line_color="#00cc96")
    assert make_light(fig) is True
    lay = fig.layout
    assert lay.paper_bgcolor is None and lay.plot_bgcolor is None
    assert lay.font.color is None and lay.title.font.color is None
    assert lay.xaxis.gridcolor is None and lay.xaxis.tickfont.color is None
    assert not has_dark_background(fig)
    assert fig.data[0].line.color == "#00cc96"          # trace colours are kept


def test_plotly_dark_template_is_made_light():
    fig = _line()
    fig.update_layout(template="plotly_dark")
    assert has_dark_background(fig)
    make_light(fig)
    assert not has_dark_background(fig)


def test_a_light_chart_is_left_alone():
    fig = _line()
    fig.update_layout(paper_bgcolor="#fafafa", font_color="#222222")
    assert make_light(fig) is False
    assert fig.layout.paper_bgcolor == "#fafafa" and fig.layout.font.color == "#222222"


def _with_note(y=-0.15, axis_title="Total Revenue", margin_b=60):
    fig = px.bar(x=[3, 2, 1], y=["a", "b", "c"], orientation="h")
    fig.update_layout(xaxis_title=axis_title, margin=dict(b=margin_b))
    fig.add_annotation(text="Note: 122 rows excluded", xref="paper", yref="paper", x=0, y=y, showarrow=False)
    return fig


def test_note_under_the_plot_moves_below_the_axis_title_with_room_made_for_it():
    # the Olist demo's bar chart note sat at y=-0.15 with a 60px margin, on top of "Total Revenue"
    fig = _with_note()
    assert place_notes(fig) == 1
    a = fig.layout.annotations[0]
    assert a.y == 0 and a.yanchor == "top" and a.yshift == -NOTE_GAP_WITH_AXIS_TITLE
    assert a.xanchor == "left" and a.font.size == 11
    assert fig.layout.margin.b >= NOTE_GAP_WITH_AXIS_TITLE + 18


def test_without_an_axis_title_the_note_sits_closer():
    fig = _with_note(axis_title="")
    place_notes(fig)
    assert fig.layout.annotations[0].yshift == -NOTE_GAP_WITHOUT_AXIS_TITLE


def test_several_notes_are_stacked_not_drawn_on_top_of_each_other():
    fig = _with_note(y=-0.15)
    fig.add_annotation(text="Second note", xref="paper", yref="paper", x=0, y=-0.25, showarrow=False)
    assert place_notes(fig) == 2
    shifts = sorted(a.yshift for a in fig.layout.annotations)
    assert shifts[1] - shifts[0] >= 18


def test_a_big_bottom_margin_is_not_shrunk():
    fig = _with_note(margin_b=200)
    place_notes(fig)
    assert fig.layout.margin.b == 200


def test_notes_inside_the_plot_are_not_moved():
    fig = _with_note(y=0.9)
    assert place_notes(fig) == 0
    assert fig.layout.annotations[0].y == 0.9


def test_tidy_figure_never_raises():
    class Broken:
        layout = None
    assert tidy_figure(Broken()) is not None


def test_tidy_figure_on_a_plain_chart_changes_nothing_visible():
    fig = go.Figure(go.Bar(x=["a"], y=[1]))
    before = fig.to_dict()
    tidy_figure(fig)
    assert fig.to_dict() == before


def test_the_dashboard_finishing_step_applies_the_style_fixes():
    import pandas as pd
    from agents.dashboard import _finish
    from tools.plan_schema import ChartSpec
    df = pd.DataFrame({"d": pd.date_range("2026-01-01", periods=6, freq="D"), "price": [1.0, 2, 9, 2, 1, 2]})
    fig = px.line(df, x="d", y="price")
    fig.update_layout(paper_bgcolor="#111111", plot_bgcolor="#111111", xaxis_title="Day")
    fig.add_annotation(text="Note", xref="paper", yref="paper", x=0, y=-0.2, showarrow=False)
    chart = ChartSpec(type="line", title="Daily Sales Revenue Trend", time="d", measure="price", agg="sum")
    fig, problems, _ = _finish(fig, chart, df)
    assert not problems
    assert not has_dark_background(fig)
    assert fig.layout.annotations[0].yshift == -NOTE_GAP_WITH_AXIS_TITLE
    assert fig.layout.title.text == "Total price peaked on 03 Jan 2026 (9)"


def test_donut_labels_go_inside_and_tiny_ones_hide_instead_of_covering_the_title():
    # the Olist donut drew "0.889% invoiced", "0.45% canceled" ... outside, on top of the title
    fig = go.Figure(go.Pie(labels=["delivered", "shipped", "invoiced"], values=[978, 13, 9], hole=0.4,
                           textposition="outside"))
    assert tidy_pie_labels(fig) == 1
    assert fig.data[0].textposition == "inside"
    assert fig.layout.uniformtext.mode == "hide" and fig.layout.uniformtext.minsize == 11


def test_log_axis_gets_whole_power_of_ten_ticks():
    fig = px.box(x=["a", "b"] * 5, y=[5, 50, 500, 5000, 20, 2, 9, 90, 900, 30], log_y=True)
    assert tidy_log_axes(fig) == 1 and fig.layout.yaxis.dtick == 1


def test_log_axis_with_ticks_chosen_by_the_code_is_left_alone():
    fig = px.box(x=["a", "b"] * 5, y=[5, 50, 500, 5000, 20, 2, 9, 90, 900, 30], log_y=True)
    fig.update_yaxes(tickvals=[1, 10, 100])
    assert tidy_log_axes(fig) == 0


def test_a_custom_title_font_is_dropped():
    fig = _line()
    fig.update_layout(title=dict(text="t", font=dict(family="Arial", size=18)))
    assert same_title_font(fig) is True
    assert fig.layout.title.font.family is None and fig.layout.title.font.size == 18


def test_long_titles_wrap_at_spaces_and_short_ones_do_not():
    fig = _line()
    fig.update_layout(title_text="Moderate positive relationship between product weight g and freight value (r=0.66)")
    assert wrap_title(fig) is True
    lines = fig.layout.title.text.split("<br>")
    assert len(lines) == 2 and all(len(l) <= 52 for l in lines)
    assert " ".join(lines) == "Moderate positive relationship between product weight g and freight value (r=0.66)"
    fig.update_layout(title_text="Short title")
    assert wrap_title(fig) is False and fig.layout.title.text == "Short title"
