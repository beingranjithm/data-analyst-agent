import pandas as pd
import plotly.express as px

from agents.dashboard import ChartOutcome, Dashboard
from tools.change_cards import ChangeCard
from tools.dashboard_html import render_dashboard_html, write_dashboard_html
from tools.plan_kpis import KpiValue
from tools.plan_schema import ChartSpec


def dashboard():
    spec = ChartSpec(type="bar", title="Bar chart", dimension="r", measure="v")
    fig = px.bar(pd.DataFrame({"r": ["a", "b"], "v": [1.0, 2.0]}), x="r", y="v", title="Bar chart")
    return Dashboard(
        title="Sales <b>2025</b>", summary="A summary", rows=3, mode="ai",
        kpis=[KpiValue("Revenue", "sum", "revenue", 1234.5, "2 rows with a missing value left out"),
              KpiValue("Broken", "sum", "x", None, "column 'x' not found")],
        charts=[ChartOutcome(spec, True, fig, "ai", 1, "code", "ok", "", ["only one data point"]),
                ChartOutcome(spec, False, None, "", 3, None, "AI: gave up | Rule-based: <script>boom</script>")],
        notes=["Used the rule-based gallery"])


def test_page_contains_everything_and_escapes_text():
    page = render_dashboard_html(dashboard())
    assert "Sales &lt;b&gt;2025&lt;/b&gt;" in page and "<b>2025</b>" not in page
    assert "1,234.50" in page and "n/a" in page and "A summary" in page and "3 rows" in page
    assert "Note: Used the rule-based gallery" in page and "Warning on" in page and "only one data point" in page
    assert "Chart skipped: Bar chart" in page and "&lt;script&gt;boom" in page and "<script>boom" not in page
    assert "Plotly.newPlot" in page


def test_page_is_written_as_utf8(tmp_path):
    d = dashboard()
    d.title = "Revenue € – 2025"
    out = write_dashboard_html(d, tmp_path / "page.html")
    assert "Revenue € – 2025" in out.read_text(encoding="utf-8")


def test_north_star_kpi_gets_a_star():
    d = dashboard()
    d.north_star = "Revenue"
    page = render_dashboard_html(d)
    assert "⭐ Revenue" in page
    assert "⭐ Broken" not in page               # only the north-star KPI gets the star


def test_all_kpis_render_in_a_single_row_grid():
    page = render_dashboard_html(dashboard())
    assert f"grid-template-columns:repeat(2, 1fr)" in page    # two KPIs -> one row of two equal columns


def test_kpi_cards_and_charts_are_bordered():
    page = render_dashboard_html(dashboard())
    # the chart sits in its own bordered div, separate from (and after) the two bordered KPI cards
    chart_div = page[page.index("<div style='border:1px solid #ccc;border-radius:8px;padding:12px;"):
                     page.index("Plotly.newPlot")]
    assert "border:1px solid #ccc" in chart_div
    assert page.count("border:1px solid #ccc") >= 3           # 2 KPI cards + 1 chart container


def test_good_delta_is_shown_in_green_and_bad_in_red():
    d = dashboard()
    d.comparison_label = "vs previous 30 days"
    d.kpis[0].delta_pct, d.kpis[0].delta_direction, d.kpis[0].delta_good = 12.3, "up", True
    d.kpis[1].delta_pct, d.kpis[1].delta_direction, d.kpis[1].delta_good = -8.0, "down", False
    page = render_dashboard_html(d)
    assert "+12.3% vs previous 30 days" in page and "#1a7f37" in page
    assert "-8.0% vs previous 30 days" in page and "#b00" in page


def test_flat_delta_is_shown_in_neutral_gray():
    d = dashboard()
    d.comparison_label = "vs previous 30 days"
    d.kpis[0].delta_pct, d.kpis[0].delta_direction, d.kpis[0].delta_good = 0.3, "flat", None
    page = render_dashboard_html(d)
    assert "+0.3% vs previous 30 days" in page and "color:#666;font-size:13px" in page


def test_no_delta_badge_when_there_is_no_usable_comparison():
    page = render_dashboard_html(dashboard())
    assert "margin-top:2px" not in page          # the delta-badge div is only emitted when there's a delta


def test_change_cards_render_as_one_bordered_row_below_the_charts():
    d = dashboard()
    d.change_cards = [ChangeCard("Revenue", "Win", "Revenue rose 23.4% vs previous 30 days.", 23.4, "up"),
                      ChangeCard("Costs", "Risk", "Costs rose 15.0% vs previous 30 days.", 15.0, "up"),
                      ChangeCard("Orders", "Watch", "Orders stayed flat vs previous 30 days.", 0.4, "flat")]
    page = render_dashboard_html(d)
    assert "What changed" in page
    assert "grid-template-columns:repeat(3, 1fr)" in page   # 3 change cards -> one row of three columns
    assert "#1a7f37" in page and "Win" in page and "Revenue rose 23.4%" in page
    assert "#b42318" in page and "Risk" in page and "Costs rose 15.0%" in page
    assert "#b54708" in page and "Watch" in page and "Orders stayed flat" in page
    # each card's own column pairs the right color with the right tag (not just present anywhere on the page)
    win_div = page[page.index("Win</div>") - 60: page.index("Win</div>")]
    risk_div = page[page.index("Risk</div>") - 60: page.index("Risk</div>")]
    assert "#1a7f37" in win_div and "#b42318" in risk_div
    # "what changed" appears after the KPI grid and after the chart, not before either
    assert page.index("grid-template-columns:repeat(2, 1fr)") < page.index("What changed")
    assert page.index("Plotly.newPlot") < page.index("What changed")


def test_no_change_cards_section_when_there_are_none():
    page = render_dashboard_html(dashboard())
    assert "What changed" not in page
