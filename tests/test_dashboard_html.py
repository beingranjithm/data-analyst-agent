import pandas as pd
import plotly.express as px

from agents.dashboard import ChartOutcome, Dashboard
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
