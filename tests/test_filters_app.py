"""Day 17 in the WHOLE app: which tabs use the filtered rows and which use all of them."""
import sys
from pathlib import Path

from streamlit.testing.v1 import AppTest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import tools.charts as charts_mod  # noqa: E402
import tools.sandbox as sandbox_mod  # noqa: E402
from ui import dashboard_tab  # noqa: E402

APP = str(Path(__file__).resolve().parent.parent / "app" / "streamlit_app.py")


def sample_app():
    at = AppTest.from_file(APP, default_timeout=90)
    at.run()
    at.sidebar.radio[0].set_value("Built-in sample sales data").run()
    assert not at.exception and not any("Filter on" in i.value for i in at.sidebar.info)     # nothing is carried over
    return at


def test_the_app_without_any_filter_behaves_as_before(monkeypatch):
    rows = []
    real = charts_mod.compute_kpis
    monkeypatch.setattr(charts_mod, "compute_kpis", lambda df, profile: (rows.append(len(df)), real(df, profile))[1])
    at = sample_app()
    total = int(at.metric[0].value.replace(",", ""))
    assert rows and rows[-1] == total and not any("Filter on" in i.value for i in at.sidebar.info)


def test_the_gallery_and_the_ai_dashboard_use_the_filtered_rows_but_the_overview_and_sandbox_use_all(monkeypatch):
    kpi_rows, dash = [], []
    real = charts_mod.compute_kpis
    monkeypatch.setattr(charts_mod, "compute_kpis", lambda df, profile: (kpi_rows.append(len(df)), real(df, profile))[1])
    monkeypatch.setattr(dashboard_tab, "default_session_factory",
                        lambda df, profile, settings, progress: dash.append((len(df), profile["dataset"]["rows"])) or None)
    at = sample_app()
    total = int(at.metric[0].value.replace(",", ""))
    box = at.sidebar.multiselect[0]
    dropped = list(box.options)[0]
    box.unselect(dropped).run()
    assert not at.exception
    cut = kpi_rows[-1]
    assert 0 < cut < total                                                                  # the gallery sees fewer rows
    assert at.metric[0].value == f"{total:,}"                                               # the Overview still shows all
    assert "Filter on" in at.sidebar.info[0].value and f"{cut:,} of {total:,} rows" in at.sidebar.info[0].value
    tables = at.session_state["tables"]
    assert len(next(iter(tables.values()))["df"]) == total                                  # the sandbox still gets all rows
    assert any(f"{cut:,} of {total:,} rows" in i.value and "dashboard will be built" in i.value for i in at.info)


def test_the_ai_dashboard_plans_from_the_full_summary_but_builds_on_the_filtered_rows(monkeypatch):
    seen = []

    class Stop(Exception):
        pass

    def factory(df, profile, settings, progress):
        seen.append((len(df), profile["dataset"]["rows"]))
        raise Stop()
    monkeypatch.setattr(dashboard_tab, "default_session_factory", factory)
    monkeypatch.setattr(dashboard_tab, "api_key_available", lambda: True)
    at = sample_app()
    total = int(at.metric[0].value.replace(",", ""))
    box = at.sidebar.multiselect[0]
    box.unselect(list(box.options)[0]).run()
    at.button(key="ai_plan").click().run()
    assert seen and seen[-1][0] < total and seen[-1][1] == total                           # rows: filtered, summary: full


def test_a_filter_that_matches_nothing_gives_a_clear_message_instead_of_a_crash():
    at = sample_app()
    box = at.sidebar.multiselect[0]
    for v in list(box.options):
        box = at.sidebar.multiselect[0]
        box.unselect(v)
    box.run()
    assert not at.exception
    assert any("No rows match the current filters" in w.value for w in at.warning)
    at.sidebar.button(key="flt_clear").click().run()
    assert not at.exception and not any("No rows match" in w.value for w in at.warning)

