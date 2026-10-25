"""Figure validator tests. Figures go through to_json/from_json first, like the sandbox's output."""
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio

from tools.figure_checks import check_figure, decode


def rt(fig):
    return pio.from_json(fig.to_json())


def test_good_chart_has_no_problems_or_warnings():
    df = pd.DataFrame({"r": ["a", "b", "c"], "v": [1.0, 2.0, 3.0]})
    c = check_figure(rt(px.bar(df, x="r", y="v", title="T")))
    assert c.ok and not c.warnings and not c.missing_title


def test_empty_figure_and_empty_data():
    assert not check_figure(go.Figure()).ok
    df = pd.DataFrame({"r": ["a"], "v": [1.0]})
    c = check_figure(rt(px.bar(df[df["v"] > 99], x="r", y="v")))
    assert not c.ok and "no data points" in c.problems[0]


def test_all_missing_values_block():
    df = pd.DataFrame({"r": ["a", "b"], "v": [float("nan"), float("nan")]})
    c = check_figure(rt(px.bar(df, x="r", y="v")))
    assert not c.ok and "missing" in c.problems[0]


def test_all_zero_values_only_warn():
    df = pd.DataFrame({"r": ["a", "b"], "v": [0.0, 0.0]})
    c = check_figure(rt(px.bar(df, x="r", y="v")))
    assert c.ok and any("zero" in w for w in c.warnings)


def test_single_point_warns_and_missing_title_is_flagged():
    df = pd.DataFrame({"r": ["a"], "v": [5.0]})
    c = check_figure(rt(px.bar(df, x="r", y="v")))
    assert c.ok and any("one data point" in w for w in c.warnings) and c.missing_title


def test_pie_heatmap_and_histogram_are_understood():
    df = pd.DataFrame({"r": ["a", "b"], "v": [1.0, 3.0]})
    assert check_figure(rt(px.pie(df, names="r", values="v", title="P"))).ok
    assert check_figure(rt(px.imshow(np.array([[1.0, 2.0], [3.0, 4.0]]), title="H"))).ok
    assert check_figure(rt(px.histogram(df, x="v", title="Hi"))).ok
    zeros = pd.DataFrame({"r": ["a", "b"], "v": [0.0, 0.0]})
    assert any("zero" in w for w in check_figure(rt(px.pie(zeros, names="r", values="v"))).warnings)


def test_decode_handles_plain_and_compressed_arrays():
    assert list(decode((1, 2, 3))) == [1, 2, 3]
    assert decode(None) is None and decode({"unknown": 1}) is None
    fig = rt(px.bar(pd.DataFrame({"r": ["a", "b"], "v": [1.5, 2.5]}), x="r", y="v"))
    assert list(decode(fig.data[0].y)) == [1.5, 2.5]
