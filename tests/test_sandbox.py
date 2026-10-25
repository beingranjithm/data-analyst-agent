import glob
import tempfile
from pathlib import Path

import pandas as pd
import pytest

from tools.code_guard import check_code
from tools.sandbox import run_code

SAMPLE = Path(__file__).parent.parent / "sample_data" / "sales_sample.csv"


@pytest.fixture(scope="module")
def df():
    return pd.read_csv(SAMPLE)


# ---------------- 1. good code must work ----------------

def test_table_result_matches_pandas(df):
    r = run_code("result = df.groupby('region')['revenue'].sum().reset_index()", df)
    assert r.ok, r.message()
    expected = df.groupby("region")["revenue"].sum().reset_index()
    assert list(r.table["region"]) == list(expected["region"])
    assert list(r.table["revenue"]) == pytest.approx(list(expected["revenue"]))


def test_scalar_and_printed_output(df):
    r = run_code("print('hello')\nresult = df['revenue'].sum()", df)
    assert r.ok and r.value == pytest.approx(df["revenue"].sum())
    assert "hello" in r.stdout


def test_series_result_becomes_a_table(df):
    r = run_code("result = df.groupby('category')['units'].sum()", df)
    assert r.ok and list(r.table.columns) == ["category", "units"]


def test_plotly_figure_comes_back_even_without_import(df):
    r = run_code("fig = px.bar(df.groupby('region', as_index=False)['revenue'].sum(), x='region', y='revenue')", df)
    assert r.ok, r.message()
    assert r.figure is not None and len(r.figure.data) == 1


def test_explicit_plotly_import_works(df):
    r = run_code("import plotly.express as px\nfig = px.histogram(df, x='revenue')", df)
    assert r.ok and r.figure is not None


def test_several_tables(df):
    other = pd.DataFrame({"a": [1, 2, 3]})
    r = run_code("result = [len(df), len(tables['other']), sorted(tables)]", {"sales": df, "other": other}, active="sales")
    assert r.ok and r.value == [600, 3, ["other", "sales"]]


def test_original_data_is_never_changed(df):
    before = df["revenue"].sum()
    r = run_code("df['revenue'] = 0\nresult = df['revenue'].sum()", df)
    assert r.ok and r.value == 0
    assert df["revenue"].sum() == before


def test_large_tables_are_cut_but_report_total(df):
    r = run_code("result = df", df)
    assert r.ok and len(r.table) == 500 and r.table_total_rows == 600


@pytest.mark.parametrize("code", [
    "result = df.groupby('region').agg(total=('revenue', 'sum'), n=('units', 'count')).reset_index()",
    "result = df.assign(share=df['revenue'] / df['revenue'].sum()).sort_values('share').head()",
    "df['order_date'] = pd.to_datetime(df['order_date'])\nresult = df.set_index('order_date')['revenue'].resample('MS').sum()",
    "result = df.pivot_table(index='region', columns='category', values='revenue', aggfunc='sum')",
    "result = df[df['revenue'] > 1000].shape[0]",
    "result = f\"{df['revenue'].sum():,.2f}\"",
    "result = '{:,}'.format(int(df['units'].sum()))",
    "result = np.where(df['units'] > 10, 'big', 'small')[:3].tolist()",
    "import math\nresult = math.sqrt(df['units'].var())",
    "def top(d, n):\n    return d.nlargest(n, 'revenue')\nresult = top(df, 3)",
    "result = {k: float(v) for k, v in df.groupby('region')['revenue'].mean().items()}",
    "result = df['product'].str.upper().value_counts().to_dict()",
    "result = df['order_date'].str[:7].value_counts().sort_index().head(3)",
    "result = pd.to_numeric(df['units']).describe()",
    "result = df.open.mean()",
    "result = df.groupby('a').apply(lambda g: g['x'].sum() / len(g))",
    "result = pd.cut(df['x'], bins=[0, 10, 20]).value_counts()",
    "fig = go.Figure(go.Bar(x=[1, 2], y=[3, 4]))\nfig.update_layout(title='t')",
    "from plotly.subplots import make_subplots\nfig = make_subplots(rows=1, cols=2)",
    "import datetime as dt\nresult = (dt.date(2025, 1, 1) - dt.date(2024, 1, 1)).days",
    "try:\n    result = 1 / 0\nexcept ZeroDivisionError:\n    result = 0",
])
def test_normal_analysis_code_passes_the_check(code):
    assert check_code(code) == []


def test_normal_analysis_code_actually_runs(df):
    r = run_code("result = df.pivot_table(index='region', columns='category', values='revenue', aggfunc='sum')", df)
    assert r.ok, r.message()


# ---------------- 2. the static check blocks attacks ----------------

BLOCKED = [
    ("import os", "not allowed"),
    ("import subprocess", "not allowed"),
    ("import socket", "not allowed"),
    ("import sys", "not allowed"),
    ("import shutil", "not allowed"),
    ("import pathlib", "not allowed"),
    ("import importlib", "not allowed"),
    ("import requests", "not allowed"),
    ("import ctypes", "not allowed"),
    ("from os import system", "not allowed"),
    ("import plotly.io as pio", "not allowed"),
    ("from plotly import io", "not allowed"),
    ("import pandas.io.common", "not allowed"),
    ("open('/etc/passwd')", "'open'"),
    ("eval('1+1')", "'eval'"),
    ("exec('x = 1')", "'exec'"),
    ("compile('1', 'a', 'exec')", "'compile'"),
    ("__import__('os')", "__import__"),
    ("getattr(df, 'sum')", "'getattr'"),
    ("globals()", "'globals'"),
    ("x = ().__class__.__bases__[0].__subclasses__()", "private attribute"),
    ("df.__class__", "private attribute"),
    ("x = '{0.__class__}'.format(df)", "double underscore"),
    ("pd.read_csv('/etc/passwd')", "read_csv"),
    ("pd.read_pickle('x.pkl')", "read_pickle"),
    ("df.to_csv('out.csv')", "to_csv"),
    ("df.to_pickle('out.pkl')", "to_pickle"),
    ("np.load('x.npy')", "'load'"),
    ("np.save('x.npy', [1])", "'save'"),
    ("pd.io.common.os.system('id')", "not allowed"),
    ("fig.write_html('a.html')", "write_html"),
    ("fig.show()", "'show'"),
    ("df.query('revenue > 5')", "'query'"),
    ("pd.eval('1 + 1')", "'eval'"),
    ("df.eval('revenue * 2')", "'eval'"),
    ("class A:\n    pass", "classes"),
    ("from pandas import read_csv", "read_csv"),
    ("from . import x", "relative"),
    ("import pandas as pd\nimport os as o", "not allowed"),
    # found by red-teaming
    ("np.lib.npyio.DataSource().open('x')", "DataSource"),
    ("pd.ExcelFile('x')", "ExcelFile"),
    ("pd.ExcelWriter('x.xlsx')", "ExcelWriter"),
    ("pd.HDFStore('x.h5')", "HDFStore"),
    ("np.lib.format.open_memmap('x.npy')", "open_memmap"),
    ("import json\nx = json.codecs", "codecs"),
    ("import pickle", "not allowed"),
    ("import tempfile", "not allowed"),
    ("import glob", "not allowed"),
]


@pytest.mark.parametrize("code,expected", BLOCKED, ids=[c[0].split("\n")[0][:40] for c in BLOCKED])
def test_dangerous_code_is_blocked_before_running(code, expected):
    problems = check_code(code)
    assert problems, f"should have been blocked: {code!r}"
    assert any(expected in p for p in problems), problems


@pytest.mark.parametrize("code,expected", [
    ("def f(:", "syntax error"),
    ("", "no code"),
    ("   \n  ", "no code"),
    ("x = 1\n" * 500, "too long"),
])
def test_unusable_code_gets_friendly_messages(code, expected):
    assert any(expected in p for p in check_code(code))


def test_blocked_code_never_starts_a_process(df):
    r = run_code("import os\nresult = os.listdir('/')", df)
    assert r.status == "blocked" and r.seconds == 0.0
    assert "not allowed" in r.message()


# ---------------- 3. inner layers hold even if the check is skipped ----------------

@pytest.mark.parametrize("code,error_type", [
    ("result = open('/etc/passwd').read()", "NameError"),
    ("import os", "ImportError"),
    ("import socket", "ImportError"),
    ("__import__('os')", "ImportError"),
    ("import subprocess", "ImportError"),
])
def test_runtime_layer_stops_attacks_when_check_is_off(df, code, error_type):
    r = run_code(code, df, check=False)
    assert r.status == "error" and r.error["type"] == error_type, r.message()


def test_network_is_blocked_at_runtime(df):
    r = run_code("result = pd.read_csv('http://example.com/data.csv').shape", df, check=False, timeout=20)
    assert r.status == "error"
    assert "blocked" in r.error["message"]


def test_secrets_are_not_passed_to_the_child_process(df, monkeypatch):
    monkeypatch.setenv("GOOGLE_API_KEY", "TOP-SECRET-VALUE")
    r = run_code("result = str(pd.io.common.os.environ.get('GOOGLE_API_KEY'))", df, check=False)
    assert r.ok and r.value == "None"
    r = run_code("result = sorted(pd.io.common.os.environ.keys())", df, check=False)
    assert "GOOGLE_API_KEY" not in r.value


# ---------------- 4. resource limits ----------------

def test_infinite_loop_is_stopped(df):
    r = run_code("while True:\n    pass", df, timeout=3)
    assert r.status == "timeout"
    assert r.seconds < 8
    assert "too long" in r.message()


def test_memory_bomb_is_stopped(df):
    r = run_code("x = [0] * (10 ** 9)", df, memory_mb=512)
    assert r.status == "error" and r.error["type"] == "MemoryError"


def test_numpy_memory_bomb_is_stopped(df):
    r = run_code("x = np.ones(10 ** 10)", df, memory_mb=512)
    assert r.status == "error" and r.error["type"] == "MemoryError"


def test_output_flood_is_cut(df):
    r = run_code("for i in range(200000):\n    print('x' * 100)", df, timeout=20)
    assert r.ok
    assert len(r.stdout) < 21_000 and "output cut" in r.stdout


def test_temporary_files_are_cleaned_up(df):
    pattern = f"{tempfile.gettempdir()}/daa_sandbox_*"
    before = len(glob.glob(pattern))
    run_code("result = 1", df)
    run_code("while True:\n    pass", df, timeout=2)
    assert len(glob.glob(pattern)) == before


# ---------------- 5. errors are reported clearly (for the self-healing loop) ----------------

def test_error_report_has_type_line_and_code(df):
    r = run_code("a = 1\nb = a / 0", df)
    assert r.status == "error"
    assert r.error["type"] == "ZeroDivisionError" and r.error["line"] == 2
    assert "b = a / 0" in r.message()


def test_wrong_column_name_is_reported(df):
    r = run_code("result = df['revenu'].sum()", df)
    assert r.status == "error" and r.error["type"] == "KeyError"
    assert "revenu" in r.message()


def test_good_run_after_a_crash_still_works(df):
    run_code("1 / 0", df)
    assert run_code("result = 2 + 2", df).value == 4


# ---------------- 6. the operating-system lock (Landlock) ----------------

@pytest.fixture(scope="module")
def lock_active(df):
    r = run_code("result = 1", df)
    if r.isolation == "none":
        pytest.skip("Landlock is not available on this computer")
    return r.isolation


def test_lock_reports_itself(df, lock_active):
    assert lock_active.startswith("landlock")


def test_lock_blocks_reading_files_even_if_the_check_is_skipped(df, lock_active, tmp_path):
    secret = tmp_path / "secret.txt"
    secret.write_text("TOP-SECRET")
    code = f"result = np.lib.npyio.DataSource().open({str(secret)!r}).read()"
    r = run_code(code, df, check=False)
    assert r.status == "error" and r.error["type"] == "PermissionError"


def test_lock_blocks_writing_files_even_if_the_check_is_skipped(df, lock_active, tmp_path):
    target = tmp_path / "pwned.npy"
    code = f"x = np.lib.format.open_memmap({str(target)!r}, mode='w+', shape=(2,), dtype='f8')"
    r = run_code(code, df, check=False)
    assert r.status == "error" and r.error["type"] == "PermissionError"
    assert not target.exists()


def test_lock_blocks_listing_directories(df, lock_active):
    r = run_code("result = pd.io.common.os.listdir('/home')", df, check=False)
    assert r.status == "error" and r.error["type"] == "PermissionError"


def test_lock_blocks_running_programs(df, lock_active):
    r = run_code("result = pd.io.common.os.system('echo pwned > /tmp/pwned_by_sandbox.txt')", df, check=False)
    assert not Path("/tmp/pwned_by_sandbox.txt").exists()


def test_lock_does_not_break_normal_work(df, lock_active):
    r = run_code("import plotly.express as px\nfig = px.histogram(df, x='revenue')\nresult = df['revenue'].describe()", df)
    assert r.ok and r.figure is not None and r.table is not None


def test_require_isolation_option_works(df, lock_active):
    assert run_code("result = 1", df, require_isolation=True).ok
