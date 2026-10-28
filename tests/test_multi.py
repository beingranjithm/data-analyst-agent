"""Day 21b: several CSV files as one data set (tools/multi.py, the loader's \\N, and the real app with uploads)."""
import io
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from tools.loader import load_csv_bytes  # noqa: E402
from tools.multi import (MAX_FILES, build_flat, build_table, calendar_time_column, check_limits, choose_focus,  # noqa: E402
                         files_signature, links_signature, _is_label_number, load_tables, looks_like_id_name, mark_ids, table_name,
                         table_order, unique_names)
from tools.profiler import profile_dataframe  # noqa: E402
from tools.relations import find_links  # noqa: E402

APP = str(Path(__file__).resolve().parent.parent / "app" / "streamlit_app.py")


def csv(df) -> bytes:
    return df.to_csv(index=False).encode()


def shop_files():
    rng = np.random.default_rng(1)
    customers = pd.DataFrame({"customer_id": range(1, 7), "name": list("ABCDEF"), "country": ["IN", "US", "US", "UK", "IN", "UK"]})
    products = pd.DataFrame({"product_id": [10, 20, 30, 40, 50], "name": ["Desk", "Chair", "Lamp", "Sofa", "Rug"],
                             "price": [100.5, 50.25, 20.0, 400.0, 75.5]})
    n = 40
    orders = pd.DataFrame({"order_id": range(1, n + 1), "customer_id": rng.integers(1, 7, n), "product_id": rng.choice([10, 20, 30, 40, 50], n),
                           "qty": rng.integers(1, 6, n), "order_date": pd.date_range("2024-01-01", periods=n, freq="5D").strftime("%Y-%m-%d")})
    return [("orders.csv", csv(orders)), ("customers.csv", csv(customers)), ("products.csv", csv(products))], orders


# ---------------------------------------------------------------- loader
def test_the_loader_reads_backslash_N_as_missing():
    df, _ = load_csv_bytes(b'id,score,label\n1,\\N,"a"\n2,5,\\N\n3,7,"c"\n4,9,"d"\n')
    assert df["score"].isna().sum() == 1 and df["label"].isna().sum() == 1
    assert pd.api.types.is_numeric_dtype(df["score"])                        # not text any more


def test_a_literal_n_in_other_words_is_not_missing():
    df, _ = load_csv_bytes(b'id,word\n1,N\n2,No\n3,\\Nope\n')
    assert list(df["word"]) == ["N", "No", "\\Nope"]


# ---------------------------------------------------------------- helpers
def test_table_names_are_safe_and_unique():
    assert table_name("Race Results (2024).csv") == "Race_Results_2024" and table_name("???.csv") == "table"
    assert unique_names(["a", "b", "a", "a", "b"]) == ["a", "b", "a_2", "a_3", "b_2"]
    assert unique_names(["a", "a_2", "a"]) == ["a", "a_2", "a_3"]


def test_files_signature_changes_with_content_and_name_but_not_order_of_nothing():
    a = [("x.csv", b"1"), ("y.csv", b"2")]
    assert files_signature(a) == files_signature(list(a))
    assert files_signature(a) != files_signature([("x.csv", b"1"), ("y.csv", b"3")])
    assert files_signature(a) != files_signature([("x.csv", b"1"), ("z.csv", b"2")])


def test_limits():
    assert check_limits([("a.csv", b"x")] * 3) == ""
    assert str(MAX_FILES) in check_limits([("a.csv", b"x")] * (MAX_FILES + 1))
    big = [("a.csv", b"x" * (200 * 1024 * 1024)), ("b.csv", b"x" * (200 * 1024 * 1024))]
    assert "MB" in check_limits(big)


def test_a_bad_file_is_reported_and_the_others_still_load():
    files, _ = shop_files()
    files = files + [("empty.csv", b""), ("header_only.csv", b"a,b\n")]
    tables, errors = load_tables(files, build_table)
    assert sorted(tables) == ["customers", "orders", "products"]
    assert [f for f, _ in errors] == ["empty.csv", "header_only.csv"]
    assert "empty" in errors[0][1].lower() and "no data rows" in errors[1][1]


def test_an_unexpected_error_in_one_file_is_caught_and_named():
    def build(name, data):
        if name == "bad":
            raise RuntimeError("boom")
        return {"name": name}
    tables, errors = load_tables([("good.csv", b"x"), ("bad.csv", b"x")], build)
    assert list(tables) == ["good"] and errors[0][0] == "bad.csv" and "RuntimeError" in errors[0][1] and "boom" in errors[0][1]


def test_duplicate_file_names_become_two_tables():
    files, _ = shop_files()
    tables, _ = load_tables([files[0], files[0]], build_table)
    assert sorted(tables) == ["orders", "orders_2"]


# ---------------------------------------------------------------- choosing the table
def frames_of(files):
    tables, _ = load_tables(files, build_table)
    return {n: t["df"] for n, t in tables.items()}, tables


def test_the_event_table_is_chosen_and_listed_first():
    files, _ = shop_files()
    fr, _ = frames_of(files)
    links = find_links(fr)
    assert choose_focus(fr, links) == "orders" and table_order(fr, links)[0] == "orders"
    assert set(table_order(fr, links)) == set(fr)


def test_with_no_links_the_biggest_table_is_chosen():
    fr = {"a": pd.DataFrame({"x": range(3)}), "b": pd.DataFrame({"y": range(10)})}
    assert choose_focus(fr, []) == "b" and table_order(fr, []) == ["b", "a"]
    with pytest.raises(ValueError):
        choose_focus({}, [])


def test_the_event_table_that_reaches_more_lookups_wins_over_a_bigger_one():
    a = pd.DataFrame({"a_id": [1, 2], "x": 1}); b = pd.DataFrame({"b_id": [1, 2], "y": 1})
    wide = pd.DataFrame({"r": range(5), "a_id": [1, 2, 1, 2, 1], "b_id": [1, 2, 1, 2, 1]})
    big = pd.DataFrame({"r": range(50), "a_id": [1, 2] * 25})
    fr = {"a": a, "b": b, "wide": wide, "big": big}
    assert choose_focus(fr, find_links(fr)) == "wide"


def test_lookups_of_lookups_count_when_choosing_the_table():
    countries = pd.DataFrame({"country_id": [1, 2], "c": 1})
    customers = pd.DataFrame({"customer_id": [1, 2, 3], "country_id": [1, 2, 2]})
    products = pd.DataFrame({"product_id": [1, 2], "p": 1})
    stores = pd.DataFrame({"store_id": [1, 2], "s": 1})
    chain = pd.DataFrame({"r": range(5), "customer_id": [1, 2, 3, 1, 2]})               # reaches customers, then countries
    direct = pd.DataFrame({"r": range(50), "product_id": [1, 2] * 25})                   # reaches one table only
    fr = {"countries": countries, "customers": customers, "products": products, "chain": chain, "direct": direct, "stores": stores}
    assert choose_focus(fr, find_links(fr)) == "chain"


def test_links_signature_depends_on_the_links_used():
    files, _ = shop_files()
    fr, _ = frames_of(files)
    links = find_links(fr)
    assert links_signature(links) == links_signature(list(reversed(links))) != links_signature(links[:1])


# ---------------------------------------------------------------- profile of the combined table
def test_link_columns_are_identifiers_not_measures():
    files, orders = shop_files()
    fr, tables = frames_of(files)
    tab, flat = build_flat(fr, "orders", find_links(fr), profile_dataframe, raw_tables=tables)
    p = tab["profile"]
    assert flat.ok and tab["df"].shape[0] == len(orders)
    assert "customer_id" in p["id_columns"] and "product_id" in p["id_columns"]
    assert "customer_id" not in p["measures"] and "product_id" not in p["measures"]
    assert "qty" in p["measures"] and "price" in p["measures"]
    assert "customers_country" in p["dimensions"]
    assert p["id_columns"] == sorted(set(p["id_columns"]), key=p["id_columns"].index)           # no duplicates
    assert any("rows unchanged" in l for l in tab["cleaning_log"])


def test_correlations_never_mention_identifiers():
    prof = {"columns": [{"name": "driverId", "role": "numeric", "notes": [], "integer_valued": True},
                        {"name": "a", "role": "numeric", "notes": []}, {"name": "b", "role": "numeric", "notes": []}],
            "correlations": [{"a": "driverId", "b": "a", "r": 0.9}, {"a": "a", "b": "b", "r": 0.6}], "time_column": None}
    out = mark_ids(prof, ["driverId"])
    assert out["correlations"] == [{"a": "a", "b": "b", "r": 0.6}]
    assert prof["columns"][0]["role"] == "numeric"                     # the input is not changed


def test_label_numbers_come_after_real_quantities():
    names = ["number", "grid", "positionOrder", "points", "drivers_number", "laps", "fastestLapSpeed", "lat", "round", "revenue", "birth_year"]
    prof = {"columns": [{"name": n, "role": "numeric", "notes": [], "integer_valued": False} for n in names],
            "correlations": [], "time_column": None}
    assert mark_ids(prof, [])["measures"] == ["points", "laps", "fastestLapSpeed", "revenue",
                                              "number", "grid", "positionOrder", "drivers_number", "lat", "round", "birth_year"]
    plain = {"columns": [{"name": n, "role": "numeric", "notes": [], "integer_valued": False} for n in ["qty", "price", "units"]],
             "correlations": [], "time_column": None}
    assert mark_ids(plain, [])["measures"] == ["qty", "price", "units"]                     # nothing to move: same order
    assert not _is_label_number("amount") and _is_label_number("Phone_no") and _is_label_number("carNumber") and not _is_label_number("nodes")


def test_only_datetime_columns_can_be_the_time_column():
    cols = [{"name": "x", "role": "numeric", "span_days": 500}, {"name": "when", "role": "datetime", "span_days": 10}]
    assert calendar_time_column(cols) == "when"


def test_a_link_column_that_is_not_named_like_an_id_is_still_an_identifier():
    prof = {"columns": [{"name": "zone", "role": "numeric", "notes": [], "integer_valued": True}], "correlations": [], "time_column": None}
    assert mark_ids(prof, ["zone"])["columns"][0]["role"] == "id" and mark_ids(prof, [])["columns"][0]["role"] == "numeric"


def test_a_text_column_named_like_an_id_is_an_identifier():
    prof = {"columns": [{"name": "region_id", "role": "categorical", "notes": []}, {"name": "region", "role": "categorical", "notes": []}],
            "correlations": [], "time_column": None}
    assert [c["role"] for c in mark_ids(prof, [])["columns"]] == ["id", "categorical"]


def test_tables_that_could_not_be_added_are_written_in_the_cleaning_log():
    from tools.relations import Link
    files, _ = shop_files()
    fr, tables = frames_of(files)
    fr["customers"] = pd.concat([fr["customers"], fr["customers"].iloc[:1]])                 # repeated key: forced in by hand
    tab, flat = build_flat(fr, "orders", [Link("orders", "customer_id", "customers", "customer_id", 1.0)], profile_dataframe, raw_tables=tables)
    assert flat.skipped and any(l.startswith("Did not add customers") and "number of rows" in l for l in tab["cleaning_log"])


def test_year_like_link_columns_stay_categories_and_text_links_stay_as_they_are():
    prof = {"columns": [{"name": "races_year", "role": "numeric", "notes": [], "integer_valued": True},
                        {"name": "country_code", "role": "categorical", "notes": []},
                        {"name": "driverId", "role": "numeric", "notes": [], "integer_valued": True}],
            "correlations": [], "time_column": None}
    out = mark_ids(prof, ["races_year", "country_code", "driverId"])
    roles = {c["name"]: c["role"] for c in out["columns"]}
    assert roles == {"races_year": "categorical", "country_code": "categorical", "driverId": "id"}
    assert out["measures"] == [] and out["dimensions"] == ["races_year", "country_code"]


def test_id_named_columns_are_identifiers_even_when_not_a_link():
    prof = {"columns": [{"name": "ticket_id", "role": "numeric", "notes": [], "integer_valued": True},
                        {"name": "width", "role": "numeric", "notes": [], "integer_valued": True},
                        {"name": "price_id", "role": "numeric", "notes": [], "integer_valued": False}],
            "correlations": [], "time_column": None}
    roles = {c["name"]: c["role"] for c in mark_ids(prof, [])["columns"]}
    assert roles == {"ticket_id": "id", "width": "numeric", "price_id": "numeric"}


def test_id_name_detection():
    assert all(looks_like_id_name(n) for n in ["driverId", "driver_id", "ID", "id", "raceID", "user id", "Customer-ID"])
    assert not any(looks_like_id_name(n) for n in ["width", "paid", "Identity", "valid", "grid", "PAID", "bid"])


def test_a_clock_or_lap_time_is_not_a_calendar_for_trends():
    cols = [{"name": "fastestLapTime", "role": "datetime", "span_days": 0},
            {"name": "races_time", "role": "datetime", "span_days": 1},
            {"name": "dob", "role": "datetime", "span_days": 36000},
            {"name": "races_date", "role": "datetime", "span_days": 27000}]
    assert calendar_time_column(cols) == "races_date"                                  # a hinted name beats an unhinted one
    assert calendar_time_column(cols[:3]) == "dob"
    assert calendar_time_column(cols[:2]) is None and calendar_time_column([]) is None


# ---------------------------------------------------------------- the real app with uploads
class Upload:
    def __init__(self, name, data):
        self.name, self._data, self.size = name, data, len(data)

    def getvalue(self):
        return self._data


@pytest.fixture
def uploads(monkeypatch):
    box = {"files": []}

    def fake(label, type=None, accept_multiple_files=False, **kw):
        files = [Upload(n, d) for n, d in box["files"]]
        return files if accept_multiple_files else (files[0] if files else None)
    monkeypatch.setattr(st, "file_uploader", fake)
    return box


def open_app(box, files):
    box["files"] = files
    return AppTest.from_file(APP, default_timeout=120).run()


def link_box(at, text):
    return next(c for c in at.checkbox if c.label.startswith(text))


def test_several_files_make_one_table_with_the_columns_of_the_tables_it_links_to(uploads):
    files, orders = shop_files()
    at = open_app(uploads, files)
    assert not at.exception, at.exception
    state = at.session_state
    assert set(state["tables"]) == {"orders", "customers", "products", "orders_flat"} and state["active_table"] == "orders_flat"
    flat = state["tables"]["orders_flat"]["df"]
    assert len(flat) == len(orders) and "customers_country" in flat.columns and "price" in flat.columns
    assert any("with columns from" in s.value for s in at.subheader)
    assert any("unchanged" in s.value for s in at.success)
    assert [m.value for m in at.metric][0] == f"{len(orders):,}"               # the Overview uses the combined table


def test_unticking_a_link_removes_its_columns_and_keeps_every_row(uploads):
    files, orders = shop_files()
    at = open_app(uploads, files)
    link_box(at, "orders.customer_id").uncheck().run()
    assert not at.exception
    flat = at.session_state["tables"]["orders_flat"]["df"]
    assert len(flat) == len(orders) and "customers_country" not in flat.columns and "price" in flat.columns


def test_unticking_every_link_gives_the_plain_table(uploads):
    files, orders = shop_files()
    at = open_app(uploads, files)
    for c in [c for c in at.checkbox if " → " in c.label]:
        c.uncheck()
    at.run()
    assert not at.exception and at.session_state["active_table"] == "orders"
    assert at.session_state["tables"]["orders"]["df"].shape == orders.shape


def test_choosing_another_table_to_analyse(uploads):
    files, _ = shop_files()
    at = open_app(uploads, files)
    focus = next(s for s in at.selectbox if s.label == "Analyse this table")
    assert focus.value == "orders" and focus.options[0].startswith("orders") and len(focus.options) == 3
    focus.select("customers").run()
    assert not at.exception
    assert any("Table: customers" in s.value for s in at.subheader) and at.session_state["active_table"] == "customers"


def test_a_bad_file_among_good_ones_is_named_and_the_rest_works(uploads):
    files, orders = shop_files()
    at = open_app(uploads, files + [("broken.csv", b"")])
    assert not at.exception
    assert any("broken.csv" in e.value and "left out" in e.value for e in at.error)
    assert len(at.session_state["tables"]["orders_flat"]["df"]) == len(orders)


def test_only_bad_files_stop_the_page_with_messages(uploads):
    at = open_app(uploads, [("a.csv", b""), ("b.csv", b"")])
    assert not at.exception and len(at.error) == 2 and "tables" not in at.session_state


def test_one_good_file_among_bad_ones_shows_the_normal_single_table_view(uploads):
    files, orders = shop_files()
    at = open_app(uploads, [files[0], ("a.csv", b"")])
    assert not at.exception and at.session_state["active_table"] == "orders" and any("a.csv" in e.value for e in at.error)
    assert any(s.value == "Table: orders" for s in at.subheader)
    assert not any(s.label == "Analyse this table" for s in at.selectbox)                      # no table picker for one table


def test_unrelated_files_are_listed_with_no_links_and_nothing_is_joined(uploads):
    a = pd.DataFrame({"alpha": range(20), "beta": np.linspace(0, 1, 20)})
    b = pd.DataFrame({"gamma": list("abcde") * 4, "delta": range(20)})
    at = open_app(uploads, [("a.csv", csv(a)), ("b.csv", csv(b))])
    assert not at.exception
    assert set(at.session_state["tables"]) == {"a", "b"} and not any("unchanged" in s.value for s in at.success)


def test_one_file_still_works_exactly_as_before(uploads):
    files, orders = shop_files()
    at = open_app(uploads, files[:1])
    assert not at.exception and at.session_state["active_table"] == "orders" and list(at.session_state["tables"]) == ["orders"]
    assert not any(e.label.startswith("Analyse this table") for e in at.selectbox)


def test_no_files_shows_the_hint(uploads):
    at = open_app(uploads, [])
    assert not at.exception and any("Upload a CSV" in i.value for i in at.info)


def test_too_many_files_are_refused_with_a_message(uploads):
    files, _ = shop_files()
    at = open_app(uploads, [(f"t{i}.csv", files[1][1]) for i in range(MAX_FILES + 1)])
    assert not at.exception and any(str(MAX_FILES) in e.value for e in at.error)


def test_the_heavy_work_is_done_once_not_on_every_click(uploads, monkeypatch):
    import ui.multi_panel as mp
    calls = {"build": 0, "links": 0}
    real_build, real_find = mp.build_table, mp.find_links
    monkeypatch.setattr(mp, "build_table", lambda n, d: (calls.__setitem__("build", calls["build"] + 1), real_build(n, d))[1])
    monkeypatch.setattr(mp, "find_links", lambda f: (calls.__setitem__("links", calls["links"] + 1), real_find(f))[1])
    files, _ = shop_files()
    at = open_app(uploads, files)
    at.run(); at.run()
    link_box(at, "orders.customer_id").uncheck().run()
    assert calls == {"build": 3, "links": 1}


def test_new_files_start_fresh_with_every_link_on(uploads):
    files, _ = shop_files()
    at = open_app(uploads, files)
    link_box(at, "orders.customer_id").uncheck().run()
    uploads["files"] = files[:2] + [("products.csv", files[2][1] + b"\n40,Extra,1.0\n")]       # a file changed
    at.run()
    assert not at.exception and link_box(at, "orders.customer_id").value is True


def test_the_filters_and_the_chat_tab_work_on_the_combined_table(uploads):
    files, orders = shop_files()
    at = open_app(uploads, files)
    assert not at.exception
    assert [t.label for t in at.tabs][:3] == ["Overview", "✨ AI dashboard", "💬 Chat"]
    assert any("customers_country" in m.label for m in at.sidebar.multiselect)                    # a lookup column can be filtered on


def test_a_skipped_table_is_shown_as_a_warning(uploads, monkeypatch):
    import ui.multi_panel as mp
    real = mp.build_flat

    def with_skip(*a, **k):
        table, flat = real(*a, **k)
        flat.skipped.append(("customers", "just a test"))
        return table, flat
    monkeypatch.setattr(mp, "build_flat", with_skip)
    files, _ = shop_files()
    at = open_app(uploads, files)
    assert not at.exception and any("Did not add" in w.value and "customers" in w.value and "just a test" in w.value for w in at.warning)


def test_a_chosen_table_that_is_not_in_the_new_files_does_not_break_the_page(uploads):
    files, _ = shop_files()
    at = open_app(uploads, files)
    next(s for s in at.selectbox if s.label == "Analyse this table").select("customers").run()
    uploads["files"] = [files[0], files[2]]                                                       # customers.csv removed
    at.run()
    assert not at.exception and at.session_state["active_table"] in ("orders", "orders_flat")


def test_the_data_key_changes_with_the_links_so_filters_and_chat_start_fresh(uploads):
    files, _ = shop_files()
    at = open_app(uploads, files)
    keys = lambda: {m.key for m in at.sidebar.multiselect}                                           # noqa: E731
    before = keys()
    assert before
    link_box(at, "orders.customer_id").uncheck().run()
    assert keys() and not (keys() & before)


def test_a_file_over_the_size_limit_is_refused_by_name(uploads, monkeypatch):
    class Huge(Upload):
        def __init__(self, name, data):
            super().__init__(name, data)
            self.size = 51 * 1024 * 1024
    files, _ = shop_files()
    monkeypatch.setattr(st, "file_uploader", lambda *a, **k: [Upload(*files[0]), Huge("huge.csv", b"a,b\n1,2\n")])
    at = AppTest.from_file(APP, default_timeout=120).run()
    assert not at.exception and any("huge.csv" in e.value and "larger than" in e.value for e in at.error)


MEMO_SCRIPT = """
import streamlit as st
from ui.memo import memo
for k in st.session_state["plan"]:
    memo(k, lambda k=k: st.session_state.setdefault("built", []).append(k) or k)
st.session_state["stored"] = list(st.session_state["_memo"])
"""


def test_the_memory_keeps_only_the_recent_results_and_remembers_what_is_used():
    from ui.memo import MAX_ENTRIES
    at = AppTest.from_string(MEMO_SCRIPT, default_timeout=30)
    at.session_state["plan"] = ["keep"] + [f"k{i}" for i in range(MAX_ENTRIES - 1)] + ["keep"] + [f"n{i}" for i in range(MAX_ENTRIES - 2)]
    at.run()
    assert not at.exception
    assert at.session_state["built"].count("keep") == 1                                       # reused, not rebuilt
    assert len(at.session_state["stored"]) == MAX_ENTRIES and "keep" in at.session_state["stored"]
    assert "k0" not in at.session_state["stored"]                                           # the least recently used went first


def test_changed_files_start_again_with_the_default_table(uploads):
    files, orders = shop_files()
    at = open_app(uploads, files)
    next(s for s in at.selectbox if s.label == "Analyse this table").select("customers").run()
    assert at.session_state["active_table"] == "customers"
    uploads["files"] = [("orders.csv", csv(orders.iloc[:30]))] + files[1:]                        # same tables, new content
    at.run()
    assert not at.exception and next(s for s in at.selectbox if s.label == "Analyse this table").value == "orders"
