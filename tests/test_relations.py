"""Day 21: finding links between tables and flattening them without changing row counts."""
import numpy as np
import pandas as pd
import pytest

from tools.relations import Link, event_tables, find_links, flatten, lookup_tables


def shop():
    customers = pd.DataFrame({"customer_id": [1, 2, 3, 4], "name": ["Ann", "Bo", "Cy", "Di"], "country": ["IN", "US", "US", "UK"]})
    products = pd.DataFrame({"product_id": [10, 20, 30], "name": ["Desk", "Chair", "Lamp"], "price": [100.0, 50.5, 20.25]})
    orders = pd.DataFrame({"order_id": range(1, 9), "customer_id": [1, 1, 2, 3, 3, 3, 4, 2],
                           "product_id": [10, 20, 10, 30, 30, 20, 10, 20], "qty": [1, 2, 1, 5, 1, 1, 2, 3]})
    return {"orders": orders, "customers": customers, "products": products}


def pairs(links):
    return {(l.child, l.child_col, l.parent, l.parent_col) for l in links}


def test_finds_the_obvious_links_and_nothing_else():
    links = find_links(shop())
    assert pairs(links) == {("orders", "customer_id", "customers", "customer_id"),
                            ("orders", "product_id", "products", "product_id")}
    assert all(l.kind == "many-to-one" and l.containment == 1.0 for l in links)


def test_roles_of_the_tables():
    t = shop()
    links = find_links(t)
    assert lookup_tables(t, links) == {"customers", "products"} and event_tables(t, links) == ["orders"]


def test_a_shared_generic_name_with_different_values_is_not_a_link():
    # both tables have a unique 'name' column, but the values are different people / things
    assert not any(l.child_col == "name" for l in find_links(shop()))


def test_a_small_id_column_inside_a_big_one_is_not_a_link_when_both_are_unique():
    big = pd.DataFrame({"result_id": range(1, 1001), "x": 1})
    small = pd.DataFrame({"result_id": range(1, 31), "y": 1})
    assert find_links({"results": big, "sprints": small}) == []


def test_a_real_one_to_one_link_is_found():
    a = pd.DataFrame({"user_id": [1, 2, 3], "nick": list("abc")})
    b = pd.DataFrame({"user_id": [3, 2, 1], "plan": list("xyz")})
    links = find_links({"a": a, "b": b})
    assert links and all(l.kind == "one-to-one" for l in links)
    assert flatten({"a": a, "b": b}, "a", links).joined == []          # one-to-one links are reported, not auto-joined


def test_the_parent_key_must_have_no_repeats():
    customers = pd.DataFrame({"customer_id": [1, 1, 2], "x": [1, 2, 3]})
    orders = pd.DataFrame({"order_id": [1, 2], "customer_id": [1, 2]})
    assert not any(l.parent == "customers" for l in find_links({"orders": orders, "customers": customers}))


def test_the_parent_key_must_have_no_gaps():
    customers = pd.DataFrame({"customer_id": [1, 2, None], "x": [1, 2, 3]})
    orders = pd.DataFrame({"order_id": [1, 2], "customer_id": [1, 2]})
    assert not any(l.parent == "customers" for l in find_links({"orders": orders, "customers": customers}))


def test_mostly_missing_parents_are_not_a_link_but_a_few_orphans_are_fine():
    customers = pd.DataFrame({"customer_id": range(1, 11), "x": 1})
    few = pd.DataFrame({"order_id": range(1, 21), "customer_id": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10] * 2})
    few.loc[0, "customer_id"] = 99                                        # one orphan: 10 of 11 distinct values found
    assert len(find_links({"orders": few, "customers": customers})) == 1
    many = pd.DataFrame({"order_id": range(1, 11), "customer_id": list(range(100, 105)) * 2})
    assert find_links({"orders": many, "customers": customers}) == []


def test_the_containment_is_reported():
    customers = pd.DataFrame({"customer_id": range(1, 10), "x": 1})
    orders = pd.DataFrame({"order_id": range(1, 12), "customer_id": list(range(1, 10)) + [50, 1]})
    (l,) = find_links({"orders": orders, "customers": customers})
    assert l.containment == pytest.approx(0.9)


def test_float_keys_with_missing_values_and_text_keys_with_spaces_still_match():
    parent = pd.DataFrame({"code": ["a1", "b2", "c3"], "label": ["A", "B", "C"]})
    child = pd.DataFrame({"row_id": [1, 2, 3, 4], "code": [" a1", "b2 ", "b2", "c3"]})
    assert len(find_links({"p": parent, "c": child})) == 1
    parent2 = pd.DataFrame({"zone_id": [1, 2, 3], "label": list("ABC")})
    child2 = pd.DataFrame({"row_id": [1, 2, 3, 4], "zone_id": [1.0, 2.0, np.nan, 3.0]})
    assert len(find_links({"p": parent2, "c": child2})) == 1


def test_decimal_and_date_columns_are_never_keys():
    a = pd.DataFrame({"price_id": [1.5, 2.5, 3.5], "x": 1})
    b = pd.DataFrame({"row": [1, 2, 3, 4], "price_id": [1.5, 2.5, 3.5, 1.5]})
    assert find_links({"a": a, "b": b}) == []
    d1 = pd.DataFrame({"day_id": pd.to_datetime(["2020-01-01", "2020-01-02", "2020-01-03"]), "x": 1})
    d2 = pd.DataFrame({"row": [1, 2, 3, 4], "day_id": pd.to_datetime(["2020-01-01"] * 4)})
    assert find_links({"a": d1, "b": d2}) == []


def test_id_naming_rule_for_a_table_with_a_plain_id_column():
    customers = pd.DataFrame({"id": [1, 2, 3], "name": list("abc")})
    orders = pd.DataFrame({"order_id": [1, 2, 3, 4], "customer_id": [1, 2, 3, 1]})
    (l,) = find_links({"customers": customers, "orders": orders})
    assert (l.child, l.child_col, l.parent, l.parent_col) == ("orders", "customer_id", "customers", "id")
    other = pd.DataFrame({"order_id": [1, 2, 3, 4], "buyer_id": [1, 2, 3, 1]})
    assert find_links({"customers": customers, "orders": other}) == []        # no name evidence, so no guess


def test_a_shared_generic_column_without_id_style_name_needs_complete_overlap():
    seasons = pd.DataFrame({"year": [2019, 2020, 2021, 2022], "url": list("abcd")})
    races = pd.DataFrame({"race": range(1, 9), "year": [2019, 2019, 2020, 2020, 2021, 2021, 2022, 2022]})
    assert len(find_links({"seasons": seasons, "races": races})) == 1          # complete overlap, so allowed
    races2 = races.copy(); races2.loc[0, "year"] = 1800
    assert find_links({"seasons": seasons, "races": races2}) == []             # 3 of 4 found: not enough for a non-id name


# ---------------------------------------------------------------- flatten
def test_flatten_adds_lookup_columns_and_keeps_every_row():
    t = shop()
    f = flatten(t, "orders", find_links(t))
    assert f.ok and f.rows_before == f.rows_after == 8 and len(f.df) == 8
    assert {"customers_country", "customers_name", "products_name", "price"} <= set(f.df.columns)
    row = f.df[f.df.order_id == 4].iloc[0]
    assert row["customers_country"] == "US" and row["price"] == 20.25 and row["qty"] == 5
    assert [l.parent for l, _ in f.joined] == ["customers", "products"]


def test_generic_and_clashing_column_names_get_the_table_name_in_front():
    t = shop()
    f = flatten(t, "orders", find_links(t))
    names = list(f.df.columns)
    assert len(names) == len(set(names))
    assert "name" not in names and "customers_name" in names and "products_name" in names
    assert "price" in names                                        # specific names stay as they are


def test_orphans_keep_their_row_with_empty_lookup_values():
    t = shop()
    t["orders"].loc[6, "customer_id"] = 3                       # customer 4 no longer has orders ...
    t["orders"].loc[0, "customer_id"] = 4                       # ... except order 1, which we make the only one
    t["customers"] = t["customers"].iloc[:3]                    # and customer 4 is missing from customers
    f = flatten(t, "orders", find_links(t, min_containment=0.7))
    assert f.ok and len(f.df) == 8 and f.df["customers_country"].isna().sum() == 1


def test_chains_are_followed_and_every_table_is_joined_once():
    countries = pd.DataFrame({"country_id": [1, 2], "country_name": ["India", "USA"]})
    customers = pd.DataFrame({"customer_id": [1, 2, 3], "country_id": [1, 2, 2]})
    orders = pd.DataFrame({"order_id": range(1, 7), "customer_id": [1, 2, 3, 1, 2, 3], "ship_country_id": [1, 1, 2, 2, 1, 2]})
    t = {"orders": orders, "customers": customers, "countries": countries}
    f = flatten(t, "orders", find_links(t))
    assert f.ok and "country_name" in f.df.columns
    assert sorted(l.parent for l, _ in f.joined) == ["countries", "customers"]
    assert f.df.loc[f.df.order_id == 1, "country_name"].iloc[0] == "India"


def test_flatten_never_changes_the_focus_values_and_does_not_touch_the_inputs():
    t = shop()
    before = {k: v.copy() for k, v in t.items()}
    f = flatten(t, "orders", find_links(t))
    assert f.df[list(t["orders"].columns)].equals(before["orders"])
    assert all(t[k].equals(before[k]) for k in t)


def test_unknown_focus_and_width_limit():
    t = shop()
    with pytest.raises(KeyError):
        flatten(t, "nope", [])
    f = flatten(t, "orders", find_links(t), max_columns=5)
    assert f.skipped and len(f.df.columns) <= 5 + 0 and f.ok


def test_no_links_means_the_table_is_returned_unchanged():
    a = pd.DataFrame({"x": [1, 2], "y": [3, 4]})
    f = flatten({"a": a, "b": pd.DataFrame({"p": [1, 2, 3], "q": [1, 1, 1]})}, "a", [])
    assert f.ok and f.df.equals(a) and f.joined == []


def test_link_text_is_readable():
    assert Link("orders", "customer_id", "customers", "customer_id", 1.0).text() == "orders.customer_id → customers.customer_id"


def test_a_big_event_table_does_not_multiply_rows_when_two_are_present():
    # results and laps both point to races; they must NOT be joined to each other
    races = pd.DataFrame({"race_id": [1, 2], "year": [2020, 2021]})
    results = pd.DataFrame({"result_id": range(1, 5), "race_id": [1, 1, 2, 2]})
    laps = pd.DataFrame({"lap_id": range(1, 11), "race_id": [1] * 5 + [2] * 5})
    t = {"races": races, "results": results, "laps": laps}
    links = find_links(t)
    assert pairs(links) == {("results", "race_id", "races", "race_id"), ("laps", "race_id", "races", "race_id")}
    assert event_tables(t, links) == ["laps", "results"]
    assert len(flatten(t, "results", links).df) == 4 and len(flatten(t, "laps", links).df) == 10


# ---------------------------------------------------------------- more edge cases (found by mutation testing)
def test_a_boolean_column_is_never_a_key():
    parent = pd.DataFrame({"flag_id": [0, 1, 2, 3], "x": 1})
    child = pd.DataFrame({"row": range(6), "flag_id": [True, False, True, False, True, True]})
    assert find_links({"p": parent, "c": child}) == []


def test_a_shared_non_id_name_needs_at_least_three_different_values():
    parent = pd.DataFrame({"grade": [1, 2, 3, 4], "x": 1})
    child = pd.DataFrame({"row": range(6), "grade": [1, 2, 1, 2, 1, 2]})
    assert find_links({"p": parent, "c": child}) == []


def test_a_shared_non_id_name_needs_complete_overlap_not_just_ninety_percent():
    parent = pd.DataFrame({"year": range(2000, 2020), "x": 1})
    child = pd.DataFrame({"row": range(21), "year": list(range(2001, 2020)) + [1999, 2001]})      # 20 distinct, 19 found
    assert len(set(child.year) & set(parent.year)) / child.year.nunique() == 0.95
    assert find_links({"p": parent, "c": child}) == []
    child_ok = pd.DataFrame({"row": range(21), "year": list(range(2000, 2020)) + [2001]})
    assert len(find_links({"p": parent, "c": child_ok})) == 1


def test_two_unique_columns_that_only_mostly_overlap_are_not_a_one_to_one_link():
    a = pd.DataFrame({"user_id": range(1, 101), "x": 1})
    b = pd.DataFrame({"user_id": range(2, 102), "y": 1})                 # 99 of 100 shared
    assert find_links({"a": a, "b": b}) == []


def test_a_table_with_a_single_row_is_not_used_as_a_parent():
    parent = pd.DataFrame({"x_id": [5], "label": ["only"]})
    child = pd.DataFrame({"row": range(3), "x_id": [5, 5, 5]})
    assert find_links({"p": parent, "c": child}) == []


def test_a_table_reachable_by_two_paths_is_joined_once():
    countries = pd.DataFrame({"country_id": [1, 2], "country_name": ["India", "USA"]})
    customers = pd.DataFrame({"customer_id": [1, 2, 3], "country_id": [1, 2, 2]})
    orders = pd.DataFrame({"order_id": range(1, 7), "customer_id": [1, 2, 3, 1, 2, 3], "country_id": [1, 1, 2, 2, 1, 2]})
    t = {"orders": orders, "customers": customers, "countries": countries}
    f = flatten(t, "orders", find_links(t))
    assert f.ok and [l.parent for l, _ in f.joined] == ["countries", "customers"]
    assert not any(c.startswith("countries_") for c in f.df.columns) and list(f.df.columns).count("country_name") == 1


def test_flatten_matches_float_keys_with_gaps_and_text_keys_with_spaces():
    zones = pd.DataFrame({"zone_id": [1, 2, 3], "zone": ["A", "B", "C"]})
    rows = pd.DataFrame({"row_id": [1, 2, 3, 4], "zone_id": [1.0, 2.0, np.nan, 3.0]})
    f = flatten({"rows": rows, "zones": zones}, "rows", find_links({"rows": rows, "zones": zones}))
    assert f.ok and list(f.df["zone"].fillna("-")) == ["A", "B", "-", "C"]
    parent = pd.DataFrame({"code": ["a1", "b2", "c3"], "label": ["A", "B", "C"]})
    child = pd.DataFrame({"row_id": [1, 2, 3, 4], "code": [" a1", "b2 ", "b2", "c3"], "n": 1})
    g = flatten({"c": child, "p": parent}, "c", find_links({"c": child, "p": parent}))
    assert g.ok and list(g.df["p_label"]) == ["A", "B", "B", "C"]


def test_a_link_that_would_change_the_row_count_is_skipped_not_applied():
    parent = pd.DataFrame({"k_id": [1, 1, 2], "v": [10, 20, 30]})            # repeats: a bad parent, forced in by hand
    child = pd.DataFrame({"row": [1, 2, 3], "k_id": [1, 2, 1]})
    bad = Link("c", "k_id", "p", "k_id", 1.0)
    f = flatten({"c": child, "p": parent}, "c", [bad])
    assert f.ok and len(f.df) == 3 and f.skipped and "number of rows" in f.skipped[0][1] and "v" not in f.df.columns


def test_hubs_are_neither_lookups_nor_events():
    countries = pd.DataFrame({"country_id": [1, 2], "n": ["a", "b"]})
    customers = pd.DataFrame({"customer_id": [1, 2, 3], "country_id": [1, 2, 2]})
    orders = pd.DataFrame({"order_id": range(1, 7), "customer_id": [1, 2, 3, 1, 2, 3]})
    t = {"orders": orders, "customers": customers, "countries": countries}
    links = find_links(t)
    assert lookup_tables(t, links) == {"countries"} and event_tables(t, links) == ["orders"]


def test_links_come_back_with_the_best_evidence_first():
    parent = pd.DataFrame({"b_id": range(1, 21), "a_id": range(1, 21), "x": 1})
    child = pd.DataFrame({"row": range(40), "b_id": list(range(1, 20)) * 2 + [1, 2], "a_id": list(range(1, 21)) * 2})
    child.loc[0, "b_id"] = 99                                                  # b_id: 19 of 20 distinct found
    got = find_links({"p": parent, "c": child})
    assert [l.child_col for l in got] == ["a_id", "b_id"] and got[0].containment > got[1].containment
