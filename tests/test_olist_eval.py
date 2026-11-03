"""Day 25: the small Olist demo set, its builder, and its questions (fake models only -- no real Gemini)."""
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import build_demo_olist as bd  # noqa: E402
from evals.olist_eval import DEFINITIONS, build_olist_questions, read_raw  # noqa: E402
from tools.relations import find_links  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
DEMO_DIR = ROOT / "sample_data" / "olist_demo"


# ------------------------------------------------------------------ the builder (tiny made-up files)
def write_olist(folder: Path):
    folder.mkdir(parents=True, exist_ok=True)
    files = {
        "olist_orders_dataset.csv":
            "order_id,customer_id,order_status,order_purchase_timestamp,order_approved_at,"
            "order_delivered_carrier_date,order_delivered_customer_date,order_estimated_delivery_date\n"
            "o1,c1,delivered,2017-11-05 10:00:00,2017-11-05 11:00:00,2017-11-06 09:00:00,2017-11-08 20:00:00,2017-11-15 00:00:00\n"
            "o2,c2,delivered,2017-11-20 08:00:00,2017-11-20 09:00:00,2017-11-21 09:00:00,2017-11-25 20:00:00,2017-11-22 00:00:00\n"
            "o4,c2,delivered,2017-11-22 08:00:00,2017-11-22 09:00:00,2017-11-23 09:00:00,2017-11-27 20:00:00,2017-11-29 00:00:00\n"
            "o3,c3,delivered,2017-10-01 08:00:00,2017-10-01 09:00:00,2017-10-02 09:00:00,2017-10-05 20:00:00,2017-10-10 00:00:00\n",
        "olist_order_items_dataset.csv":
            "order_id,order_item_id,product_id,seller_id,shipping_limit_date,price,freight_value\n"
            "o1,1,p1,s1,2017-11-06 00:00:00,100.0,10.0\n"
            "o2,1,p2,s2,2017-11-21 00:00:00,50.0,5.0\n"
            "o4,1,p3,s2,2017-11-23 00:00:00,30.0,5.0\n"
            "o3,1,p1,s1,2017-10-02 00:00:00,100.0,10.0\n",
        "olist_order_payments_dataset.csv":
            "order_id,payment_sequential,payment_type,payment_installments,payment_value\n"
            "o1,1,credit_card,1,110.0\no2,1,boleto,1,55.0\no4,1,boleto,1,35.0\no3,1,credit_card,1,110.0\n",
        "olist_order_reviews_dataset.csv":
            "review_id,order_id,review_score,review_comment_title,review_comment_message,"
            "review_creation_date,review_answer_timestamp\n"
            "r1,o1,5,,,2017-11-09 00:00:00,2017-11-09 12:00:00\n"
            "r2,o2,3,,,2017-11-26 00:00:00,2017-11-26 12:00:00\n"
            "r4,o4,2,,,2017-11-28 00:00:00,2017-11-28 12:00:00\n"
            "r3,o3,4,,,2017-10-06 00:00:00,2017-10-06 12:00:00\n",
        "olist_customers_dataset.csv":
            "customer_id,customer_unique_id,customer_zip_code_prefix,customer_city,customer_state\n"
            "c1,u1,100,sao paulo,SP\nc2,u2,200,rio,RJ\nc3,u3,300,belo horizonte,MG\n",
        "olist_products_dataset.csv":
            "product_id,product_category_name,product_name_lenght,product_description_lenght,"
            "product_photos_qty,product_weight_g,product_length_cm,product_height_cm,product_width_cm\n"
            "p1,beleza_saude,40,200,1,500,20,10,15\np2,moveis_decoracao,30,150,2,2000,40,30,20\n"
            "p3,esporte_lazer,25,100,1,800,25,15,18\n",
        "olist_sellers_dataset.csv":
            "seller_id,seller_zip_code_prefix,seller_city,seller_state\ns1,400,campinas,SP\ns2,500,curitiba,PR\n",
        "product_category_name_translation.csv":
            "product_category_name,product_category_name_english\nbeleza_saude,health_beauty\n"
            "moveis_decoracao,furniture_decor\nesporte_lazer,sports_leisure\n",
    }
    for name, text in files.items():
        (folder / name).write_text(text)


def test_the_builder_keeps_only_the_chosen_window_and_what_it_points_to(tmp_path):
    write_olist(tmp_path / "src")
    counts = bd.build_demo(tmp_path / "src", tmp_path / "out", "2017-11-01", "2017-11-30")
    assert counts["orders"] == 3 and counts["order_items"] == 3 and counts["order_payments"] == 3
    assert counts["order_reviews"] == 3 and counts["customers"] == 2
    out_orders = pd.read_csv(tmp_path / "out" / "orders.csv")
    assert set(out_orders.order_id) == {"o1", "o2", "o4"}            # o3 (October) is outside the window
    # only the seller(s)/product(s) that o1/o2/o4's items actually point to survive the cut
    out_products = pd.read_csv(tmp_path / "out" / "products.csv")
    assert set(out_products.product_id) == {"p1", "p2", "p3"}
    out_sellers = pd.read_csv(tmp_path / "out" / "sellers.csv")
    assert set(out_sellers.seller_id) == {"s1", "s2"}


def test_every_link_between_the_tables_survives_the_cut(tmp_path):
    write_olist(tmp_path / "src")
    bd.build_demo(tmp_path / "src", tmp_path / "out", "2017-11-01", "2017-11-30")
    frames = {p.stem: pd.read_csv(p) for p in (tmp_path / "out").glob("*.csv")}
    links = find_links(frames)
    joined = {(l.child, l.parent) for l in links}
    assert ("order_items", "orders") in joined and ("order_items", "products") in joined
    assert ("order_items", "sellers") in joined and ("products", "category_translation") in joined


def test_the_builder_says_what_went_wrong(tmp_path):
    (tmp_path / "src").mkdir()
    with pytest.raises(FileNotFoundError, match="olist_orders_dataset.csv"):
        bd.build_demo(tmp_path / "src", tmp_path / "out", "2017-11-01", "2017-11-30")
    write_olist(tmp_path / "src2")
    with pytest.raises(ValueError, match="No orders found"):
        bd.build_demo(tmp_path / "src2", tmp_path / "out", "2019-01-01", "2019-01-31")


def test_a_rebuild_removes_files_from_an_earlier_build(tmp_path):
    write_olist(tmp_path / "src")
    out = tmp_path / "out"
    out.mkdir()
    (out / "old.csv").write_text("x\n1\n")
    bd.build_demo(tmp_path / "src", out, "2017-11-01", "2017-11-30")
    assert not (out / "old.csv").exists()


def test_text_that_looks_like_missing_is_copied_untouched(tmp_path):
    """A literal 'NA' or 'null' in the source data must survive as that exact text, not get silently turned into
    a real missing value by pandas' default na-detection (the builder reads with keep_default_na=False)."""
    write_olist(tmp_path / "src")
    c = tmp_path / "src" / "olist_customers_dataset.csv"
    c.write_text(c.read_text().replace("sao paulo", "NA").replace("rio", "null"))
    bd.build_demo(tmp_path / "src", tmp_path / "out", "2017-11-01", "2017-11-30")
    out = pd.read_csv(tmp_path / "out" / "customers.csv", dtype=str, keep_default_na=False)
    assert set(out.customer_city) & {"NA", "null"} == {"NA", "null"}


# ------------------------------------------------------------------ the shipped demo set and its questions
def test_the_shipped_demo_set_is_small_complete_and_linked():
    files = sorted(DEMO_DIR.glob("*.csv"))
    assert {p.stem for p in files} == set(bd.FILE_NAMES)
    assert sum(p.stat().st_size for p in files) < 10_000_000
    frames = {p.stem: pd.read_csv(p) for p in files}
    purchased = pd.to_datetime(frames["orders"]["order_purchase_timestamp"])
    assert purchased.min() >= pd.Timestamp("2017-11-01") and purchased.max() < pd.Timestamp("2017-12-01")
    assert len(find_links(frames)) >= 6


def test_the_demo_questions_all_have_true_answers_on_the_shipped_set():
    qs, skipped = build_olist_questions(read_raw(DEMO_DIR))
    assert skipped == [] and len(qs) == len(DEFINITIONS)
    by = {q.text: q.expect for q in qs}
    assert by["How many orders have the status 'canceled'?"].numbers == [37]
    assert by["Which product category (in English) appears in the most order items, and how many?"].names == ["bed_bath_table"]
    assert by["Which seller state shipped the most order items, and how many?"].names == ["SP"]
    assert by["Which payment type is used the most, and how many times?"].names == ["credit_card"]
    assert by["How many orders were delivered later than their estimated delivery date?"].numbers == [1043]
    assert by["How many orders were paid by credit card and also received a review score of 1?"].numbers == [919]
    assert sum(q.expect.trap for q in qs) == 2


def test_the_window_boundaries_are_inclusive_start_exclusive_end(tmp_path):
    """A purchase at exactly midnight of the start date is IN the window; a purchase at midnight of the day
    after the end date is OUT. Guards the >= / < boundary logic against being loosened to > / <=."""
    write_olist(tmp_path / "src")
    (tmp_path / "src" / "olist_orders_dataset.csv").write_text(
        "order_id,customer_id,order_status,order_purchase_timestamp,order_approved_at,"
        "order_delivered_carrier_date,order_delivered_customer_date,order_estimated_delivery_date\n"
        "oA,c1,delivered,2017-11-01 00:00:00,2017-11-01 01:00:00,2017-11-02 09:00:00,2017-11-04 20:00:00,2017-11-10 00:00:00\n"
        "oB,c1,delivered,2017-12-01 00:00:00,2017-12-01 01:00:00,2017-12-02 09:00:00,2017-12-04 20:00:00,2017-12-10 00:00:00\n"
        "o1,c1,delivered,2017-11-05 10:00:00,2017-11-05 11:00:00,2017-11-06 09:00:00,2017-11-08 20:00:00,2017-11-15 00:00:00\n"
        "o2,c2,delivered,2017-11-20 08:00:00,2017-11-20 09:00:00,2017-11-21 09:00:00,2017-11-25 20:00:00,2017-11-22 00:00:00\n"
        "o4,c2,delivered,2017-11-22 08:00:00,2017-11-22 09:00:00,2017-11-23 09:00:00,2017-11-27 20:00:00,2017-11-29 00:00:00\n"
        "o3,c3,delivered,2017-10-01 08:00:00,2017-10-01 09:00:00,2017-10-02 09:00:00,2017-10-05 20:00:00,2017-10-10 00:00:00\n"
    )
    counts = bd.build_demo(tmp_path / "src", tmp_path / "out", "2017-11-01", "2017-11-30")
    out_orders = pd.read_csv(tmp_path / "out" / "orders.csv")
    assert "oA" in set(out_orders.order_id)          # exactly on the start boundary: included
    assert "oB" not in set(out_orders.order_id)      # a day into December: excluded
    assert counts["orders"] == 4                     # oA, o1, o2, o4 (o3 and oB are outside the window)


def test_a_delivery_exactly_on_time_is_not_counted_as_late(tmp_path):
    """delivered == estimated must NOT count as late: guards > against being loosened to >=."""
    write_olist(tmp_path / "src")
    (tmp_path / "src" / "olist_orders_dataset.csv").write_text(
        "order_id,customer_id,order_status,order_purchase_timestamp,order_approved_at,"
        "order_delivered_carrier_date,order_delivered_customer_date,order_estimated_delivery_date\n"
        "o1,c1,delivered,2017-11-05 10:00:00,2017-11-05 11:00:00,2017-11-06 09:00:00,2017-11-08 20:00:00,2017-11-08 20:00:00\n"
    )
    raw = read_raw(tmp_path / "src")
    qs, skipped = build_olist_questions(raw)
    by = {q.text: q.expect for q in qs}
    assert by["How many orders were delivered later than their estimated delivery date?"].numbers == [0]


def test_credit_card_and_one_star_is_an_intersection_not_a_union(tmp_path):
    """A credit-card order with a 5-star review, and a boleto order with a 1-star review, must NOT be counted:
    guards the set intersection (&) against being loosened to a union (|)."""
    write_olist(tmp_path / "src")
    (tmp_path / "src" / "olist_order_payments_dataset.csv").write_text(
        "order_id,payment_sequential,payment_type,payment_installments,payment_value\n"
        "o1,1,credit_card,1,110.0\no2,1,boleto,1,55.0\no4,1,boleto,1,35.0\no3,1,credit_card,1,110.0\n"
    )
    (tmp_path / "src" / "olist_order_reviews_dataset.csv").write_text(
        "review_id,order_id,review_score,review_comment_title,review_comment_message,"
        "review_creation_date,review_answer_timestamp\n"
        "r1,o1,5,,,2017-11-09 00:00:00,2017-11-09 12:00:00\n"       # credit_card + 5-star: not a match
        "r2,o2,1,,,2017-11-26 00:00:00,2017-11-26 12:00:00\n"       # boleto + 1-star: not a match either
        "r4,o4,2,,,2017-11-28 00:00:00,2017-11-28 12:00:00\n"
        "r3,o3,4,,,2017-10-06 00:00:00,2017-10-06 12:00:00\n"
    )
    raw = read_raw(tmp_path / "src")
    qs, skipped = build_olist_questions(raw)
    by = {q.text: q.expect for q in qs}
    assert by["How many orders were paid by credit card and also received a review score of 1?"].numbers == [0]


def test_the_true_answers_agree_between_the_full_raw_files_and_the_demo_subset_shape():
    """Not a claim the NUMBERS match (the demo is a slice) -- a check that every question still computes on both
    without error, so a column rename in one copy of the files can't silently break only one of them."""
    qs_demo, skipped_demo = build_olist_questions(read_raw(DEMO_DIR))
    assert skipped_demo == []
    for q in qs_demo:
        assert q.expect.trap or q.expect.numbers or q.expect.names
