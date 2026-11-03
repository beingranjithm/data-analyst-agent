"""Day 25: how accurate is the chat when the answer needs SEVERAL tables? (the Olist Brazilian e-commerce files)

Same idea as f1_eval.py (it reuses chat_eval's scoring): every question has a true answer computed here with plain
pandas straight from the raw files, never from the chat's own tables. The questions are chosen so the answer needs
columns from other tables (a product's English category name lives in product_category_name_translation, a seller's
state lives in sellers) and some need two big "event" tables (order_payments and order_reviews, or order_items and
order_reviews) at once, which is where careless joins give wrong totals.

Unlike f1_eval.py, none of the questions below hardcode a year or date range: they work out their own answer from
whatever orders/items/payments/reviews tables they are given. That means the SAME definitions answer correctly on
the full downloaded dataset and on the small demo subset (sample_data/olist_demo) -- there is no separate "demo
version" of each question to keep in sync, just a smaller set of rows to compute over. DEMO_DEFINITIONS below is
therefore the same list as DEFINITIONS; it exists only so the rest of the code (and scripts/eval_olist.py) can
import a "the demo's questions" name the same way it does for F1.
"""
from __future__ import annotations

from typing import Callable

import pandas as pd

from evals.chat_eval import Expectation, Question

NEEDED = ("orders", "order_items", "order_payments", "order_reviews", "customers", "products", "sellers",
          "category_translation")


def read_raw(folder) -> dict:
    """The raw files exactly as pandas reads them. Independent of the app's own loader."""
    from pathlib import Path
    names = {
        "orders": "olist_orders_dataset.csv", "order_items": "olist_order_items_dataset.csv",
        "order_payments": "olist_order_payments_dataset.csv", "order_reviews": "olist_order_reviews_dataset.csv",
        "customers": "olist_customers_dataset.csv", "products": "olist_products_dataset.csv",
        "sellers": "olist_sellers_dataset.csv", "category_translation": "product_category_name_translation.csv",
    }
    out = {}
    for key, filename in names.items():
        f = Path(folder) / filename
        if f.exists():
            out[key] = pd.read_csv(f)
        else:
            f2 = Path(folder) / f"{key}.csv"               # the demo set ships with the shorter names
            if f2.exists():
                out[key] = pd.read_csv(f2)
    return out


def _top(series: pd.Series):
    top = series.sort_values(ascending=False)
    if len(top) > 1 and top.iloc[0] == top.iloc[1]:
        raise ValueError("the top two are tied, so the question has no single answer")
    return top.index[0], top.iloc[0]


def _items_with_category(t) -> pd.DataFrame:
    p = t["products"].merge(t["category_translation"], on="product_category_name", how="left")
    return t["order_items"].merge(p[["product_id", "product_category_name_english"]], on="product_id", how="left")


def _q_canceled_orders(t):
    n = int((t["orders"].order_status == "canceled").sum())
    return Expectation(numbers=[n], describe=f"{n} orders with order_status 'canceled'")


def _q_distinct_sellers(t):
    n = int(t["sellers"].seller_id.nunique())
    return Expectation(numbers=[n], describe=f"{n} distinct sellers (unique seller_id in sellers)")


def _q_late_deliveries(t):
    """Needs only the orders table, but two of its own date columns: a classic single-table trap for off-by-one
    date comparisons (strictly later than the estimate, not on or after it)."""
    o = t["orders"].dropna(subset=["order_delivered_customer_date", "order_estimated_delivery_date"])
    delivered = pd.to_datetime(o["order_delivered_customer_date"])
    estimated = pd.to_datetime(o["order_estimated_delivery_date"])
    n = int((delivered > estimated).sum())
    return Expectation(numbers=[n], describe=f"{n} orders delivered later than their estimated delivery date")


def _q_top_category(t):
    ic = _items_with_category(t)
    name, n = _top(ic.product_category_name_english.value_counts())
    return Expectation(numbers=[int(n)], names=[name], describe=f"{name}, {n} order items")


def _q_top_seller_state(t):
    isell = t["order_items"].merge(t["sellers"][["seller_id", "seller_state"]], on="seller_id", how="left")
    name, n = _top(isell.seller_state.value_counts())
    return Expectation(numbers=[int(n)], names=[name], describe=f"{name}, {n} order items shipped by sellers there")


def _q_top_customer_state(t):
    ocust = t["orders"].merge(t["customers"][["customer_id", "customer_state"]], on="customer_id", how="left")
    name, n = _top(ocust.customer_state.value_counts())
    return Expectation(numbers=[int(n)], names=[name], describe=f"{name}, {n} orders placed by customers there")


def _q_total_payments(t):
    total = float(t["order_payments"].payment_value.sum())
    return Expectation(numbers=[(round(total), round(total, 1), round(total, 2))],
                       describe=f"R$ {total:,.2f} total (sum of order_payments.payment_value)")


def _q_five_star_reviews(t):
    n = int((t["order_reviews"].review_score == 5).sum())
    return Expectation(numbers=[n], describe=f"{n} reviews with review_score == 5")


def _q_avg_review_score(t):
    m = float(t["order_reviews"].review_score.mean())
    return Expectation(numbers=[(round(m), round(m, 1), round(m, 2), round(m, 3))],
                       describe=f"{m:.3f}: mean of order_reviews.review_score")


def _q_top_payment_type(t):
    name, n = _top(t["order_payments"].payment_type.value_counts())
    return Expectation(numbers=[int(n)], names=[name], describe=f"{name}, used {n} times in order_payments")


def _q_credit_card_and_one_star(t):
    """Needs order_payments AND order_reviews for the same orders: two event tables, joined only on order_id,
    never on rows directly (an order can have several payment rows and, rarely, several review rows)."""
    cc_orders = set(t["order_payments"][t["order_payments"].payment_type == "credit_card"].order_id)
    bad_orders = set(t["order_reviews"][t["order_reviews"].review_score == 1].order_id)
    n = len(cc_orders & bad_orders)
    return Expectation(numbers=[n],
                       describe=f"{n} orders paid by credit_card that also have a review_score of 1")


def _q_health_beauty_avg_score(t):
    """Needs order_items (+products +category_translation, to find the category) AND order_reviews (for the
    score) for the same orders: another two-event-table question, this time an average rather than a count."""
    ic = _items_with_category(t)
    hb_order_ids = set(ic[ic.product_category_name_english == "health_beauty"].order_id)
    scores = t["order_reviews"][t["order_reviews"].order_id.isin(hb_order_ids)].review_score
    m = float(scores.mean())
    return Expectation(numbers=[(round(m), round(m, 1), round(m, 2), round(m, 3))],
                       describe=f"{m:.3f}: mean review_score for orders containing a health_beauty item "
                                f"({len(hb_order_ids)} such orders)")


def _q_trap_customer_age(t):
    return Expectation(trap=True, describe="no age or birth-date column exists in any table")


def _q_trap_2025(t):
    n = int((pd.to_datetime(t["orders"]["order_purchase_timestamp"]).dt.year == 2025).sum())
    return Expectation(numbers=[n], trap=True, allow_zero=True,
                       describe=f"{n}: the data runs from September 2016 to October 2018, so there is no 2025")


DEFINITIONS: list = [
    ("one table", "How many orders have the status 'canceled'?", _q_canceled_orders),
    ("one table", "How many distinct sellers are there?", _q_distinct_sellers),
    ("one table", "How many orders were delivered later than their estimated delivery date?", _q_late_deliveries),
    ("lookup columns", "Which product category (in English) appears in the most order items, and how many?", _q_top_category),
    ("lookup columns", "Which seller state shipped the most order items, and how many?", _q_top_seller_state),
    ("lookup columns", "Which customer state placed the most orders, and how many?", _q_top_customer_state),
    ("other event table", "What is the total value recorded in order payments?", _q_total_payments),
    ("other event table", "How many reviews have a score of 5?", _q_five_star_reviews),
    ("other event table", "What is the average review score across all reviews?", _q_avg_review_score),
    ("other event table", "Which payment type is used the most, and how many times?", _q_top_payment_type),
    ("two event tables", "How many orders were paid by credit card and also received a review score of 1?", _q_credit_card_and_one_star),
    ("two event tables", "What is the average review score for orders that contain a health_beauty item?", _q_health_beauty_avg_score),
    ("traps", "What is the average customer age?", _q_trap_customer_age),
    ("traps", "How many orders were placed in 2025?", _q_trap_2025),
]


def build_olist_questions(raw: dict, definitions=None) -> tuple:
    """(questions, skipped) like chat_eval.build_questions. A question whose tables or answer are not available
    (a missing file, a tie) is skipped with the reason, never asked."""
    questions, skipped = [], []
    for n, (category, text, fn) in enumerate(definitions or DEFINITIONS, 1):
        try:
            questions.append(Question(n, category, text, fn(raw)))
        except (KeyError, ValueError, TypeError, AttributeError, IndexError) as e:
            skipped.append((n, text, f"{type(e).__name__}: {str(e).strip(chr(39) + chr(34))}"))
    return questions, skipped


# ------------------------------------------------------------------ the same questions on the small demo set
# (no separate DEMO_DEFINITIONS needed -- see the module docstring). These are the questions the public demo can
# answer without a key, from saved replies.
DEMO_DEFINITIONS = DEFINITIONS
DEMO_QUESTIONS = [text for _, text, _ in DEMO_DEFINITIONS]


def build_demo_questions(raw: dict) -> tuple:
    return build_olist_questions(raw, DEMO_DEFINITIONS)
