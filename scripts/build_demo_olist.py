"""Day 25: makes the small Olist Brazilian e-commerce demo set from the 8 full files. Run from the project root:
    python scripts/build_demo_olist.py "/path/to/olist files folder"                 (default window: November 2017)
    python scripts/build_demo_olist.py "/path/to/olist files folder" --start 2017-11-01 --end 2017-11-30
It keeps every order placed in that window, every row that belongs to those orders (items, payments, reviews), and
only the customers, products and sellers those rows point to, so all the links between the tables still work.
Output goes to sample_data/olist_demo/. Check the data's licence (CC BY-NC-SA 4.0, Olist/Kaggle) and keep its
attribution before making a repo public.
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "sample_data" / "olist_demo"
ORDER_CHILD_TABLES = ("order_items", "order_payments", "order_reviews")
FILE_NAMES = {
    "orders": "olist_orders_dataset.csv",
    "order_items": "olist_order_items_dataset.csv",
    "order_payments": "olist_order_payments_dataset.csv",
    "order_reviews": "olist_order_reviews_dataset.csv",
    "customers": "olist_customers_dataset.csv",
    "products": "olist_products_dataset.csv",
    "sellers": "olist_sellers_dataset.csv",
    "category_translation": "product_category_name_translation.csv",
}
DEFAULT_START, DEFAULT_END = "2017-11-01", "2017-11-30"           # November 2017: the Black Friday order spike


def read(folder: Path, key: str) -> pd.DataFrame:
    """Everything as text, nothing turned into missing: the output has exactly the values the input had."""
    return pd.read_csv(folder / FILE_NAMES[key], dtype=str, keep_default_na=False)


def build_demo(src: Path, dst: Path, start: str, end: str) -> dict:
    """Writes the subset into dst. Returns {table: rows}."""
    frames = {k: read(src, k) for k in FILE_NAMES if (src / FILE_NAMES[k]).exists()}
    missing = [k for k in FILE_NAMES if k not in frames]
    if missing:
        raise FileNotFoundError("Missing files: " + ", ".join(FILE_NAMES[m] for m in missing))

    purchased = pd.to_datetime(frames["orders"]["order_purchase_timestamp"])
    window = (purchased >= pd.Timestamp(start)) & (purchased < pd.Timestamp(end) + pd.Timedelta(days=1))
    frames["orders"] = frames["orders"][window]
    if frames["orders"].empty:
        raise ValueError(f"No orders found between {start} and {end}.")
    order_ids = set(frames["orders"]["order_id"])
    for t in ORDER_CHILD_TABLES:
        frames[t] = frames[t][frames[t]["order_id"].isin(order_ids)]

    frames["customers"] = frames["customers"][frames["customers"]["customer_id"].isin(set(frames["orders"]["customer_id"]))]
    frames["products"] = frames["products"][frames["products"]["product_id"].isin(set(frames["order_items"]["product_id"]))]
    frames["sellers"] = frames["sellers"][frames["sellers"]["seller_id"].isin(set(frames["order_items"]["seller_id"]))]
    # category_translation is tiny (71 rows): keep it whole, it's just an English name lookup

    dst.mkdir(parents=True, exist_ok=True)
    for old in dst.glob("*.csv"):
        old.unlink()
    for name, f in frames.items():
        f.to_csv(dst / f"{name}.csv", index=False)
    return {name: len(f) for name, f in frames.items()}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("folder")
    ap.add_argument("--start", default=DEFAULT_START)
    ap.add_argument("--end", default=DEFAULT_END)
    args = ap.parse_args()
    src = Path(args.folder).expanduser()
    try:
        counts = build_demo(src, OUT, args.start, args.end)
    except (FileNotFoundError, ValueError) as e:
        sys.exit(str(e))
    size = sum(p.stat().st_size for p in OUT.glob("*.csv")) / 1e6
    print(f"Wrote {len(counts)} files to {OUT.relative_to(ROOT)} ({size:.1f} MB):")
    for n, c in counts.items():
        print(f"  {n}: {c:,} rows")


if __name__ == "__main__":
    main()
