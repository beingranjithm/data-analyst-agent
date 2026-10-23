import numpy as np
import pandas as pd

rng = np.random.default_rng(42)
n = 600

products = {
    "Laptop": ("Electronics", 900),
    "Phone": ("Electronics", 600),
    "Headphones": ("Electronics", 120),
    "Desk": ("Furniture", 250),
    "Chair": ("Furniture", 150),
    "Notebook": ("Stationery", 5),
    "Pen Set": ("Stationery", 12),
}

names = rng.choice(list(products), n)
dates = pd.date_range("2025-01-01", "2025-12-31", freq="D")
units = rng.integers(1, 20, n)
price = np.array([products[p][1] for p in names])
discount = rng.choice([0.0, 0.05, 0.10], n)

df = pd.DataFrame({
    "order_date": pd.to_datetime(rng.choice(dates, n)).strftime("%Y-%m-%d"),
    "region": rng.choice(["North", "South", "East", "West", "Europe"], n),
    "category": [products[p][0] for p in names],
    "product": names,
    "units": units,
    "unit_price": price,
    "discount": discount,
})
df["revenue"] = (df["units"] * df["unit_price"] * (1 - df["discount"])).round(2)
df = df.sort_values("order_date").reset_index(drop=True)

# Add a few missing values on purpose, like real messy data
df.loc[rng.choice(n, 15, replace=False), "discount"] = np.nan
df.loc[rng.choice(n, 10, replace=False), "region"] = np.nan

df.to_csv("sample_data/sales_sample.csv", index=False)
print("Created sample_data/sales_sample.csv with", len(df), "rows")
