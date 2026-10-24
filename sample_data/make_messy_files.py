"""Creates deliberately messy (fake) CSV files to stress-test the profiler."""
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path(__file__).parent / "messy"
OUT.mkdir(exist_ok=True)
rng = np.random.default_rng(7)
n = 80

# 1. Mixed date formats + junk markers for "missing"
dates = pd.date_range("2025-01-01", periods=n, freq="3D")
mixed = []
for i, d in enumerate(dates):
    mixed.append([d.strftime("%Y-%m-%d"), d.strftime("%d/%m/%Y"), d.strftime("%b %d, %Y")][i % 3])
for i in (5, 17, 40):
    mixed[i] = "N/A"
pd.DataFrame({"order_date": mixed, "amount": rng.integers(50, 500, n)}).to_csv(OUT / "1_mixed_dates.csv", index=False)

# 2. Ugly headers: spaces, blank, duplicates, symbols
df = pd.DataFrame(rng.integers(1, 100, (n, 5)), columns=[" Sales Amount ", "", "Region", "Region", "Profit (%)"])
df["Region"] = None
df.iloc[:, 2] = rng.choice(["N", "S"], n)
df.iloc[:, 3] = rng.choice(["E", "W"], n)
df.to_csv(OUT / "2_ugly_headers.csv", index=False)

# 3. Numbers stored as text, with currency, percent and placeholders
price = [f"${v:,.2f}" for v in rng.integers(100, 9000, n)]
pct = [f"{v}%" for v in rng.integers(1, 60, n)]
for i in (3, 9, 30):
    price[i] = "-"
    pct[i] = "n/a"
pd.DataFrame({"price": price, "growth": pct, "qty": rng.integers(1, 9, n)}).to_csv(OUT / "3_currency_text.csv", index=False)

# 4. Empty and constant columns, plus a text column
pd.DataFrame({
    "id": range(1, n + 1),
    "empty_col": [None] * n,
    "constant": ["same"] * n,
    "notes": [f"Customer comment number {i} with a long free text description of the issue" for i in range(n)],
    "score": rng.normal(50, 10, n).round(1),
}).to_csv(OUT / "4_empty_constant.csv", index=False)

# 5. Tiny file
pd.DataFrame({"a": [1, 2, 3, 4, 5], "b": list("vwxyz")}).to_csv(OUT / "5_tiny.csv", index=False)

# 6. Semicolon-delimited, European decimals, latin-1 encoding
lines = ["produkt;preis;menge;datum"]
for i in range(n):
    lines.append(f"Käse {i % 4};{rng.integers(100, 999) / 10:.1f}".replace(".", ",") + f";{rng.integers(1, 9)};2025-02-{(i % 27) + 1:02d}")
(OUT / "6_semicolon_latin1.csv").write_bytes("\n".join(lines).encode("latin-1"))

# 7. Header only, no rows
(OUT / "7_header_only.csv").write_text("a,b,c\n")

# 8. Wide file (200 columns)
pd.DataFrame(rng.normal(0, 1, (n, 200)), columns=[f"m{i}" for i in range(200)]).to_csv(OUT / "8_wide.csv", index=False)

# 9. Dates as day-first text only, plus a year column and a zip-like code
pd.DataFrame({
    "year": rng.choice([2022, 2023, 2024, 2025], n),
    "zip_code": rng.integers(10000, 99999, n),
    "joined": [f"{(i % 28) + 1:02d}-{(i % 12) + 1:02d}-2024" for i in range(n)],
    "active": rng.choice(["Yes", "No"], n),
}).to_csv(OUT / "9_year_zip_dayfirst.csv", index=False)

print("Created", len(list(OUT.glob('*.csv'))), "messy files in", OUT)
