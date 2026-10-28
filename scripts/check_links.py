"""Day 21: show how a folder of CSV files links together. No AI, no internet.
Usage:  python scripts/check_links.py FOLDER_WITH_CSV_FILES"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tools.relations import event_tables, find_links, flatten, lookup_tables  # noqa: E402


def main(folder: str) -> int:
    files = sorted(Path(folder).expanduser().glob("*.csv"))
    if len(files) < 2:
        print(f"Found {len(files)} CSV file(s) in {folder}. Put at least two CSV files in one folder.")
        return 1
    tables = {}
    for f in files:
        try:
            tables[f.stem] = pd.read_csv(f, na_values=["\\N"], keep_default_na=True)
        except Exception as e:                                   # one odd file must not stop the rest
            print(f"  skipped {f.name}: {type(e).__name__}: {str(e)[:100]}")
    print(f"Loaded {len(tables)} tables: " + ", ".join(f"{k} ({len(v):,} rows)" for k, v in tables.items()))
    links = find_links(tables)
    print(f"\nLinks found: {len(links)}")
    for l in links:
        print(f"  {l.text():60s} {l.kind}, {l.containment:.0%} of values found ({l.reason})")
    print("\nLookup tables (only pointed at):", ", ".join(sorted(lookup_tables(tables, links))) or "none")
    events = event_tables(tables, links)
    print("Event tables (point at others):  ", ", ".join(events) or "none")
    print("\nFlattening each event table (rows must not change):")
    bad = 0
    for ev in events:
        f = flatten(tables, ev, links)
        bad += not f.ok
        print(f"  {ev:22s} {f.rows_before:>9,} rows -> {f.rows_after:>9,}  {'OK' if f.ok else 'CHANGED!'}   "
              f"{len(tables[ev].columns)} -> {len(f.df.columns)} columns, joined: {', '.join(l.parent for l, _ in f.joined) or 'nothing'}")
        for t, why in f.skipped:
            print(f"      skipped {t}: {why}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]) if len(sys.argv) > 1 else print(__doc__) or 1)
