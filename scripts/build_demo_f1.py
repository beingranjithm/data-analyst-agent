"""Day 23b: makes the small Formula 1 demo set from the 14 full files. Run from the project root:
    python scripts/build_demo_f1.py "/path/to/F1 Data Set"                 (default: seasons 2021 and 2022)
    python scripts/build_demo_f1.py "/path/to/F1 Data Set" --years 2021 2022
It keeps every race of those seasons, every row that belongs to those races, and only the drivers, constructors,
circuits and statuses that those rows point to, so all the links between the tables still work. Output goes to
sample_data/f1_demo/ (about 2 MB). Check the data's licence and keep its attribution before making a repo public."""
import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "sample_data" / "f1_demo"
RACE_TABLES = ("results", "qualifying", "pit_stops", "lap_times", "driver_standings", "constructor_standings",
               "constructor_results", "sprint_results")
ALL = ("races", "seasons", "circuits", "drivers", "constructors", "status") + RACE_TABLES


def read(folder: Path, name: str) -> pd.DataFrame:
    """Everything as text, nothing turned into missing: the output has exactly the values the input had."""
    return pd.read_csv(folder / f"{name}.csv", dtype=str, keep_default_na=False)


def ids(frames: dict, col: str, tables) -> set:
    out = set()
    for t in tables:
        if t in frames and col in frames[t].columns:
            out |= set(frames[t][col])
    return out - {"", "\\N"}


def build_demo(src: Path, dst: Path, years) -> dict:
    """Writes the subset into dst. Returns {table: rows}."""
    years = {str(y) for y in years}
    frames = {n: read(src, n) for n in ALL if (src / f"{n}.csv").exists()}
    missing = [n for n in ALL if n not in frames]
    if missing:
        raise FileNotFoundError("Missing files: " + ", ".join(f"{m}.csv" for m in missing))
    frames["races"] = frames["races"][frames["races"]["year"].isin(years)]
    if frames["races"].empty:
        raise ValueError(f"No races found for the years {sorted(years)}.")
    race_ids = set(frames["races"]["raceId"])
    for t in RACE_TABLES:
        frames[t] = frames[t][frames[t]["raceId"].isin(race_ids)]
    frames["seasons"] = frames["seasons"][frames["seasons"]["year"].isin(years)]
    frames["circuits"] = frames["circuits"][frames["circuits"]["circuitId"].isin(set(frames["races"]["circuitId"]))]
    frames["drivers"] = frames["drivers"][frames["drivers"]["driverId"].isin(ids(frames, "driverId", RACE_TABLES))]
    frames["constructors"] = frames["constructors"][
        frames["constructors"]["constructorId"].isin(ids(frames, "constructorId", RACE_TABLES))]
    frames["status"] = frames["status"][frames["status"]["statusId"].isin(ids(frames, "statusId", ("results", "sprint_results")))]
    dst.mkdir(parents=True, exist_ok=True)
    for old in dst.glob("*.csv"):
        old.unlink()
    for n, f in frames.items():
        f.to_csv(dst / f"{n}.csv", index=False)
    return {n: len(f) for n, f in frames.items()}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("folder")
    ap.add_argument("--years", nargs="+", type=int, default=[2021, 2022])
    args = ap.parse_args()
    src = Path(args.folder).expanduser()
    try:
        counts = build_demo(src, OUT, args.years)
    except (FileNotFoundError, ValueError) as e:
        sys.exit(str(e))
    size = sum(p.stat().st_size for p in OUT.glob("*.csv")) / 1e6
    print(f"Wrote {len(counts)} files to {OUT.relative_to(ROOT)} ({size:.1f} MB):")
    for n, c in counts.items():
        print(f"  {n}: {c:,} rows")


if __name__ == "__main__":
    main()
