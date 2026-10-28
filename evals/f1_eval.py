"""Day 22: how accurate is the chat when the answer needs SEVERAL tables? (the Formula 1 files)

Same idea as chat_eval.py (it reuses its scoring): every question has a true answer computed here with plain pandas
straight from the raw files, never from the chat's own tables. The questions are chosen so that the answer needs
columns from other tables (a driver's name lives in 'drivers', a race's year in 'races') and some need two big
"event" tables (results and pit_stops) at once, which is where careless joins give wrong totals.
"""
from __future__ import annotations

from typing import Callable

import pandas as pd

from evals.chat_eval import Expectation, Question

NEEDED = ("results", "races", "drivers", "constructors", "circuits", "pit_stops", "lap_times")


def read_raw(folder) -> dict:
    """The raw files exactly as pandas reads them (\\N = missing). Independent of the app's own loader."""
    from pathlib import Path
    out = {}
    for name in NEEDED:
        f = Path(folder) / f"{name}.csv"
        if f.exists():
            out[name] = pd.read_csv(f, na_values=["\\N"])
    return out


def _top(series: pd.Series):
    top = series.sort_values(ascending=False)
    if len(top) > 1 and top.iloc[0] == top.iloc[1]:
        raise ValueError("the top two are tied, so the question has no single answer")
    return top.index[0], top.iloc[0]


def _driver(t, surname, forename=None):
    d = t["drivers"]
    d = d[d.surname == surname] if forename is None else d[(d.surname == surname) & (d.forename == forename)]
    if len(d) != 1:
        raise ValueError(f"{surname} is not exactly one driver")
    return int(d.driverId.iloc[0])


def _wins(t):
    r = t["results"]
    return r[r.position == 1]


def _q_races_2019(t):
    n = int((t["races"].year == 2019).sum())
    return Expectation(numbers=[n], describe=f"{n} races in 2019 (count of rows in races)")


def _q_most_wins(t):
    w = _wins(t).merge(t["drivers"], on="driverId").groupby(["forename", "surname"]).size()
    (fn, sn), n = _top(w)
    return Expectation(numbers=[int(n)], names=[sn], describe=f"{fn} {sn}, {n} wins")


def _q_constructor_wins(t):
    w = _wins(t).merge(t["constructors"], on="constructorId").groupby("name").size()
    name, n = _top(w)
    return Expectation(numbers=[int(n)], names=[name], describe=f"{name}, {n} wins")


def _q_distinct_winners(t):
    n = int(_wins(t).driverId.nunique())
    return Expectation(numbers=[n], describe=f"{n} different drivers have won a race")


def _q_hamilton_points(t):
    h = _driver(t, "Hamilton", "Lewis")
    pts = float(t["results"][t["results"].driverId == h].points.sum())
    return Expectation(numbers=[pts], describe=f"{pts:g} points (sum of results.points for Hamilton)")


def _q_top_circuit(t):
    c = t["races"][["circuitId"]].merge(t["circuits"][["circuitId", "name"]], on="circuitId").groupby("name").size()
    name, n = _top(c)
    return Expectation(numbers=[int(n)], names=[name], describe=f"{name}, {n} races")


def _q_schumacher_wins(t):
    sid = _driver(t, "Schumacher", "Michael")
    n = int((_wins(t).driverId == sid).sum())
    return Expectation(numbers=[n], describe=f"Michael Schumacher: {n} wins")


def _q_country(t):
    c = t["races"].merge(t["circuits"], on="circuitId").groupby("country").size()
    name, n = _top(c)
    return Expectation(numbers=[int(n)], names=[name], describe=f"{name}, {n} races")


def _q_points_2019(t):
    r = t["results"].merge(t["races"][["raceId", "year"]], on="raceId")
    r = r[r.year == 2019].merge(t["drivers"], on="driverId").groupby("surname").points.sum()
    sn, pts = _top(r)
    return Expectation(numbers=[float(pts)], names=[sn], describe=f"{sn}, {pts:g} points in 2019")


def _q_brazil(t):
    n = int((t["drivers"].nationality == "Brazilian").sum())
    return Expectation(numbers=[n], describe=f"{n} drivers with nationality Brazilian")


def _q_constructors_2020(t):
    r = t["results"].merge(t["races"][["raceId", "year"]], on="raceId")
    n = int(r[r.year == 2020].constructorId.nunique())
    return Expectation(numbers=[n], describe=f"{n} different constructors scored/entered a result in 2020")


def _q_verstappen_pits(t):
    v = _driver(t, "Verstappen", "Max")
    p = t["pit_stops"].merge(t["races"][["raceId", "year"]], on="raceId")
    n = int(((p.driverId == v) & (p.year == 2021)).sum())
    return Expectation(numbers=[n], describe=f"{n} pit-stop rows for Verstappen in 2021")


def _q_laps_2019(t):
    h = _driver(t, "Hamilton", "Lewis")
    l = t["lap_times"].merge(t["races"][["raceId", "year"]], on="raceId")
    n = int(((l.driverId == h) & (l.year == 2019)).sum())
    return Expectation(numbers=[n], describe=f"{n} lap rows for Hamilton in 2019")


def _q_avg_lap(t):
    h = _driver(t, "Hamilton", "Lewis")
    l = t["lap_times"].merge(t["races"][["raceId", "year"]], on="raceId")
    m = float(l[(l.driverId == h) & (l.year == 2019)].milliseconds.mean())
    return Expectation(numbers=[(round(m), round(m, 1), round(m, 2))],
                       describe=f"{m:,.1f} ms: mean of lap_times.milliseconds for Hamilton in 2019")


def _q_pits_of_winner(t):
    """Needs results (who won) AND pit_stops (how many stops) for the same race: two event tables."""
    ra = t["races"]
    race = ra[(ra.year == 2019) & (ra.name == "Monaco Grand Prix")]
    if len(race) != 1:
        raise ValueError("no unique 2019 Monaco Grand Prix")
    rid = int(race.raceId.iloc[0])
    w = t["results"][(t["results"].raceId == rid) & (t["results"].position == 1)]
    if len(w) != 1:
        raise ValueError("no unique winner")
    did = int(w.driverId.iloc[0])
    p = t["pit_stops"]
    n = int(((p.raceId == rid) & (p.driverId == did)).sum())
    sn = t["drivers"].set_index("driverId").surname[did]
    return Expectation(numbers=[n], names=[sn], describe=f"winner {sn}, {n} pit stops in the 2019 Monaco Grand Prix")


def _q_fastest_lap_driver(t):
    l = t["lap_times"]
    row = l.loc[l.milliseconds.idxmin()]
    if (l.milliseconds == row.milliseconds).sum() > 1:
        raise ValueError("tied fastest lap")
    sn = t["drivers"].set_index("driverId").surname[int(row.driverId)]
    return Expectation(names=[sn], describe=f"{sn} ({int(row.milliseconds):,} ms is the shortest lap in lap_times)")


def _q_trap_attendance(t):
    return Expectation(trap=True, describe="no attendance data exists in any table")


def _q_trap_future(t):
    return Expectation(trap=True, allow_zero=True, describe="the data ends in 2024; there is no 2030 season")


DEFINITIONS: list = [
    ("one table", "How many races were held in 2019?", _q_races_2019),
    ("one table", "How many drivers are Brazilian?", _q_brazil),
    ("lookup columns", "Which driver has won the most races, and how many?", _q_most_wins),
    ("lookup columns", "Which constructor has won the most races, and how many?", _q_constructor_wins),
    ("lookup columns", "How many different drivers have ever won a race?", _q_distinct_winners),
    ("lookup columns", "How many races did Michael Schumacher win?", _q_schumacher_wins),
    ("lookup columns", "What is the total number of points Lewis Hamilton has scored?", _q_hamilton_points),
    ("lookup columns", "Which circuit has hosted the most races, and how many?", _q_top_circuit),
    ("lookup columns", "Which country has hosted the most races, and how many?", _q_country),
    ("lookup columns", "Which driver scored the most points in 2019, and how many?", _q_points_2019),
    ("lookup columns", "How many different constructors took part in 2020?", _q_constructors_2020),
    ("other event table", "How many pit stops did Max Verstappen make in 2021?", _q_verstappen_pits),
    ("other event table", "How many laps did Lewis Hamilton drive in 2019 according to the lap times?", _q_laps_2019),
    ("other event table", "What was Lewis Hamilton's average lap time in milliseconds in 2019?", _q_avg_lap),
    ("other event table", "Which driver set the fastest lap in the lap times data?", _q_fastest_lap_driver),
    ("two event tables", "How many pit stops did the winner of the 2019 Monaco Grand Prix make, and who was it?", _q_pits_of_winner),
    ("traps", "What was the average attendance at each race?", _q_trap_attendance),
    ("traps", "Who won the 2030 world championship?", _q_trap_future),
]


def build_f1_questions(raw: dict, definitions=None) -> tuple:
    """(questions, skipped) like chat_eval.build_questions. A question whose tables or answer are not available
    (a missing file, a tie) is skipped with the reason, never asked."""
    questions, skipped = [], []
    for n, (category, text, fn) in enumerate(definitions or DEFINITIONS, 1):
        try:
            questions.append(Question(n, category, text, fn(raw)))
        except (KeyError, ValueError, TypeError, AttributeError, IndexError) as e:
            skipped.append((n, text, f"{type(e).__name__}: {str(e).strip(chr(39) + chr(34))}"))
    return questions, skipped
