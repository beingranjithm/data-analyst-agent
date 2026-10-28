"""Day 20: does a chart's title say what the chart measures? (A chart titled 'Top Products by Sales Volume' that adds
up revenue misleads: volume usually means units.) A simple word check, shown as a warning and never blocking.
It only speaks up for the two clear cases: a money measure under a title about units or volume, and a units measure
under a title about money."""
from __future__ import annotations

import re
from typing import Optional

MONEY = {"revenue", "sales", "amount", "price", "prices", "income", "cost", "costs", "spend", "profit", "turnover", "gmv"}
STRONG_VOLUME = {"units", "unit", "quantity", "qty", "volume", "pieces"}
WEAK_VOLUME = {"count", "counts", "orders", "number", "items"}


def _words(text: Optional[str]) -> set:
    return set(re.findall(r"[a-z]+", str(text or "").lower()))


def title_warnings(title: Optional[str], measure: Optional[str]) -> list:
    """Warnings (usually none) about a title that disagrees with the measure's name."""
    if not title or not measure:
        return []
    t, m = _words(title), _words(measure)
    money_measure = bool(m & MONEY) and not (m & (STRONG_VOLUME | WEAK_VOLUME))
    volume_measure = bool(m & (STRONG_VOLUME | WEAK_VOLUME)) and not (m & MONEY)
    if money_measure and (t & STRONG_VOLUME):
        word = sorted(t & STRONG_VOLUME)[0]
        return [f"The title says '{word}' (usually units sold), but this chart measures '{measure}'. "
                "Check that the title says what the chart shows."]
    if volume_measure and (t & MONEY) and not (t & (STRONG_VOLUME | WEAK_VOLUME)):
        word = sorted(t & MONEY)[0]
        return [f"The title says '{word}' (an amount of money), but this chart measures '{measure}'. "
                "Check that the title says what the chart shows."]
    return []
