""""What changed and why" cards (dashboard SOP, Phase C).

Grounded only, per an explicit product decision: a card states what changed and by how much,
computed from the data (reusing tools/kpi_deltas.py's already-computed, already-tested period-
over-period deltas), and never guesses a cause. There is no "why" beyond the number itself unless
a future dataset carries an explicit events/annotations column to join against — until then, no
causal language ("likely linked to...") is ever produced.

One card per KPI that has a usable delta (Phase B). No usable date column/span -> no deltas ->
no cards, same graceful fallback as everywhere else in the SOP.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ChangeCard:
    label: str          # the KPI's label, e.g. "Revenue"
    tag: str              # "Win", "Risk" or "Watch"
    text: str            # "Revenue rose 23.4% vs previous 30 days."
    pct: float
    direction: str        # "up", "down" or "flat"


def _tag_for(direction: str, good) -> str:
    if direction == "flat" or good is None:
        return "Watch"
    return "Win" if good else "Risk"


def build_change_cards(kpis, comparison_label: str) -> list[ChangeCard]:
    """One ChangeCard per KPI with a usable delta, Risks first, then Wins, then Watch,
    each group ordered by the size of the move. Never raises: a bad KPI is just skipped."""
    cards = []
    for k in kpis:
        try:
            if k.delta_pct is None or k.delta_direction is None:
                continue
            tag = _tag_for(k.delta_direction, k.delta_good)
            verb = {"up": "rose", "down": "fell", "flat": "stayed flat"}[k.delta_direction]
            amount = "" if k.delta_direction == "flat" else f" {abs(k.delta_pct):.1f}%"
            text = f"{k.label} {verb}{amount} {comparison_label}.".replace(" .", ".")
            cards.append(ChangeCard(k.label, tag, text, k.delta_pct, k.delta_direction))
        except Exception:          # a card must never break the dashboard
            continue
    order = {"Risk": 0, "Win": 1, "Watch": 2}
    cards.sort(key=lambda c: (order.get(c.tag, 3), -abs(c.pct)))
    return cards
