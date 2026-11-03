""""What changed and why" cards (dashboard SOP, Phase C): grounded only, reuses Phase B's
already-computed deltas, never invents a cause. Pure function, no model."""
from tools.change_cards import build_change_cards
from tools.plan_kpis import KpiValue


def kpi(label, pct, direction, good):
    return KpiValue(label, "sum", label.lower(), 100.0, "",
                    delta_pct=pct, delta_direction=direction, delta_good=good)


def test_a_good_rise_is_tagged_win():
    cards = build_change_cards([kpi("Revenue", 23.4, "up", True)], "vs previous 30 days")
    assert len(cards) == 1
    assert cards[0].tag == "Win" and cards[0].text == "Revenue rose 23.4% vs previous 30 days."


def test_a_bad_rise_is_tagged_risk():
    cards = build_change_cards([kpi("Costs", 15.0, "up", False)], "vs previous 30 days")
    assert cards[0].tag == "Risk" and cards[0].text == "Costs rose 15.0% vs previous 30 days."


def test_a_good_fall_is_tagged_win_with_fell():
    cards = build_change_cards([kpi("Churn", 9.5, "down", True)], "vs previous 30 days")
    assert cards[0].tag == "Win" and cards[0].text == "Churn fell 9.5% vs previous 30 days."


def test_a_flat_change_is_tagged_watch_with_no_percentage():
    cards = build_change_cards([kpi("Orders", 0.4, "flat", None)], "vs previous 30 days")
    assert cards[0].tag == "Watch" and cards[0].text == "Orders stayed flat vs previous 30 days."


def test_kpis_with_no_delta_produce_no_card():
    k = KpiValue("Revenue", "sum", "revenue", 100.0, "")      # delta fields default to None
    assert build_change_cards([k], "vs previous 30 days") == []


def test_no_comparison_label_means_no_cards_were_ever_requested_for_a_no_date_dataset():
    # mirrors the graceful-fallback case: compute_kpi_deltas() returned {} upstream, so every
    # KpiValue still has delta_pct=None, and build_change_cards naturally produces nothing
    kpis = [KpiValue("Revenue", "sum", "revenue", 100.0, "")]
    assert build_change_cards(kpis, "") == []


def test_cards_are_ordered_risks_first_then_wins_then_watch_by_size_of_move():
    cards = build_change_cards([
        kpi("SmallWin", 2.0, "up", True),
        kpi("BigRisk", 40.0, "up", False),
        kpi("Flat", 0.5, "flat", None),
        kpi("BigWin", 30.0, "up", True),
        kpi("SmallRisk", 5.0, "down", False),
    ], "vs previous 30 days")
    assert [c.label for c in cards] == ["BigRisk", "SmallRisk", "BigWin", "SmallWin", "Flat"]


def test_a_broken_kpi_is_skipped_rather_than_crashing():
    class Weird:
        label = "Oops"
        delta_pct = 5.0
        delta_direction = "sideways"    # not a recognised direction -> KeyError inside, caught
        delta_good = True
    assert build_change_cards([Weird()], "vs previous 30 days") == []


def test_no_causal_language_appears_in_any_card_text():
    cards = build_change_cards([kpi("Revenue", 23.4, "up", True),
                                kpi("Costs", 15.0, "up", False)], "vs previous 30 days")
    banned = ("likely", "linked to", "caused by", "because", "due to")
    for c in cards:
        assert not any(b in c.text.lower() for b in banned)
