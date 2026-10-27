"""Day 16 in the Chat tab: a healed calculation shows all its tries."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_chat_calc_tab import BAD, EAST_SUM, expander_labels, new_app, say  # noqa: E402

TYPO = "```python\nresult = df[df['region'] == 'east']['revenue'].sum()\nresult = df[df['region'] == 'east']\n```"


def test_a_filter_typo_is_fixed_and_the_tab_shows_both_tries():
    at = say(new_app([TYPO, EAST_SUM, "East is 480."], max_code_runs=3), "East total?")
    assert not at.exception
    assert expander_labels(at) == ["🧮 Show the calculation"]
    assert len(at.code) == 2 and any("(found nothing)" in c.value for c in at.caption)
    assert any("East is 480." in m.value for m in at.markdown) and at.session_state["llm"].calls == 3


def test_an_error_then_a_fix_shows_the_failed_try_and_the_good_one():
    at = say(new_app([BAD, EAST_SUM, "East is 480."], max_code_runs=3), "q")
    assert len(at.code) == 2 and any("(failed)" in c.value for c in at.caption)
    assert not any("could not" in m.value.lower() for m in at.markdown)
