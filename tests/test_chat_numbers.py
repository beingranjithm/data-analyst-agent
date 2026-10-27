"""Day 16c: the 'is this number from the data?' check must read results the way they are written (CSV tables)."""
import agents.chat_agent as ca
from agents.chat_agent import numbers_in, ungrounded_numbers

PROFILE_TEXT = "ROWS 600, revenue min 4.50 max 16,245.00"
# the exact kind of result the live run produced: a date column first, then a number
RESULT = "order_date,revenue\n2025-01-01,13297.5\n2025-01-02,3150.0\n2025-01-03,63.0\n2025-01-04,5985.0"


def test_the_comma_in_a_csv_row_does_not_glue_a_date_and_a_number_together():
    assert {"3150", "13297.5", "2025"} <= numbers_in("2025-01-02,3150.0 and 2025-01-01,13297.5")
    assert not any(n.startswith("0") and len(n) > 4 for n in numbers_in("2025-01-02,3150.0"))   # no 023150.0


def test_real_thousands_separators_still_work():
    assert numbers_in("1,708,450.5 and 6,094.") == {"1708450.5", "6094"}
    assert numbers_in("revenue 13,297.50 ok") == {"13297.5"}
    assert numbers_in("12,34") == {"12", "34"}                 # not a thousands group
    assert numbers_in("1,2345") == {"1", "2345"}


def test_an_answer_that_quotes_a_csv_result_is_not_flagged():
    answer = "On 2025-01-01 revenue was 13,297.50, then 3,150.00, then 63.00 and 5,985.00."
    assert ungrounded_numbers(answer, PROFILE_TEXT, results=RESULT) == []


def test_numbers_that_are_not_in_the_result_are_still_flagged():
    answer = "Revenue was 13,297.50 and also 777,777."
    assert ungrounded_numbers(answer, PROFILE_TEXT, results=RESULT) == ["777777"]


def test_adjacent_three_digit_columns_are_read_both_ways():
    results = "a,b\n100,200\n300,400"                       # could be 100200 or the two numbers 100 and 200
    assert ungrounded_numbers("the values are 100 and 200, then 300 and 400", PROFILE_TEXT, results=results) == []
    assert ungrounded_numbers("the value is 100,200", PROFILE_TEXT, results=results) == []
    assert ungrounded_numbers("the value is 555", PROFILE_TEXT, results=results) == ["555"]


def test_the_profile_text_is_read_the_same_way():
    assert ungrounded_numbers("max is 16,245.00", PROFILE_TEXT) == [] and ungrounded_numbers("min is 4.5", PROFILE_TEXT) == []
