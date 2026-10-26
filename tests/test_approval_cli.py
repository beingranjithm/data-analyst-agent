import pytest

from tools.approval_cli import parse_answer


def test_what_people_type():
    assert parse_answer("").approve and parse_answer("   ").drop == []
    assert parse_answer("2").drop == [2]
    assert parse_answer("2, 3").drop == [2, 3] and parse_answer("1 4").drop == [1, 4]
    for word in ("q", "Q", "cancel", "no"):
        assert parse_answer(word).approve is False


def test_nonsense_is_explained():
    for text in ("two", "1;2", "-1", "2.5"):
        with pytest.raises(ValueError, match="did not understand"):
            parse_answer(text)
