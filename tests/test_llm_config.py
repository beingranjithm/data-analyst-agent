"""Sanity checks on the model configuration (agents/llm_factory.py)."""
from agents.llm_factory import MIN_INTERVAL, ROLE_MODELS


def test_every_configured_model_has_its_own_pacing_interval():
    for role, names in ROLE_MODELS.items():
        for name in names:
            assert name in MIN_INTERVAL, f"{name} ({role}) would be paced with the slow default"


def test_no_model_is_listed_twice_in_one_role():
    for role, names in ROLE_MODELS.items():
        assert len(set(names)) == len(names), role


def test_every_role_has_a_fallback_for_when_a_daily_quota_runs_out():
    for role, names in ROLE_MODELS.items():
        assert len(names) >= 3, role
