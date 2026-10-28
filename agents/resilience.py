"""Waiting out TEMPORARY Google errors (503 busy server, 429 rate limit) instead of failing at once."""
from __future__ import annotations

import asyncio
import re
from typing import Awaitable, Callable, Optional

DEFAULT_DELAYS = (5.0, 15.0, 30.0)          # seconds to wait before retry 1, 2 and 3
_TRANSIENT_CODES = {429, 500, 502, 503, 504}
_TRANSIENT_WORDS = ("503", "429", "UNAVAILABLE", "RESOURCE_EXHAUSTED", "high demand", "overloaded")


_DAILY_WORDS = re.compile(r"PerDay|per day", re.IGNORECASE)
_RETRY_DELAY = re.compile(r"retryDelay['\"]?\s*:\s*['\"]?(\d+(?:\.\d+)?)s")
_RETRY_IN = re.compile(r"retry in (?:(\d+)h)?(?:(\d+)m)?", re.IGNORECASE)


def is_daily_quota_error(text: str) -> bool:
    """The free tier's DAILY limit: waiting seconds or minutes will not help."""
    if _DAILY_WORDS.search(text):
        return True
    m = _RETRY_DELAY.search(text)
    return bool(m and float(m.group(1)) > 120)


def is_transient_message(text: str) -> bool:
    return not is_daily_quota_error(text) and any(w in text for w in _TRANSIENT_WORDS)


def is_transient_error(exc: BaseException) -> bool:
    """True for errors that usually go away by themselves: busy server, per-minute rate limit.
    False for the daily quota (it resets once a day) and for every other kind of error."""
    text = f"{type(exc).__name__}: {exc}"
    if is_daily_quota_error(text):
        return False
    code = getattr(exc, "code", None)
    if isinstance(code, int) and code in _TRANSIENT_CODES:
        return True
    return is_transient_message(text)


def describe_model_error(exc: BaseException) -> str:
    """A short message for people: Google's long error dumps are cut down."""
    if type(exc).__name__ == "CacheMiss":                      # Day 20: plain words for "no saved reply"
        return ("There is no saved answer for this yet, and 'Replay only' never contacts Google. Choose the "
                "option that uses the AI (under 'AI replies') to get a fresh answer.")
    text = f"{type(exc).__name__}: {exc}"
    if is_daily_quota_error(text):
        m = _RETRY_IN.search(text)
        hours, minutes = (m.group(1), m.group(2)) if m else (None, None)
        when = " It should reset in about " + " ".join(p for p in (hours and f"{hours}h", minutes and f"{minutes}m") if p) + "." \
            if (hours or minutes) else ""
        return "The free daily Gemini quota is used up." + when
    return text[:300]


def quiet_adk_logs() -> None:
    """Hide ADK's multi-page error dumps (our own messages already report what went wrong)."""
    import logging
    logging.getLogger("google_adk").setLevel(logging.CRITICAL)


async def with_backoff(fn: Callable[[], Awaitable], *, delays=DEFAULT_DELAYS, sleep=asyncio.sleep,
                       on_wait: Optional[Callable[[int, float, str], None]] = None):
    """Await fn(); on a transient error wait and try again (len(delays) extra tries at most).
    Other errors, and the last transient one, are raised unchanged. on_wait(n, seconds, reason)."""
    for i in range(len(delays) + 1):
        try:
            return await fn()
        except Exception as e:
            if i == len(delays) or not is_transient_error(e):
                raise
            if on_wait:
                on_wait(i + 1, delays[i], f"{type(e).__name__}: {e}"[:200])
            await sleep(delays[i])
