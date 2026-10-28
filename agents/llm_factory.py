"""One place that says which model does which job (Day 12a).

Free-tier numbers seen in AI Studio (Oct 2026): gemini-3.6-flash 5 requests/min and 20/day;
gemini-3.5-flash-lite 15 requests/min and 500/day. Edit ROLE_MODELS if your table changes."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

from agents.llm_cache import DEFAULT_CACHE_PATH, CachingLlm
from agents.llm_fallback import FallbackLlm

# first = preferred, the rest are used only when the earlier model's DAILY quota is used up
ROLE_MODELS = {
    "planner": ["gemini-3.6-flash", "gemini-3.5-flash-lite", "gemini-3.1-flash-lite"],   # 1 request per dashboard
    "chart": ["gemini-3.5-flash-lite", "gemini-3.1-flash-lite", "gemini-3.6-flash"],     # many requests per dashboard
    "chat": ["gemini-3.5-flash-lite", "gemini-3.1-flash-lite", "gemini-3.6-flash"],
}
# seconds between real requests = 60 / requests-per-minute, plus a little margin
MIN_INTERVAL = {"gemini-3.6-flash": 12.5, "gemini-3.5-flash-lite": 4.5, "gemini-3.1-flash-lite": 4.5}
DEFAULT_INTERVAL = 12.5


DEMO_CACHE_PATH = Path(__file__).resolve().parent.parent / "deploy" / "demo_cache.json"


def default_cache_path(mode: Optional[str]) -> Path:
    """Where saved replies live. Normally .cache/llm_cache.json. To RECORD the public demo's answers, start the app or
    the script with DAA_CACHE_PATH=deploy/demo_cache.json. On the public host, 'Replay only' reads the shipped demo file."""
    env = os.getenv("DAA_CACHE_PATH")
    if env:
        return Path(env).expanduser()
    from tools.hosting import is_hosted
    return DEMO_CACHE_PATH if (is_hosted() and mode == "replay") else Path(DEFAULT_CACHE_PATH)


def make_llm(role: str = "chart", *, mode: Optional[str] = None, path=None, models: Optional[list] = None,
             api_key: Optional[str] = None):
    """The model object to pass to the agents. mode: "use" (default), "replay" or "off"
    (or set the environment variable LLM_CACHE_MODE).
    api_key: a visitor's own key (hosted version). It goes only into this model object, never into the environment."""
    if models is None:
        if role not in ROLE_MODELS:
            raise ValueError(f"unknown role {role!r}; choose one of {sorted(ROLE_MODELS)}")
        models = ROLE_MODELS[role]
    mode = mode or os.getenv("LLM_CACHE_MODE", "use")
    chain = []
    for name in models:
        inner = None
        if mode != "replay":
            from google.adk.models import Gemini
            inner = Gemini(model=name, client_kwargs={"api_key": api_key}) if api_key else Gemini(model=name)
        chain.append(CachingLlm(model=name, inner=inner, mode=mode, path=str(path or default_cache_path(mode)),
                                min_interval=MIN_INTERVAL.get(name, DEFAULT_INTERVAL)))
    return chain[0] if len(chain) == 1 else FallbackLlm(chain=chain)
