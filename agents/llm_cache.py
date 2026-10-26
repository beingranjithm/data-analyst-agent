"""Record/replay cache around any ADK model, with request pacing (Day 12a).

Same prompt asked twice -> the saved reply is replayed and NO request is sent to Google.
Modes:  "use"    replay if saved, otherwise ask the model and save the reply (default)
        "replay" never contact the model; a prompt that was not saved raises CacheMiss
        "off"    behave like the plain model (no cache at all)
Only the model's TEXT reply is stored (never the prompt, never your data rows). The reply is derived
from the dataset profile, so the cache folder is git-ignored. Chart code still runs on the real data.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from google.adk.models.base_llm import BaseLlm, LlmCapabilities
from google.adk.models.llm_response import LlmResponse
from google.genai import types
from pydantic import PrivateAttr, field_validator

DEFAULT_CACHE_PATH = Path(__file__).resolve().parent.parent / ".cache" / "llm_cache.json"
MODES = ("use", "replay", "off")
_now = time.monotonic          # replaced in tests
_sleep = asyncio.sleep         # replaced in tests


def _trace(message: str) -> None:
    """One line per model event when the environment variable LLM_TRACE is set."""
    if os.getenv("LLM_TRACE"):
        print(f"[llm] {message}", file=sys.stderr, flush=True)


class CacheMiss(RuntimeError):
    """replay mode: the cache has no saved reply for this request."""


def request_key(model: str, llm_request) -> str:
    """A fingerprint of everything that decides the model's answer."""
    cfg = llm_request.config
    schema = getattr(cfg, "response_schema", None) if cfg else None
    if schema is not None and hasattr(schema, "model_json_schema"):
        schema = schema.model_json_schema()
    parts = {
        "model": model,
        "system": str(getattr(cfg, "system_instruction", "") or ""),
        "contents": [c.model_dump_json(exclude_none=True) for c in llm_request.contents],
        "schema": json.dumps(schema, sort_keys=True, default=str) if schema is not None else "",
        "mime": str(getattr(cfg, "response_mime_type", "") or ""),
        "temperature": getattr(cfg, "temperature", None),
        "tools": str(getattr(cfg, "tools", "") or ""),
    }
    return hashlib.sha256(json.dumps(parts, sort_keys=True, default=str).encode()).hexdigest()


def _final_text(responses: list) -> Optional[str]:
    """The reply text worth saving, or None (errors, tool calls, empty replies are never saved)."""
    if len(responses) != 1:
        return None
    r = responses[0]
    if getattr(r, "error_code", None) or getattr(r, "error_message", None):
        return None
    parts = r.content.parts if r.content and r.content.parts else []
    if any(p.function_call or p.function_response for p in parts):
        return None
    return "".join(p.text for p in parts if p.text and not p.thought) or None


class CachingLlm(BaseLlm):
    model: str = "gemini-3.6-flash"          # the REAL model name; it is also sent to `inner`
    inner: Any = None                        # the model that answers on a cache miss (None in replay mode)
    mode: str = "use"
    path: str = str(DEFAULT_CACHE_PATH)
    min_interval: float = 0.0                # minimum seconds between REAL requests (60 / requests-per-minute)
    hits: int = 0
    real_calls: int = 0
    _store: Optional[dict] = PrivateAttr(default=None)
    _last_real: Optional[float] = PrivateAttr(default=None)

    @field_validator("mode")
    @classmethod
    def _check_mode(cls, v):
        if v not in MODES:
            raise ValueError(f"mode must be one of {MODES}, not {v!r}")
        return v

    @property
    def capabilities(self):
        return LlmCapabilities(output_schema_and_tools=False)

    # ---- storage ----
    def _load(self) -> dict:
        if self._store is None:
            try:
                data = json.loads(Path(self.path).read_text())
                self._store = data if isinstance(data, dict) else {}
            except (OSError, ValueError):
                self._store = {}
        return self._store

    def _read_disk(self) -> dict:
        try:
            data = json.loads(Path(self.path).read_text())
            return data if isinstance(data, dict) else {}
        except (OSError, ValueError):
            return {}

    def _save(self) -> None:
        p = Path(self.path)
        p.parent.mkdir(parents=True, exist_ok=True)
        merged = self._read_disk()          # other CachingLlm objects may have saved since we loaded
        merged.update(self._store)
        self._store = merged
        tmp = p.with_suffix(p.suffix + ".tmp")
        tmp.write_text(json.dumps(self._store, indent=1))
        os.replace(tmp, p)

    def clear(self) -> None:
        self._store = {}
        Path(self.path).unlink(missing_ok=True)

    def summary(self) -> str:
        return (f"{self.model}: {self.real_calls} real request(s), {self.hits} replayed from cache "
                f"({len(self._load())} replies saved)")

    # ---- the model interface ----
    async def _pace(self) -> None:
        if self.min_interval > 0 and self._last_real is not None:
            wait = self._last_real + self.min_interval - _now()
            if wait > 0:
                _trace(f"{self.model}: waiting {wait:.1f}s (request pacing)")
                await _sleep(wait)
        self._last_real = _now()

    async def generate_content_async(self, llm_request, stream=False):
        use_cache = self.mode != "off" and not stream
        key = request_key(self.model, llm_request) if use_cache else None
        if use_cache:
            hit = self._load().get(key)
            if hit is None:                  # another CachingLlm object may have saved it since we loaded
                hit = self._read_disk().get(key)
            if hit is not None:
                self.hits += 1
                _trace(f"{self.model}: replayed from cache")
                yield LlmResponse(content=types.Content(role="model", parts=[types.Part(text=hit["text"])]))
                return
        if self.mode == "replay":
            _trace(f"{self.model}: not in the cache (replay-only mode)")
            raise CacheMiss("No saved reply for this request (replay-only mode).")
        if self.inner is None:
            raise RuntimeError("CachingLlm has no inner model to call.")
        await self._pace()
        self.real_calls += 1
        _trace(f"{self.model}: REAL request")
        request = llm_request.model_copy(update={"model": self.model})
        responses = []
        try:
            async for r in self.inner.generate_content_async(request, stream=stream):
                responses.append(r)
                yield r
        except Exception as e:
            _trace(f"{self.model}: ERROR {type(e).__name__}: {' '.join(str(e).split())[:110]}")
            raise
        if use_cache:
            text = _final_text(responses)
            _trace(f"{self.model}: answered ({len(text or '')} characters{', saved' if text else ', not saved'})")
            if text:
                self._load()[key] = {"text": text, "model": self.model,
                                     "saved": datetime.now(timezone.utc).isoformat(timespec="seconds")}
                self._save()
