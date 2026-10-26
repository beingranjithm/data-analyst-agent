"""Try several models in order (Day 12a). A model whose DAILY free quota is used up is skipped for the
rest of this process. Busy-server and other errors are NOT handled here: they are raised unchanged so the
wait-and-retry layer can deal with them."""
from __future__ import annotations

from typing import Any

from google.adk.models.base_llm import BaseLlm, LlmCapabilities
from pydantic import PrivateAttr

from agents.llm_cache import CacheMiss, _trace
from agents.resilience import is_daily_quota_error


class FallbackLlm(BaseLlm):
    model: str = "fallback"          # only a label: each member of the chain uses its own real name
    chain: list = []                 # BaseLlm objects in order of preference
    _exhausted: set = PrivateAttr(default_factory=set)

    @property
    def capabilities(self):
        return LlmCapabilities(output_schema_and_tools=False)

    @property
    def exhausted(self) -> set:
        return set(self._exhausted)

    def summary(self) -> str:
        lines = [m.summary() if hasattr(m, "summary") else m.model for m in self.chain]
        if self._exhausted:
            lines.append("daily quota used up: " + ", ".join(sorted(self._exhausted)))
        return "; ".join(lines)

    async def generate_content_async(self, llm_request, stream=False):
        last: Any = None
        for llm in self.chain:
            if llm.model in self._exhausted:
                _trace(f"{llm.model}: skipped (daily quota already used up)")
                continue
            request = llm_request.model_copy(update={"model": llm.model})
            try:
                async for r in llm.generate_content_async(request, stream=stream):
                    yield r
                return
            except CacheMiss as e:                       # replay mode: maybe another model's reply was saved
                last = e
            except Exception as e:
                if not is_daily_quota_error(f"{type(e).__name__}: {e}"):
                    raise
                self._exhausted.add(llm.model)
                _trace(f"{llm.model}: daily quota used up, trying the next model")
                last = e
        raise last or RuntimeError("No model is available (the whole chain is used up).")
