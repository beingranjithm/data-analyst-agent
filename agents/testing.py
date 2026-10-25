"""Test helper: a model wrapper that makes the FIRST reply deliberately broken, then hands every
later request to a real (or fake) model. Used to prove the retry loop works against real Gemini."""
from __future__ import annotations

from typing import Any

from google.adk.models.base_llm import BaseLlm, LlmCapabilities
from google.adk.models.llm_response import LlmResponse
from google.genai import types


class ForcedFirstFailure(BaseLlm):
    model: str = "gemini-3.5-flash"     # must be a real model name when `inner` is a real Gemini
    inner: Any = None                   # the model that answers from the 2nd call on
    first_reply: str = ""               # the broken reply returned on the 1st call
    calls: int = 0
    prompts: list = []                  # system prompt of every call (so you can inspect what was sent)

    @property
    def capabilities(self):
        return LlmCapabilities(output_schema_and_tools=False)

    async def generate_content_async(self, llm_request, stream=False):
        self.calls += 1
        self.prompts.append(str(llm_request.config.system_instruction))
        if self.calls == 1:
            yield LlmResponse(content=types.Content(role="model", parts=[types.Part(text=self.first_reply)]))
            return
        async for resp in self.inner.generate_content_async(llm_request, stream=stream):
            yield resp
