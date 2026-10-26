"""Cache objects sharing one file, and the LLM_TRACE event lines. Fake models only."""
import asyncio
import json

import pytest
from google.adk.models.base_llm import BaseLlm, LlmCapabilities
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse
from google.genai import types

from agents.llm_cache import CacheMiss, CachingLlm
from agents.llm_fallback import FallbackLlm


class Inner(BaseLlm):
    model: str = "inner"
    replies: list = []
    calls: int = 0

    @property
    def capabilities(self):
        return LlmCapabilities(output_schema_and_tools=False)

    async def generate_content_async(self, llm_request, stream=False):
        self.calls += 1
        item = self.replies.pop(0)
        if isinstance(item, BaseException):
            raise item
        yield LlmResponse(content=types.Content(role="model", parts=[types.Part(text=item)]))


def req(text="hi"):
    return LlmRequest(model="label", contents=[types.Content(role="user", parts=[types.Part(text=text)])],
                      config=types.GenerateContentConfig(system_instruction="sys"))


def ask(llm, request):
    async def go():
        return [r async for r in llm.generate_content_async(request)]
    return asyncio.run(go())


def text_of(responses):
    return "".join(p.text for p in responses[0].content.parts)


@pytest.fixture
def path(tmp_path):
    return str(tmp_path / "shared.json")


def test_objects_sharing_a_file_do_not_overwrite_each_others_replies(path):
    a = CachingLlm(model="m", inner=Inner(replies=["A1", "A2"]), path=path)
    b = CachingLlm(model="m", inner=Inner(replies=["B1"]), path=path)
    ask(a, req("first"))
    ask(b, req("second"))
    ask(a, req("third"))                           # a's copy was stale: it must not erase b's entry
    assert len(json.loads(open(path).read())) == 3
    replay = CachingLlm(model="m", inner=None, mode="replay", path=path)
    assert [text_of(ask(replay, req(t))) for t in ("first", "second", "third")] == ["A1", "B1", "A2"]


def test_a_reply_saved_by_another_object_is_found_without_restarting(path):
    a = CachingLlm(model="m", inner=Inner(replies=["A1"]), path=path)
    b_inner = Inner(replies=[])
    b = CachingLlm(model="m", inner=b_inner, path=path)
    b._load()                                      # b loaded the (empty) cache long ago
    ask(a, req("q"))                               # a saves a reply afterwards
    assert text_of(ask(b, req("q"))) == "A1" and b_inner.calls == 0 and b.hits == 1


def test_trace_prints_one_line_per_event_only_when_asked(path, monkeypatch, capsys):
    monkeypatch.delenv("LLM_TRACE", raising=False)
    quiet = CachingLlm(model="m1", inner=Inner(replies=["x"]), path=path)
    ask(quiet, req("silent"))
    assert capsys.readouterr().err == ""

    monkeypatch.setenv("LLM_TRACE", "1")
    llm = CachingLlm(model="m1", inner=Inner(replies=["x", RuntimeError("503 UNAVAILABLE\nhigh demand")]), path=path)
    ask(llm, req("q1"))                            # real request, saved
    ask(llm, req("q1"))                            # replayed
    with pytest.raises(RuntimeError):
        ask(llm, req("q2"))                        # real request that fails
    err = capsys.readouterr().err
    assert "[llm] m1: REAL request" in err and "answered (1 characters, saved)" in err
    assert "m1: replayed from cache" in err and "ERROR RuntimeError: 503 UNAVAILABLE high demand" in err
    assert err.count("\n") == err.count("[llm]")   # exactly one line per event


def test_trace_shows_why_a_model_was_skipped(path, monkeypatch, capsys):
    monkeypatch.setenv("LLM_TRACE", "1")
    daily = RuntimeError("429 RESOURCE_EXHAUSTED. requests per day limit exceeded")
    a = CachingLlm(model="A", inner=Inner(replies=[daily]), path=path)
    b = CachingLlm(model="B", inner=Inner(replies=["b1", "b2"]), path=path)
    llm = FallbackLlm(chain=[a, b])
    ask(llm, req("q1"))
    ask(llm, req("q2"))
    err = capsys.readouterr().err
    assert "A: daily quota used up, trying the next model" in err and "A: skipped (daily quota already used up)" in err


def test_replay_miss_is_traced(path, monkeypatch, capsys):
    monkeypatch.setenv("LLM_TRACE", "1")
    llm = CachingLlm(model="m", inner=None, mode="replay", path=path)
    with pytest.raises(CacheMiss):
        ask(llm, req("never saved"))
    assert "not in the cache (replay-only mode)" in capsys.readouterr().err
