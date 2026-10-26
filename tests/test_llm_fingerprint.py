"""LLM_TRACE fingerprints: they must show WHICH part of a request changed between two runs."""
import asyncio

from google.adk.models.base_llm import BaseLlm, LlmCapabilities
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse
from google.genai import types

from agents.llm_cache import CachingLlm


class Inner(BaseLlm):
    model: str = "inner"
    calls: int = 0

    @property
    def capabilities(self):
        return LlmCapabilities(output_schema_and_tools=False)

    async def generate_content_async(self, llm_request, stream=False):
        self.calls += 1
        yield LlmResponse(content=types.Content(role="model", parts=[types.Part(text="answer")]))


def req(text="hi", system="sys"):
    return LlmRequest(model="label", contents=[types.Content(role="user", parts=[types.Part(text=text)])],
                      config=types.GenerateContentConfig(system_instruction=system))


def ask(llm, request):
    async def go():
        return [r async for r in llm.generate_content_async(request)]
    return asyncio.run(go())


def keys(err):
    return [line.split("[key ")[1].rstrip("]") for line in err.splitlines() if "[key " in line]


def test_trace_lines_carry_a_fingerprint_that_is_equal_for_equal_requests(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("LLM_TRACE", "1")
    llm = CachingLlm(model="m", inner=Inner(), path=str(tmp_path / "c.json"))
    ask(llm, req("a"))                  # REAL request
    ask(llm, req("a"))                  # replayed
    ask(llm, req("b"))                  # a different request
    k = keys(capsys.readouterr().err)
    assert len(k) == 3 and k[0] == k[1] and k[2] != k[0] and all(len(x) == 6 for x in k)


def test_level_2_shows_which_part_of_the_request_changed(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("LLM_TRACE", "2")
    llm = CachingLlm(model="m", inner=Inner(), path=str(tmp_path / "c.json"))
    ask(llm, req("a", system="one"))
    ask(llm, req("a", system="two"))    # only the system prompt differs
    lines = [l for l in capsys.readouterr().err.splitlines() if "request parts" in l]
    assert len(lines) == 2
    first, second = (dict(p.split("=") for p in l.split("request parts ")[1].split()) for l in lines)
    changed = sorted(k for k in first if first[k] != second[k])
    assert changed == ["system"] and set(first) >= {"model", "system", "contents", "schema", "tools"}
