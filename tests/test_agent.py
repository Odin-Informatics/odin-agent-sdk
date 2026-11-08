"""Agent loop tests with a scripted fake server (httpx.MockTransport)."""

from __future__ import annotations

import json
from typing import Any

import httpx
import pytest

from odin_agent import AsyncOdinClient, OdinAgent, OdinClient

BASE = "http://test.local/v1"


def _scripted(replies: list[dict[str, Any]], seen: list[dict[str, Any]]) -> httpx.MockTransport:
    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(json.loads(request.content))
        msg = replies[min(len(seen) - 1, len(replies) - 1)]
        return httpx.Response(200, json={"id": "r", "model": "m", "choices": [{"message": msg}]})

    return httpx.MockTransport(handler)


TOOL_TURN = {
    "role": "assistant",
    "content": None,
    "tool_calls": [{"id": "c1", "type": "function", "function": {"name": "stok", "arguments": '{"kod": "A1"}'}}],
}
FINAL = {"role": "assistant", "content": "A1 stokta 42 adet var."}


def _agent(seen: list[dict[str, Any]], replies: list[dict[str, Any]]) -> OdinAgent:
    client = OdinClient(base_url=BASE, http_client=httpx.Client(transport=_scripted(replies, seen)))
    agent = OdinAgent(model="m", client=client)
    agent.register_tool(
        "stok",
        "Stok sorgular",
        {"type": "object", "properties": {"kod": {"type": "string"}}},
        lambda kod: {"kod": kod, "adet": 42},
    )
    return agent


def test_run_executes_tool_then_answers() -> None:
    seen: list[dict[str, Any]] = []
    answer = _agent(seen, [TOOL_TURN, FINAL]).run("A1 stok?")
    assert answer == "A1 stokta 42 adet var."
    tool_msg = seen[1]["messages"][-1]
    assert tool_msg["role"] == "tool" and json.loads(tool_msg["content"]) == {"kod": "A1", "adet": 42}
    # the assistant tool call is replayed with string arguments
    assert seen[1]["messages"][-2]["tool_calls"][0]["function"]["arguments"] == '{"kod": "A1"}'


def test_run_stops_after_max_turns() -> None:
    seen: list[dict[str, Any]] = []
    out = _agent(seen, [TOOL_TURN]).run("loop", max_turns=2)
    assert len(seen) == 2 and "42" in out


async def test_arun() -> None:
    seen: list[dict[str, Any]] = []
    aclient = AsyncOdinClient(base_url=BASE, http_client=httpx.AsyncClient(transport=_scripted([FINAL], seen)))
    agent = OdinAgent(model="m", async_client=aclient)
    assert await agent.arun("selam") == "A1 stokta 42 adet var."


async def test_arun_with_tools() -> None:
    seen: list[dict[str, Any]] = []
    aclient = AsyncOdinClient(
        base_url=BASE, http_client=httpx.AsyncClient(transport=_scripted([TOOL_TURN, FINAL], seen))
    )
    agent = OdinAgent(model="m", async_client=aclient)
    agent.register_tool("stok", "d", {"type": "object"}, lambda kod: {"adet": 1})
    assert await agent.arun("x") == FINAL["content"]


@pytest.mark.parametrize(
    ("name", "args", "expect"),
    [
        ("yok", "{}", "not registered"),
        ("stok", "{bozuk", "not valid JSON"),
        ("stok", "[1]", "must be a JSON object"),
        ("patlak", "{}", "Execution error"),
    ],
)
def test_tool_errors_are_returned_to_model(name: str, args: str, expect: str) -> None:
    agent = OdinAgent(model="m", client=OdinClient(base_url=BASE))
    agent.register_tool("stok", "d", {"type": "object"}, lambda **_: {})
    agent.register_tool("patlak", "d", {"type": "object"}, lambda: 1 / 0)
    assert expect in agent._execute_tool(name, args)["error"]


def test_default_client_needs_base_url(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ODIN_BASE_URL", raising=False)
    agent = OdinAgent(model="m")
    with pytest.raises(ValueError):
        _ = agent.client
    with pytest.raises(ValueError):
        _ = agent.async_client
