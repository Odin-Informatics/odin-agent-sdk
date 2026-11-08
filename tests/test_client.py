"""Client tests. All HTTP traffic goes to httpx.MockTransport; nothing leaves the machine."""

from __future__ import annotations

import json
from typing import Any

import httpx
import pytest

from odin_agent import AsyncOdinClient, AuthenticationError, Message, OdinClient, Role
from odin_agent.client import _build_payload
from odin_agent.types import FunctionCall, ToolCall, ToolDefinition

BASE = "http://test.local/v1"


def _completion(content: str = "hi", tool_calls: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    msg: dict[str, Any] = {"role": "assistant", "content": content}
    if tool_calls:
        msg["tool_calls"] = tool_calls
    return {
        "id": "r1",
        "model": "m",
        "choices": [{"message": msg, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5},
    }


def test_base_url_is_required(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ODIN_BASE_URL", raising=False)
    with pytest.raises(ValueError, match="base_url is required"):
        OdinClient()
    with pytest.raises(ValueError):
        AsyncOdinClient()


def test_base_url_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ODIN_BASE_URL", "http://env.local/v1/")
    assert OdinClient().base_url == "http://env.local/v1"


def test_no_auth_header_without_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ODIN_API_KEY", raising=False)
    seen: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["auth"] = request.headers.get("authorization")
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json=_completion("merhaba"))

    client = OdinClient(base_url=BASE, http_client=httpx.Client(transport=httpx.MockTransport(handler)))
    res = client.chat_complete([Message(role=Role.USER, content="selam")], model="llama3")
    assert res.message.content == "merhaba"
    assert res.usage is not None and res.usage.total_tokens == 5
    assert seen["auth"] is None
    assert seen["body"]["model"] == "llama3"
    client.close()


def test_auth_header_and_401_mapping() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers["authorization"] == "Bearer k"
        return httpx.Response(401, text="bad key")

    transport = httpx.MockTransport(handler)
    with (
        OdinClient(base_url=BASE, api_key="k", http_client=httpx.Client(transport=transport)) as c,
        pytest.raises(AuthenticationError),
    ):
        c.chat_complete([Message(role=Role.USER, content="x")], model="m")


def test_tool_call_arguments_string_is_parsed_and_sent_back_as_string() -> None:
    raw = [{"id": "c1", "type": "function", "function": {"name": "f", "arguments": '{"a": 1}'}}]

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=_completion("", raw))

    c = OdinClient(base_url=BASE, http_client=httpx.Client(transport=httpx.MockTransport(handler)))
    msg = c.chat_complete([Message(role=Role.USER, content="x")], model="m").message
    assert msg.tool_calls and msg.tool_calls[0].function.arguments == '{"a": 1}'

    dict_args = Message(
        role=Role.ASSISTANT, tool_calls=[ToolCall(id="c2", function=FunctionCall(name="g", arguments={"b": 2}))]
    )
    tool = ToolDefinition(name="g", description="d", parameters={"type": "object"})
    payload = _build_payload([dict_args], "m", 0.1, 50, [tool])
    sent = payload["messages"][0]["tool_calls"][0]["function"]["arguments"]
    assert json.loads(sent) == {"b": 2}
    assert payload["max_tokens"] == 50 and payload["tools"][0]["function"]["name"] == "g"


def test_stream_chat() -> None:
    body = "\n".join(
        [
            'data: {"choices":[{"delta":{"content":"Mer"}}]}',
            'data: {"choices":[{"delta":{"content":"haba"}}]}',
            'data: {"choices":[{"delta":{}}]}',
            "data: [DONE]",
        ]
    )

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text=body)

    c = OdinClient(base_url=BASE, http_client=httpx.Client(transport=httpx.MockTransport(handler)))
    assert "".join(c.stream_chat([Message(role=Role.USER, content="x")], model="m")) == "Merhaba"


async def test_async_client_and_stream() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if json.loads(request.content).get("stream"):
            return httpx.Response(200, text='data: {"choices":[{"delta":{"content":"ok"}}]}\ndata: [DONE]')
        return httpx.Response(200, json=_completion("async"))

    async with AsyncOdinClient(
        base_url=BASE, http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler))
    ) as c:
        res = await c.chat_complete([Message(role=Role.USER, content="x")], model="m")
        assert res.message.content == "async"
        chunks = [t async for t in c.stream_chat([Message(role=Role.USER, content="x")], model="m")]
        assert chunks == ["ok"]


async def test_async_error_mapping() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, text="no")

    c = AsyncOdinClient(base_url=BASE, http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)))
    with pytest.raises(AuthenticationError):
        await c.chat_complete([Message(role=Role.USER, content="x")], model="m")
    await c.close()
