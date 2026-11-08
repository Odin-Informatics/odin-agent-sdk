"""Synchronous and asynchronous clients for OpenAI-compatible chat completion APIs.

The SDK has no built-in endpoint: the caller must pass ``base_url`` (or set ``ODIN_BASE_URL``)
pointing at a server they run or trust, for example a local Ollama or vLLM instance.
"""

from __future__ import annotations

import json
import os
from collections.abc import AsyncIterator, Iterator
from typing import Any

import httpx

from ._version import __version__ as _version
from .exceptions import (
    AuthenticationError,
    NotFoundError,
    PermissionDeniedError,
    RateLimitError,
)
from .retry import with_retry, with_retry_async
from .types import Message, ModelResponse, Role, ToolDefinition, Usage


def _map_http_error(err: httpx.HTTPStatusError) -> None:
    code = err.response.status_code
    msg = err.response.text
    if code == 401:
        raise AuthenticationError(f"Invalid API key: {msg}", status_code=code) from err
    if code == 403:
        raise PermissionDeniedError(f"Access forbidden: {msg}", status_code=code) from err
    if code == 404:
        raise NotFoundError(f"Resource not found: {msg}", status_code=code) from err
    if code == 429:
        raise RateLimitError(f"Rate limit exceeded: {msg}", status_code=code) from err


def _resolve_base_url(base_url: str | None) -> str:
    url = base_url or os.getenv("ODIN_BASE_URL")
    if not url:
        raise ValueError(
            "base_url is required: pass base_url=... or set ODIN_BASE_URL "
            "(e.g. http://localhost:11434/v1 for a local Ollama server)."
        )
    return url.rstrip("/")


def _headers(api_key: str, flavour: str) -> dict[str, str]:
    headers = {"Content-Type": "application/json", "User-Agent": f"odin-agent-sdk/{_version} ({flavour})"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    return headers


def _message_payload(m: Message) -> dict[str, Any]:
    data = m.model_dump(exclude_none=True, mode="json")
    # OpenAI-compatible servers expect tool call arguments as a JSON string.
    for call in data.get("tool_calls") or []:
        args = call.get("function", {}).get("arguments")
        if not isinstance(args, str):
            call["function"]["arguments"] = json.dumps(args or {}, ensure_ascii=False)
    return data


def _build_payload(
    messages: list[Message],
    model: str,
    temperature: float,
    max_tokens: int | None,
    tools: list[ToolDefinition] | None,
    stream: bool = False,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": model,
        "messages": [_message_payload(m) for m in messages],
        "temperature": temperature,
        "stream": stream,
    }
    if max_tokens:
        payload["max_tokens"] = max_tokens
    if tools:
        payload["tools"] = [
            {"type": "function", "function": {"name": t.name, "description": t.description, "parameters": t.parameters}}
            for t in tools
        ]
    return payload


def _parse_response(data: dict[str, Any], model: str) -> ModelResponse:
    choice = (data.get("choices") or [{}])[0]
    msg_data = choice.get("message") or {}
    return ModelResponse(
        id=data.get("id", "res-unknown"),
        model=data.get("model", model),
        message=Message(
            role=Role(msg_data.get("role", "assistant")),
            content=msg_data.get("content") or "",
            tool_calls=msg_data.get("tool_calls"),
        ),
        finish_reason=choice.get("finish_reason"),
        usage=Usage(**(data.get("usage") or {})),
    )


def _delta_text(line: str) -> str | None:
    if not line.startswith("data: ") or line == "data: [DONE]":
        return None
    chunk = json.loads(line[6:])
    delta = (chunk.get("choices") or [{}])[0].get("delta") or {}
    content = delta.get("content")
    return content if isinstance(content, str) and content else None


class OdinClient:
    """Synchronous client with retries for an OpenAI-compatible ``/chat/completions`` endpoint."""

    def __init__(
        self,
        base_url: str | None = None,
        api_key: str | None = None,
        timeout: float = 60.0,
        max_retries: int = 3,
        http_client: httpx.Client | None = None,
    ) -> None:
        self.base_url = _resolve_base_url(base_url)
        self.api_key = api_key if api_key is not None else os.getenv("ODIN_API_KEY", "")
        self.timeout = timeout
        self.max_retries = max_retries
        self._headers = _headers(self.api_key, "python")
        self._client = http_client or httpx.Client(timeout=httpx.Timeout(timeout, connect=10.0))

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> OdinClient:
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()

    def chat_complete(
        self,
        messages: list[Message],
        model: str,
        temperature: float = 0.7,
        max_tokens: int | None = None,
        tools: list[ToolDefinition] | None = None,
    ) -> ModelResponse:
        """Send one chat completion request, retrying transient failures with backoff."""
        payload = _build_payload(messages, model, temperature, max_tokens, tools, stream=False)

        def _call() -> dict[str, Any]:
            try:
                resp = self._client.post(f"{self.base_url}/chat/completions", headers=self._headers, json=payload)
                resp.raise_for_status()
                data: dict[str, Any] = resp.json()
                return data
            except httpx.HTTPStatusError as err:
                _map_http_error(err)
                raise

        return _parse_response(with_retry(_call, max_retries=self.max_retries), model)

    def stream_chat(self, messages: list[Message], model: str, temperature: float = 0.7) -> Iterator[str]:
        """Yield response text chunks from a streaming completion."""
        payload = _build_payload(messages, model, temperature, None, None, stream=True)
        with self._client.stream(
            "POST", f"{self.base_url}/chat/completions", headers=self._headers, json=payload
        ) as response:
            try:
                response.raise_for_status()
            except httpx.HTTPStatusError as err:
                _map_http_error(err)
                raise
            for line in response.iter_lines():
                text = _delta_text(line)
                if text:
                    yield text


class AsyncOdinClient:
    """Asynchronous counterpart of :class:`OdinClient`."""

    def __init__(
        self,
        base_url: str | None = None,
        api_key: str | None = None,
        timeout: float = 60.0,
        max_retries: int = 3,
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        self.base_url = _resolve_base_url(base_url)
        self.api_key = api_key if api_key is not None else os.getenv("ODIN_API_KEY", "")
        self.timeout = timeout
        self.max_retries = max_retries
        self._headers = _headers(self.api_key, "python-async")
        self._client = http_client or httpx.AsyncClient(timeout=httpx.Timeout(timeout, connect=10.0))

    async def close(self) -> None:
        await self._client.aclose()

    async def __aenter__(self) -> AsyncOdinClient:
        return self

    async def __aexit__(self, *args: Any) -> None:
        await self.close()

    async def chat_complete(
        self,
        messages: list[Message],
        model: str,
        temperature: float = 0.7,
        max_tokens: int | None = None,
        tools: list[ToolDefinition] | None = None,
    ) -> ModelResponse:
        """Send one chat completion request asynchronously, retrying transient failures."""
        payload = _build_payload(messages, model, temperature, max_tokens, tools, stream=False)

        async def _call() -> dict[str, Any]:
            try:
                resp = await self._client.post(f"{self.base_url}/chat/completions", headers=self._headers, json=payload)
                resp.raise_for_status()
                data: dict[str, Any] = resp.json()
                return data
            except httpx.HTTPStatusError as err:
                _map_http_error(err)
                raise

        return _parse_response(await with_retry_async(_call, max_retries=self.max_retries), model)

    async def stream_chat(self, messages: list[Message], model: str, temperature: float = 0.7) -> AsyncIterator[str]:
        """Yield response text chunks from a streaming completion."""
        payload = _build_payload(messages, model, temperature, None, None, stream=True)
        async with self._client.stream(
            "POST", f"{self.base_url}/chat/completions", headers=self._headers, json=payload
        ) as response:
            try:
                response.raise_for_status()
            except httpx.HTTPStatusError as err:
                _map_http_error(err)
                raise
            async for line in response.aiter_lines():
                text = _delta_text(line)
                if text:
                    yield text
