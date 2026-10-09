"""Resilient Synchronous and Asynchronous HTTP clients for Odin AI Platform."""

from __future__ import annotations

import os
import json
from typing import Any, AsyncIterator, Dict, Iterator, List, Optional
import httpx

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
        raise AuthenticationError(f"Invalid API Key: {msg}", status_code=code) from err
    if code == 403:
        raise PermissionDeniedError(f"Access forbidden: {msg}", status_code=code) from err
    if code == 404:
        raise NotFoundError(f"Resource not found: {msg}", status_code=code) from err
    if code == 429:
        raise RateLimitError(f"Rate limit exceeded: {msg}", status_code=code) from err


def _build_payload(
    messages: List[Message],
    model: str,
    temperature: float,
    max_tokens: Optional[int],
    tools: Optional[List[ToolDefinition]],
    stream: bool = False,
) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "model": model,
        "messages": [m.model_dump(exclude_none=True) for m in messages],
        "temperature": temperature,
        "stream": stream,
    }
    if max_tokens:
        payload["max_tokens"] = max_tokens
    if tools:
        payload["tools"] = [
            {
                "type": "function",
                "function": {
                    "name": t.name,
                    "description": t.description,
                    "parameters": t.parameters,
                },
            }
            for t in tools
        ]
    return payload


class OdinClient:
    """Production-grade synchronous client with automatic retries and connection pooling."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout: float = 60.0,
        max_retries: int = 3,
    ) -> None:
        self.api_key = api_key or os.getenv("ODIN_API_KEY", "")
        self.base_url = (base_url or os.getenv("ODIN_BASE_URL", "https://api.odinbilisim.com/v1")).rstrip("/")
        self.timeout = timeout
        self.max_retries = max_retries
        self._headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "User-Agent": "Odin-Agent-SDK/0.1.0-python",
        }
        self._client = httpx.Client(
            timeout=httpx.Timeout(timeout, connect=10.0),
            limits=httpx.Limits(max_keepalive_connections=20, max_connections=100),
        )

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> "OdinClient":
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()

    def chat_complete(
        self,
        messages: List[Message],
        model: str = "odin-v2-chat",
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        tools: Optional[List[ToolDefinition]] = None,
    ) -> ModelResponse:
        """Send a synchronous chat completion request with exponential backoff."""
        payload = _build_payload(messages, model, temperature, max_tokens, tools, stream=False)

        def _call() -> Dict[str, Any]:
            try:
                resp = self._client.post(
                    f"{self.base_url}/chat/completions",
                    headers=self._headers,
                    json=payload,
                )
                resp.raise_for_status()
                return resp.json()
            except httpx.HTTPStatusError as err:
                _map_http_error(err)
                raise

        data = with_retry(_call, max_retries=self.max_retries)
        choice = data.get("choices", [{}])[0]
        msg_data = choice.get("message", {})
        return ModelResponse(
            id=data.get("id", "res-unknown"),
            model=data.get("model", model),
            message=Message(
                role=Role(msg_data.get("role", "assistant")),
                content=msg_data.get("content") or "",
                tool_calls=msg_data.get("tool_calls"),
            ),
            finish_reason=choice.get("finish_reason"),
            usage=Usage(**data.get("usage", {})),
        )

    def stream_chat(
        self,
        messages: List[Message],
        model: str = "odin-v2-chat",
        temperature: float = 0.7,
    ) -> Iterator[str]:
        """Stream response tokens synchronously."""
        payload = _build_payload(messages, model, temperature, None, None, stream=True)
        with self._client.stream(
            "POST",
            f"{self.base_url}/chat/completions",
            headers=self._headers,
            json=payload,
        ) as response:
            try:
                response.raise_for_status()
            except httpx.HTTPStatusError as err:
                _map_http_error(err)
                raise
            for line in response.iter_lines():
                if line.startswith("data: ") and line != "data: [DONE]":
                    chunk = json.loads(line[6:])
                    delta = chunk.get("choices", [{}])[0].get("delta", {})
                    if "content" in delta and delta["content"]:
                        yield delta["content"]


class AsyncOdinClient:
    """Production-grade asynchronous client for asyncio environments."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout: float = 60.0,
        max_retries: int = 3,
    ) -> None:
        self.api_key = api_key or os.getenv("ODIN_API_KEY", "")
        self.base_url = (base_url or os.getenv("ODIN_BASE_URL", "https://api.odinbilisim.com/v1")).rstrip("/")
        self.timeout = timeout
        self.max_retries = max_retries
        self._headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "User-Agent": "Odin-Agent-SDK/0.1.0-async-python",
        }
        self._client = httpx.AsyncClient(
            timeout=httpx.Timeout(timeout, connect=10.0),
            limits=httpx.Limits(max_keepalive_connections=20, max_connections=100),
        )

    async def close(self) -> None:
        await self._client.aclose()

    async def __aenter__(self) -> "AsyncOdinClient":
        return self

    async def __aexit__(self, *args: Any) -> None:
        await self.close()

    async def chat_complete(
        self,
        messages: List[Message],
        model: str = "odin-v2-chat",
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        tools: Optional[List[ToolDefinition]] = None,
    ) -> ModelResponse:
        """Asynchronously send chat completion with exponential backoff retry."""
        payload = _build_payload(messages, model, temperature, max_tokens, tools, stream=False)

        async def _call() -> Dict[str, Any]:
            try:
                resp = await self._client.post(
                    f"{self.base_url}/chat/completions",
                    headers=self._headers,
                    json=payload,
                )
                resp.raise_for_status()
                return resp.json()
            except httpx.HTTPStatusError as err:
                _map_http_error(err)
                raise

        data = await with_retry_async(_call, max_retries=self.max_retries)
        choice = data.get("choices", [{}])[0]
        msg_data = choice.get("message", {})
        return ModelResponse(
            id=data.get("id", "res-unknown"),
            model=data.get("model", model),
            message=Message(
                role=Role(msg_data.get("role", "assistant")),
                content=msg_data.get("content") or "",
                tool_calls=msg_data.get("tool_calls"),
            ),
            finish_reason=choice.get("finish_reason"),
            usage=Usage(**data.get("usage", {})),
        )

    async def stream_chat(
        self,
        messages: List[Message],
        model: str = "odin-v2-chat",
        temperature: float = 0.7,
    ) -> AsyncIterator[str]:
        """Asynchronously stream response tokens."""
        payload = _build_payload(messages, model, temperature, None, None, stream=True)
        async with self._client.stream(
            "POST",
            f"{self.base_url}/chat/completions",
            headers=self._headers,
            json=payload,
        ) as response:
            try:
                response.raise_for_status()
            except httpx.HTTPStatusError as err:
                _map_http_error(err)
                raise
            async for line in response.aiter_lines():
                if line.startswith("data: ") and line != "data: [DONE]":
                    chunk = json.loads(line[6:])
                    delta = chunk.get("choices", [{}])[0].get("delta", {})
                    if "content" in delta and delta["content"]:
                        yield delta["content"]
