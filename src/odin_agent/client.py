"""HTTP and streaming client for Odin AI Platform."""

from __future__ import annotations

import os
import json
from typing import Any, AsyncIterator, Dict, Iterator, List, Optional
import httpx

from .types import Message, ModelResponse, Role, ToolDefinition, Usage


class OdinClient:
    """Synchronous and asynchronous client for communicating with Odin AI instances."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout: float = 60.0,
    ) -> None:
        self.api_key = api_key or os.getenv("ODIN_API_KEY", "")
        self.base_url = (base_url or os.getenv("ODIN_BASE_URL", "https://api.odinbilisim.com/v1")).rstrip("/")
        self.timeout = timeout
        self._headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "User-Agent": "Odin-Agent-SDK/0.1.0-python",
        }

    def chat_complete(
        self,
        messages: List[Message],
        model: str = "odin-v2-chat",
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        tools: Optional[List[ToolDefinition]] = None,
    ) -> ModelResponse:
        """Send a synchronous chat completion request to Odin AI."""
        payload: Dict[str, Any] = {
            "model": model,
            "messages": [m.model_dump(exclude_none=True) for m in messages],
            "temperature": temperature,
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

        with httpx.Client(timeout=self.timeout) as client:
            resp = client.post(
                f"{self.base_url}/chat/completions",
                headers=self._headers,
                json=payload,
            )
            resp.raise_for_status()
            data = resp.json()

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
        """Stream response tokens synchronously from Odin AI."""
        payload = {
            "model": model,
            "messages": [m.model_dump(exclude_none=True) for m in messages],
            "temperature": temperature,
            "stream": True,
        }

        with httpx.Client(timeout=self.timeout) as client:
            with client.stream(
                "POST",
                f"{self.base_url}/chat/completions",
                headers=self._headers,
                json=payload,
            ) as response:
                response.raise_for_status()
                for line in response.iter_lines():
                    if line.startswith("data: ") and line != "data: [DONE]":
                        chunk = json.loads(line[6:])
                        delta = chunk.get("choices", [{}])[0].get("delta", {})
                        if "content" in delta and delta["content"]:
                            yield delta["content"]
