"""A minimal multi-turn, tool-calling agent loop on top of :mod:`odin_agent.client`."""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

from .client import AsyncOdinClient, OdinClient
from .memory import ConversationMemory
from .types import Message, ModelResponse, Role, ToolCall, ToolDefinition


class OdinAgent:
    """Runs a chat model in a loop, executing the tools it asks for until it answers in text.

    Pass ``client`` and/or ``async_client`` explicitly. If one is missing it is created on first
    use from ``ODIN_BASE_URL`` / ``ODIN_API_KEY``; there is no built-in endpoint.
    """

    def __init__(
        self,
        model: str,
        client: OdinClient | None = None,
        async_client: AsyncOdinClient | None = None,
        system_prompt: str = "You are a helpful assistant.",
        name: str = "assistant",
        max_memory_messages: int = 40,
    ) -> None:
        self.name = name
        self.model = model
        self.system_prompt = system_prompt
        self._client = client
        self._async_client = async_client
        self.memory = ConversationMemory(max_messages=max_memory_messages, system_prompt=system_prompt)
        self._tools: dict[str, ToolDefinition] = {}

    @property
    def client(self) -> OdinClient:
        if self._client is None:
            self._client = OdinClient()
        return self._client

    @property
    def async_client(self) -> AsyncOdinClient:
        if self._async_client is None:
            self._async_client = AsyncOdinClient()
        return self._async_client

    def register_tool(
        self, name: str, description: str, parameters: dict[str, Any], handler: Callable[..., Any]
    ) -> None:
        """Register a Python function the model may call. ``parameters`` is a JSON Schema object."""
        self._tools[name] = ToolDefinition(name=name, description=description, parameters=parameters, handler=handler)

    def _execute_tool(self, tool_name: str, args: Any) -> Any:
        if isinstance(args, str):
            try:
                args = json.loads(args) if args.strip() else {}
            except json.JSONDecodeError:
                return {"error": f"Arguments for '{tool_name}' are not valid JSON."}
        if not isinstance(args, dict):
            return {"error": f"Arguments for '{tool_name}' must be a JSON object."}
        tool_def = self._tools.get(tool_name)
        if not tool_def or not tool_def.handler:
            return {"error": f"Tool '{tool_name}' is not registered."}
        try:
            return tool_def.handler(**args)
        except Exception as err:  # the model sees the error and can recover
            return {"error": f"Execution error in '{tool_name}': {err}"}

    def _tool_message(self, call: ToolCall) -> Message:
        result = self._execute_tool(call.function.name, call.function.arguments)
        return Message(
            role=Role.TOOL,
            name=call.function.name,
            content=json.dumps(result, ensure_ascii=False, default=str),
            tool_call_id=call.id,
        )

    def _after_response(self, response: ModelResponse) -> str | None:
        """Store the reply; return its text when the model is done, otherwise run the tools."""
        msg = response.message
        self.memory.add_message(msg)
        if not msg.tool_calls:
            return msg.content
        for call in msg.tool_calls:
            self.memory.add_message(self._tool_message(call))
        return None

    def _tools_or_none(self) -> list[ToolDefinition] | None:
        return list(self._tools.values()) if self._tools else None

    def run(self, user_prompt: str, max_turns: int = 6) -> str:
        """Run the loop synchronously and return the model's final text answer."""
        self.memory.add_message(Message(role=Role.USER, content=user_prompt))
        for _ in range(max_turns):
            response = self.client.chat_complete(
                messages=self.memory.get_messages(), model=self.model, tools=self._tools_or_none()
            )
            answer = self._after_response(response)
            if answer is not None:
                return answer
        return self.memory.get_messages()[-1].content

    async def arun(self, user_prompt: str, max_turns: int = 6) -> str:
        """Run the loop asynchronously and return the model's final text answer."""
        self.memory.add_message(Message(role=Role.USER, content=user_prompt))
        for _ in range(max_turns):
            response = await self.async_client.chat_complete(
                messages=self.memory.get_messages(), model=self.model, tools=self._tools_or_none()
            )
            answer = self._after_response(response)
            if answer is not None:
                return answer
        return self.memory.get_messages()[-1].content
