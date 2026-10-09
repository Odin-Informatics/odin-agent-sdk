"""Autonomous Multi-Turn Agent implementation for Odin AI models."""

from __future__ import annotations

import json
from typing import Any, Callable, Dict, List, Optional

from .client import AsyncOdinClient, OdinClient
from .memory import ConversationMemory
from .types import Message, ModelResponse, Role, ToolDefinition


class OdinAgent:
    """An autonomous conversational and tool-using agent powered by Odin models."""

    def __init__(
        self,
        name: str = "OdinAssistant",
        system_prompt: str = "Sen Odin Informatics tarafından geliştirilmiş kurumsal bir yapay zeka asistanısın.",
        client: Optional[OdinClient] = None,
        async_client: Optional[AsyncOdinClient] = None,
        model: str = "odin-v2-chat",
        max_memory_messages: int = 40,
    ) -> None:
        self.name = name
        self.system_prompt = system_prompt
        self.client = client or OdinClient()
        self.async_client = async_client or AsyncOdinClient()
        self.model = model
        self.memory = ConversationMemory(
            max_messages=max_memory_messages,
            system_prompt=system_prompt,
        )
        self._tools: Dict[str, ToolDefinition] = {}

    def register_tool(
        self,
        name: str,
        description: str,
        parameters: Dict[str, Any],
        handler: Callable[..., Any],
    ) -> None:
        """Register an executable tool function to the agent."""
        self._tools[name] = ToolDefinition(
            name=name,
            description=description,
            parameters=parameters,
            handler=handler,
        )

    def _execute_tool(self, tool_name: str, args: Any) -> Any:
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except Exception:
                args = {}
        tool_def = self._tools.get(tool_name)
        if not tool_def or not tool_def.handler:
            return {"error": f"Tool '{tool_name}' is not registered."}
        try:
            return tool_def.handler(**args)
        except Exception as err:
            return {"error": f"Execution error in '{tool_name}': {str(err)}"}

    def run(self, user_prompt: str, max_turns: int = 6) -> str:
        """Synchronously execute multi-turn agentic reasoning loop."""
        self.memory.add_message(Message(role=Role.USER, content=user_prompt))

        for _ in range(max_turns):
            response = self.client.chat_complete(
                messages=self.memory.get_messages(),
                model=self.model,
                tools=list(self._tools.values()) if self._tools else None,
            )

            assistant_msg = response.message
            self.memory.add_message(assistant_msg)

            if not assistant_msg.tool_calls:
                return assistant_msg.content

            # Dispatch each requested tool call
            for tool_call in assistant_msg.tool_calls:
                result = self._execute_tool(
                    tool_call.function.name,
                    tool_call.function.arguments,
                )
                self.memory.add_message(
                    Message(
                        role=Role.TOOL,
                        name=tool_call.function.name,
                        content=json.dumps(result, ensure_ascii=False),
                        tool_call_id=tool_call.id,
                    )
                )

        return self.memory.get_messages()[-1].content

    async def arun(self, user_prompt: str, max_turns: int = 6) -> str:
        """Asynchronously execute multi-turn agentic reasoning loop."""
        self.memory.add_message(Message(role=Role.USER, content=user_prompt))

        for _ in range(max_turns):
            response = await self.async_client.chat_complete(
                messages=self.memory.get_messages(),
                model=self.model,
                tools=list(self._tools.values()) if self._tools else None,
            )

            assistant_msg = response.message
            self.memory.add_message(assistant_msg)

            if not assistant_msg.tool_calls:
                return assistant_msg.content

            for tool_call in assistant_msg.tool_calls:
                result = self._execute_tool(
                    tool_call.function.name,
                    tool_call.function.arguments,
                )
                self.memory.add_message(
                    Message(
                        role=Role.TOOL,
                        name=tool_call.function.name,
                        content=json.dumps(result, ensure_ascii=False),
                        tool_call_id=tool_call.id,
                    )
                )

        return self.memory.get_messages()[-1].content
