"""Autonomous Agent implementation for Odin AI models."""

from __future__ import annotations

import json
from typing import Any, Callable, Dict, List, Optional

from .client import OdinClient
from .types import Message, ModelResponse, Role, ToolDefinition


class OdinAgent:
    """An autonomous conversational and tool-using agent powered by Odin models."""

    def __init__(
        self,
        name: str = "OdinAssistant",
        system_prompt: str = "Sen Odin Informatics tarafından geliştirilmiş kurumsal bir yapay zeka asistanısın.",
        client: Optional[OdinClient] = None,
        model: str = "odin-v2-chat",
    ) -> None:
        self.name = name
        self.system_prompt = system_prompt
        self.client = client or OdinClient()
        self.model = model
        self.history: List[Message] = [Message(role=Role.SYSTEM, content=system_prompt)]
        self._tools: Dict[str, ToolDefinition] = {}

    def register_tool(
        self,
        name: str,
        description: str,
        parameters: Dict[str, Any],
        handler: Callable[..., Any],
    ) -> None:
        """Register an executable tool to the agent."""
        self._tools[name] = ToolDefinition(
            name=name,
            description=description,
            parameters=parameters,
            handler=handler,
        )

    def run(self, user_prompt: str, max_turns: int = 5) -> str:
        """Run an autonomous interaction turn, executing tools when necessary."""
        self.history.append(Message(role=Role.USER, content=user_prompt))

        for _ in range(max_turns):
            response = self.client.chat_complete(
                messages=self.history,
                model=self.model,
                tools=list(self._tools.values()) if self._tools else None,
            )

            assistant_msg = response.message
            self.history.append(assistant_msg)

            if not assistant_msg.tool_calls:
                return assistant_msg.content

            # Execute tool calls
            for tool_call in assistant_msg.tool_calls:
                fn_name = tool_call.function.name
                fn_args = tool_call.function.arguments
                if isinstance(fn_args, str):
                    fn_args = json.loads(fn_args)

                tool_def = self._tools.get(fn_name)
                if tool_def and tool_def.handler:
                    try:
                        result = tool_def.handler(**fn_args)
                    except Exception as err:
                        result = {"error": str(err)}
                else:
                    result = {"error": f"Tool '{fn_name}' not found."}

                self.history.append(
                    Message(
                        role=Role.TOOL,
                        name=fn_name,
                        content=json.dumps(result, ensure_ascii=False),
                        tool_call_id=tool_call.id,
                    )
                )

        return self.history[-1].content
