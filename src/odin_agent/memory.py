"""Conversation memory management for autonomous agents."""

from __future__ import annotations

from typing import List, Optional
from .types import Message, Role


class ConversationMemory:
    """Manages conversational state with message count limits and system prompt preservation."""

    def __init__(self, max_messages: int = 40, system_prompt: Optional[str] = None) -> None:
        self.max_messages = max_messages
        self._messages: List[Message] = []
        if system_prompt:
            self._messages.append(Message(role=Role.SYSTEM, content=system_prompt))

    def add_message(self, message: Message) -> None:
        """Add a message to the memory buffer, maintaining sliding window constraints."""
        self._messages.append(message)
        self._truncate_if_needed()

    def get_messages(self) -> List[Message]:
        """Return the current conversation history."""
        return list(self._messages)

    def clear(self, keep_system: bool = True) -> None:
        """Reset conversation memory."""
        if keep_system and self._messages and self._messages[0].role == Role.SYSTEM:
            system_msg = self._messages[0]
            self._messages = [system_msg]
        else:
            self._messages = []

    def _truncate_if_needed(self) -> None:
        """Preserve system prompt while removing oldest user/assistant turns if buffer exceeds capacity."""
        if len(self._messages) <= self.max_messages:
            return

        has_system = self._messages and self._messages[0].role == Role.SYSTEM
        if has_system:
            system_msg = self._messages[0]
            # Keep system prompt + the most recent (max_messages - 1) items
            self._messages = [system_msg] + self._messages[-(self.max_messages - 1) :]
        else:
            self._messages = self._messages[-self.max_messages :]
