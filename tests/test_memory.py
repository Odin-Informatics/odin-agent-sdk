import pytest
from odin_agent.memory import ConversationMemory
from odin_agent.types import Message, Role


def test_memory_preserves_system_prompt():
    memory = ConversationMemory(max_messages=3, system_prompt="System Prompt")
    memory.add_message(Message(role=Role.USER, content="Hello 1"))
    memory.add_message(Message(role=Role.ASSISTANT, content="Hi 1"))
    memory.add_message(Message(role=Role.USER, content="Hello 2"))

    messages = memory.get_messages()
    assert len(messages) == 3
    assert messages[0].role == Role.SYSTEM
    assert messages[0].content == "System Prompt"
    assert messages[-1].content == "Hello 2"


def test_memory_clear():
    memory = ConversationMemory(max_messages=10, system_prompt="System")
    memory.add_message(Message(role=Role.USER, content="Msg"))
    assert len(memory.get_messages()) == 2

    memory.clear(keep_system=True)
    assert len(memory.get_messages()) == 1
    assert memory.get_messages()[0].role == Role.SYSTEM

    memory.clear(keep_system=False)
    assert len(memory.get_messages()) == 0
