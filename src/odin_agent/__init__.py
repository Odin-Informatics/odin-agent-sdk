"""odin-agent-sdk: a small, experimental agent toolkit for OpenAI-compatible chat APIs."""

from ._version import __version__
from .agent import OdinAgent
from .client import AsyncOdinClient, OdinClient
from .exceptions import (
    APIConnectionError,
    AuthenticationError,
    InternalServerError,
    NotFoundError,
    OdinError,
    PermissionDeniedError,
    RateLimitError,
)
from .memory import ConversationMemory
from .types import (
    FunctionCall,
    Message,
    ModelResponse,
    Role,
    ToolCall,
    ToolDefinition,
    Usage,
)

__all__ = [
    "__version__",
    "OdinAgent",
    "OdinClient",
    "AsyncOdinClient",
    "ConversationMemory",
    "Message",
    "Role",
    "FunctionCall",
    "ToolCall",
    "ToolDefinition",
    "ModelResponse",
    "Usage",
    "OdinError",
    "AuthenticationError",
    "PermissionDeniedError",
    "NotFoundError",
    "RateLimitError",
    "APIConnectionError",
    "InternalServerError",
]
