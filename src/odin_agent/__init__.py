"""Odin Agent SDK - Official Python SDK for Odin AI Models and Intelligent Agents."""

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

__version__ = "0.2.0"
__all__ = [
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
