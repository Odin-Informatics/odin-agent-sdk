"""Custom exception hierarchy for Odin Agent SDK."""

from __future__ import annotations
from typing import Optional


class OdinError(Exception):
    """Base exception for all Odin SDK errors."""

    def __init__(self, message: str, status_code: Optional[int] = None) -> None:
        super().__init__(message)
        self.message = message
        self.status_code = status_code


class AuthenticationError(OdinError):
    """Raised when authentication fails (HTTP 401)."""
    pass


class PermissionDeniedError(OdinError):
    """Raised when access is forbidden (HTTP 403)."""
    pass


class NotFoundError(OdinError):
    """Raised when a requested resource is missing (HTTP 404)."""
    pass


class RateLimitError(OdinError):
    """Raised when API rate limits are exceeded (HTTP 429)."""
    pass


class APIConnectionError(OdinError):
    """Raised when network or DNS connection to Odin API fails."""
    pass


class InternalServerError(OdinError):
    """Raised when Odin Cloud experiences an internal error (HTTP 5xx)."""
    pass
