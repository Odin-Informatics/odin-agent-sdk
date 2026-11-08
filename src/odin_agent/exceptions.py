"""Custom exception hierarchy for Odin Agent SDK."""

from __future__ import annotations


class OdinError(Exception):
    """Base exception for all odin-agent-sdk errors."""

    def __init__(self, message: str, status_code: int | None = None) -> None:
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
    """Raised when the configured endpoint cannot be reached."""
    pass


class InternalServerError(OdinError):
    """Raised when the server returns HTTP 5xx."""
    pass
