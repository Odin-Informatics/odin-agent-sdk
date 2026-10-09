"""Exponential backoff with jitter retry strategy for resilient network calls."""

from __future__ import annotations

import asyncio
import random
import time
from collections.abc import Callable, Coroutine
from typing import Any, TypeVar

import httpx

from .exceptions import APIConnectionError, InternalServerError, RateLimitError

T = TypeVar("T")

RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}


def calculate_backoff(attempt: int, base_delay: float = 0.5, max_delay: float = 10.0) -> float:
    """Calculate exponential backoff with full jitter to avoid retry storms."""
    exp = min(max_delay, base_delay * (2 ** attempt))
    return random.uniform(0, exp)


def with_retry(
    fn: Callable[[], T],
    max_retries: int = 3,
    base_delay: float = 0.5,
    max_delay: float = 10.0,
) -> T:
    """Execute a synchronous function with automatic exponential backoff retry."""
    last_err: Exception = Exception("Unknown error")
    for attempt in range(max_retries + 1):
        try:
            return fn()
        except httpx.HTTPStatusError as err:
            status = err.response.status_code
            if status in RETRYABLE_STATUS_CODES and attempt < max_retries:
                delay = calculate_backoff(attempt, base_delay, max_delay)
                time.sleep(delay)
                last_err = err
                continue
            if status == 429:
                raise RateLimitError("Rate limit exceeded.", status_code=status) from err
            if status >= 500:
                raise InternalServerError("Odin Cloud internal server error.", status_code=status) from err
            raise
        except (httpx.ConnectError, httpx.TimeoutException) as err:
            if attempt < max_retries:
                delay = calculate_backoff(attempt, base_delay, max_delay)
                time.sleep(delay)
                last_err = err
                continue
            raise APIConnectionError("Failed to connect to Odin API endpoints.") from err
    raise last_err


async def with_retry_async(
    fn: Callable[[], Coroutine[Any, Any, T]],
    max_retries: int = 3,
    base_delay: float = 0.5,
    max_delay: float = 10.0,
) -> T:
    """Execute an asynchronous coroutine with exponential backoff retry."""
    last_err: Exception = Exception("Unknown error")
    for attempt in range(max_retries + 1):
        try:
            return await fn()
        except httpx.HTTPStatusError as err:
            status = err.response.status_code
            if status in RETRYABLE_STATUS_CODES and attempt < max_retries:
                delay = calculate_backoff(attempt, base_delay, max_delay)
                await asyncio.sleep(delay)
                last_err = err
                continue
            if status == 429:
                raise RateLimitError("Rate limit exceeded.", status_code=status) from err
            if status >= 500:
                raise InternalServerError("Odin Cloud internal server error.", status_code=status) from err
            raise
        except (httpx.ConnectError, httpx.TimeoutException) as err:
            if attempt < max_retries:
                delay = calculate_backoff(attempt, base_delay, max_delay)
                await asyncio.sleep(delay)
                last_err = err
                continue
            raise APIConnectionError("Failed to connect to Odin API endpoints.") from err
    raise last_err
