"""Comprehensive tests for retry logic — sync, async, backoff, and storm scenarios."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from odin_agent.exceptions import APIConnectionError, InternalServerError, RateLimitError
from odin_agent.retry import calculate_backoff, with_retry, with_retry_async


# ---------------------------------------------------------------------------
# calculate_backoff
# ---------------------------------------------------------------------------


def test_calculate_backoff_within_bounds():
    """Backoff must be in [0, max_delay] at every attempt."""
    for attempt in range(10):
        delay = calculate_backoff(attempt=attempt, base_delay=0.1, max_delay=1.0)
        assert 0.0 <= delay <= 1.0, f"Delay {delay} out of bounds at attempt {attempt}"


def test_calculate_backoff_never_exceeds_max_delay():
    """Exponential factor must be capped at max_delay regardless of attempt count."""
    for attempt in range(30):
        delay = calculate_backoff(attempt=attempt, base_delay=1.0, max_delay=5.0)
        assert delay <= 5.0


def test_calculate_backoff_zero_attempt():
    """Attempt 0 should return a delay between 0 and base_delay."""
    delay = calculate_backoff(attempt=0, base_delay=0.5, max_delay=10.0)
    assert 0.0 <= delay <= 0.5


# ---------------------------------------------------------------------------
# with_retry — sync
# ---------------------------------------------------------------------------


def test_with_retry_success_first_call():
    """A function that succeeds immediately should return without retry."""
    calls: list[int] = []

    def fn():
        calls.append(1)
        return "ok"

    result = with_retry(fn, max_retries=3)
    assert result == "ok"
    assert len(calls) == 1


def test_with_retry_succeeds_after_transient_failure():
    """Function should succeed after retryable failures."""
    attempts = [0]

    mock_response = MagicMock()
    mock_response.status_code = 503

    def fn():
        attempts[0] += 1
        if attempts[0] < 3:
            raise httpx.HTTPStatusError(
                "Service Unavailable",
                request=MagicMock(),
                response=mock_response,
            )
        return "recovered"

    with patch("odin_agent.retry.time.sleep"):
        result = with_retry(fn, max_retries=3, base_delay=0.0)

    assert result == "recovered"
    assert attempts[0] == 3


def test_with_retry_raises_rate_limit_error():
    """HTTP 429 must be translated to RateLimitError after exhausting retries."""
    mock_response = MagicMock()
    mock_response.status_code = 429

    def fn():
        raise httpx.HTTPStatusError("Too Many Requests", request=MagicMock(), response=mock_response)

    with patch("odin_agent.retry.time.sleep"), pytest.raises(RateLimitError):
        with_retry(fn, max_retries=2, base_delay=0.0)


def test_with_retry_raises_internal_server_error():
    """HTTP 500 must be translated to InternalServerError after exhausting retries."""
    mock_response = MagicMock()
    mock_response.status_code = 500

    def fn():
        raise httpx.HTTPStatusError("Internal Server Error", request=MagicMock(), response=mock_response)

    with patch("odin_agent.retry.time.sleep"), pytest.raises(InternalServerError):
        with_retry(fn, max_retries=2, base_delay=0.0)


def test_with_retry_raises_api_connection_error():
    """Persistent connection errors must be translated to APIConnectionError."""

    def fn():
        raise httpx.ConnectError("Connection refused")

    with patch("odin_agent.retry.time.sleep"), pytest.raises(APIConnectionError):
        with_retry(fn, max_retries=2, base_delay=0.0)


def test_with_retry_non_retryable_http_status_reraises():
    """Non-retryable HTTP status codes (e.g. 400) must be re-raised immediately."""
    mock_response = MagicMock()
    mock_response.status_code = 400

    def fn():
        raise httpx.HTTPStatusError("Bad Request", request=MagicMock(), response=mock_response)

    with pytest.raises(httpx.HTTPStatusError):
        with_retry(fn, max_retries=3, base_delay=0.0)


def test_with_retry_storm_prevention():
    """All retries must sleep — verifying no retry-storm (sleep called N times)."""
    mock_response = MagicMock()
    mock_response.status_code = 503
    max_retries = 4

    def fn():
        raise httpx.HTTPStatusError("Service Unavailable", request=MagicMock(), response=mock_response)

    with patch("odin_agent.retry.time.sleep") as mock_sleep, pytest.raises(InternalServerError):
        with_retry(fn, max_retries=max_retries, base_delay=0.0)

    assert mock_sleep.call_count == max_retries, (
        f"Expected {max_retries} sleeps to prevent retry storm, got {mock_sleep.call_count}"
    )


# ---------------------------------------------------------------------------
# with_retry_async — async
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_with_retry_async_success():
    """Async function that succeeds immediately should return without retry."""
    calls: list[int] = []

    async def fn():
        calls.append(1)
        return "async_ok"

    result = await with_retry_async(fn, max_retries=3)
    assert result == "async_ok"
    assert len(calls) == 1


@pytest.mark.asyncio
async def test_with_retry_async_succeeds_after_transient_failure():
    """Async function should succeed after retryable failures."""
    attempts = [0]
    mock_response = MagicMock()
    mock_response.status_code = 502

    async def fn():
        attempts[0] += 1
        if attempts[0] < 3:
            raise httpx.HTTPStatusError("Bad Gateway", request=MagicMock(), response=mock_response)
        return "async_recovered"

    with patch("odin_agent.retry.asyncio.sleep", new_callable=AsyncMock):
        result = await with_retry_async(fn, max_retries=3, base_delay=0.0)

    assert result == "async_recovered"
    assert attempts[0] == 3


@pytest.mark.asyncio
async def test_with_retry_async_raises_rate_limit_error():
    """Async: HTTP 429 must become RateLimitError."""
    mock_response = MagicMock()
    mock_response.status_code = 429

    async def fn():
        raise httpx.HTTPStatusError("Too Many Requests", request=MagicMock(), response=mock_response)

    with patch("odin_agent.retry.asyncio.sleep", new_callable=AsyncMock), pytest.raises(RateLimitError):
        await with_retry_async(fn, max_retries=2, base_delay=0.0)


@pytest.mark.asyncio
async def test_with_retry_async_storm_prevention():
    """Async: all retries must sleep — verifying no retry-storm."""
    mock_response = MagicMock()
    mock_response.status_code = 503
    max_retries = 3

    async def fn():
        raise httpx.HTTPStatusError("Service Unavailable", request=MagicMock(), response=mock_response)

    with patch("odin_agent.retry.asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
        with pytest.raises(InternalServerError):
            await with_retry_async(fn, max_retries=max_retries, base_delay=0.0)

    assert mock_sleep.call_count == max_retries


@pytest.mark.asyncio
async def test_with_retry_async_connection_error():
    """Async: persistent connection errors must become APIConnectionError."""

    async def fn():
        raise httpx.ConnectError("Connection refused")

    with patch("odin_agent.retry.asyncio.sleep", new_callable=AsyncMock), pytest.raises(APIConnectionError):
        await with_retry_async(fn, max_retries=2, base_delay=0.0)
