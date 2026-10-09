import pytest
from odin_agent.retry import calculate_backoff, with_retry


def test_calculate_backoff():
    delay = calculate_backoff(attempt=2, base_delay=0.1, max_delay=1.0)
    assert 0 <= delay <= 1.0


def test_with_retry_success():
    calls = 0

    def flaky():
        nonlocal calls
        calls += 1
        return "success"

    result = with_retry(flaky, max_retries=2)
    assert result == "success"
    assert calls == 1
