"""Tests for CancellationToken.

Pure Python tests - no Qt, no xvfb required.
"""

import time
from concurrent.futures import ThreadPoolExecutor

from caliscope.task_manager import CancellationToken


def test_initial_state_not_cancelled():
    token = CancellationToken()
    assert not token.is_cancelled


def test_cancel_sets_is_cancelled():
    token = CancellationToken()
    token.cancel()
    assert token.is_cancelled


def test_sleep_unless_cancelled_waits_full_duration():
    token = CancellationToken()
    start = time.perf_counter()
    cancelled = token.sleep_unless_cancelled(0.1)
    elapsed = time.perf_counter() - start

    assert not cancelled
    assert elapsed >= 0.1


def test_sleep_unless_cancelled_returns_immediately_when_cancelled():
    token = CancellationToken()
    token.cancel()

    start = time.perf_counter()
    cancelled = token.sleep_unless_cancelled(10.0)  # Would wait 10s if not cancelled
    elapsed = time.perf_counter() - start

    assert cancelled
    assert elapsed < 0.1  # Should return nearly instantly


def test_sleep_wakes_when_cancelled_from_another_thread():
    token = CancellationToken()
    result: dict[str, float | bool] = {}

    def sleeper():
        result["cancelled"] = token.sleep_unless_cancelled(10.0)
        result["time"] = time.perf_counter()

    with ThreadPoolExecutor() as executor:
        start = time.perf_counter()
        future = executor.submit(sleeper)

        time.sleep(0.1)  # Let sleeper start
        token.cancel()
        future.result(timeout=1.0)

    elapsed = result["time"] - start
    assert result["cancelled"]
    assert elapsed < 1.0  # Should wake up quickly after cancel


if __name__ == "__main__":
    test_initial_state_not_cancelled()
    test_cancel_sets_is_cancelled()
    test_sleep_unless_cancelled_waits_full_duration()
    test_sleep_unless_cancelled_returns_immediately_when_cancelled()
    test_sleep_wakes_when_cancelled_from_another_thread()
    print("All CancellationToken tests passed!")
