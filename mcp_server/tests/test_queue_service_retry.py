"""Tests for transient-error retry handling in the episode processing queue."""

import pytest

from services.queue_service import (
    FAILED_EPISODES,
    RETRY_DELAYS_SECONDS,
    QueueService,
    _is_retryable_error,
)


@pytest.fixture(autouse=True)
def _clear_failed_episodes():
    """Ensure FAILED_EPISODES doesn't leak state between tests."""
    FAILED_EPISODES.clear()
    yield
    FAILED_EPISODES.clear()


@pytest.fixture(autouse=True)
def _fast_sleep(monkeypatch):
    """Replace the retry backoff sleep with a no-op so tests don't actually wait."""
    sleeps: list[float] = []

    async def fake_sleep(seconds: float) -> None:
        sleeps.append(seconds)

    monkeypatch.setattr('services.queue_service._sleep', fake_sleep)
    return sleeps


async def _drain(queue_service: QueueService, group_id: str) -> None:
    """Wait until the queue for group_id is fully processed."""
    queue = queue_service._episode_queues[group_id]
    await queue.join()


class Rate429Error(Exception):
    pass


class ServiceUnavailable(Exception):
    """Named to match the connection-error retry check by class name."""


async def test_retries_transient_error_then_succeeds(_fast_sleep):
    service = QueueService()
    calls = 0

    async def process_func():
        nonlocal calls
        calls += 1
        if calls < 3:
            raise Rate429Error('429 RESOURCE_EXHAUSTED')

    await service.add_episode_task('g1', process_func, label='ep-1 (g1)')
    await _drain(service, 'g1')

    assert calls == 3
    assert FAILED_EPISODES == {}
    # Two retries happened before success, using the first two backoff delays.
    assert _fast_sleep == RETRY_DELAYS_SECONDS[:2]


async def test_exhausts_retries_and_records_failure(_fast_sleep):
    service = QueueService()
    calls = 0

    async def process_func():
        nonlocal calls
        calls += 1
        raise Rate429Error('429 RESOURCE_EXHAUSTED')

    await service.add_episode_task('g2', process_func, label='ep-2 (g2)')
    await _drain(service, 'g2')

    # Initial attempt + 3 retries = 4 total calls, no 5th attempt.
    assert calls == 4
    assert FAILED_EPISODES['ep-2 (g2)'] == 1
    assert _fast_sleep == RETRY_DELAYS_SECONDS


async def test_non_retryable_error_fails_immediately(_fast_sleep):
    service = QueueService()
    calls = 0

    async def process_func():
        nonlocal calls
        calls += 1
        raise ValueError('bad input')

    await service.add_episode_task('g3', process_func, label='ep-3 (g3)')
    await _drain(service, 'g3')

    assert calls == 1
    assert not _fast_sleep
    assert FAILED_EPISODES['ep-3 (g3)'] == 1


async def test_connection_error_by_class_name_is_retryable():
    assert _is_retryable_error(ServiceUnavailable('db unreachable'))


async def test_timeout_and_connection_errors_are_retryable():
    assert _is_retryable_error(TimeoutError('timed out'))
    assert _is_retryable_error(ConnectionError('connection reset'))


async def test_rate_limit_message_is_retryable():
    assert _is_retryable_error(Exception('rate limit exceeded'))


async def test_generic_error_is_not_retryable():
    assert not _is_retryable_error(ValueError('bad input'))


async def test_add_episode_task_without_label_defaults_to_none(_fast_sleep):
    """Existing callers that don't pass a label keep the old unlabeled behavior."""
    service = QueueService()
    calls = 0

    async def process_func():
        nonlocal calls
        calls += 1
        raise ValueError('bad input')

    await service.add_episode_task('g4', process_func)
    await _drain(service, 'g4')

    assert calls == 1
    assert FAILED_EPISODES[None] == 1


async def test_worker_processes_second_episode_after_first_fails(_fast_sleep):
    """Queue processing continues for subsequent episodes after a failure."""
    service = QueueService()
    results: list[str] = []

    async def failing():
        raise ValueError('bad input')

    async def succeeding():
        results.append('ok')

    await service.add_episode_task('g5', failing, label='fail (g5)')
    await service.add_episode_task('g5', succeeding, label='ok (g5)')
    await _drain(service, 'g5')

    assert results == ['ok']
    assert FAILED_EPISODES['fail (g5)'] == 1
