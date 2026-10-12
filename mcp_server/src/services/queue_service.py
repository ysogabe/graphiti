"""Queue service for managing episode processing."""

import asyncio
import collections
import logging
from collections.abc import Awaitable, Callable
from datetime import datetime, timezone
from typing import Any

logger = logging.getLogger(__name__)

# Backoff schedule (seconds) for transient-error retries: initial attempt + 3 retries.
# Exposed as a module constant so tests can monkeypatch it to avoid real waits.
RETRY_DELAYS_SECONDS: list[float] = [30, 120, 300]

# Labels of episodes that exhausted all retries, keyed by the label passed to
# add_episode_task (or None if no label was supplied). Read by tests/operators
# to see which episodes were dropped, since the old behavior silently discarded them.
FAILED_EPISODES: collections.Counter = collections.Counter()

# Exception class names (matched by name, not isinstance, so we don't need a hard
# dependency on e.g. neo4j.exceptions across every graph backend) treated as transient.
_RETRYABLE_EXCEPTION_NAMES = ('TimeoutError', 'ConnectionError', 'ServiceUnavailable')
_RETRYABLE_MESSAGE_MARKERS = ('429', 'RESOURCE_EXHAUSTED', 'rate limit')


async def _sleep(seconds: float) -> None:
    """Wait between retry attempts. A module-level indirection so tests can monkeypatch it."""
    await asyncio.sleep(seconds)


def _is_retryable_error(exc: BaseException) -> bool:
    """Return True for transient errors (rate limits, connection issues) worth retrying."""
    if isinstance(exc, (TimeoutError, ConnectionError)):
        return True
    if type(exc).__name__ in _RETRYABLE_EXCEPTION_NAMES:
        return True
    message = str(exc)
    return any(marker.lower() in message.lower() for marker in _RETRYABLE_MESSAGE_MARKERS)


async def _process_with_retry(
    process_func: Callable[[], Awaitable[None]],
    label: str | None,
    group_id: str,
) -> None:
    """Run process_func, retrying transient errors with exponential backoff.

    Re-raises the final exception if all attempts are exhausted, or immediately if
    the error isn't retryable (e.g. validation/programming errors).
    """
    # Read RETRY_DELAYS_SECONDS at call time (not import time) so tests can
    # monkeypatch it to shorten both the delays and the number of attempts.
    max_attempts = len(RETRY_DELAYS_SECONDS) + 1
    for attempt in range(1, max_attempts + 1):
        try:
            await process_func()
            if attempt > 1:
                logger.info(
                    f'Episode (label={label}) for group_id {group_id} succeeded '
                    f'on attempt {attempt}/{max_attempts}'
                )
            return
        except Exception as e:
            if attempt >= max_attempts or not _is_retryable_error(e):
                raise
            delay = RETRY_DELAYS_SECONDS[attempt - 1]
            logger.warning(
                f'Transient error processing episode (label={label}) for group_id {group_id} '
                f'on attempt {attempt}/{max_attempts}: {e}. Retrying in {delay}s'
            )
            await _sleep(delay)


class QueueService:
    """Service for managing sequential episode processing queues by group_id."""

    def __init__(self):
        """Initialize the queue service."""
        # Dictionary to store queues for each group_id
        self._episode_queues: dict[str, asyncio.Queue] = {}
        # Dictionary to track if a worker is running for each group_id
        self._queue_workers: dict[str, bool] = {}
        # Store the graphiti client after initialization
        self._graphiti_client: Any = None

    async def add_episode_task(
        self,
        group_id: str,
        process_func: Callable[[], Awaitable[None]],
        label: str | None = None,
    ) -> int:
        """Add an episode processing task to the queue.

        Args:
            group_id: The group ID for the episode
            process_func: The async function to process the episode
            label: Optional human-readable label (e.g. episode name) used in logs
                and in FAILED_EPISODES if processing ultimately fails. Defaults to
                None, matching the previous unlabeled behavior.

        Returns:
            The position in the queue
        """
        # Initialize queue for this group_id if it doesn't exist
        if group_id not in self._episode_queues:
            self._episode_queues[group_id] = asyncio.Queue()

        # Add the episode processing function (and its label) to the queue
        await self._episode_queues[group_id].put((process_func, label))

        # Start a worker for this queue if one isn't already running
        if not self._queue_workers.get(group_id, False):
            asyncio.create_task(self._process_episode_queue(group_id))

        return self._episode_queues[group_id].qsize()

    async def _process_episode_queue(self, group_id: str) -> None:
        """Process episodes for a specific group_id sequentially.

        This function runs as a long-lived task that processes episodes
        from the queue one at a time.
        """
        logger.info(f'Starting episode queue worker for group_id: {group_id}')
        self._queue_workers[group_id] = True

        try:
            while True:
                # Get the next episode processing function (and its label) from the queue
                # This will wait if the queue is empty
                process_func, label = await self._episode_queues[group_id].get()

                try:
                    # Process the episode, retrying transient errors
                    await _process_with_retry(process_func, label, group_id)
                except Exception as e:
                    logger.error(
                        f'Error processing queued episode for group_id {group_id} '
                        f'(label={label}): {str(e)}'
                    )
                    FAILED_EPISODES[label] += 1
                finally:
                    # Mark the task as done regardless of success/failure
                    self._episode_queues[group_id].task_done()
        except asyncio.CancelledError:
            logger.info(f'Episode queue worker for group_id {group_id} was cancelled')
        except Exception as e:
            logger.error(f'Unexpected error in queue worker for group_id {group_id}: {str(e)}')
        finally:
            self._queue_workers[group_id] = False
            logger.info(f'Stopped episode queue worker for group_id: {group_id}')

    def get_queue_size(self, group_id: str) -> int:
        """Get the current queue size for a group_id."""
        if group_id not in self._episode_queues:
            return 0
        return self._episode_queues[group_id].qsize()

    def is_worker_running(self, group_id: str) -> bool:
        """Check if a worker is running for a group_id."""
        return self._queue_workers.get(group_id, False)

    async def initialize(self, graphiti_client: Any) -> None:
        """Initialize the queue service with a graphiti client.

        Args:
            graphiti_client: The graphiti client instance to use for processing episodes
        """
        self._graphiti_client = graphiti_client
        logger.info('Queue service initialized with graphiti client')

    async def add_episode(
        self,
        group_id: str,
        name: str,
        content: str,
        source_description: str,
        episode_type: Any,
        entity_types: Any,
        uuid: str | None,
        reference_time: datetime | None = None,
        edge_types: Any = None,
        edge_type_map: Any = None,
        excluded_entity_types: list[str] | None = None,
        previous_episode_uuids: list[str] | None = None,
        custom_extraction_instructions: str | None = None,
        update_communities: bool = False,
        saga: str | None = None,
        saga_previous_episode_uuid: str | None = None,
    ) -> int:
        """Add an episode for processing.

        Args:
            group_id: The group ID for the episode
            name: Name of the episode
            content: Episode content
            source_description: Description of the episode source
            episode_type: Type of the episode
            entity_types: Entity types for extraction
            uuid: Episode UUID
            reference_time: Event occurrence time for the episode. Defaults to
                the current UTC time when not provided (bi-temporal model).
            edge_types: Optional mapping of edge (fact) type name to Pydantic model
            edge_type_map: Optional mapping of (source, target) entity type pairs to
                allowed edge type names
            excluded_entity_types: Optional list of entity type names to exclude
                from extraction
            previous_episode_uuids: Optional explicit list of prior episode UUIDs to
                use as context (overrides automatic retrieval)
            custom_extraction_instructions: Optional extra natural-language
                instructions for the extraction LLM
            update_communities: Whether to incrementally update communities after
                ingestion
            saga: Optional saga name/id to attach this episode to
            saga_previous_episode_uuid: Optional UUID of the prior episode in the saga

        Returns:
            The position in the queue
        """
        if self._graphiti_client is None:
            raise RuntimeError('Queue service not initialized. Call initialize() first.')

        async def process_episode():
            """Process the episode using the graphiti client."""
            try:
                logger.info(f'Processing episode {uuid} for group {group_id}')

                # Process the episode using the graphiti client
                await self._graphiti_client.add_episode(
                    name=name,
                    episode_body=content,
                    source_description=source_description,
                    source=episode_type,
                    group_id=group_id,
                    reference_time=reference_time or datetime.now(timezone.utc),
                    entity_types=entity_types,
                    edge_types=edge_types,
                    edge_type_map=edge_type_map,
                    excluded_entity_types=excluded_entity_types,
                    previous_episode_uuids=previous_episode_uuids,
                    custom_extraction_instructions=custom_extraction_instructions,
                    update_communities=update_communities,
                    saga=saga,
                    saga_previous_episode_uuid=saga_previous_episode_uuid,
                    uuid=uuid,
                )

                logger.info(f'Successfully processed episode {uuid} for group {group_id}')

            except Exception as e:
                logger.error(f'Failed to process episode {uuid} for group {group_id}: {str(e)}')
                raise

        # Use the existing add_episode_task method to queue the processing
        return await self.add_episode_task(group_id, process_episode, label=f'{name} ({group_id})')
