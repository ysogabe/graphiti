"""Structured-output helpers for the ingest pipeline: validate, retry once, then fail open.

`add_episode` is queued and fire-and-forget. When one dedupe call comes back in the wrong
*shape* the exception escapes `resolve_edge` / node resolution, the worker drops the episode,
and it leaves **no node at all** (sometimes not even a failure line). Observed failure shapes on
b.ai models: a JSON-Schema echo instead of an instance, `{"answer": {...}}` wrapping, and prose
appended after a valid object. Those are transport/format problems, not "no facts to dedupe", so
the safe answer is the *empty* judgement — no merge, no invalidation — while the episode, its
entities and its timestamps are still persisted.

Use `generate_structured` where an empty model instance is semantically valid; keep plain
`generate_response` where an empty result would silently drop extracted content.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any, TypeVar

from pydantic import BaseModel

from ...llm_client.client import LLMClient
from ...llm_client.config import ModelSize
from ...prompts.models import Message

logger = logging.getLogger(__name__)

ModelT = TypeVar('ModelT', bound=BaseModel)


def _clone(messages: list[Message]) -> list[Message]:
    """Copy the prompt: `generate_response` mutates its last/first message in place."""
    return [Message(role=m.role, content=m.content) for m in messages]


def _with_correction(
    messages: list[Message], response_model: type[BaseModel], error: Exception
) -> list[Message]:
    keys = ', '.join(sorted(response_model.model_fields))
    corrected = _clone(messages)
    corrected.append(
        Message(
            role='user',
            content=(
                'Your previous reply could not be used, because it did not match the required'
                f' shape ({type(error).__name__}: {str(error)[:300]}).\n'
                f'Reply with ONE JSON object that is an instance of {response_model.__name__}.'
                f' Top-level keys: {keys}. Use [] for empty lists. Do not repeat the schema and do'
                ' not add any text around the JSON.'
            ),
        )
    )
    return corrected


async def generate_structured(
    llm_client: LLMClient,
    messages: list[Message],
    response_model: type[ModelT],
    *,
    model_size: ModelSize = ModelSize.small,
    prompt_name: str,
    fallback: Callable[[], ModelT] | None = None,
    **kwargs: Any,
) -> ModelT:
    """Return a validated `response_model`, retrying once, then failing open if allowed.

    `fallback` decides what "open" means for the caller (e.g. `EdgeDuplicate()` = no merge and no
    invalidation). When it is None the last error is re-raised, so callers whose empty result
    would lose content keep the strict behaviour.
    """
    attempt_messages = messages
    last_error: Exception | None = None

    for attempt in (1, 2):
        try:
            raw = await llm_client.generate_response(
                attempt_messages,
                response_model=response_model,
                model_size=model_size,
                prompt_name=prompt_name,
                **kwargs,
            )
            return response_model.model_validate(raw)
        except Exception as e:  # noqa: BLE001 - any failure here is a shape/transport problem
            last_error = e
            logger.warning(
                '%s: attempt %d/2 did not match %s (%s: %s)',
                prompt_name,
                attempt,
                response_model.__name__,
                type(e).__name__,
                str(e)[:200],
            )
            attempt_messages = _with_correction(messages, response_model, e)

    if fallback is None:
        assert last_error is not None
        raise last_error

    logger.error(
        '%s: %s still invalid after retry (%s) — failing open, episode continues',
        prompt_name,
        response_model.__name__,
        last_error,
    )
    return fallback()
