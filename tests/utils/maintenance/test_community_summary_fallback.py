import json
import logging
from unittest.mock import AsyncMock, MagicMock

import pytest

from graphiti_core.llm_client.errors import EmptyResponseError, ResponseTruncatedError
from graphiti_core.nodes import EntityNode
from graphiti_core.utils.maintenance import community_operations
from graphiti_core.utils.maintenance.community_operations import (
    COMMUNITY_SUMMARY_FALLBACKS,
    RESCUE_SUMMARY_CHARS,
    build_community,
    generate_summary_description,
    summarize_pair,
)
from graphiti_core.utils.text_utils import MAX_SUMMARY_CHARS


@pytest.fixture(autouse=True)
def reset_fallback_counter():
    COMMUNITY_SUMMARY_FALLBACKS.clear()
    yield
    COMMUNITY_SUMMARY_FALLBACKS.clear()


def _llm(side_effect) -> MagicMock:
    client = MagicMock()
    client.generate_response = AsyncMock(side_effect=side_effect)
    return client


def _sent_user_content(client: MagicMock, call_index: int) -> str:
    messages = client.generate_response.call_args_list[call_index].args[0]
    return messages[-1].content


def _long(label: str, sentences: int = 40) -> str:
    # ~1200 chars of distinct sentences so shortening is observable.
    return ' '.join(f'{label} fact number {i} is recorded here.' for i in range(sentences))


def _truncated() -> ResponseTruncatedError:
    return ResponseTruncatedError('LLM returned an empty response (finish_reason=length)')


@pytest.mark.asyncio
async def test_summarize_pair_success_path_is_single_call():
    client = _llm([{'summary': 'Merged.'}])

    result = await summarize_pair(client, ('Left.', 'Right.'))

    assert result == 'Merged.'
    assert client.generate_response.await_count == 1
    assert sum(COMMUNITY_SUMMARY_FALLBACKS.values()) == 0


@pytest.mark.asyncio
async def test_summarize_pair_rescue1_retries_once_with_shortened_inputs():
    left, right = _long('Alpha'), _long('Beta')
    client = _llm([_truncated(), {'summary': 'Rescued summary.'}])

    result = await summarize_pair(client, (left, right))

    assert result == 'Rescued summary.'
    assert client.generate_response.await_count == 2
    first, second = _sent_user_content(client, 0), _sent_user_content(client, 1)
    assert left in first and right in first
    assert left not in second and right not in second
    assert left[: RESCUE_SUMMARY_CHARS // 2] in second
    assert right[: RESCUE_SUMMARY_CHARS // 2] in second
    assert len(second) < len(first)
    assert sum(COMMUNITY_SUMMARY_FALLBACKS.values()) == 0


@pytest.mark.asyncio
async def test_summarize_pair_rescue2_merges_deterministically_after_two_failures(caplog):
    left, right = _long('Alpha'), _long('Beta')
    client = _llm([_truncated(), json.JSONDecodeError('bad', '', 0), {'summary': 'unused'}])

    with caplog.at_level(logging.WARNING, logger=community_operations.logger.name):
        result = await summarize_pair(client, (left, right))

    # Exactly one shortened retry, never a third call at the same length.
    assert client.generate_response.await_count == 2
    assert 'Alpha fact number 0' in result
    assert 'Beta fact number 0' in result
    assert len(result) <= MAX_SUMMARY_CHARS
    assert COMMUNITY_SUMMARY_FALLBACKS['summarize_pair'] == 1
    warnings = [r for r in caplog.records if 'deterministic merge' in r.getMessage()]
    assert len(warnings) == 1


@pytest.mark.asyncio
async def test_summarize_pair_skips_rescue1_when_inputs_already_short():
    client = _llm([EmptyResponseError('empty'), {'summary': 'unused'}])

    result = await summarize_pair(client, ('Short left.', 'Short right.'))

    assert client.generate_response.await_count == 1
    assert result == 'Short left. Short right.'
    assert COMMUNITY_SUMMARY_FALLBACKS['summarize_pair'] == 1


@pytest.mark.asyncio
async def test_summarize_pair_does_not_swallow_other_errors():
    client = _llm([RuntimeError('auth failed')])

    with pytest.raises(RuntimeError, match='auth failed'):
        await summarize_pair(client, (_long('Alpha'), _long('Beta')))

    assert client.generate_response.await_count == 1
    assert sum(COMMUNITY_SUMMARY_FALLBACKS.values()) == 0


@pytest.mark.asyncio
async def test_summarize_pair_does_not_swallow_errors_during_rescue1():
    client = _llm([_truncated(), RuntimeError('server down')])

    with pytest.raises(RuntimeError, match='server down'):
        await summarize_pair(client, (_long('Alpha'), _long('Beta')))

    assert sum(COMMUNITY_SUMMARY_FALLBACKS.values()) == 0


@pytest.mark.asyncio
async def test_summary_description_falls_back_to_summary_prefix():
    summary = 'The first sentence names the group. ' + _long('Gamma')
    client = _llm([_truncated(), _truncated()])

    name = await generate_summary_description(client, summary)

    assert client.generate_response.await_count == 2
    assert name.startswith('The first sentence names the group.')
    assert len(name) <= community_operations.FALLBACK_NAME_CHARS
    assert COMMUNITY_SUMMARY_FALLBACKS['summary_description'] == 1


@pytest.mark.asyncio
async def test_build_community_completes_when_every_llm_call_is_truncated():
    nodes = [
        EntityNode(name=f'n{i}', group_id='g', labels=['Entity'], summary=_long(f'E{i}'))
        for i in range(5)
    ]
    client = _llm(lambda *args, **kwargs: (_ for _ in ()).throw(_truncated()))

    community, edges = await build_community(client, nodes)

    assert community.summary
    assert community.name
    assert len(community.summary) <= MAX_SUMMARY_CHARS
    assert len(edges) == len(nodes)
    # 5 leaves -> 2 + 1 + 1 pair merges, then one name.
    assert COMMUNITY_SUMMARY_FALLBACKS['summarize_pair'] == 4
    assert COMMUNITY_SUMMARY_FALLBACKS['summary_description'] == 1


@pytest.mark.asyncio
async def test_build_community_uses_fallback_name_when_only_description_fails():
    nodes = [
        EntityNode(name='a', group_id='g', labels=['Entity'], summary='A.'),
        EntityNode(name='b', group_id='g', labels=['Entity'], summary='B.'),
    ]
    client = _llm([{'summary': 'Merged summary.'}, _truncated()])

    community, _ = await build_community(client, nodes)

    assert community.summary == 'Merged summary.'
    assert community.name == 'Merged summary.'
    assert COMMUNITY_SUMMARY_FALLBACKS['summarize_pair'] == 0
    assert COMMUNITY_SUMMARY_FALLBACKS['summary_description'] == 1
