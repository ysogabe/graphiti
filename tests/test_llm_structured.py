"""Shape-failure handling: parse tolerance, one corrective retry, then fail open."""

from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from pydantic import BaseModel, Field

from graphiti_core.graphiti_types import GraphitiClients
from graphiti_core.llm_client.config import ModelSize
from graphiti_core.llm_client.openai_generic_client import OpenAIGenericClient
from graphiti_core.prompts.dedupe_edges import EdgeDuplicate
from graphiti_core.prompts.dedupe_nodes import NodeResolutions
from graphiti_core.prompts.models import Message
from graphiti_core.utils.maintenance.llm_structured import generate_structured


class _FakeClient:
    """Minimal stand-in: returns queued payloads, repeating the last one when exhausted."""

    def __init__(self, payloads: list[Any]) -> None:
        self.payloads = list(payloads)
        self.calls: list[list[Message]] = []

    async def generate_response(self, messages, **_kwargs) -> Any:
        self.calls.append(list(messages))
        payload = self.payloads.pop(0) if len(self.payloads) > 1 else self.payloads[0]
        if isinstance(payload, Exception):
            raise payload
        return payload


# --- parsing tolerance -------------------------------------------------------------------


@pytest.mark.parametrize(
    'raw,expected',
    [
        ('{"duplicate_facts": [1], "contradicted_facts": []}', {'duplicate_facts': [1], 'contradicted_facts': []}),
        (  # valid object followed by commentary (b.ai "Extra data" case)
            '{"duplicate_facts": [], "contradicted_facts": [2]}\n\nNote: I ignored unrelated facts.',
            {'duplicate_facts': [], 'contradicted_facts': [2]},
        ),
        (  # answer-wrapped payload
            '{"answer": {"duplicate_facts": [0], "contradicted_facts": []}}',
            {'duplicate_facts': [0], 'contradicted_facts': []},
        ),
        (  # prose before the object
            'Sure, here it is: {"duplicate_facts": [], "contradicted_facts": []} done.',
            {'duplicate_facts': [], 'contradicted_facts': []},
        ),
    ],
)
def test_extract_json_payload(raw: str, expected: dict[str, Any]) -> None:
    assert OpenAIGenericClient._extract_json_payload(raw) == expected


def test_extract_json_payload_raises_without_json() -> None:
    import json

    with pytest.raises(json.JSONDecodeError):
        OpenAIGenericClient._extract_json_payload('no json at all')


# --- models tolerate missing keys (schema echo / omitted keys) ----------------------------


def test_edge_duplicate_defaults_to_empty() -> None:
    dup = EdgeDuplicate(**{'duplicate_facts': [], 'contradicted_facts': []})
    assert (dup.duplicate_facts, dup.contradicted_facts) == ([], [])
    assert EdgeDuplicate().duplicate_facts == []
    # A JSON-Schema echo has extra keys and no values: still valid, still harmless.
    echo = EdgeDuplicate(**{'properties': {'duplicate_facts': {}}, 'type': 'object'})
    assert (echo.duplicate_facts, echo.contradicted_facts) == ([], [])


def test_node_resolutions_defaults_to_empty() -> None:
    assert NodeResolutions().entity_resolutions == []
    assert NodeResolutions(**{'properties': {}}).entity_resolutions == []


# --- retry once, then fail open ----------------------------------------------------------


@pytest.mark.asyncio
async def test_generate_structured_retries_once_then_succeeds() -> None:
    class Model(BaseModel):
        values: list[int] = Field(default_factory=list)

    client = _FakeClient([{'values': 'not a list of ints'}, {'values': [1, 2]}])
    out = await generate_structured(
        client,  # type: ignore[arg-type]
        [Message(role='user', content='go')],
        Model,
        model_size=ModelSize.small,
        prompt_name='test.retry',
        fallback=Model,
    )
    assert out.values == [1, 2]
    assert len(client.calls) == 2
    # the corrective retry must carry the error as an extra user turn
    assert len(client.calls[1]) == 2 and 'did not match the required' in client.calls[1][-1].content


@pytest.mark.asyncio
async def test_generate_structured_fails_open_after_retry() -> None:
    class Model(BaseModel):
        values: list[int] = Field(default_factory=list)

    client = _FakeClient([ValueError('bad shape')])
    out = await generate_structured(
        client,  # type: ignore[arg-type]
        [Message(role='user', content='go')],
        Model,
        prompt_name='test.failopen',
        fallback=Model,
    )
    assert out.values == []
    assert len(client.calls) == 2


@pytest.mark.asyncio
async def test_generate_structured_without_fallback_raises() -> None:
    class Model(BaseModel):
        values: list[int] = Field(default_factory=list)

    client = _FakeClient([ValueError('bad shape')])
    with pytest.raises(ValueError):
        await generate_structured(
            client,  # type: ignore[arg-type]
            [Message(role='user', content='go')],
            Model,
            prompt_name='test.strict',
            fallback=None,
        )


@pytest.mark.asyncio
async def test_generate_structured_does_not_mutate_caller_messages() -> None:
    class Model(BaseModel):
        values: list[int] = Field(default_factory=list)

    messages = [Message(role='system', content='sys'), Message(role='user', content='go')]
    client = _FakeClient([{'values': [1]}])
    await generate_structured(
        client,  # type: ignore[arg-type]
        messages,
        Model,
        prompt_name='test.copy',
        fallback=Model,
    )
    assert [m.content for m in messages] == ['sys', 'go']


@pytest.mark.asyncio
async def test_generate_structured_routes_through_clients_when_prompt_routing_is_set() -> None:
    from graphiti_core.prompts import create_prompt_library

    class Model(BaseModel):
        values: list[int] = Field(default_factory=list)

    llm_client = _FakeClient([{'values': [9]}])
    clients = GraphitiClients.model_construct(
        driver=MagicMock(),
        embedder=MagicMock(),
        cross_encoder=MagicMock(),
        llm_client=llm_client,
        tracer=MagicMock(),
        prompt_library=create_prompt_library({}),
    )
    complete_prompt = AsyncMock(side_effect=[{'values': 'not a list'}, {'values': [3]}])
    object.__setattr__(clients, 'complete_prompt', complete_prompt)

    result = await generate_structured(
        llm_client,  # type: ignore[arg-type]
        [Message(role='user', content='go')],
        Model,
        prompt_name='dedupe_edges.resolve_edge',
        fallback=Model,
        clients=clients,
        context={'new_edge': 'fact'},
    )

    # Both attempts went through the routed prompt with the caller's context, the direct client
    # was never called, and the second (valid) answer is returned.
    assert result == Model(values=[3])
    assert complete_prompt.await_count == 2
    for call in complete_prompt.await_args_list:
        assert call.args == ('dedupe_edges.resolve_edge', {'new_edge': 'fact'})
    assert llm_client.calls == []
