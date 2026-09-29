"""Server-less unit tests for the search_communities MCP tool."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from graphiti_core.driver.driver import GraphProvider
from graphiti_core.nodes import CommunityNode
from graphiti_core.search.search_config import SearchResults
from graphiti_core.search.search_config_recipes import COMBINED_HYBRID_SEARCH_RRF
from graphiti_core.search.search_utils import DEFAULT_MIN_SCORE, rrf

import graphiti_mcp_server
from graphiti_mcp_server import (
    MAX_SEARCH_COMMUNITIES,
    _community_only_search_config,
    _order_and_truncate,
    _require_group_ids,
    search_communities,
)


def _community(
    uuid: str, name: str = 'c', summary: str = 's', group_id: str = 'g'
) -> CommunityNode:
    return CommunityNode(uuid=uuid, name=name, group_id=group_id, summary=summary)


def _install_fake_service(
    monkeypatch,
    results: SearchResults,
    member_rows: list[dict],
    provider: GraphProvider = GraphProvider.NEO4J,
):
    client = Mock()
    client.driver.provider = provider
    client.search_ = AsyncMock(return_value=results)
    client.driver.execute_query = AsyncMock(return_value=(member_rows, None, None))
    service = Mock()
    service.get_client = AsyncMock(return_value=client)
    monkeypatch.setattr(graphiti_mcp_server, 'graphiti_service', service)
    return client


@pytest.mark.unit
def test_config_runs_community_leg_only():
    cfg = _community_only_search_config(7, 0.9)
    assert cfg.edge_config is None
    assert cfg.node_config is None
    assert cfg.episode_config is None
    assert cfg.community_config is not None
    assert cfg.community_config.search_methods == (
        COMBINED_HYBRID_SEARCH_RRF.community_config.search_methods
    )
    assert cfg.limit == 7
    assert cfg.community_config.sim_min_score == 0.9


@pytest.mark.unit
def test_config_clamps_negative_floor_and_leaves_recipe_intact():
    original_floor = COMBINED_HYBRID_SEARCH_RRF.community_config.sim_min_score
    original_limit = COMBINED_HYBRID_SEARCH_RRF.limit
    cfg = _community_only_search_config(3, -1)
    assert cfg.community_config.sim_min_score == 0.0
    assert COMBINED_HYBRID_SEARCH_RRF.edge_config is not None
    assert COMBINED_HYBRID_SEARCH_RRF.node_config is not None
    assert COMBINED_HYBRID_SEARCH_RRF.episode_config is not None
    assert COMBINED_HYBRID_SEARCH_RRF.community_config.sim_min_score == original_floor
    assert COMBINED_HYBRID_SEARCH_RRF.limit == original_limit


# --- #1: the floor defaults to the core recipe's, not a value borrowed from another knob ---


@pytest.mark.unit
def test_config_none_floor_keeps_core_default():
    cfg = _community_only_search_config(5, None)
    assert cfg.community_config.sim_min_score == DEFAULT_MIN_SCORE
    assert _community_only_search_config(5).community_config.sim_min_score == DEFAULT_MIN_SCORE


@pytest.mark.unit
async def test_tool_default_floor_is_core_default(monkeypatch):
    client = _install_fake_service(monkeypatch, SearchResults(), [])
    await search_communities('q', group_ids=['g'])
    cfg = client.search_.await_args.kwargs['config']
    assert cfg.community_config.sim_min_score == DEFAULT_MIN_SCORE


@pytest.mark.unit
async def test_returns_hits_with_member_counts(monkeypatch):
    results = SearchResults(
        communities=[_community('a', 'Alpha'), _community('b', 'Beta')],
        community_reranker_scores=[1.5, 0.33333],
    )
    client = _install_fake_service(monkeypatch, results, [{'uuid': 'a', 'members': 4}])

    resp = await search_communities('q', group_ids='g', max_communities=2, sim_min_score=0.8)

    # Neo4j path: members are counted, so the message carries no caveat.
    assert resp['message'] == 'Communities retrieved successfully'
    assert [h['uuid'] for h in resp['communities']] == ['a', 'b']
    assert resp['communities'][0]['members'] == 4
    assert resp['communities'][1]['members'] == 0  # no HAS_MEMBER row -> 0
    assert resp['communities'][0]['score'] == 1.5
    assert resp['communities'][1]['score'] == 0.3333
    assert resp['communities'][0]['summary'] == 's'

    kwargs = client.search_.await_args.kwargs
    assert kwargs['group_ids'] == ['g']
    assert kwargs['config'].node_config is None
    assert kwargs['config'].limit == 2
    assert kwargs['config'].community_config.sim_min_score == 0.8
    # One member-count query covering every returned community.
    assert client.driver.execute_query.await_count == 1
    assert client.driver.execute_query.await_args.kwargs['uuids'] == ['a', 'b']


@pytest.mark.unit
async def test_score_is_none_when_lengths_differ(monkeypatch):
    results = SearchResults(communities=[_community('a')], community_reranker_scores=[])
    _install_fake_service(monkeypatch, results, [])
    resp = await search_communities('q', group_ids=['g'])
    assert resp['communities'][0]['score'] is None
    assert 'score=None' in resp['message']


@pytest.mark.unit
async def test_empty_result_skips_member_query(monkeypatch):
    client = _install_fake_service(monkeypatch, SearchResults(), [])
    resp = await search_communities('q', group_ids=['g'])
    assert resp == {'message': 'No relevant communities found', 'communities': []}
    client.driver.execute_query.assert_not_awaited()


@pytest.mark.unit
async def test_rejects_non_positive_max(monkeypatch):
    client = _install_fake_service(monkeypatch, SearchResults(), [])
    resp = await search_communities('q', group_ids=['g'], max_communities=0)
    assert 'error' in resp
    client.search_.assert_not_awaited()


@pytest.mark.unit
async def test_errors_are_returned_not_raised(monkeypatch):
    client = _install_fake_service(monkeypatch, SearchResults(), [])
    client.search_.side_effect = RuntimeError('boom')
    resp = await search_communities('q', group_ids=['g'])
    assert resp == {'error': 'Error searching communities: boom'}


@pytest.mark.unit
async def test_uninitialized_service(monkeypatch):
    monkeypatch.setattr(graphiti_mcp_server, 'graphiti_service', None)
    resp = await search_communities('q')
    assert resp == {'error': 'Graphiti service not initialized'}


# --- #3: an empty allow-list must not widen to every group ---


@pytest.mark.unit
@pytest.mark.parametrize('bad', [[], [''], ['g', ''], ['  ']])
async def test_rejects_empty_or_blank_group_ids(monkeypatch, bad):
    client = _install_fake_service(monkeypatch, SearchResults(), [])
    resp = await search_communities('q', group_ids=bad)
    assert 'error' in resp
    client.search_.assert_not_awaited()
    client.driver.execute_query.assert_not_awaited()


@pytest.mark.unit
async def test_omitted_group_ids_without_server_default_is_rejected(monkeypatch):
    client = _install_fake_service(monkeypatch, SearchResults(), [])
    fake_config = SimpleNamespace(graphiti=SimpleNamespace(group_id=''))
    monkeypatch.setattr(graphiti_mcp_server, 'config', fake_config, raising=False)
    resp = await search_communities('q')
    assert 'error' in resp
    client.search_.assert_not_awaited()


@pytest.mark.unit
async def test_omitted_group_ids_uses_server_default(monkeypatch):
    client = _install_fake_service(monkeypatch, SearchResults(), [])
    fake_config = SimpleNamespace(graphiti=SimpleNamespace(group_id='minamo'))
    monkeypatch.setattr(graphiti_mcp_server, 'config', fake_config, raising=False)
    await search_communities('q')
    assert client.search_.await_args.kwargs['group_ids'] == ['minamo']


@pytest.mark.unit
def test_require_group_ids_pure():
    assert _require_group_ids(None) is not None
    assert _require_group_ids([]) is not None
    assert _require_group_ids(['']) is not None
    assert _require_group_ids(['minamo', 'haruo']) is None


# --- #4: members is None (not 0) where it cannot be counted correctly ---


@pytest.mark.unit
@pytest.mark.parametrize(
    'provider', [GraphProvider.FALKORDB, GraphProvider.KUZU, GraphProvider.NEPTUNE]
)
async def test_members_none_off_neo4j(monkeypatch, provider):
    results = SearchResults(communities=[_community('a')], community_reranker_scores=[1.0])
    client = _install_fake_service(monkeypatch, results, [{'uuid': 'a', 'members': 9}], provider)
    resp = await search_communities('q', group_ids=['g'])
    assert resp['communities'][0]['members'] is None
    assert 'members=None' in resp['message']
    client.driver.execute_query.assert_not_awaited()


# --- #5: FalkorDB multi-group merge -> one global score order and one global limit ---


def _per_group(group_id: str, names: list[str], limit: int) -> SearchResults:
    """What community_search returns for one group: rrf output sliced to limit."""
    uuids, scores = rrf([names, names[::2]])  # two legs, like BM25 + cosine
    return SearchResults(
        communities=[_community(u, u, group_id=group_id) for u in uuids][:limit],
        community_reranker_scores=scores[:limit],
    )


@pytest.mark.unit
async def test_falkordb_multi_group_is_resorted_and_capped(monkeypatch):
    limit = 3
    merged = SearchResults.merge(
        [
            _per_group('minamo', ['m1', 'm2', 'm3', 'm4'], limit),
            _per_group('haruo', ['h1', 'h2', 'h3', 'h4'], limit),
        ]
    )
    assert len(merged.communities) == 2 * limit  # the per-group behaviour being fixed
    by_uuid = dict(
        zip([c.uuid for c in merged.communities], merged.community_reranker_scores, strict=True)
    )
    _install_fake_service(monkeypatch, merged, [], GraphProvider.FALKORDB)

    resp = await search_communities('q', group_ids=['minamo', 'haruo'], max_communities=limit)

    hits = resp['communities']
    assert len(hits) == limit
    got = [h['score'] for h in hits]
    assert got == sorted(got, reverse=True)
    assert {h['group_id'] for h in hits} == {'minamo', 'haruo'}
    # Each score is still the one the core assigned to that community.
    for h in hits:
        assert h['score'] == round(by_uuid[h['uuid']], 4)


@pytest.mark.unit
def test_order_and_truncate_is_noop_on_sorted_input_with_ties():
    cs = ['a', 'b', 'c', 'd']
    ss = [2.0, 1.0, 1.0, 0.5]
    assert _order_and_truncate(cs, ss, 10) == (cs, ss, True)
    assert _order_and_truncate(cs, ss, 3) == (cs[:3], ss[:3], True)


@pytest.mark.unit
def test_order_and_truncate_misaligned_scores_are_dropped():
    out_c, out_s, aligned = _order_and_truncate(['a', 'b', 'c'], [1.0, 2.0], 2)
    assert out_c == ['a', 'b']
    assert out_s == [None, None]
    assert aligned is False


# --- #6: max_communities is capped ---


@pytest.mark.unit
async def test_rejects_max_over_cap(monkeypatch):
    client = _install_fake_service(monkeypatch, SearchResults(), [])
    resp = await search_communities(
        'q', group_ids=['g'], max_communities=MAX_SEARCH_COMMUNITIES + 1
    )
    assert 'error' in resp
    client.search_.assert_not_awaited()
    ok = await search_communities('q', group_ids=['g'], max_communities=MAX_SEARCH_COMMUNITIES)
    assert 'error' not in ok
