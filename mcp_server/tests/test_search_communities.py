"""Server-less unit tests for the search_communities MCP tool."""

from unittest.mock import AsyncMock, Mock

import pytest
from graphiti_core.nodes import CommunityNode
from graphiti_core.search.search_config import SearchResults
from graphiti_core.search.search_config_recipes import COMBINED_HYBRID_SEARCH_RRF

import graphiti_mcp_server
from graphiti_mcp_server import _community_only_search_config, search_communities


def _community(uuid: str, name: str = 'c', summary: str = 's') -> CommunityNode:
    return CommunityNode(uuid=uuid, name=name, group_id='g', summary=summary)


def _install_fake_service(monkeypatch, results: SearchResults, member_rows: list[dict]):
    client = Mock()
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


@pytest.mark.unit
async def test_returns_hits_with_member_counts(monkeypatch):
    results = SearchResults(
        communities=[_community('a', 'Alpha'), _community('b', 'Beta')],
        community_reranker_scores=[1.5, 0.33333],
    )
    client = _install_fake_service(monkeypatch, results, [{'uuid': 'a', 'members': 4}])

    resp = await search_communities('q', group_ids='g', max_communities=2, sim_min_score=0.8)

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
