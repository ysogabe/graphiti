from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from graphiti_core.driver.driver import GraphProvider
from graphiti_core.driver.falkordb.operations.search_ops import FalkorSearchOperations
from graphiti_core.driver.neo4j.operations.search_ops import Neo4jSearchOperations
from graphiti_core.graph_queries import (
    NEO4J_TO_FALKORDB_MAPPING,
    get_fulltext_indices,
)
from graphiti_core.models.nodes.node_db_queries import get_community_node_save_query
from graphiti_core.search.search_utils import community_similarity_search


def test_community_fulltext_indices_include_summary_without_renaming_index():
    neo4j_indices = get_fulltext_indices(GraphProvider.NEO4J, analyzer='standard')
    neo4j_community_index = next(index for index in neo4j_indices if 'community_name' in index)

    assert 'community_name' in neo4j_community_index
    assert '[n.name, n.summary, n.group_id]' in neo4j_community_index
    assert NEO4J_TO_FALKORDB_MAPPING['community_name'] == 'Community'

    falkordb_indices = get_fulltext_indices(GraphProvider.FALKORDB)
    falkordb_community_index = next(
        index for index in falkordb_indices if "label: 'Community'" in index
    )

    assert "'name', 'summary', 'group_id'" in falkordb_community_index


@pytest.mark.parametrize(
    'operations',
    [Neo4jSearchOperations, FalkorSearchOperations],
)
@pytest.mark.asyncio
async def test_provider_community_similarity_uses_summary_with_name_fallback(operations):
    executor = MagicMock()
    executor.execute_query = AsyncMock(return_value=([], [], None))

    await operations().community_similarity_search(executor, [0.1, 0.2])

    query = executor.execute_query.await_args.args[0]
    assert 'coalesce(c.summary_embedding, c.name_embedding)' in query


@pytest.mark.parametrize('provider', [GraphProvider.NEO4J, GraphProvider.FALKORDB])
@pytest.mark.asyncio
async def test_shared_community_similarity_fallback_uses_summary_embedding(provider):
    driver = SimpleNamespace(
        provider=provider,
        search_interface=None,
        execute_query=AsyncMock(return_value=([], [], None)),
    )

    await community_similarity_search(driver, [0.1, 0.2])

    query = driver.execute_query.await_args.args[0]
    assert 'coalesce(c.summary_embedding, c.name_embedding)' in query


@pytest.mark.parametrize('provider', [GraphProvider.NEO4J, GraphProvider.FALKORDB])
def test_community_save_queries_persist_summary_embedding(provider):
    query = get_community_node_save_query(provider)

    assert '$summary_embedding' in query
