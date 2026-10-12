import json
import logging
from collections import Counter, defaultdict

from graphiti_core.driver.driver import GraphDriver, GraphProvider

# Single implementation shared with driver/*/operations/graph_ops.py; re-exported here so existing
# imports from this module keep working.
from graphiti_core.driver.operations.graph_utils import (  # noqa: F401
    MAX_LABEL_PROPAGATION_ITERATIONS,
    Neighbor,
    label_propagation,
)
from graphiti_core.edges import CommunityEdge
from graphiti_core.embedder import EmbedderClient
from graphiti_core.graphiti_types import GraphitiClients, generate_prompt_response
from graphiti_core.helpers import semaphore_gather
from graphiti_core.llm_client import LLMClient
from graphiti_core.llm_client.errors import EmptyResponseError
from graphiti_core.models.nodes.node_db_queries import COMMUNITY_NODE_RETURN
from graphiti_core.nodes import CommunityNode, EntityNode, get_community_node_from_record
from graphiti_core.prompts import prompt_library
from graphiti_core.prompts.summarize_nodes import Summary, SummaryDescription
from graphiti_core.utils.datetime_utils import utc_now
from graphiti_core.utils.maintenance.edge_operations import build_community_edges
from graphiti_core.utils.text_utils import MAX_SUMMARY_CHARS, truncate_at_sentence

MAX_COMMUNITY_BUILD_CONCURRENCY = 10

# Per-summary budget for the shortened-input retry and the deterministic fallback merge.
RESCUE_SUMMARY_CHARS = MAX_SUMMARY_CHARS // 2
# Upper bound for a community name derived without the LLM.
FALLBACK_NAME_CHARS = 200

# Failures that mean "this prompt produced no usable output" (e.g. the output budget was spent
# on reasoning). Anything else (auth, rate limit, server errors) propagates unchanged.
_RECOVERABLE_SUMMARY_ERRORS = (EmptyResponseError, json.JSONDecodeError)

# How many times a community summary/name fell back to the non-LLM path, keyed by
# 'summarize_pair' / 'summary_description'. Read by callers and tests.
COMMUNITY_SUMMARY_FALLBACKS: Counter[str] = Counter()

logger = logging.getLogger(__name__)


async def get_community_clusters(
    driver: GraphDriver,
    group_ids: list[str] | None,
    max_coroutines: int | None = None,
) -> list[list[EntityNode]]:
    if driver.graph_operations_interface:
        try:
            return await driver.graph_operations_interface.get_community_clusters(driver, group_ids)
        except NotImplementedError:
            pass

    community_clusters: list[list[EntityNode]] = []

    if group_ids is None:
        group_id_values, _, _ = await driver.execute_query(
            """
            MATCH (n:Entity)
            WHERE n.group_id IS NOT NULL
            RETURN
                collect(DISTINCT n.group_id) AS group_ids
            """
        )

        group_ids = group_id_values[0]['group_ids'] if group_id_values else []

    for group_id in group_ids:
        projection: dict[str, list[Neighbor]] = {}
        nodes = await EntityNode.get_by_group_ids(driver, [group_id])
        for node in nodes:
            match_query = """
                MATCH (n:Entity {group_id: $group_id, uuid: $uuid})-[e:RELATES_TO]-(m: Entity {group_id: $group_id})
            """
            if driver.provider == GraphProvider.KUZU:
                match_query = """
                MATCH (n:Entity {group_id: $group_id, uuid: $uuid})-[:RELATES_TO]-(e:RelatesToNode_)-[:RELATES_TO]-(m: Entity {group_id: $group_id})
                """
            records, _, _ = await driver.execute_query(
                match_query
                + """
                WITH count(e) AS count, m.uuid AS uuid
                RETURN
                    uuid,
                    count
                """,
                uuid=node.uuid,
                group_id=group_id,
            )

            projection[node.uuid] = [
                Neighbor(node_uuid=record['uuid'], edge_count=record['count']) for record in records
            ]

        cluster_uuids = label_propagation(projection)

        community_clusters.extend(
            list(
                await semaphore_gather(
                    *[EntityNode.get_by_uuids(driver, cluster) for cluster in cluster_uuids],
                    max_coroutines=max_coroutines,
                )
            )
        )

    return community_clusters


async def _summarize_pair_llm(
    llm_client: LLMClient,
    summary_pair: tuple[str, str],
    *,
    clients: GraphitiClients | None = None,
) -> str:
    # Prepare context for LLM
    context = {
        'node_summaries': [{'summary': summary} for summary in summary_pair],
    }

    llm_response = await generate_prompt_response(
        llm_client,
        'summarize_nodes.summarize_pair',
        prompt_library.summarize_nodes.summarize_pair,
        context,
        clients=clients,
        response_model=Summary,
    )

    pair_summary = llm_response.get('summary', '')

    return truncate_at_sentence(pair_summary, MAX_SUMMARY_CHARS)


def _merge_summaries_deterministically(summary_pair: tuple[str, str]) -> str:
    # Split the budget before joining: truncating the joined text instead would cut back to the
    # last sentence end, which drops the whole right side when it has no sentence terminator.
    left, right = (s.strip() for s in summary_pair)
    if not right:
        return truncate_at_sentence(left, MAX_SUMMARY_CHARS)
    if not left:
        return truncate_at_sentence(right, MAX_SUMMARY_CHARS)
    left = truncate_at_sentence(left, RESCUE_SUMMARY_CHARS)
    # The right side gets whatever the left left over, minus the separating space. Since
    # truncate_at_sentence hard-cuts when there is no sentence end, it never comes back empty.
    right = truncate_at_sentence(right, max(MAX_SUMMARY_CHARS - len(left) - 1, 0))
    return f'{left} {right}'.strip() if right else left


async def summarize_pair(
    llm_client: LLMClient,
    summary_pair: tuple[str, str],
    *,
    clients: GraphitiClients | None = None,
) -> str:
    """Merge two summaries, surviving an exhausted LLM output budget.

    clients: optional bundle that selects prompt overrides and model routes.

    At most 2 ``generate_response`` calls are made (the original prompt, then one retry with
    shortened inputs); each call retries internally up to 4 attempts, so this issues at most
    8 LLM requests before falling back to a deterministic merge.
    """
    try:
        return await _summarize_pair_llm(llm_client, summary_pair, clients=clients)
    except _RECOVERABLE_SUMMARY_ERRORS as e:
        first_error: Exception = e

    # Rescue 1: retry once with each input shortened. Skip it when shortening changes nothing,
    # since the identical prompt would fail the same way.
    shortened = (
        truncate_at_sentence(summary_pair[0], RESCUE_SUMMARY_CHARS),
        truncate_at_sentence(summary_pair[1], RESCUE_SUMMARY_CHARS),
    )
    if shortened != summary_pair:
        try:
            return await _summarize_pair_llm(llm_client, shortened, clients=clients)
        except _RECOVERABLE_SUMMARY_ERRORS as e:
            first_error = e

    # Rescue 2: deterministic merge without the LLM. The left side is capped at half the budget
    # so a long left summary cannot crowd out the right one.
    COMMUNITY_SUMMARY_FALLBACKS['summarize_pair'] += 1
    logger.warning(
        f'summarize_pair fell back to a deterministic merge after LLM failure: {first_error!r}'
    )
    return _merge_summaries_deterministically(summary_pair)


async def _generate_summary_description_llm(
    llm_client: LLMClient,
    summary: str,
    *,
    clients: GraphitiClients | None = None,
) -> str:
    context = {
        'summary': summary,
    }

    llm_response = await generate_prompt_response(
        llm_client,
        'summarize_nodes.summary_description',
        prompt_library.summarize_nodes.summary_description,
        context,
        clients=clients,
        response_model=SummaryDescription,
    )

    description = llm_response.get('description', '')

    return description


async def generate_summary_description(
    llm_client: LLMClient,
    summary: str,
    *,
    clients: GraphitiClients | None = None,
) -> str:
    """Name a community from its summary, surviving an exhausted LLM output budget.

    clients: optional bundle that selects prompt overrides and model routes.

    At most 2 ``generate_response`` calls are made (the original prompt, then one retry with a
    shortened summary); each call retries internally up to 4 attempts, so this issues at most
    8 LLM requests before falling back to the summary's leading sentence.
    """
    try:
        return await _generate_summary_description_llm(llm_client, summary, clients=clients)
    except _RECOVERABLE_SUMMARY_ERRORS as e:
        first_error: Exception = e

    # Rescue 1: retry once with a shortened summary, only if that changes the prompt.
    shortened = truncate_at_sentence(summary, RESCUE_SUMMARY_CHARS)
    if shortened != summary:
        try:
            return await _generate_summary_description_llm(llm_client, shortened, clients=clients)
        except _RECOVERABLE_SUMMARY_ERRORS as e:
            first_error = e

    # Rescue 2: name the community after the leading sentence of its summary.
    COMMUNITY_SUMMARY_FALLBACKS['summary_description'] += 1
    logger.warning(
        'generate_summary_description fell back to a summary prefix after LLM failure: '
        f'{first_error!r}'
    )
    return truncate_at_sentence(summary, FALLBACK_NAME_CHARS)


async def build_community(
    llm_client: LLMClient,
    community_cluster: list[EntityNode],
    *,
    clients: GraphitiClients | None = None,
    max_coroutines: int | None = None,
) -> tuple[CommunityNode, list[CommunityEdge]]:
    """Build one community from its nodes.

    clients: optional bundle that selects prompt overrides and model routes.
    """
    summaries = [entity.summary for entity in community_cluster]
    length = len(summaries)
    while length > 1:
        odd_one_out: str | None = None
        if length % 2 == 1:
            odd_one_out = summaries.pop()
            length -= 1
        new_summaries: list[str] = list(
            await semaphore_gather(
                *[
                    summarize_pair(
                        llm_client,
                        (str(left_summary), str(right_summary)),
                        clients=clients,
                    )
                    for left_summary, right_summary in zip(
                        summaries[: int(length / 2)], summaries[int(length / 2) :], strict=False
                    )
                ],
                max_coroutines=max_coroutines,
            )
        )
        if odd_one_out is not None:
            new_summaries.append(odd_one_out)
        summaries = new_summaries
        length = len(summaries)

    summary = truncate_at_sentence(summaries[0], MAX_SUMMARY_CHARS)
    name = await generate_summary_description(llm_client, summary, clients=clients)
    now = utc_now()
    community_node = CommunityNode(
        name=name,
        group_id=community_cluster[0].group_id,
        labels=['Community'],
        created_at=now,
        summary=summary,
    )
    community_edges = build_community_edges(community_cluster, community_node, now)

    logger.debug(f'Built community {community_node.uuid} with {len(community_edges)} edges')

    return community_node, community_edges


async def build_communities(
    driver: GraphDriver,
    llm_client: LLMClient,
    group_ids: list[str] | None,
    *,
    clients: GraphitiClients | None = None,
    max_coroutines: int | None = None,
    embedder: EmbedderClient | None = None,
) -> tuple[list[CommunityNode], list[CommunityEdge]]:
    """Build communities via label propagation and, given an embedder, embed their summaries.

    Community summary search is wired up for Neo4j and FalkorDB only. Kuzu's `community_name` FTS
    index covers `name` but not `summary`, and Neptune's AOSS mirror only carries `name`/`uuid`/
    `group_id` (see `save_to_aoss` in `driver/neptune/operations/community_node_ops.py`) — on
    those two providers `summary_embedding` is still computed and saved, but unreachable by search.

    clients: optional bundle that selects prompt overrides and model routes.
    """
    community_clusters = await get_community_clusters(
        driver, group_ids, max_coroutines=max_coroutines
    )

    # Honour the per-instance `max_coroutines` cap when set; otherwise fall back to the
    # historical built-in cap of `MAX_COMMUNITY_BUILD_CONCURRENCY` to preserve behavior.
    # We let `semaphore_gather` enforce the cap directly rather than wrapping each
    # coroutine in a redundant `asyncio.Semaphore` — both mechanisms would bound the
    # same fan-out at the same value, and the inner cap obscures the precedence.
    communities: list[tuple[CommunityNode, list[CommunityEdge]]] = list(
        await semaphore_gather(
            *[
                build_community(
                    llm_client,
                    cluster,
                    clients=clients,
                    max_coroutines=max_coroutines,
                )
                for cluster in community_clusters
            ],
            max_coroutines=max_coroutines or MAX_COMMUNITY_BUILD_CONCURRENCY,
        )
    )

    community_nodes: list[CommunityNode] = []
    community_edges: list[CommunityEdge] = []
    for community in communities:
        community_nodes.append(community[0])
        community_edges.extend(community[1])

    if embedder is not None:
        await _embed_community_summaries(embedder, community_nodes)

    return community_nodes, community_edges


async def _embed_community_summaries(
    embedder: EmbedderClient, community_nodes: list[CommunityNode]
) -> None:
    """Embed every community's summary, batching the embedder call when the client supports it.

    `build_communities` can produce many communities in one run, so a single `create_batch` call
    is far cheaper than one `create` round trip per community. `EmbedderClient.create_batch`
    raises `NotImplementedError` unless a concrete client overrides it, so clients that don't
    support batching fall back to embedding one summary at a time, same as before.
    """
    pending: list[tuple[CommunityNode, str]] = []
    for node in community_nodes:
        text = node.summary.replace('\n', ' ').strip()
        if not text:
            node.summary_embedding = None
            continue
        pending.append((node, text))

    if not pending:
        return

    try:
        embeddings = await embedder.create_batch([text for _, text in pending])
    except NotImplementedError:
        await semaphore_gather(
            *[node.generate_summary_embedding(embedder) for node, _ in pending],
            max_coroutines=MAX_COMMUNITY_BUILD_CONCURRENCY,
        )
        return

    for (node, _), embedding in zip(pending, embeddings, strict=True):
        node.summary_embedding = embedding


async def remove_communities(driver: GraphDriver, group_ids: list[str] | None = None):
    if driver.graph_operations_interface:
        try:
            return await driver.graph_operations_interface.remove_communities(
                driver, group_ids=group_ids
            )
        except NotImplementedError:
            pass

    if driver.graph_ops:
        return await driver.graph_ops.remove_communities(driver, group_ids=group_ids)

    if group_ids:
        await driver.execute_query(
            """
            MATCH (c:Community)
            WHERE c.group_id IN $group_ids
            DETACH DELETE c
            """,
            group_ids=group_ids,
        )
    else:
        await driver.execute_query(
            """
            MATCH (c:Community)
            DETACH DELETE c
            """
        )


async def determine_entity_community(
    driver: GraphDriver, entity: EntityNode
) -> tuple[CommunityNode | None, bool]:
    if driver.graph_operations_interface:
        try:
            return await driver.graph_operations_interface.determine_entity_community(
                driver, entity
            )
        except NotImplementedError:
            pass

    # Check if the node is already part of a community
    records, _, _ = await driver.execute_query(
        """
        MATCH (c:Community)-[:HAS_MEMBER]->(n:Entity {uuid: $entity_uuid})
        RETURN
        """
        + COMMUNITY_NODE_RETURN,
        entity_uuid=entity.uuid,
    )

    if len(records) > 0:
        return get_community_node_from_record(records[0]), False

    # If the node has no community, add it to the mode community of surrounding entities
    match_query = """
        MATCH (c:Community)-[:HAS_MEMBER]->(m:Entity)-[:RELATES_TO]-(n:Entity {uuid: $entity_uuid})
    """
    if driver.provider == GraphProvider.KUZU:
        match_query = """
            MATCH (c:Community)-[:HAS_MEMBER]->(m:Entity)-[:RELATES_TO]-(e:RelatesToNode_)-[:RELATES_TO]-(n:Entity {uuid: $entity_uuid})
        """
    records, _, _ = await driver.execute_query(
        match_query
        + """
        RETURN
        """
        + COMMUNITY_NODE_RETURN,
        entity_uuid=entity.uuid,
    )

    communities: list[CommunityNode] = [
        get_community_node_from_record(record) for record in records
    ]

    community_map: dict[str, int] = defaultdict(int)
    for community in communities:
        community_map[community.uuid] += 1

    community_uuid = None
    max_count = 0
    for uuid, count in community_map.items():
        if count > max_count:
            community_uuid = uuid
            max_count = count

    if max_count == 0:
        return None, False

    for community in communities:
        if community.uuid == community_uuid:
            return community, True

    return None, False


async def update_community(
    driver: GraphDriver,
    llm_client: LLMClient,
    embedder: EmbedderClient,
    entity: EntityNode,
    *,
    clients: GraphitiClients | None = None,
) -> tuple[list[CommunityNode], list[CommunityEdge]]:
    """Update the community that contains an entity.

    clients: optional bundle that selects prompt overrides and model routes.
    """
    community, is_new = await determine_entity_community(driver, entity)

    if community is None:
        return [], []

    new_summary = await summarize_pair(
        llm_client,
        (entity.summary, community.summary),
        clients=clients,
    )
    new_name = await generate_summary_description(llm_client, new_summary, clients=clients)

    community.summary = new_summary
    community.name = new_name
    # Single-community update (one entity at a time), so there is nothing to batch the way
    # `_embed_community_summaries` batches embedder calls across many communities.
    await community.generate_summary_embedding(embedder)

    community_edges = []
    if is_new:
        community_edge = (build_community_edges([entity], community, utc_now()))[0]
        await community_edge.save(driver)
        community_edges.append(community_edge)

    await community.generate_name_embedding(embedder)

    await community.save(driver)

    return [community], community_edges
