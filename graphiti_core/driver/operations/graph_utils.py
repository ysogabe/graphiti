"""
Copyright 2024, Zep Software, Inc.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import logging
from collections import defaultdict

from pydantic import BaseModel

# Floor for the sweep cap. The effective default is max(this, 2 * len(projection)); see
# label_propagation for why the cap exists at all and why it scales with the node count.
MAX_LABEL_PROPAGATION_ITERATIONS = 50

logger = logging.getLogger(__name__)


class Neighbor(BaseModel):
    node_uuid: str
    edge_count: int


def label_propagation(
    projection: dict[str, list[Neighbor]],
    max_iterations: int | None = None,
) -> list[list[str]]:
    """Cluster nodes by label propagation.

    Every node starts in its own community and repeatedly adopts the community with the largest
    total edge weight among its neighbors (ties go to the larger community id) until a full
    sweep changes nothing.

    Updates are applied asynchronously (in place, sweeping in descending uuid order) and initial
    community ids are assigned in ascending uuid order, so the result depends only on the graph,
    not on the order the projection was built in. Sweeping against the id order lets the highest
    id travel along an already-sorted chain in one sweep instead of one hop per sweep.

    With synchronous updates two groups of nodes can adopt each other's label in the same sweep
    and swap forever. A single mutual edge of weight >= 2 is enough, but weight-1 edges alone can
    do it too (e.g. K2,2 plus a leaf).

    For a symmetric projection (every edge listed from both ends with the same weight, which is
    what an undirected MATCH yields) the asynchronous version always terminates: each label change
    strictly increases (total weight of same-label edges, sum of label ids) lexicographically.
    The projection is, however, built by one query per node, so edges ingested concurrently can
    make it asymmetric, and then it can oscillate forever (e.g. a->b:3, a->c:2, b->c:4, c->a:3
    cycles with period 2). `max_iterations` is the safety valve for that case: when it is hit
    a warning is logged and the current, unconverged partition is returned. The warning can also
    fire if the last permitted sweep still changed a label but the next one would have confirmed
    convergence.

    For the same reason a neighbor can name a node that is not a key of the projection (it was
    added to the group after the node list was read). Such neighbors are ignored, i.e. the edge
    is not counted; a node whose only neighbors are unknown keeps its own community. The number
    of ignored adjacency entries is logged as a warning.

    `max_iterations` defaults to max(MAX_LABEL_PROPAGATION_ITERATIONS, 2 * len(projection)).
    A fixed cap is wrong because even on symmetric inputs the sweeps needed grow linearly with
    the graph's diameter: a path with the same weight 2 in both directions whose uuids are in
    random order needs ~n/2 sweeps (measured: n=10/100/500/1300 -> 5/44/140/581 sweeps; n=1000
    over many orders: median 374, max 470). The worst observed ratio (20000 random symmetric
    graphs, adversarial search on n <= 20) is n sweeps, so 2n leaves a 2x margin over that.

    Cost: each sweep is O(n + m) (m = adjacency entries), so the worst case is O(n * (n + m)),
    and it is reached by symmetric long chains, not only by asymmetric input: a path has
    m = 2(n - 1) and needs O(n) sweeps, so its total cost is quadratic in n (n=1300 took ~0.9 s
    in pure Python). Graphs with a small diameter converge in few sweeps: synthetic sparse random
    graphs of 1300 nodes (the size of a 1.3k-entity group) with 1300/2600/5200 edges converged
    in 4/6-7/13-17 sweeps (< 0.1 s). With the default cap a run never exceeds max(50, 2n) sweeps.

    `max_iterations` must be an int >= 1 (bool is rejected); anything else raises ValueError.
    Silently clamping or rounding it hid misconfiguration (0 and -1 used to run one sweep and
    2.5 three), and every caller in graphiti_core uses the default, so rejecting is safe.
    """
    if max_iterations is None:
        max_iterations = max(MAX_LABEL_PROPAGATION_ITERATIONS, 2 * len(projection))
    if (
        not isinstance(max_iterations, int)
        or isinstance(max_iterations, bool)
        or max_iterations < 1
    ):
        raise ValueError(f'max_iterations must be an int >= 1 or None, got {max_iterations!r}')

    community_map = {uuid: i for i, uuid in enumerate(sorted(projection))}
    order = sorted(projection, reverse=True)

    # Drop neighbors that are not in the projection once, instead of on every sweep.
    adjacency: dict[str, list[Neighbor]] = {}
    unknown_entries = 0
    unknown_uuids: set[str] = set()
    for uuid, neighbors in projection.items():
        known = [neighbor for neighbor in neighbors if neighbor.node_uuid in community_map]
        unknown_entries += len(neighbors) - len(known)
        unknown_uuids.update(
            neighbor.node_uuid for neighbor in neighbors if neighbor.node_uuid not in community_map
        )
        adjacency[uuid] = known
    edge_entries = sum(len(neighbors) for neighbors in adjacency.values())
    if unknown_entries:
        logger.warning(
            'label_propagation ignored %d adjacency entries pointing to %d uuids that are not '
            'in the projection (%d nodes); they were probably added while it was being read.',
            unknown_entries,
            len(unknown_uuids),
            len(projection),
        )

    converged = False
    sweeps = 0
    while sweeps < max_iterations:
        sweeps += 1
        no_change = True

        for uuid in order:
            neighbors = adjacency[uuid]
            curr_community = community_map[uuid]

            community_candidates: dict[int, int] = defaultdict(int)
            for neighbor in neighbors:
                community_candidates[community_map[neighbor.node_uuid]] += neighbor.edge_count
            community_lst = [
                (count, community) for community, count in community_candidates.items()
            ]

            community_lst.sort(reverse=True)
            candidate_rank, community_candidate = community_lst[0] if community_lst else (0, -1)
            if community_candidate != -1 and candidate_rank > 1:
                new_community = community_candidate
            else:
                new_community = max(community_candidate, curr_community)

            if new_community != curr_community:
                community_map[uuid] = new_community
                no_change = False

        if no_change:
            converged = True
            break

    if converged:
        logger.debug(
            'label_propagation converged after %d sweeps (%d nodes, %d adjacency entries)',
            sweeps,
            len(projection),
            edge_entries,
        )
    else:
        logger.warning(
            'label_propagation did not converge within %d sweeps (max_iterations=%d, '
            '%d nodes, %d adjacency entries); returning an unconverged partition. '
            'Possible causes: asymmetric adjacency (edges changed while the projection was '
            'being read), a large chain-like graph that needs ~n/2 sweeps, or max_iterations '
            'set too small.',
            sweeps,
            max_iterations,
            len(projection),
            edge_entries,
        )

    community_cluster_map: dict[int, list[str]] = defaultdict(list)
    for uuid, community in community_map.items():
        community_cluster_map[community].append(uuid)

    return list(community_cluster_map.values())
