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

# label_propagation は反復上限を持つので、実装が振動してもハングせずに返る。
# そのためテスト側のタイムアウト（SIGALRM 依存）は不要。

import logging
import random
from collections import defaultdict

import graphiti_core.driver.operations.graph_utils as graph_utils
import graphiti_core.utils.maintenance.community_operations as community_operations
from graphiti_core.utils.maintenance.community_operations import Neighbor, label_propagation

LOGGER = graph_utils.__name__


def _projection(edges: dict[str, list[tuple[str, int]]]) -> dict[str, list[Neighbor]]:
    return {
        uuid: [Neighbor(node_uuid=n, edge_count=w) for n, w in neighbors]
        for uuid, neighbors in edges.items()
    }


def _chain_of(names: list[str], weight: int) -> dict[str, list[tuple[str, int]]]:
    edges: dict[str, list[tuple[str, int]]] = {name: [] for name in names}
    for a, b in zip(names, names[1:], strict=False):
        edges[a].append((b, weight))
        edges[b].append((a, weight))
    return edges


def _partition(clusters: list[list[str]]) -> frozenset[frozenset[str]]:
    return frozenset(frozenset(c) for c in clusters)


def _assert_each_node_once(clusters: list[list[str]], nodes) -> None:
    assigned = sum(clusters, [])
    assert sorted(assigned) == sorted(nodes), '全ノードが必ずどれかのコミュニティに入る'
    assert len(assigned) == len(set(assigned)), '同じノードが複数のコミュニティに入ってはいけない'


def _synchronous_label_propagation(
    projection: dict[str, list[Neighbor]], max_iterations: int
) -> bool:
    """修正前（同期更新）のアルゴリズム。max_iterations 以内に収束したかを返す。"""
    community_map = {uuid: i for i, uuid in enumerate(projection)}
    for _ in range(max_iterations):
        new_map: dict[str, int] = {}
        for uuid, neighbors in projection.items():
            candidates: dict[int, int] = defaultdict(int)
            for neighbor in neighbors:
                candidates[community_map[neighbor.node_uuid]] += neighbor.edge_count
            ranked = sorted(((c, comm) for comm, c in candidates.items()), reverse=True)
            rank, candidate = ranked[0] if ranked else (0, -1)
            new_map[uuid] = (
                candidate if candidate != -1 and rank > 1 else max(candidate, community_map[uuid])
            )
        if new_map == community_map:
            return True
        community_map = new_map
    return False


def test_single_implementation_shared_with_driver_graph_ops():
    """community_operations と driver/*/graph_ops が同じ実装を使う（片方だけ直る事故の防止）。"""
    assert community_operations.label_propagation is graph_utils.label_propagation
    assert community_operations.Neighbor is graph_utils.Neighbor


def test_mutual_heavy_edge_converges(caplog):
    """重み 2 の相互辺: 同期更新だと両端が互いのラベルを取り合って永久に振動する。"""
    proj = _projection({'a': [('b', 2)], 'b': [('a', 2)]})
    assert not _synchronous_label_propagation(proj, 200)

    with caplog.at_level(logging.WARNING, logger=LOGGER):
        clusters = label_propagation(proj)

    assert sorted(sum(clusters, [])) == ['a', 'b']
    assert len(clusters) == 1, '重み 2 の 1 辺で結ばれた 2 ノードは同じコミュニティになる'
    assert not caplog.records


def test_weight_one_edges_alone_oscillate_synchronously(caplog):
    """重み 1 の辺だけでも同期更新は振動する（K2,2 + 葉）。非同期版は収束する。"""
    proj = _projection(
        {
            '0': [('2', 1), ('3', 1)],
            '1': [('2', 1), ('3', 1)],
            '2': [('0', 1), ('1', 1), ('4', 1)],
            '3': [('0', 1), ('1', 1)],
            '4': [('2', 1)],
        }
    )
    assert not _synchronous_label_propagation(proj, 200)

    with caplog.at_level(logging.WARNING, logger=LOGGER):
        clusters = label_propagation(proj)

    _assert_each_node_once(clusters, proj)
    assert not caplog.records


def test_star_with_heavy_leaves_converges(caplog):
    """実障害の最小再現（2026-09-29: 32 ノード/58 隣接の group で 7 ノードが永久振動）。

    中心 y.sogabe に重み 2 以上の葉がぶら下がり、葉は中心だけを見ている星型。
    同期更新では各葉と中心が互いのラベルを交換し続ける。
    """
    core = {
        'y.sogabe': [
            ('codex', 2),
            ('genai-lab', 3),
            ('subdomain-akebi', 3),
            ('claude-code', 4),
            ('star-office-ui', 1),
        ],
        'genai-lab': [('y.sogabe', 3)],
        'codex': [('y.sogabe', 2)],
        'subdomain-akebi': [('y.sogabe', 3)],
        'claude-code': [('y.sogabe', 4)],
        'star-office-ui': [('y.sogabe', 1), ('url-star-office-ui', 2)],
        'url-star-office-ui': [('star-office-ui', 2)],
    }
    assert not _synchronous_label_propagation(_projection(core), 200)

    with caplog.at_level(logging.WARNING, logger=LOGGER):
        clusters = label_propagation(_projection(core))

    _assert_each_node_once(clusters, core)
    assert not caplog.records

    # 重み 2 で結ばれた URL とエンティティは同一コミュニティに落ち着く
    group_of = {uuid: i for i, cluster in enumerate(clusters) for uuid in cluster}
    assert group_of['star-office-ui'] == group_of['url-star-office-ui']


def test_long_chain_converges_under_default_cap(caplog):
    """鎖は O(n) sweep を要する（uuid 順がばらばらだと ~n/2）。固定上限 50 では打ち切られる長さでも
    既定上限は正常収束を打ち切らない（警告なし・1 クラスタ）。"""
    names = [f'n{i:03d}' for i in range(200)]
    random.Random(0).shuffle(names)
    proj = _projection(_chain_of(names, 2))
    assert len(label_propagation(proj, max_iterations=50)) > 1, '前提: 50 sweep では収束しない'
    caplog.clear()

    with caplog.at_level(logging.WARNING, logger=LOGGER):
        clusters = label_propagation(proj)

    _assert_each_node_once(clusters, proj)
    assert len(clusters) == 1
    assert not caplog.records


def test_asymmetric_projection_hits_cap_and_warns(caplog):
    """非対称な隣接（並行取り込み中の逐次クエリで起こりうる）は非同期版でも永久に振動する。

    有向 3 サイクル 0->2->1->0: 各ノードが唯一の隣接先のラベルを取り、ラベルが回り続ける。
    上限で必ず返り、未収束の分割を返したことを警告で記録する。
    """
    proj = _projection({'0': [('2', 2)], '1': [('0', 5)], '2': [('1', 5)]})

    with caplog.at_level(logging.WARNING, logger=LOGGER):
        clusters = label_propagation(proj, max_iterations=3)

    _assert_each_node_once(clusters, proj)
    assert len(clusters) > 1, '未収束のまま返る'
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1
    assert 'did not converge within 3 sweeps' in warnings[0].getMessage()

    # 上限を大きくしても収束しない（安全弁が本質的に必要なケース）
    caplog.clear()
    with caplog.at_level(logging.WARNING, logger=LOGGER):
        label_propagation(proj)
    assert 'did not converge' in caplog.records[0].getMessage()


def test_iteration_cap_truncates_symmetric_input(caplog):
    """対称入力でも上限が小さすぎれば打ち切られ、警告が出る（打ち切りの検出）。"""
    # uuid 順がばらばらな鎖は ~n/2 sweep を要する
    names = [f'n{i:02d}' for i in range(20)]
    random.Random(0).shuffle(names)
    proj = _projection(_chain_of(names, 2))

    with caplog.at_level(logging.WARNING, logger=LOGGER):
        clusters = label_propagation(proj, max_iterations=2)

    _assert_each_node_once(clusters, proj)
    assert len(clusters) > 1
    assert any('did not converge' in r.getMessage() for r in caplog.records)


def test_partition_independent_of_insertion_order():
    """初期ラベル id は uuid の sorted 順で振るので、dict の挿入順（DB 返却順）に依存しない。"""
    rng = random.Random(42)
    nodes = [f'u{i:03d}' for i in range(60)]
    weights: dict[str, dict[str, int]] = defaultdict(dict)
    for _ in range(90):
        a, b = rng.sample(nodes, 2)
        w = rng.randint(1, 4)
        weights[a][b] = w
        weights[b][a] = w

    partitions = set()
    for _ in range(30):
        order = nodes[:]
        rng.shuffle(order)
        edges = {}
        for node in order:
            neighbors = list(weights[node].items())
            rng.shuffle(neighbors)
            edges[node] = neighbors
        partitions.add(_partition(label_propagation(_projection(edges))))

    assert len(partitions) == 1
