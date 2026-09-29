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

import signal
from contextlib import contextmanager

from graphiti_core.utils.maintenance.community_operations import Neighbor, label_propagation


@contextmanager
def time_limit(seconds: float):
    """収束しない実装をハングではなく失敗として検出するための締切。"""

    def _handler(signum, frame):  # noqa: ANN001
        raise TimeoutError(f'label_propagation が {seconds}s 以内に収束しなかった')

    old = signal.signal(signal.SIGALRM, _handler)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, old)


def _projection(edges: dict[str, list[tuple[str, int]]]) -> dict[str, list[Neighbor]]:
    return {
        uuid: [Neighbor(node_uuid=n, edge_count=w) for n, w in neighbors]
        for uuid, neighbors in edges.items()
    }


def test_mutual_heavy_edge_converges():
    """重み 2 の相互辺: 同期更新だと両端が互いのラベルを取り合って永久に振動する。"""
    proj = _projection({'a': [('b', 2)], 'b': [('a', 2)]})

    with time_limit(10):
        clusters = label_propagation(proj)

    assert sorted(sum(clusters, [])) == ['a', 'b']
    assert len(clusters) == 1, '重み 2 の 1 辺で結ばれた 2 ノードは同じコミュニティになる'


def test_star_with_heavy_leaves_converges():
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

    with time_limit(10):
        clusters = label_propagation(_projection(core))

    assigned = sum(clusters, [])
    assert sorted(assigned) == sorted(core), '全ノードが必ずどれかのコミュニティに入る'
    assert len(assigned) == len(set(assigned)), '同じノードが複数のコミュニティに入ってはいけない'

    # 重み 2 で結ばれた URL とエンティティは同一コミュニティに落ち着く
    group_of = {uuid: i for i, cluster in enumerate(clusters) for uuid in cluster}
    assert group_of['star-office-ui'] == group_of['url-star-office-ui']


def test_iteration_cap_bounds_work():
    """収束しない入力でも反復上限で必ず返る（安全弁）。"""
    n = 8
    proj = _projection(
        {str(i): [(str((i + 1) % n), 2), (str((i - 1) % n), 2)] for i in range(n)}
    )

    with time_limit(10):
        clusters = label_propagation(proj, max_iterations=3)

    assert sorted(sum(clusters, [])) == sorted(proj.keys())
