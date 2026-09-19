#!/usr/bin/env python3
"""Daily Graphiti ledger evaluation report (accumulation + quality + token estimate).

Queries Neo4j for each profile group (minamo / haruo), computes accumulation
state, quality signals, hook usefulness (source breakdown), and an estimated LLM
token cost (episodes x benchmark avg). Emits a Japanese report to stdout.

Also carries an embedding-health section (duplicate-vector groups + a live embedder
probe) because a route that silently stops encoding a script is invisible in every
other metric: the ledger keeps growing while cosine search and duplicate detection
quietly stop working.
"""

from __future__ import annotations

import asyncio
import os
from datetime import datetime, timedelta, timezone

from neo4j import GraphDatabase

from embedding_health_check import embedder_findings, ledger_findings, probe_available

# Ledger host moved off this box: Neo4j now lives in mahal's LXC `lxc-graphiti`
# (2026-09-10 migration). Override with NEO4J_URI for ad-hoc runs.
NEO4J_URI = os.environ.get('NEO4J_URI', 'bolt://192.168.0.158:7687')
NEO4J_USER = os.environ.get('NEO4J_USER', 'neo4j')
NEO4J_PASS = os.environ.get('NEO4J_PASSWORD', 'testpass')
GROUPS = ['minamo', 'haruo']

# Benchmark averages from benchmark_sync.py (glm-5.3-flash, effort high): per add_episode.
# input ~4100 / output ~1111, 2 LLM calls. Used as an estimate until the server
# exposes real token counts.
TOKENS_PER_EPISODE = 5211
LLM_CALLS_PER_EPISODE = 2.0

SOURCE_LABEL = {
    'session': 'on_session_end',
    'compress': 'on_pre_compress',
    'delegation': 'on_delegation',
    'mem': 'on_memory_write',
    'turn': 'sync_turn',
}


def _q(s, cypher, **kw):
    return s.run(cypher, **kw).data()


def group_eval(driver, group: str) -> dict:
    with driver.session() as s:
        eps = _q(s, 'MATCH (e:Episodic) WHERE e.group_id=$g RETURN count(e) AS c', g=group)[0]['c']
        ents = _q(s, 'MATCH (e:Entity) WHERE e.group_id=$g RETURN count(e) AS c', g=group)[0]['c']
        facts = _q(
            s,
            'MATCH (:Entity)-[r:RELATES_TO]->(:Entity) WHERE r.group_id=$g RETURN count(r) AS c',
            g=group,
        )[0]['c']
        # facts added in last 24h
        since = datetime.now(timezone.utc) - timedelta(hours=24)
        facts_24h = _q(
            s,
            'MATCH (:Entity)-[r:RELATES_TO]->(:Entity) WHERE r.group_id=$g '
            'AND r.created_at >= $since RETURN count(r) AS c',
            g=group,
            since=since,
        )[0]['c']
        # episodes by name-prefix (hook usefulness) — source is restricted to
        # EpisodeType enum and falls back to "text", so the name prefix is the tag.
        by_src = {
            r['s']: r['c']
            for r in _q(
                s,
                'MATCH (e:Episodic) WHERE e.group_id=$g '
                "RETURN split(e.name, '-')[0] AS s, count(*) AS c ORDER BY c DESC",
                g=group,
            )
        }
        # exact-duplicate fact texts
        dups = _q(
            s,
            'MATCH (:Entity)-[r:RELATES_TO]->(:Entity) WHERE r.group_id=$g '
            'WITH r.fact AS f, count(*) AS c WHERE c > 1 '
            'RETURN count(*) AS c',
            g=group,
        )[0]['c']
        # near-duplicate proxy: facts sharing the same subject+object entity pair
        same_pair = _q(
            s,
            'MATCH (a:Entity)-[r:RELATES_TO]->(b:Entity) WHERE a.group_id=$g '
            'WITH a.uuid AS au, b.uuid AS bu, count(*) AS c WHERE c > 1 '
            'RETURN count(*) AS c',
            g=group,
        )[0]['c']
        # top entities by incident-relationship degree (count{()-[*]->()} as an
        # ORDER BY expression made the planner scan the whole graph and hang).
        top_ents = [
            r['name']
            for r in _q(
                s,
                'MATCH (e:Entity) WHERE e.group_id=$g WITH e, count { (e)--() } AS deg '
                'RETURN e.name AS name ORDER BY deg DESC LIMIT 8',
                g=group,
            )
        ]
    return {
        'episodes': eps,
        'entities': ents,
        'facts': facts,
        'facts_24h': facts_24h,
        'by_source': by_src,
        'exact_dups': dups,
        'same_pair_dups': same_pair,
        'top_entities': top_ents,
    }


def embedding_health(driver) -> list[str]:
    """Ledger duplicate-vector counts + a live embedder probe, as report lines."""
    findings, stats = ledger_findings(driver)
    lines = ['## 埋め込み健全性(別テキストで同一ベクトル / ライブ埋め込みプローブ)']
    for label in ('Entity.name_embedding', 'RELATES_TO.fact_embedding'):
        total = stats.get(label, 0)
        distinct = stats.get(f'{label}.distinct_vectors', 0)
        colliding = stats.get(f'{label}.colliding_groups', 0)
        mark = '⚠️ ' if colliding else ''
        lines.append(
            f'- {mark}{label}: {total} 行 / {distinct} distinct'
            f'(別テキストが同一ベクトル: {colliding} 組)'
        )

    if probe_available():
        probe_findings, hashes = asyncio.run(embedder_findings())
        findings.extend(probe_findings)
        distinct = len(set(hashes.values()))
        ok = '✓' if distinct == len(hashes) else '⚠️ '
        lines.append(
            f'- {ok} ライブプローブ: {len(hashes)}件中 {distinct} distinct'
            f'(みなも/トヨタ/エコーショー + ASCII対照)'
        )
    else:
        lines.append('- ライブプローブ: スキップ(GOOGLE_API_KEY 未設定 → 台帳側のみ判定)')

    if findings:
        lines.append(f'- ⚠️ 検出 {len(findings)} 件:')
        lines.extend(f'    - {f}' for f in findings[:6])
        lines.append(
            '    - 詳細: `embedding_health_check.py` / 再埋め込み: `reembed_embeddings.py`'
        )
    lines.append('')
    return lines


def main() -> None:
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASS))
    day = datetime.now(timezone.utc).astimezone().strftime('%Y-%m-%d %H:%M')
    lines = [f'# Graphiti 台帳 日次レポート({day})', '']

    total_eps = 0
    for g in GROUPS:
        m = group_eval(driver, g)
        total_eps += m['episodes']
        facts_per_ep = (m['facts'] / m['episodes']) if m['episodes'] else 0.0
        lines.append(f'## {g}')
        lines.append(
            f'- 蓄積: {m["episodes"]} episodes / {m["entities"]} entities / {m["facts"]} facts'
            f'(直近24h +{m["facts_24h"]} facts)'
        )
        lines.append(
            f'- 品質: {facts_per_ep:.1f} facts/episode, '
            f'完全重複 {m["exact_dups"]}, 同一ペア複数 {m["same_pair_dups"]}'
        )
        lines.append(
            '- フック(episode by source): '
            + (
                ', '.join(f'{SOURCE_LABEL.get(k, k)}={v}' for k, v in m['by_source'].items())
                if m['by_source']
                else 'なし(まだ発火なし)'
            )
        )
        lines.append(f'- 上位実体: {", ".join(m["top_entities"]) or "なし"}')
        lines.append('')

    # Surface any groups that exist but aren't part of the ledger (test/reference
    # data that the report otherwise hides — prevents the "blind spot" of only ever
    # looking at GROUPS).
    with driver.session() as s:
        present = {r['g'] for r in s.run('MATCH (n) RETURN DISTINCT n.group_id AS g').data()}
    others = sorted(present - set(GROUPS))
    if others:
        lines.append('## ⚠️ 台帳対象外のグループ(参考/テストデータ)')
        lines.append(
            f'- {", ".join(g if g else "(空 group_id)" for g in others)}'
            f' (レポート対象外。台帳グループは {", ".join(GROUPS)})'
        )
        lines.append('')

    # embedding health (duplicate vectors + live embedder probe) — a silent embedder
    # regression corrupts dedupe/search without moving any other number in this report.
    lines.extend(embedding_health(driver))

    # token estimate + hook usefulness summary
    tok = total_eps * TOKENS_PER_EPISODE
    calls = total_eps * LLM_CALLS_PER_EPISODE
    lines.append('## トークン推定(実測は server 再起動後の拡張で可能)')
    lines.append(
        f'- 全エピソード: {total_eps} (推定 {tok:,.0f} tokens / {calls:.0f} LLM calls)'
        f'  ※ glm-5.3-flash high で1エピソード≈{TOKENS_PER_EPISODE} tokens'
    )
    lines.append('')

    # hook usefulness summary (did the configured hooks actually fire?)
    # Aggregate-only would hide a group that is missing a hook, so track per group.
    hooks_by_group: dict[str, set[str]] = {}
    for g in GROUPS:
        hooks_by_group[g] = set(
            str(SOURCE_LABEL.get(k, k)) for k in group_eval(driver, g)['by_source']
        )
    live_hooks = set().union(*hooks_by_group.values()) if hooks_by_group else set()
    lines.append('## 設定フックの有用性')
    lines.append(f'- 発火確認済: {", ".join(sorted(live_hooks)) or "なし"}')
    expected = ('on_session_end', 'on_pre_compress', 'on_delegation', 'on_memory_write')
    for g, fired in hooks_by_group.items():
        missing = [h for h in expected if h not in fired]
        if missing:
            lines.append(
                f'- ⚠️ {g}: まだ発火なし {", ".join(missing)}'
                f'(セッション境界/圧縮/委譲/メモリ書込が未発生のため)'
            )
    missing = [
        h for h in ('on_session_end', 'on_pre_compress', 'on_delegation') if h not in live_hooks
    ]
    if missing:
        lines.append(
            f'- ⚠️ 全体でまだ発火なし: {", ".join(missing)}(セッション境界/圧縮/委譲が未発生のため)'
        )

    print('\n'.join(lines))
    driver.close()


if __name__ == '__main__':
    main()
