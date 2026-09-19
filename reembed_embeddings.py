#!/usr/bin/env python3
"""Re-embed the Graphiti ledger with the configured embedder (dry-run by default).

Needed whenever the embedder model/route changes, because the ledger stores *vectors*: rows
written under the old model stay in the old vector space and are incomparable with new ones.
Measured case: Vertex text-embedding-005 over an API-key route does not encode CJK, so every
pure-CJK name collapsed onto one constant vector; gemini-embedding-001 encodes them properly
and therefore requires this migration.

Usage:

    NEO4J_URI=bolt://192.168.0.158:7687 NEO4J_PASSWORD=testpass GOOGLE_API_KEY=... \
    EMBEDDER_MODEL=gemini-embedding-001 EMBEDDER_DIMENSIONS=768 EMBEDDER_VERTEXAI=true \
    GOOGLE_CLOUD_PROJECT=... GOOGLE_CLOUD_LOCATION=global \
    /opt/graphiti/.venv/bin/python reembed_embeddings.py            # dry-run report

    ... reembed_embeddings.py --apply --backup-dir ~/backups        # write (backup first)

Exit code 0 on success; a dry run always exits 0 unless the new vectors themselves are broken.
"""

from __future__ import annotations

import argparse
import asyncio
import gzip
import hashlib
import json
import math
import os
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

from neo4j import GraphDatabase

READ_ENTITIES = (
    'MATCH (e:Entity) WHERE e.name IS NOT NULL RETURN e.uuid AS uuid, e.name AS text '
    'ORDER BY e.uuid'
)
READ_EDGES = (
    'MATCH ()-[r:RELATES_TO]->() WHERE r.fact IS NOT NULL RETURN r.uuid AS uuid, r.fact AS text '
    'ORDER BY r.uuid'
)
OLD_ENTITIES = (
    'MATCH (e:Entity) WHERE e.name IS NOT NULL RETURN e.name_embedding AS emb ORDER BY e.uuid'
)
OLD_EDGES = (
    'MATCH ()-[r:RELATES_TO]->() WHERE r.fact IS NOT NULL RETURN r.fact_embedding AS emb '
    'ORDER BY r.uuid'
)
WRITE_ENTITIES = (
    'UNWIND $rows AS row MATCH (e:Entity {uuid: row.uuid}) SET e.name_embedding = row.embedding'
)
WRITE_EDGES = (
    'UNWIND $rows AS row MATCH ()-[r:RELATES_TO {uuid: row.uuid}]->() '
    'SET r.fact_embedding = row.embedding'
)


def vector_hash(vector: list[float]) -> str:
    return hashlib.sha256(','.join(f'{value:.6f}' for value in vector).encode()).hexdigest()[:12]


def cosine(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b, strict=True))


def norm(vector: list[float]) -> float:
    return math.sqrt(sum(value * value for value in vector))


def build_embedder():
    from graphiti_core.embedder.gemini import GeminiEmbedder, GeminiEmbedderConfig

    vertexai = os.environ.get('EMBEDDER_VERTEXAI', 'true').lower() in ('1', 'true', 'yes')
    return GeminiEmbedder(
        config=GeminiEmbedderConfig(
            api_key=os.environ['GOOGLE_API_KEY'],
            embedding_model=os.environ.get('EMBEDDER_MODEL', 'gemini-embedding-001'),
            embedding_dim=int(os.environ.get('EMBEDDER_DIMENSIONS', '768')),
            vertexai=vertexai,
            project_id=os.environ.get('GOOGLE_CLOUD_PROJECT'),
            location=os.environ.get('GOOGLE_CLOUD_LOCATION', 'global'),
        )
    )


def read_rows(driver, limit: int | None) -> tuple[list[dict], list[dict]]:
    with driver.session() as session:
        entities = [dict(r) for r in session.run(READ_ENTITIES)]
        edges = [dict(r) for r in session.run(READ_EDGES)]
    if limit:
        entities = entities[:limit]
        edges = edges[:limit]
    return entities, edges


async def embed_rows(embedder, rows: list[dict], batch_size: int) -> list[list[float]]:
    vectors: list[list[float]] = []
    for start in range(0, len(rows), batch_size):
        chunk = rows[start : start + batch_size]
        vectors.extend(await embedder.create_batch([row['text'] for row in chunk]))
        print(f'    embedded {min(start + batch_size, len(rows))}/{len(rows)}', flush=True)
    return vectors


def describe(label: str, rows: list[dict], old: list[list[float]], new: list[list[float]]) -> None:
    old_groups: dict[str, set[str]] = defaultdict(set)
    new_groups: dict[str, set[str]] = defaultdict(set)
    for row, old_vector, new_vector in zip(rows, old, new, strict=True):
        old_groups[vector_hash(old_vector)].add(row['text'])
        new_groups[vector_hash(new_vector)].add(row['text'])
    old_collisions = sum(1 for texts in old_groups.values() if len(texts) > 1)
    new_collisions = {h: t for h, t in new_groups.items() if len(t) > 1}
    changed = sum(
        1 for old_vector, new_vector in zip(old, new, strict=True) if old_vector != new_vector
    )
    norms = [norm(v) for v in new]
    print(f'\n  {label}: rows={len(rows)}')
    print(f'    distinct vectors before={len(old_groups)} (colliding groups={old_collisions})')
    print(f'    distinct vectors after ={len(new_groups)} (colliding groups={len(new_collisions)})')
    print(f'    rows whose vector changes: {changed}')
    print(f'    new |v| min={min(norms):.6f} max={max(norms):.6f}')
    for h, texts in sorted(new_collisions.items(), key=lambda kv: -len(kv[1]))[:5]:
        print(f'    still-colliding {h}: {sorted(texts)[:4]}')


def semantic_sample(rows: list[dict], new: list[list[float]]) -> None:
    by_text = {row['text']: vector for row, vector in zip(rows, new, strict=True)}
    pairs = [
        ('トヨタ', '自動車'),
        ('トヨタ', '料理'),
        ('みなも', 'はるお'),
        ('Discord bot の設定', 'ボットの構成'),
    ]
    printed = False
    for a, b in pairs:
        if a in by_text and b in by_text:
            print(f'    cos({a!r}, {b!r}) = {cosine(by_text[a], by_text[b]):+.4f}')
            printed = True
    if not printed:
        print('    (sample pairs not present in this ledger slice)')


async def embed_all(
    embedder, entities: list[dict], edges: list[dict], batch_size: int
) -> tuple[list[list[float]], list[list[float]]]:
    """Embed both kinds inside a single event loop.

    Two separate asyncio.run() calls would leave the embedder's client bound to the first
    (now closed) loop, which fails with 'Event loop is closed' and silently drops the
    batched path.
    """
    print('  embedding entities...', flush=True)
    new_entities = await embed_rows(embedder, entities, batch_size)
    print('  embedding edge facts...', flush=True)
    new_edges = await embed_rows(embedder, edges, batch_size)
    return new_entities, new_edges


def backup(path: Path, entities, edges, vectors) -> None:
    payload = {
        'created_at': datetime.now(timezone.utc).isoformat(),
        'model': os.environ.get('EMBEDDER_MODEL', 'gemini-embedding-001'),
        'entities': {
            row['uuid']: [round(v, 7) for v in vec]
            for row, vec in zip(entities, vectors[0], strict=True)
        },
        'edges': {
            row['uuid']: [round(v, 7) for v in vec]
            for row, vec in zip(edges, vectors[1], strict=True)
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, 'wt', encoding='utf-8') as handle:
        json.dump(payload, handle)
    print(f'  backup written: {path} ({path.stat().st_size / 1e6:.1f} MB)')


def apply(driver, rows: list[dict], vectors: list[list[float]], query: str, batch: int) -> int:
    written = 0
    with driver.session() as session:
        for start in range(0, len(rows), batch):
            chunk = [
                {'uuid': row['uuid'], 'embedding': vector}
                for row, vector in zip(
                    rows[start : start + batch], vectors[start : start + batch], strict=True
                )
            ]
            session.run(query, rows=chunk)
            written += len(chunk)
    return written


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--apply', action='store_true', help='write the new vectors (default: dry run)'
    )
    parser.add_argument('--backup-dir', default='~/backups/graphiti-embeddings')
    parser.add_argument('--batch-size', type=int, default=100)
    parser.add_argument('--limit', type=int, default=None, help='only the first N rows per kind')
    args = parser.parse_args()

    driver = GraphDatabase.driver(
        os.environ.get('NEO4J_URI', 'bolt://192.168.0.158:7687'),
        auth=(
            os.environ.get('NEO4J_USER', 'neo4j'),
            os.environ.get('NEO4J_PASSWORD', 'testpass'),
        ),
    )
    embedder = build_embedder()
    model = os.environ.get('EMBEDDER_MODEL', 'gemini-embedding-001')
    print(f'model={model} dim={os.environ.get("EMBEDDER_DIMENSIONS", "768")} apply={args.apply}')

    entities, edges = read_rows(driver, args.limit)
    print(f'rows: entities={len(entities)} edges={len(edges)}')

    with driver.session() as session:
        old_entities = [list(r['emb']) for r in session.run(OLD_ENTITIES)]
        old_edges = [list(r['emb']) for r in session.run(OLD_EDGES)]

    print('  embedding entities...')
    started = time.time()
    new_entities, new_edges = asyncio.run(embed_all(embedder, entities, edges, args.batch_size))
    print(f'  done in {time.time() - started:.1f}s')

    describe('entities', entities, old_entities, new_entities)
    semantic_sample(entities, new_entities)
    describe('edges', edges, old_edges, new_edges)

    if not args.apply:
        print('\ndry run: nothing written. Re-run with --apply to write.')
        driver.close()
        return 0

    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    backup(
        Path(os.path.expanduser(args.backup_dir)) / f'embeddings-{stamp}.json.gz',
        entities,
        edges,
        (old_entities, old_edges),
    )
    written = apply(driver, entities, new_entities, WRITE_ENTITIES, args.batch_size)
    print(f'  entities written: {written}')
    written = apply(driver, edges, new_edges, WRITE_EDGES, args.batch_size)
    print(f'  edges written: {written}')
    driver.close()
    print('done — verify with embedding_health_check.py')
    return 0


if __name__ == '__main__':
    sys.exit(main())
