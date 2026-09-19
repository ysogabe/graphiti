#!/usr/bin/env python3
"""Read-only health check for the Graphiti ledger's embeddings.

Two questions, both answerable without writing anything:

1. Does the ledger hold distinct vectors for distinct names/facts? (Duplicate-vector groups
   make Graphiti's cosine-based duplicate detection treat unrelated nodes as each other's
   strongest merge candidate.)
2. Does the *live* embedder actually encode the scripts we store? A route that drops CJK
   returns one constant vector for every CJK input, which question 1 alone cannot see once
   every node already shares that constant.

Usage (production credentials are the ones the MCP server uses):

    NEO4J_URI=bolt://192.168.0.158:7687 NEO4J_PASSWORD=testpass \
    GOOGLE_API_KEY=... EMBEDDER_MODEL=gemini-embedding-001 EMBEDDER_DIMENSIONS=768 \
    EMBEDDER_VERTEXAI=true GOOGLE_CLOUD_PROJECT=... GOOGLE_CLOUD_LOCATION=global \
    /opt/graphiti/.venv/bin/python embedding_health_check.py

Exit code 0 = clean, 1 = at least one finding.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import os
import sys
from collections import defaultdict

from neo4j import GraphDatabase

PROBE_TEXTS = ['みなも', 'トヨタ', 'エコーショー']  # distinct, same script -> must not collide
CONTROL_TEXTS = ['minamo', 'toyota']  # ASCII control: these always worked


def vector_hash(vector: list[float]) -> str:
    return hashlib.sha256(','.join(f'{value:.6f}' for value in vector).encode()).hexdigest()[:12]


def build_embedder():
    from graphiti_core.embedder.gemini import GeminiEmbedder, GeminiEmbedderConfig

    vertexai = os.environ.get('EMBEDDER_VERTEXAI', 'true').lower() in ('1', 'true', 'yes')
    config = GeminiEmbedderConfig(
        api_key=os.environ['GOOGLE_API_KEY'],
        embedding_model=os.environ.get('EMBEDDER_MODEL', 'gemini-embedding-001'),
        embedding_dim=int(os.environ.get('EMBEDDER_DIMENSIONS', '768')),
        vertexai=vertexai,
        project_id=os.environ.get('GOOGLE_CLOUD_PROJECT'),
        location=os.environ.get('GOOGLE_CLOUD_LOCATION', 'global'),
    )
    return GeminiEmbedder(config=config)


def ledger_findings(driver) -> tuple[list[str], dict[str, int]]:
    """Return (findings, stats) for entity names and edge facts."""
    findings: list[str] = []
    stats: dict[str, int] = {}
    with driver.session() as session:
        for label, query in (
            (
                'Entity.name_embedding',
                'MATCH (e:Entity) WHERE e.name_embedding IS NOT NULL '
                'RETURN e.name AS text, e.name_embedding AS emb',
            ),
            (
                'RELATES_TO.fact_embedding',
                'MATCH ()-[r:RELATES_TO]->() WHERE r.fact_embedding IS NOT NULL '
                'RETURN r.fact AS text, r.fact_embedding AS emb',
            ),
        ):
            groups: dict[str, set[str]] = defaultdict(set)
            total = 0
            for record in session.run(query):
                total += 1
                groups[vector_hash(list(record['emb']))].add(record['text'] or '')
            stats[label] = total
            distinct_texts = sum(len(texts) for texts in groups.values())
            stats[f'{label}.distinct_vectors'] = len(groups)
            collisions = {h: t for h, t in groups.items() if len(t) > 1}
            if collisions:
                stats[f'{label}.colliding_groups'] = len(collisions)
                findings.append(
                    f'{label}: {len(collisions)} vector group(s) shared by distinct texts '
                    f'({sum(len(t) for t in collisions.values())} rows)'
                )
                for h, texts in sorted(collisions.items(), key=lambda kv: -len(kv[1]))[:5]:
                    sample = sorted(texts)[:4]
                    findings.append(f'    {h} n={len(texts)} e.g. {sample}')
            del distinct_texts
    return findings, stats


async def embedder_findings() -> tuple[list[str], dict[str, str]]:
    embedder = build_embedder()
    findings: list[str] = []
    hashes: dict[str, str] = {}

    for text in PROBE_TEXTS + CONTROL_TEXTS:
        hashes[text] = vector_hash(await embedder.create(input_data=[text]))

    for group in (PROBE_TEXTS, CONTROL_TEXTS):
        seen: dict[str, list[str]] = defaultdict(list)
        for text in group:
            seen[hashes[text]].append(text)
        for h, texts in seen.items():
            if len(texts) > 1:
                findings.append(
                    f'embedder collapsed {texts} onto one vector ({h}) — this route does not '
                    f'encode these characters'
                )
    return findings, hashes


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--skip-probe', action='store_true', help='ledger only, no embedder call')
    args = parser.parse_args()

    driver = GraphDatabase.driver(
        os.environ.get('NEO4J_URI', 'bolt://192.168.0.158:7687'),
        auth=(
            os.environ.get('NEO4J_USER', 'neo4j'),
            os.environ.get('NEO4J_PASSWORD', 'testpass'),
        ),
    )
    findings, stats = ledger_findings(driver)
    driver.close()

    print('=== ledger ===')
    for key, value in stats.items():
        print(f'  {key}: {value}')

    hashes: dict[str, str] = {}
    if not args.skip_probe:
        probe_findings, hashes = asyncio.run(embedder_findings())
        findings.extend(probe_findings)
        print('\n=== live embedder probe ===')
        for text, h in hashes.items():
            print(f'  {text!r:14} -> {h}')

    print('\n=== findings ===')
    if not findings:
        print('  none')
        return 0
    for finding in findings:
        print(f'  {finding}')
    return 1


if __name__ == '__main__':
    sys.exit(main())
