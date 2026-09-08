#!/usr/bin/env python3
"""Graphiti smoke test: Neo4j + b.ai LLM + Vertex AI embedding → add_episode → search.

Run from the graphiti repo with the venv that has graphiti-core (editable) + deps:
    set -a; source <env with BAI_API_KEY + GOOGLE_API_KEY>; set +a
    .venv/bin/python smoke_test.py [model] [effort]
"""
import asyncio
import os
import sys
from datetime import datetime, timezone

from graphiti_core import Graphiti
from graphiti_core.driver.neo4j_driver import Neo4jDriver
from graphiti_core.embedder.gemini import GeminiEmbedder, GeminiEmbedderConfig
from graphiti_core.llm_client.config import LLMConfig
from graphiti_core.llm_client.openai_generic_client import OpenAIGenericClient
from graphiti_core.cross_encoder.client import CrossEncoderClient

NEO4J_URI = os.environ.get("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.environ.get("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.environ.get("NEO4J_PASSWORD", "testpass")

BAI_KEY = os.environ.get("BAI_API_KEY")
GOOGLE_KEY = os.environ.get("GOOGLE_API_KEY")
BAI_BASE = os.environ.get("BAI_BASE_URL", "https://api.b.ai/v1")
MODEL = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("BAI_MODEL", "gpt-5.6-luna")
EFFORT = sys.argv[2] if len(sys.argv) > 2 else os.environ.get("BAI_EFFORT", "high")

GCP_PROJECT = os.environ.get("GOOGLE_CLOUD_PROJECT", "257425471273")
GCP_LOCATION = os.environ.get("GOOGLE_CLOUD_LOCATION", "global")
EMBED_MODEL = os.environ.get("EMBED_MODEL", "text-embedding-005")
EMBED_DIM = int(os.environ.get("EMBED_DIM", "768"))


class PassthroughCrossEncoder(CrossEncoderClient):
    """No-op reranker so the smoke test does not need OpenAI/BGE/Vertex reranking."""

    async def rank(self, query: str, passages: list[str]) -> list[tuple[str, float]]:
        return [(p, 0.0) for p in passages]


async def main() -> None:
    if not BAI_KEY:
        raise SystemExit("BAI_API_KEY not set")
    if not GOOGLE_KEY:
        raise SystemExit("GOOGLE_API_KEY not set")

    llm = OpenAIGenericClient(
        config=LLMConfig(api_key=BAI_KEY, model=MODEL, base_url=BAI_BASE),
        structured_output_mode="json_object",
        reasoning_effort=EFFORT,
    )
    embedder = GeminiEmbedder(
        config=GeminiEmbedderConfig(
            api_key=GOOGLE_KEY,
            embedding_model=EMBED_MODEL,
            embedding_dim=EMBED_DIM,
            vertexai=True,
            project_id=GCP_PROJECT,
            location=GCP_LOCATION,
        )
    )

    print(f"[init] llm={MODEL}(effort={EFFORT}) embedder={EMBED_MODEL}({EMBED_DIM}d, vertex) "
          f"neo4j={NEO4J_URI}")
    graphiti = Graphiti(
        uri=NEO4J_URI,
        user=NEO4J_USER,
        password=NEO4J_PASSWORD,
        llm_client=llm,
        embedder=embedder,
        cross_encoder=PassthroughCrossEncoder(),
    )
    print("[init] Graphiti connected")

    episode_body = (
        "Rescue Vet Ledger (RVL) is a system for managing shelter intake. "
        "The 221-PR7 feature transfers pet records from the legacy ledger to the new ledger "
        "using a pull model, runs nightly at 02:00 JST, and is owned by ysogabe."
    )
    print("[add_episode] ...")
    res = await graphiti.add_episode(
        name="smoke-1",
        episode_body=episode_body,
        source_description="smoke test",
        reference_time=datetime.now(timezone.utc),
    )
    print("[add_episode] OK", "nodes:", len(res.nodes), "edges:", len(res.edges))

    print("[search] ...")
    results = await graphiti.search(query="Which system transfers pet records via a pull model?")
    print("[search] OK results:", len(results))
    for i, r in enumerate(results[:5], 1):
        print(f"  {i}. {r}")

    print("\nSMOKE TEST PASSED")


if __name__ == "__main__":
    asyncio.run(main())
