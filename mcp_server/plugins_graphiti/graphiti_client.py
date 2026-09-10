"""Minimal MCP client for the Graphiti server (streamable-http, mcp 2.x).

Runs in the Hermes venv (mcp 2.x), so it uses the 2.x client API
(``mcp.client.streamable_http.streamable_http_client`` yielding 2 tuple items).
The memory-provider hooks (prefetch / on_memory_write / handle_tool_call) are
synchronous and may be called from inside the host event loop, so each MCP call
runs in a dedicated daemon thread with its own event loop.
"""
from __future__ import annotations

import asyncio
import threading
from typing import Any

try:
    from mcp import ClientSession
    from mcp.client.streamable_http import streamable_http_client
except Exception:  # pragma: no cover - mcp missing
    ClientSession = None
    streamable_http_client = None


class GraphitiMCPClient:
    """Calls the Graphiti MCP server tools over streamable-http (mcp 2.x)."""

    def __init__(self, mcp_url: str) -> None:
        if not streamable_http_client:
            raise RuntimeError("mcp client not available (need 'mcp' 2.x)")
        self._mcp_url = mcp_url

    async def _session_call(self, tool_name: str, args: dict[str, Any]) -> str:
        async with streamable_http_client(self._mcp_url) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                result = await session.call_tool(tool_name, args)
                texts: list[str] = []
                for c in getattr(result, "content", []) or []:
                    if getattr(c, "type", None) == "text" and c.text:
                        texts.append(c.text)
                if texts:
                    return "\n".join(texts)
                if hasattr(result, "model_dump"):
                    import json as _json

                    return _json.dumps(result.model_dump(), ensure_ascii=False)
                return str(result)

    def call(self, tool_name: str, args: dict[str, Any], *, timeout: int = 60) -> str:
        """Run an MCP tool call synchronously (fresh event loop in a daemon thread)."""
        box: dict[str, Any] = {}

        def _runner() -> None:
            try:
                box["value"] = asyncio.run(self._session_call(tool_name, args))
            except Exception as exc:  # surfaced to the caller below
                box["error"] = exc

        t = threading.Thread(target=_runner, daemon=True, name="graphiti-mcp-call")
        t.start()
        t.join(timeout=timeout)
        if "error" in box:
            raise box["error"]
        if "value" not in box:
            raise TimeoutError(f"graphiti MCP call '{tool_name}' timed out")
        return box["value"]

    # Convenience wrappers mirroring the Graphiti MCP tools ------------------------
    def add_memory(self, name: str, episode_body: str, group_id: str, *, timeout: int = 180, **kw: Any) -> str:
        return self.call(
            "add_memory",
            {"name": name, "episode_body": episode_body, "group_id": group_id, **kw},
            timeout=timeout,
        )

    def search_memory_facts(
        self, query: str, group_ids: list[str], max_facts: int = 8, **kw: Any
    ) -> str:
        return self.call(
            "search_memory_facts",
            {"query": query, "group_ids": group_ids, "max_facts": max_facts, **kw},
        )

    def search_nodes(self, query: str, group_ids: list[str], max_nodes: int = 8, **kw: Any) -> str:
        return self.call(
            "search_nodes",
            {"query": query, "group_ids": group_ids, "max_nodes": max_nodes, **kw},
        )
