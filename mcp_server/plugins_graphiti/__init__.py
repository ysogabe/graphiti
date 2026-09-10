"""graphiti-memory — Hermes external memory provider backed by the Graphiti MCP server.

Writes (add_memory) go to ``write_group``; reads (search) span ``read_groups``.
This lets each Hermes profile isolate its memory (its own group) while a profile
may opt-in to read another group read-only (e.g. minamo reads the haruo group for
development ideas). Per-turn sync is OFF by default to avoid Graphiti ingestion
cost; durable writes flow through ``on_memory_write`` (built-in memory mirror) and
the ``graphiti_add_memory`` tool.
"""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

from agent.memory_provider import MemoryProvider, is_trivial_prompt
from .graphiti_client import GraphitiMCPClient

logger = logging.getLogger(__name__)

_CONFIG_FILE = "graphiti_memory.json"
DEFAULT_PREFETCH_MAX = 6
DEFAULT_SESSION_SUMMARY_MAX = 6000


def _load_config(hermes_home: str) -> dict[str, Any]:
    cfg: dict[str, Any] = {}
    p = Path(hermes_home) / _CONFIG_FILE
    if p.exists():
        try:
            cfg.update(json.loads(p.read_text(encoding="utf-8-sig")))
        except Exception as e:  # pragma: no cover
            logger.warning("Could not read %s: %s", p, e)
    cfg["mcp_url"] = os.environ.get("GRAPHITI_MCP_URL", cfg.get("mcp_url", ""))
    cfg["write_group"] = os.environ.get("GRAPHITI_WRITE_GROUP", cfg.get("write_group", ""))
    raw_read = os.environ.get("GRAPHITI_READ_GROUPS")
    if raw_read:
        cfg["read_groups"] = [g.strip() for g in raw_read.split(",") if g.strip()]
    cfg.setdefault("read_groups", [])
    if cfg["write_group"] and cfg["write_group"] not in cfg["read_groups"]:
        cfg["read_groups"] = cfg["read_groups"] + [cfg["write_group"]]
    cfg.setdefault("sync_turn", False)
    cfg.setdefault("prefetch_max", DEFAULT_PREFETCH_MAX)
    cfg.setdefault("session_summary", True)
    cfg.setdefault("pre_compress", True)
    cfg.setdefault("delegation", True)
    cfg.setdefault("session_summary_max_chars", DEFAULT_SESSION_SUMMARY_MAX)
    return cfg


def _format_facts(raw: str, limit: int) -> str:
    """Best-effort: turn a search_memory_facts JSON response into a fact list."""
    try:
        data = json.loads(raw)
    except Exception:
        return raw[:3000]
    facts = data.get("facts", data.get("results", data))
    if isinstance(facts, dict):
        facts = [facts]
    if not isinstance(facts, list):
        return raw[:3000]
    lines: list[str] = []
    for f in facts[:limit]:
        if isinstance(f, dict):
            fact = f.get("fact") or f.get("name") or f.get("content")
            if fact:
                lines.append(f"- {fact}")
        elif isinstance(f, str):
            lines.append(f"- {f}")
    return "\n".join(lines) if lines else raw[:3000]


def _format_session_messages(messages: list[Any], max_chars: int) -> str:
    """Turn a session transcript into a capped ``role: content`` episode body.

    Used by ``on_session_end`` to hand the whole conversation to Graphiti's
    extractor. Keeps the most recent portion when over ``max_chars`` (the last
    activity matters most for a session summary).
    """
    parts: list[str] = []
    for m in messages or []:
        if not isinstance(m, dict):
            continue
        role = m.get("role", "?")
        content = m.get("content")
        if isinstance(content, list):
            content = " ".join(
                c.get("text", "") if isinstance(c, dict) else str(c) for c in content
            )
        content = str(content or "").strip()
        if content:
            parts.append(f"{role}: {content[:800]}")
    joined = "\n\n".join(parts).strip()
    if not joined:
        return ""
    if len(joined) <= max_chars:
        return joined
    return "...(earlier turns truncated)...\n\n" + joined[-max_chars:]


class GraphitiMemoryProvider(MemoryProvider):
    def __init__(self) -> None:
        self._client: GraphitiMCPClient | None = None
        self._cfg: dict[str, Any] = {}
        self._hermes_home = ""
        self._agent_identity = ""
        self._session_id = ""

    @property
    def name(self) -> str:
        return "graphiti-memory"

    def is_available(self) -> bool:
        """No network: config + client import must be present."""
        home = os.environ.get("HERMES_HOME", "")
        cfg = _load_config(home)
        return bool(cfg.get("mcp_url")) and bool(cfg.get("write_group")) and bool(_mcp_ok())

    def unavailable_reason(self) -> str:
        return (
            "Set GRAPHITI_MCP_URL + GRAPHITI_WRITE_GROUP, or write "
            f"{os.environ.get('HERMES_HOME', '~/.hermes')}/{_CONFIG_FILE}."
        )

    def initialize(self, session_id: str, **kwargs: Any) -> None:
        self._hermes_home = kwargs.get("hermes_home", os.environ.get("HERMES_HOME", ""))
        self._agent_identity = kwargs.get("agent_identity", "") or ""
        self._session_id = session_id
        self._cfg = _load_config(self._hermes_home)
        self._client = GraphitiMCPClient(self._cfg["mcp_url"])
        logger.info(
            "graphiti-memory initialized: write=%s read=%s (identity=%s)",
            self._cfg.get("write_group"), self._cfg.get("read_groups"), self._agent_identity,
        )

    def shutdown(self) -> None:
        self._client = None

    # -- Recall ------------------------------------------------------------------
    def prefetch(self, query: str, *, session_id: str = "") -> str:
        if is_trivial_prompt(query):
            return ""
        if not self._client:
            return ""
        groups = self._cfg.get("read_groups", []) or [
            self._cfg.get("write_group", "")
        ]
        limit = int(self._cfg.get("prefetch_max", DEFAULT_PREFETCH_MAX))
        try:
            raw = self._client.search_memory_facts(
                query, groups, max_facts=limit
            )
            return _format_facts(raw, limit)
        except Exception as e:  # pragma: no cover
            logger.warning("graphiti prefetch failed: %s", e)
            return ""

    # -- Persistence -------------------------------------------------------------
    def on_memory_write(self, action: str, target: str, content: str, metadata: Any = None) -> None:
        """Mirror built-in memory writes into the Graphiti write_group (curated)."""
        if not self._client or action == "remove":
            return
        try:
            self._client.add_memory(
                name=f"mem-{target}-{action}",
                episode_body=content,
                group_id=self._cfg.get("write_group", ""),
                source="hermes-memory",
                source_description=f"built-in memory {action} to {target}",
            )
        except Exception as e:  # pragma: no cover
            logger.warning("graphiti on_memory_write failed: %s", e)

    def sync_turn(self, user_content: str, assistant_content: str, *, session_id: str = "", messages=None) -> None:
        """Off by default (Graphiti ingestion is LLM-heavy); enable via sync_turn: true."""
        if not self._cfg.get("sync_turn") or not self._client:
            return
        body = f"{user_content}\n\n{assistant_content}"
        try:
            self._client.add_memory(
                name=f"turn-{session_id[:12] or 'session'}",
                episode_body=body,
                group_id=self._cfg.get("write_group", ""),
                source="hermes-turn",
            )
        except Exception as e:  # pragma: no cover
            logger.warning("graphiti sync_turn failed: %s", e)

    def on_session_end(self, messages: list[Any]) -> None:
        """Write an end-of-session summary into the Graphiti write_group.

        Fires at real session boundaries (CLI exit, /reset, /new, gateway session
        expiry). Formats the full transcript (capped) and hands it to Graphiti's
        extractor so the session's durable facts are captured in one write. Gated
        behind ``session_summary`` (default on); set false to disable.
        """
        if not self._cfg.get("session_summary") or not self._client:
            return
        if not isinstance(messages, list) or not messages:
            return
        max_chars = int(self._cfg.get("session_summary_max_chars", DEFAULT_SESSION_SUMMARY_MAX))
        body = _format_session_messages(messages, max_chars)
        if not body:
            return
        try:
            self._client.add_memory(
                name=f"session-{self._session_id[:12] or 'summary'}",
                episode_body=body,
                group_id=self._cfg.get("write_group", ""),
                source="hermes-session",
                source_description="end-of-session summary (on_session_end)",
                timeout=300,
            )
        except Exception as e:  # pragma: no cover
            logger.warning("graphiti on_session_end failed: %s", e)

    def on_pre_compress(self, messages: list[Any]) -> str:
        """Preserve to-be-compressed messages into the Graphiti write_group.

        Fires before context compression discards old messages. Hands the doomed
        slice to Graphiti's extractor so the session's facts survive compression
        (recallable later, not lost in the summary). Returns a short marker for the
        compression prompt; gated behind ``pre_compress`` (default on).
        """
        if not self._cfg.get("pre_compress") or not self._client:
            return ""
        if not isinstance(messages, list) or not messages:
            return ""
        max_chars = int(self._cfg.get("session_summary_max_chars", DEFAULT_SESSION_SUMMARY_MAX))
        body = _format_session_messages(messages, max_chars)
        if not body:
            return ""
        try:
            self._client.add_memory(
                name=f"compress-{self._session_id[:12] or 'summary'}",
                episode_body=body,
                group_id=self._cfg.get("write_group", ""),
                source="hermes-compress",
                source_description="pre-compression fact capture (on_pre_compress)",
                timeout=300,
            )
        except Exception as e:  # pragma: no cover
            logger.warning("graphiti on_pre_compress failed: %s", e)
        return ""

    def on_delegation(self, task: str, result: str, *,
                      child_session_id: str = "", **kwargs: Any) -> None:
        """Record a delegation (task + subagent result) into the write_group.

        Fires on the PARENT when a subagent completes — useful to put delegated
        work into the shared ledger. Gated behind ``delegation`` (default on).
        """
        if not self._cfg.get("delegation") or not self._client:
            return
        if not task:
            return
        body = f"DELEGATED TASK: {task}\n\nSUBAGENT RESULT: {result}"
        try:
            self._client.add_memory(
                name=f"delegation-{child_session_id[:12] or 'task'}",
                episode_body=body,
                group_id=self._cfg.get("write_group", ""),
                source="hermes-delegation",
                source_description=f"delegation to {child_session_id or 'subagent'}",
                timeout=300,
            )
        except Exception as e:  # pragma: no cover
            logger.warning("graphiti on_delegation failed: %s", e)

    def on_session_switch(self, new_session_id: str, *,
                          parent_session_id: str = "", reset: bool = False,
                          rewound: bool = False, **kwargs: Any) -> None:
        """Track the session_id across /resume, /branch, /reset, /new, compression.

        The provider is stateless (no per-session cache besides ``_session_id``),
        so this only keeps future writes (on_session_end names) bound to the
        current session id. ``reset`` means a genuinely new conversation.
        """
        self._session_id = new_session_id

    # -- Tools -------------------------------------------------------------------
    def get_tool_schemas(self) -> list[dict[str, Any]]:
        return [
            {
                "name": "graphiti_add_memory",
                "description": (
                    "Persist a fact/decision to the Graphiti shared memory ledger "
                    "(write_group). Use for durable facts, decisions, procedures, PRD "
                    "records, merge records."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "content": {"type": "string", "description": "The fact/decision to store."},
                        "group_id": {"type": "string", "description": "Optional group (defaults to write_group)."},
                    },
                    "required": ["content"],
                },
            },
            {
                "name": "graphiti_search_memory",
                "description": (
                    "Search the Graphiti memory graph (read_groups) for relevant facts. "
                    "Use to recall prior decisions/procedures before acting."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Search query."},
                        "max_facts": {"type": "integer", "description": "Max facts (default 8)."},
                    },
                    "required": ["query"],
                },
            },
            {
                "name": "graphiti_search_nodes",
                "description": "Search Graphiti entities (nodes) in read_groups by query.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Search query."},
                        "max_nodes": {"type": "integer", "description": "Max nodes (default 8)."},
                    },
                    "required": ["query"],
                },
            },
        ]

    def handle_tool_call(self, tool_name: str, args: dict[str, Any], **kwargs: Any) -> str:
        if not self._client:
            return json.dumps({"error": "graphiti client not initialized"})
        groups = self._cfg.get("read_groups", []) or [self._cfg.get("write_group", "")]
        if tool_name == "graphiti_add_memory":
            return self._client.add_memory(
                name="tool-add",
                episode_body=args.get("content", ""),
                group_id=args.get("group_id") or self._cfg.get("write_group", ""),
                source="hermes-tool",
            )
        if tool_name == "graphiti_search_memory":
            return self._client.search_memory_facts(
                args.get("query", ""), groups, max_facts=int(args.get("max_facts", 8))
            )
        if tool_name == "graphiti_search_nodes":
            return self._client.search_nodes(
                args.get("query", ""), groups, max_nodes=int(args.get("max_nodes", 8))
            )
        raise NotImplementedError(f"graphiti-memory does not handle {tool_name}")

    # -- Config (for `hermes memory setup`) -------------------------------------
    def get_config_schema(self) -> list[dict[str, Any]]:
        return [
            {
                "key": "mcp_url",
                "description": "Graphiti MCP server URL (e.g. http://127.0.0.1:8100/mcp/)",
                "type": "text",
                "required": True,
            },
            {
                "key": "write_group",
                "description": "group_id to write memory to (your profile's namespace)",
                "type": "text",
                "required": True,
            },
            {
                "key": "read_groups",
                "description": "Comma-separated group_ids to read from (include write_group; "
                        "add others to read-only, e.g. haruo)",
                "type": "text",
                "required": False,
            },
            {
                "key": "sync_turn",
                "description": "Persist every turn (Graphiti ingestion is LLM-heavy; keep false)",
                "type": "boolean",
                "default": False,
            },
        ]

    def save_config(self, values: dict[str, Any], hermes_home: str) -> None:
        cfg = _load_config(hermes_home)
        cfg.update(values)
        # read_groups arrives as a comma string -> normalize
        rg = cfg.get("read_groups")
        if isinstance(rg, str):
            cfg["read_groups"] = [g.strip() for g in rg.split(",") if g.strip()]
        wg = cfg.get("write_group", "")
        if wg and wg not in cfg.get("read_groups", []):
            cfg["read_groups"] = cfg.get("read_groups", []) + [wg]
        p = Path(hermes_home) / _CONFIG_FILE
        p.write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")


def _mcp_ok() -> bool:
    return bool(GraphitiMCPClient)
