"""
MCP server exposing CognitiveMemory (Postgres-first brain) tools.

This is intentionally thin: the database owns state and logic.

Install with:
  pip install -e .

Run:
  python -m apps.mcp.hexis_mcp_server
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from dataclasses import asdict, is_dataclass
from datetime import date, datetime
from enum import Enum
from importlib.metadata import PackageNotFoundError, version
from typing import Any
from uuid import UUID

from dotenv import load_dotenv

from core.cognitive_memory_api import (
    CognitiveMemory,
    GoalPriority,
    MemoryInput,
    MemoryType,
    RelationshipInput,
    RelationshipType,
)


def _print_err(msg: str) -> None:
    sys.stderr.write(msg + "\n")


def _env_dsn() -> str:
    db_host = os.getenv("POSTGRES_HOST", "localhost")
    db_port = os.getenv("POSTGRES_PORT", "43815")
    db_name = os.getenv("POSTGRES_DB", "hexis_memory")
    db_user = os.getenv("POSTGRES_USER", "hexis_user")
    db_password = os.getenv("POSTGRES_PASSWORD", "hexis_password")
    return f"postgresql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}"


def _jsonable(obj: Any) -> Any:
    if obj is None:
        return None
    if isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, (UUID, datetime, date)):
        return str(obj)
    if isinstance(obj, Enum):
        return obj.value
    if is_dataclass(obj):
        return {k: _jsonable(v) for k, v in asdict(obj).items()}
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [_jsonable(v) for v in obj]
    return str(obj)


def _tool(name: str, description: str, schema: dict[str, Any]):
    # Tool class is provided by MCP; imported lazily in main().
    from mcp.types import Tool

    return Tool(name=name, description=description, inputSchema=schema)


def _require(args: dict[str, Any], key: str, tool: str) -> Any:
    if key not in args:
        raise ValueError(f"Missing required argument '{key}' for tool '{tool}'")
    return args[key]


async def _dispatch_tool(client: CognitiveMemory, name: str, args: dict[str, Any]) -> Any:
    if name == "hydrate":
        query = _require(args, "query", name)
        return await client.hydrate(
            query,
            memory_limit=int(args.get("memory_limit", 10)),
            include_partial=bool(args.get("include_partial", True)),
            include_identity=bool(args.get("include_identity", True)),
            include_worldview=bool(args.get("include_worldview", True)),
            include_emotional_state=bool(args.get("include_emotional_state", True)),
            include_goals=bool(args.get("include_goals", False)),
            include_drives=bool(args.get("include_drives", True)),
        )

    if name == "hydrate_batch":
        queries = _require(args, "queries", name)
        if not isinstance(queries, list):
            raise ValueError("queries must be an array of strings")
        return await client.hydrate_batch(
            queries,
            memory_limit=int(args.get("memory_limit", 10)),
            include_partial=bool(args.get("include_partial", True)),
            include_identity=bool(args.get("include_identity", True)),
            include_worldview=bool(args.get("include_worldview", True)),
            include_emotional_state=bool(args.get("include_emotional_state", True)),
            include_goals=bool(args.get("include_goals", False)),
            include_drives=bool(args.get("include_drives", True)),
        )

    if name == "recall":
        query = _require(args, "query", name)
        limit = int(args.get("limit", 10))
        include_partial = bool(args.get("include_partial", True))

        memory_types = args.get("memory_types")
        parsed_types: list[MemoryType] | None = None
        if memory_types is not None:
            if not isinstance(memory_types, list):
                raise ValueError("memory_types must be an array of strings")
            parsed_types = [MemoryType(t) for t in memory_types]

        return await client.recall(
            query,
            limit=limit,
            memory_types=parsed_types,
            min_importance=float(args.get("min_importance", 0.0)),
            include_partial=include_partial,
        )

    if name == "recall_by_id":
        memory_id = UUID(_require(args, "memory_id", name))
        return await client.recall_by_id(memory_id)

    if name == "recall_recent":
        limit = int(args.get("limit", 10))
        memory_type = args.get("memory_type")
        mt: MemoryType | None = MemoryType(memory_type) if memory_type else None
        return await client.recall_recent(limit=limit, memory_type=mt)

    if name == "remember":
        content = _require(args, "content", name)
        type_raw = args.get("type", MemoryType.EPISODIC.value)
        return await client.remember(
            content,
            type=MemoryType(type_raw),
            importance=float(args.get("importance", 0.5)),
            emotional_valence=float(args.get("emotional_valence", 0.0)),
            context=args.get("context"),
            concepts=args.get("concepts"),
        )

    if name == "remember_batch":
        items = _require(args, "memories", name)
        if not isinstance(items, list):
            raise ValueError("memories must be an array")
        memories: list[MemoryInput] = []
        for item in items:
            if not isinstance(item, dict):
                raise ValueError("memories items must be objects")
            memories.append(
                MemoryInput(
                    content=str(item.get("content", "")),
                    type=MemoryType(item.get("type", MemoryType.EPISODIC.value)),
                    importance=float(item.get("importance", 0.5)),
                    emotional_valence=float(item.get("emotional_valence", 0.0)),
                    context=item.get("context"),
                    concepts=item.get("concepts"),
                )
            )
        return await client.remember_batch(memories)

    if name == "remember_batch_raw":
        contents = _require(args, "contents", name)
        embeddings = _require(args, "embeddings", name)
        if not isinstance(contents, list) or not isinstance(embeddings, list):
            raise ValueError("contents and embeddings must be arrays")
        type_raw = args.get("type", MemoryType.EPISODIC.value)
        return await client.remember_batch_raw(
            contents,
            embeddings,
            type=MemoryType(type_raw),
            importance=float(args.get("importance", 0.5)),
        )

    if name == "connect":
        from_id = UUID(_require(args, "from_id", name))
        to_id = UUID(_require(args, "to_id", name))
        rel = RelationshipType(_require(args, "relationship", name))
        await client.connect_memories(
            from_id,
            to_id,
            rel,
            confidence=float(args.get("confidence", 0.8)),
            context=args.get("context"),
        )
        return {"ok": True}

    if name == "connect_batch":
        items = _require(args, "relationships", name)
        if not isinstance(items, list):
            raise ValueError("relationships must be an array")
        rels: list[RelationshipInput] = []
        for item in items:
            if not isinstance(item, dict):
                raise ValueError("relationships items must be objects")
            rels.append(
                RelationshipInput(
                    from_id=UUID(str(item["from_id"])),
                    to_id=UUID(str(item["to_id"])),
                    relationship_type=RelationshipType(str(item["relationship_type"])),
                    confidence=float(item.get("confidence", 0.8)),
                    context=item.get("context"),
                )
            )
        await client.connect_batch(rels)
        return {"ok": True}

    if name == "find_causes":
        memory_id = UUID(_require(args, "memory_id", name))
        return await client.find_causes(memory_id, depth=int(args.get("depth", 3)))

    if name == "find_contradictions":
        memory_id = args.get("memory_id")
        mid = UUID(memory_id) if memory_id else None
        return await client.find_contradictions(mid)

    if name == "find_supporting_evidence":
        worldview_id = UUID(_require(args, "worldview_id", name))
        return await client.find_supporting_evidence(worldview_id)

    if name == "link_concept":
        memory_id = UUID(_require(args, "memory_id", name))
        concept = _require(args, "concept", name)
        strength = float(args.get("strength", 1.0))
        return await client.link_concept(memory_id, concept, strength=strength)

    if name == "find_by_concept":
        concept = _require(args, "concept", name)
        return await client.find_by_concept(concept, limit=int(args.get("limit", 10)))

    if name == "hold":
        content = _require(args, "content", name)
        return await client.hold(content, ttl_seconds=int(args.get("ttl_seconds", 3600)))

    if name == "search_working":
        query = _require(args, "query", name)
        return await client.search_working(query, limit=int(args.get("limit", 5)))

    if name == "get_health":
        return await client.get_health()

    if name == "get_drives":
        return await client.get_drives()

    if name == "get_identity":
        return await client.get_identity()

    if name == "get_worldview":
        return await client.get_worldview()

    if name == "get_goals":
        pri = args.get("priority")
        gp: GoalPriority | None = GoalPriority(pri) if pri else None
        return await client.get_goals(gp)

    if name == "batch":
        ops = _require(args, "operations", name) or []
        continue_on_error = bool(args.get("continue_on_error", False))
        results: list[dict[str, Any]] = []
        for op in ops:
            op_name = op.get("name") if isinstance(op, dict) else None
            op_args = op.get("arguments") if isinstance(op, dict) else {}
            try:
                if not isinstance(op_name, str):
                    raise ValueError("operation.name must be a string")
                r = await _dispatch_tool(client, op_name, op_args or {})
                results.append({"name": op_name, "result": _jsonable(r)})
            except Exception as exc:
                results.append({"name": op_name or "<unknown>", "error": str(exc)})
                if not continue_on_error:
                    raise
        return results

    raise ValueError(f"Unknown tool '{name}'")


def _tools() -> list[Any]:
    return [
        _tool(
            "hydrate",
            "Hydrate a query with relevant memories + (optional) identity/worldview/drives/emotion.",
            {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "memory_limit": {"type": "integer", "minimum": 1, "maximum": 50, "default": 10},
                    "include_partial": {"type": "boolean", "default": True},
                    "include_identity": {"type": "boolean", "default": True},
                    "include_worldview": {"type": "boolean", "default": True},
                    "include_emotional_state": {"type": "boolean", "default": True},
                    "include_goals": {"type": "boolean", "default": False},
                    "include_drives": {"type": "boolean", "default": True},
                },
                "required": ["query"],
                "additionalProperties": False,
            },
        ),
        _tool(
            "hydrate_batch",
            "Hydrate multiple queries (sequential) for throughput.",
            {
                "type": "object",
                "properties": {
                    "queries": {"type": "array", "items": {"type": "string"}, "minItems": 1},
                    "memory_limit": {"type": "integer", "minimum": 1, "maximum": 50, "default": 10},
                    "include_partial": {"type": "boolean", "default": True},
                    "include_identity": {"type": "boolean", "default": True},
                    "include_worldview": {"type": "boolean", "default": True},
                    "include_emotional_state": {"type": "boolean", "default": True},
                    "include_goals": {"type": "boolean", "default": False},
                    "include_drives": {"type": "boolean", "default": True},
                },
                "required": ["queries"],
                "additionalProperties": False,
            },
        ),
        _tool(
            "recall",
            "Recall relevant memories by semantic similarity (fast_recall).",
            {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "limit": {"type": "integer", "minimum": 1, "maximum": 50, "default": 10},
                    "memory_types": {
                        "type": ["array", "null"],
                        "items": {"type": "string", "enum": [t.value for t in MemoryType]},
                    },
                    "min_importance": {"type": "number", "minimum": 0.0, "maximum": 1.0, "default": 0.0},
                    "include_partial": {"type": "boolean", "default": True},
                },
                "required": ["query"],
                "additionalProperties": False,
            },
        ),
        _tool(
            "recall_by_id",
            "Fetch a memory by id.",
            {
                "type": "object",
                "properties": {"memory_id": {"type": "string"}},
                "required": ["memory_id"],
                "additionalProperties": False,
            },
        ),
        _tool(
            "recall_recent",
            "Fetch recent memories.",
            {
                "type": "object",
                "properties": {
                    "limit": {"type": "integer", "minimum": 1, "maximum": 200, "default": 10},
                    "memory_type": {"type": ["string", "null"], "enum": [t.value for t in MemoryType] + [None]},
                },
                "additionalProperties": False,
            },
        ),
        _tool(
            "remember",
            "Store a memory (embedding generated in DB).",
            {
                "type": "object",
                "properties": {
                    "content": {"type": "string"},
                    "type": {"type": "string", "enum": [t.value for t in MemoryType], "default": MemoryType.EPISODIC.value},
                    "importance": {"type": "number", "minimum": 0.0, "maximum": 1.0, "default": 0.5},
                    "emotional_valence": {"type": "number", "minimum": -1.0, "maximum": 1.0, "default": 0.0},
                    "context": {"type": ["object", "null"]},
                    "concepts": {"type": ["array", "null"], "items": {"type": "string"}},
                },
                "required": ["content"],
                "additionalProperties": False,
            },
        ),
        _tool(
            "remember_batch",
            "Store many memories (single DB connection).",
            {
                "type": "object",
                "properties": {
                    "memories": {
                        "type": "array",
                        "minItems": 1,
                        "items": {
                            "type": "object",
                            "properties": {
                                "content": {"type": "string"},
                                "type": {"type": "string", "enum": [t.value for t in MemoryType], "default": MemoryType.EPISODIC.value},
                                "importance": {"type": "number", "minimum": 0.0, "maximum": 1.0, "default": 0.5},
                                "emotional_valence": {"type": "number", "minimum": -1.0, "maximum": 1.0, "default": 0.0},
                                "context": {"type": ["object", "null"]},
                                "concepts": {"type": ["array", "null"], "items": {"type": "string"}},
                            },
                            "required": ["content"],
                            "additionalProperties": False,
                        },
                    }
                },
                "required": ["memories"],
                "additionalProperties": False,
            },
        ),
        _tool(
            "remember_batch_raw",
            "Store many memories using precomputed embeddings (bypasses get_embedding).",
            {
                "type": "object",
                "properties": {
                    "contents": {"type": "array", "items": {"type": "string"}, "minItems": 1},
                    "embeddings": {"type": "array", "items": {"type": "array", "items": {"type": "number"}}},
                    "type": {"type": "string", "enum": [t.value for t in MemoryType], "default": MemoryType.EPISODIC.value},
                    "importance": {"type": "number", "minimum": 0.0, "maximum": 1.0, "default": 0.5},
                },
                "required": ["contents", "embeddings"],
                "additionalProperties": False,
            },
        ),
        _tool(
            "connect",
            "Create a relationship between two memories (graph).",
            {
                "type": "object",
                "properties": {
                    "from_id": {"type": "string"},
                    "to_id": {"type": "string"},
                    "relationship": {"type": "string", "enum": [t.value for t in RelationshipType]},
                    "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0, "default": 0.8},
                    "context": {"type": ["string", "null"]},
                },
                "required": ["from_id", "to_id", "relationship"],
                "additionalProperties": False,
            },
        ),
        _tool(
            "connect_batch",
            "Create multiple memory relationships (single DB connection).",
            {
                "type": "object",
                "properties": {
                    "relationships": {
                        "type": "array",
                        "minItems": 1,
                        "items": {
                            "type": "object",
                            "properties": {
                                "from_id": {"type": "string"},
                                "to_id": {"type": "string"},
                                "relationship_type": {"type": "string", "enum": [t.value for t in RelationshipType]},
                                "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0, "default": 0.8},
                                "context": {"type": ["string", "null"]},
                            },
                            "required": ["from_id", "to_id", "relationship_type"],
                            "additionalProperties": False,
                        },
                    }
                },
                "required": ["relationships"],
                "additionalProperties": False,
            },
        ),
        _tool(
            "find_causes",
            "Find causal chain leading to a memory.",
            {
                "type": "object",
                "properties": {
                    "memory_id": {"type": "string"},
                    "depth": {"type": "integer", "minimum": 1, "maximum": 10, "default": 3},
                },
                "required": ["memory_id"],
                "additionalProperties": False,
            },
        ),
        _tool(
            "find_contradictions",
            "Find contradictions in the graph (optionally scoped to a memory_id).",
            {
                "type": "object",
                "properties": {"memory_id": {"type": ["string", "null"]}},
                "additionalProperties": False,
            },
        ),
        _tool(
            "find_supporting_evidence",
            "Find supporting evidence for a worldview primitive.",
            {
                "type": "object",
                "properties": {"worldview_id": {"type": "string"}},
                "required": ["worldview_id"],
                "additionalProperties": False,
            },
        ),
        _tool(
            "link_concept",
            "Link a memory to a concept.",
            {
                "type": "object",
                "properties": {
                    "memory_id": {"type": "string"},
                    "concept": {"type": "string"},
                    "strength": {"type": "number", "minimum": 0.0, "maximum": 1.0, "default": 1.0},
                },
                "required": ["memory_id", "concept"],
                "additionalProperties": False,
            },
        ),
        _tool(
            "find_by_concept",
            "Retrieve memories linked to a concept.",
            {
                "type": "object",
                "properties": {"concept": {"type": "string"}, "limit": {"type": "integer", "minimum": 1, "maximum": 200, "default": 10}},
                "required": ["concept"],
                "additionalProperties": False,
            },
        ),
        _tool(
            "hold",
            "Add content to working memory (auto-expires).",
            {
                "type": "object",
                "properties": {
                    "content": {"type": "string"},
                    "ttl_seconds": {"type": "integer", "minimum": 1, "maximum": 604800, "default": 3600},
                },
                "required": ["content"],
                "additionalProperties": False,
            },
        ),
        _tool(
            "search_working",
            "Search working memory.",
            {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "limit": {"type": "integer", "minimum": 1, "maximum": 50, "default": 5},
                },
                "required": ["query"],
                "additionalProperties": False,
            },
        ),
        _tool("get_health", "Get cognitive health metrics.", {"type": "object", "properties": {}, "additionalProperties": False}),
        _tool("get_drives", "Get current drive levels.", {"type": "object", "properties": {}, "additionalProperties": False}),
        _tool("get_identity", "Get identity aspects.", {"type": "object", "properties": {}, "additionalProperties": False}),
        _tool("get_worldview", "Get worldview primitives.", {"type": "object", "properties": {}, "additionalProperties": False}),
        _tool(
            "get_goals",
            "Get goals (optionally filtered by priority).",
            {
                "type": "object",
                "properties": {"priority": {"type": ["string", "null"], "enum": [g.value for g in GoalPriority] + [None]}},
                "additionalProperties": False,
            },
        ),
        _tool(
            "batch",
            "Run multiple tool calls sequentially (use batch_* tools where available).",
            {
                "type": "object",
                "properties": {
                    "operations": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "name": {"type": "string"},
                                "arguments": {"type": "object", "default": {}},
                            },
                            "required": ["name"],
                            "additionalProperties": False,
                        },
                        "minItems": 1,
                    },
                    "continue_on_error": {"type": "boolean", "default": False},
                },
                "required": ["operations"],
                "additionalProperties": False,
            },
        ),
    ]


async def _run_server(dsn: str) -> None:
    try:
        from mcp.server import Server
        from mcp.server.models import InitializationOptions
        from mcp.server.stdio import stdio_server
        from mcp.types import ServerCapabilities, TextContent, ToolsCapability
    except Exception as e:
        raise RuntimeError(
            "MCP dependencies not installed. Install with: pip install -e ."
        ) from e

    server = Server("hexis-mcp")

    async with CognitiveMemory.connect(dsn) as client:

        @server.list_tools()
        async def list_tools():
            return _tools()

        @server.call_tool()
        async def call_tool(name: str, arguments: dict[str, Any]):
            try:
                result = await _dispatch_tool(client, name, arguments or {})
                text = json.dumps(_jsonable(result), indent=2, sort_keys=True)
            except Exception as exc:
                text = f"Error: {exc}"
            return [TextContent(type="text", text=text)]

        try:
            server_version = version("hexis")
        except PackageNotFoundError:  # local dev
            server_version = "dev"

        init_opts = InitializationOptions(
            server_name="hexis-mcp",
            server_version=server_version,
            capabilities=ServerCapabilities(tools=ToolsCapability()),
        )

        async with stdio_server() as (read_stream, write_stream):
            await server.run(read_stream, write_stream, init_opts)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="hexis-mcp", description="MCP server exposing CognitiveMemory tools over stdio.")
    p.add_argument(
        "--dsn",
        default=os.getenv("HEXIS_DB_DSN") or None,
        help="Postgres DSN; defaults to POSTGRES_* env vars",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    load_dotenv()
    args = build_parser().parse_args(argv)
    dsn = args.dsn or _env_dsn()
    try:
        asyncio.run(_run_server(dsn))
        return 0
    except KeyboardInterrupt:
        return 130
    except Exception as e:
        _print_err(str(e))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
