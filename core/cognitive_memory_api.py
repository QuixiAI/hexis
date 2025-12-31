"""
Cognitive Memory API

Thin async client for the Postgres-backed cognitive memory system.

Design:
- The database owns state and behavior (functions/views in db/schema.sql).
- This module is a convenience layer for application integration.
"""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Any, AsyncIterator, Iterable, Optional
from uuid import UUID

import asyncpg

from core.prompt_resources import compose_personhood_prompt

class MemoryType(str, Enum):
    EPISODIC = "episodic"
    SEMANTIC = "semantic"
    PROCEDURAL = "procedural"
    STRATEGIC = "strategic"


class GoalPriority(str, Enum):
    ACTIVE = "active"
    QUEUED = "queued"
    BACKBURNER = "backburner"
    COMPLETED = "completed"
    ABANDONED = "abandoned"

class GoalSource(str, Enum):
    CURIOSITY = "curiosity"
    USER_REQUEST = "user_request"
    IDENTITY = "identity"
    DERIVED = "derived"
    EXTERNAL = "external"


class RelationshipType(str, Enum):
    TEMPORAL_NEXT = "TEMPORAL_NEXT"
    CAUSES = "CAUSES"
    DERIVED_FROM = "DERIVED_FROM"
    CONTRADICTS = "CONTRADICTS"
    SUPPORTS = "SUPPORTS"
    INSTANCE_OF = "INSTANCE_OF"
    PARENT_OF = "PARENT_OF"
    ASSOCIATED = "ASSOCIATED"


@dataclass(frozen=True)
class Memory:
    id: UUID
    type: MemoryType
    content: str
    importance: float
    relevance_score: float | None = None
    similarity: float | None = None
    source: str | None = None  # retrieval source: 'vector', 'association', 'temporal'
    trust_level: float | None = None  # epistemic trust [0..1] (DB-computed)
    source_attribution: dict[str, Any] | None = None  # primary provenance (DB-stored JSON)
    created_at: datetime | None = None
    emotional_valence: float | None = None


@dataclass(frozen=True)
class PartialActivation:
    cluster_id: UUID
    cluster_name: str
    keywords: list[str]
    emotional_signature: dict[str, Any] | None
    cluster_similarity: float
    best_memory_similarity: float


@dataclass(frozen=True)
class RecallResult:
    memories: list[Memory]
    partial_activations: list[PartialActivation]
    query: str


@dataclass(frozen=True)
class HydratedContext:
    memories: list[Memory]
    partial_activations: list[PartialActivation]
    identity: list[dict[str, Any]]
    worldview: list[dict[str, Any]]
    emotional_state: dict[str, Any] | None
    goals: dict[str, Any] | None
    urgent_drives: list[dict[str, Any]]


@dataclass(frozen=True)
class MemoryInput:
    content: str
    type: MemoryType = MemoryType.EPISODIC
    importance: float = 0.5
    emotional_valence: float = 0.0
    context: dict[str, Any] | None = None
    concepts: list[str] | None = None
    source_attribution: dict[str, Any] | None = None
    source_references: Any | None = None  # JSONB for semantic memories (dict or list[dict])
    trust_level: float | None = None


@dataclass(frozen=True)
class RelationshipInput:
    from_id: UUID
    to_id: UUID
    relationship_type: RelationshipType
    confidence: float = 0.8
    context: str | None = None


async def _init_connection(conn: asyncpg.Connection) -> None:
    try:
        await conn.execute("LOAD 'age';")
    except Exception:
        pass
    try:
        await conn.execute("SET search_path = ag_catalog, public;")
    except Exception:
        pass

def _to_jsonb_arg(val: Any) -> Any:
    if val is None:
        return None
    if isinstance(val, (dict, list)):
        import json

        return json.dumps(val)
    return val


def _cypher_escape(value: str) -> str:
    return value.replace("'", "''")


class CognitiveMemory:
    """
    Async client for the cognitive memory database.

    Two common flows:
    - RAG hydration: `hydrate()`
    - Agent operations: `recall()`, `remember()`, `connect_memories()`
    """

    def __init__(self, pool: asyncpg.Pool):
        self._pool = pool

    @classmethod
    @asynccontextmanager
    async def connect(
        cls,
        dsn: str,
        **pool_kwargs: Any,
    ) -> AsyncIterator["CognitiveMemory"]:
        """
        Async context manager that owns the underlying pool.

        Usage:
            async with CognitiveMemory.connect(dsn) as mem:
                ctx = await mem.hydrate("...")
        """
        pool = await asyncpg.create_pool(dsn, init=_init_connection, **pool_kwargs)
        client = cls(pool)
        try:
            yield client
        finally:
            await pool.close()

    @classmethod
    async def create(cls, dsn: str, **pool_kwargs: Any) -> "CognitiveMemory":
        """Create a pool and return a client; call `close()` when done."""
        pool = await asyncpg.create_pool(dsn, init=_init_connection, **pool_kwargs)
        return cls(pool)

    async def close(self) -> None:
        await self._pool.close()

    # =========================================================================
    # RAG: HYDRATION
    # =========================================================================

    async def hydrate(
        self,
        query: str,
        *,
        memory_limit: int = 10,
        include_partial: bool = True,
        include_identity: bool = True,
        include_worldview: bool = True,
        include_emotional_state: bool = True,
        include_goals: bool = False,
        include_drives: bool = True,
    ) -> HydratedContext:
        """
        Hydrate a query with relevant context for RAG prompt augmentation.

        This uses:
        - `fast_recall(query, limit)` for relevant memories
        - `find_partial_activations(query)` for tip-of-tongue clusters (optional)
        - `gather_turn_context()` for identity/worldview/emotions/drives/goals (optional subsets)
        """
        async with self._pool.acquire() as conn:
            memories = await self._recall_memories(conn, query, memory_limit)
            partial = await self._find_partial_activations(conn, query) if include_partial else []

            ctx_row = await conn.fetchval("SELECT gather_turn_context()")
            ctx = _coerce_json(ctx_row)

            identity = ctx.get("identity", []) if include_identity else []
            worldview = ctx.get("worldview", []) if include_worldview else []
            emotional_state = ctx.get("emotional_state") if include_emotional_state else None
            goals = ctx.get("goals") if include_goals else None
            urgent_drives = ctx.get("urgent_drives", []) if include_drives else []

            return HydratedContext(
                memories=memories,
                partial_activations=partial,
                identity=list(identity) if isinstance(identity, list) else [],
                worldview=list(worldview) if isinstance(worldview, list) else [],
                emotional_state=dict(emotional_state) if isinstance(emotional_state, dict) else None,
                goals=dict(goals) if isinstance(goals, dict) else None,
                urgent_drives=list(urgent_drives) if isinstance(urgent_drives, list) else [],
            )

    async def hydrate_batch(
        self,
        queries: list[str],
        *,
        max_concurrency: int = 5,
        **kwargs: Any,
    ) -> list[HydratedContext]:
        """
        Hydrate multiple queries concurrently (pool-backed).

        Note: `asyncpg.Connection` cannot run concurrent queries, so batching here
        means concurrent hydrations across pooled connections.
        """
        sem = asyncio.Semaphore(max(1, max_concurrency))

        async def _one(q: str) -> HydratedContext:
            async with sem:
                return await self.hydrate(q, **kwargs)

        return list(await asyncio.gather(*[_one(q) for q in queries]))

    # =========================================================================
    # RECALL
    # =========================================================================

    async def recall(
        self,
        query: str,
        *,
        limit: int = 10,
        memory_types: list[MemoryType] | None = None,
        min_importance: float = 0.0,
        include_partial: bool = True,
    ) -> RecallResult:
        async with self._pool.acquire() as conn:
            memories = await self._recall_memories(
                conn,
                query,
                limit,
                memory_types=memory_types,
                min_importance=min_importance,
            )
            partial = await self._find_partial_activations(conn, query) if include_partial else []
            return RecallResult(memories=memories, partial_activations=partial, query=query)

    async def recall_by_id(self, memory_id: UUID) -> Memory | None:
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                SELECT
                    m.id,
                    m.type,
                    m.content,
                    m.importance,
                    m.trust_level,
                    m.source_attribution,
                    m.created_at,
                    em.emotional_valence
                FROM memories m
                LEFT JOIN episodic_memories em ON m.id = em.memory_id
                WHERE m.id = $1
                """,
                memory_id,
            )
            if not row:
                return None
            return Memory(
                id=row["id"],
                type=MemoryType(row["type"]),
                content=row["content"],
                importance=float(row["importance"]),
                trust_level=float(row["trust_level"]) if row["trust_level"] is not None else None,
                source_attribution=_coerce_json(row["source_attribution"]) if row["source_attribution"] is not None else None,
                created_at=row["created_at"],
                emotional_valence=row["emotional_valence"],
            )

    async def recall_recent(
        self,
        *,
        limit: int = 10,
        memory_type: MemoryType | None = None,
    ) -> list[Memory]:
        async with self._pool.acquire() as conn:
            if memory_type is None:
                rows = await conn.fetch(
                    """
                    SELECT
                        m.id,
                        m.type,
                        m.content,
                        m.importance,
                        m.trust_level,
                        m.source_attribution,
                        m.created_at,
                        em.emotional_valence
                    FROM memories m
                    LEFT JOIN episodic_memories em ON m.id = em.memory_id
                    WHERE m.status = 'active'
                    ORDER BY m.created_at DESC
                    LIMIT $1
                    """,
                    limit,
                )
            else:
                rows = await conn.fetch(
                    """
                    SELECT
                        m.id,
                        m.type,
                        m.content,
                        m.importance,
                        m.trust_level,
                        m.source_attribution,
                        m.created_at,
                        em.emotional_valence
                    FROM memories m
                    LEFT JOIN episodic_memories em ON m.id = em.memory_id
                    WHERE m.status = 'active' AND m.type = $2::memory_type
                    ORDER BY m.created_at DESC
                    LIMIT $1
                    """,
                    limit,
                    memory_type.value,
                )
            return [self._row_to_memory(row) for row in rows]

    # =========================================================================
    # REMEMBER
    # =========================================================================

    async def remember(
        self,
        content: str,
        *,
        type: MemoryType = MemoryType.EPISODIC,
        importance: float = 0.5,
        emotional_valence: float = 0.0,
        context: dict[str, Any] | None = None,
        concepts: list[str] | None = None,
        source_attribution: dict[str, Any] | None = None,
        source_references: Any | None = None,
        trust_level: float | None = None,
    ) -> UUID:
        async with self._pool.acquire() as conn:
            memory_id = await self._create_memory(
                conn,
                content,
                type,
                importance,
                emotional_valence,
                context,
                source_attribution=source_attribution,
                source_references=source_references,
                trust_level=trust_level,
            )

            if concepts:
                for concept in concepts:
                    await conn.fetchval("SELECT link_memory_to_concept($1::uuid, $2::text, 1.0)", memory_id, concept)

            return memory_id

    async def add_source(self, memory_id: UUID, source: dict[str, Any]) -> None:
        """Attach an additional source reference to a semantic memory and recompute trust."""
        async with self._pool.acquire() as conn:
            await conn.execute(
                "SELECT add_semantic_source_reference($1::uuid, $2::jsonb)",
                memory_id,
                _to_jsonb_arg(source),
            )

    async def get_truth_profile(self, memory_id: UUID) -> dict[str, Any]:
        """Return DB-computed provenance/trust details for a memory."""
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow("SELECT get_memory_truth_profile($1::uuid) AS profile", memory_id)
            if not row or row["profile"] is None:
                return {}
            return dict(_coerce_json(row["profile"]))

    async def remember_batch(self, memories: Iterable[MemoryInput]) -> list[UUID]:
        async with self._pool.acquire() as conn:
            items: list[dict[str, Any]] = []
            mem_list = list(memories)
            for m in mem_list:
                item: dict[str, Any] = {"type": m.type.value, "content": m.content, "importance": m.importance}
                if m.source_attribution is not None:
                    item["source_attribution"] = m.source_attribution
                if m.trust_level is not None:
                    item["trust_level"] = m.trust_level
                if m.type == MemoryType.EPISODIC:
                    item["context"] = m.context
                    item["emotional_valence"] = m.emotional_valence
                elif m.type == MemoryType.SEMANTIC:
                    item["source_references"] = m.source_references if m.source_references is not None else m.context
                elif m.type == MemoryType.PROCEDURAL:
                    item["steps"] = m.context if m.context is not None else {"steps": []}
                elif m.type == MemoryType.STRATEGIC:
                    item["supporting_evidence"] = m.context
                items.append(item)

            import json

            created = await conn.fetchval("SELECT batch_create_memories($1::jsonb)", json.dumps(items))
            ids = list(created or [])

            # Link concepts (still per-memory).
            for mid, m in zip(ids, mem_list):
                if m.concepts:
                    for concept in m.concepts:
                        await conn.fetchval("SELECT link_memory_to_concept($1::uuid, $2::text, 1.0)", mid, concept)

            return ids

    async def remember_batch_raw(
        self,
        contents: list[str],
        embeddings: list[list[float]],
        *,
        type: MemoryType = MemoryType.EPISODIC,
        importance: float = 0.5,
    ) -> list[UUID]:
        """
        Insert memories with pre-computed embeddings (bypasses get_embedding()).

        Notes:
        - Graph nodes are created to keep AGE state consistent.
        - Embedding dimension must match the DB typmod.
        """
        if len(contents) != len(embeddings):
            raise ValueError("contents and embeddings must have same length")

        async with self._pool.acquire() as conn:
            expected_dim = int(await conn.fetchval("SELECT embedding_dimension()"))
            for embedding in embeddings:
                if len(embedding) != expected_dim:
                    raise ValueError(f"embedding dimension mismatch: expected {expected_dim}, got {len(embedding)}")

            created = await conn.fetchval(
                """
                SELECT batch_create_memories_with_embeddings(
                    $1::memory_type,
                    $2::text[],
                    $3::jsonb,
                    $4::float
                )
                """,
                type.value,
                contents,
                _to_jsonb_arg(embeddings),
                float(importance),
            )
            return list(created or [])

    async def touch_memories(self, memory_ids: Iterable[UUID]) -> int:
        """Increment access_count/last_accessed for the given memory ids."""
        ids = list(memory_ids)
        if not ids:
            return 0
        async with self._pool.acquire() as conn:
            n = await conn.execute(
                """
                UPDATE memories
                SET access_count = access_count + 1,
                    last_accessed = CURRENT_TIMESTAMP
                WHERE id = ANY($1::uuid[])
                """,
                ids,
            )
            # asyncpg returns "UPDATE <count>"
            try:
                return int(str(n).split()[-1])
            except Exception:
                return 0

    # =========================================================================
    # GRAPH / RELATIONSHIPS
    # =========================================================================

    async def connect_memories(
        self,
        from_id: UUID,
        to_id: UUID,
        relationship: RelationshipType,
        *,
        confidence: float = 0.8,
        context: str | None = None,
    ) -> None:
        async with self._pool.acquire() as conn:
            await conn.execute(
                """
                SELECT discover_relationship($1::uuid, $2::uuid, $3::graph_edge_type, $4::float, 'api', NULL, $5::text)
                """,
                from_id,
                to_id,
                relationship.value,
                confidence,
                context,
            )

    async def connect_batch(self, relationships: Iterable[RelationshipInput]) -> None:
        async with self._pool.acquire() as conn:
            for r in relationships:
                await conn.execute(
                    """
                    SELECT discover_relationship($1::uuid, $2::uuid, $3::graph_edge_type, $4::float, 'api', NULL, $5::text)
                    """,
                    r.from_id,
                    r.to_id,
                    r.relationship_type.value,
                    r.confidence,
                    r.context,
                )

    async def find_causes(self, memory_id: UUID, *, depth: int = 3) -> list[dict[str, Any]]:
        async with self._pool.acquire() as conn:
            rows = await conn.fetch("SELECT * FROM find_causal_chain($1::uuid, $2::int)", memory_id, depth)
            return [dict(row) for row in rows]

    async def find_contradictions(self, memory_id: UUID | None = None) -> list[dict[str, Any]]:
        async with self._pool.acquire() as conn:
            rows = await conn.fetch("SELECT * FROM find_contradictions($1::uuid)", memory_id)
            return [dict(row) for row in rows]

    async def find_supporting_evidence(self, worldview_id: UUID) -> list[dict[str, Any]]:
        async with self._pool.acquire() as conn:
            rows = await conn.fetch("SELECT * FROM find_supporting_evidence($1::uuid)", worldview_id)
            return [dict(row) for row in rows]

    # =========================================================================
    # CONCEPTS
    # =========================================================================

    async def link_concept(self, memory_id: UUID, concept: str, *, strength: float = 1.0) -> UUID:
        async with self._pool.acquire() as conn:
            return await conn.fetchval("SELECT link_memory_to_concept($1::uuid, $2::text, $3::float)", memory_id, concept, strength)

    async def find_by_concept(self, concept: str, *, limit: int = 10) -> list[Memory]:
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT m.id, m.type, m.content, m.importance, m.created_at, em.emotional_valence
                FROM memories m
                JOIN memory_concepts mc ON m.id = mc.memory_id
                JOIN concepts c ON mc.concept_id = c.id
                LEFT JOIN episodic_memories em ON m.id = em.memory_id
                WHERE c.name = $1 AND m.status = 'active'
                ORDER BY mc.strength DESC, m.importance DESC
                LIMIT $2
                """,
                concept,
                limit,
            )
            return [self._row_to_memory(row) for row in rows]

    # =========================================================================
    # WORKING MEMORY
    # =========================================================================

    async def hold(self, content: str, *, ttl_seconds: int = 3600) -> UUID:
        async with self._pool.acquire() as conn:
            return await conn.fetchval(
                "SELECT add_to_working_memory($1::text, $2::int * interval '1 second')",
                content,
                ttl_seconds,
            )

    async def search_working(self, query: str, *, limit: int = 5) -> list[dict[str, Any]]:
        async with self._pool.acquire() as conn:
            rows = await conn.fetch("SELECT * FROM search_working_memory($1::text, $2::int)", query, limit)
            return [dict(row) for row in rows]

    # =========================================================================
    # STATE / INTROSPECTION
    # =========================================================================

    async def get_emotional_state(self) -> dict[str, Any] | None:
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow("SELECT * FROM current_emotional_state")
            return dict(row) if row else None

    async def get_drives(self) -> list[dict[str, Any]]:
        async with self._pool.acquire() as conn:
            rows = await conn.fetch("SELECT * FROM drive_status")
            return [dict(row) for row in rows]

    async def get_health(self) -> dict[str, Any]:
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow("SELECT * FROM cognitive_health")
            return dict(row) if row else {}

    async def get_identity(self) -> list[dict[str, Any]]:
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT aspect_type, content, stability
                FROM identity_aspects
                WHERE stability > 0.3
                ORDER BY stability DESC
                LIMIT 5
                """
            )
            return [dict(row) for row in rows]

    async def get_worldview(self) -> list[dict[str, Any]]:
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT category, belief, confidence
                FROM worldview_primitives
                WHERE confidence > 0.5
                ORDER BY confidence DESC
                LIMIT 5
                """
            )
            return [dict(row) for row in rows]

    async def get_goals(self, *, priority: GoalPriority | None = None) -> list[dict[str, Any]]:
        async with self._pool.acquire() as conn:
            if priority is None:
                rows = await conn.fetch(
                    """
                    SELECT * FROM goals
                    WHERE priority IN ('active', 'queued')
                    ORDER BY priority, last_touched DESC
                    """
                )
            else:
                rows = await conn.fetch(
                    "SELECT * FROM goals WHERE priority = $1::goal_priority ORDER BY last_touched DESC",
                    priority.value,
                )
            return [dict(row) for row in rows]

    async def create_goal(
        self,
        title: str,
        *,
        description: str | None = None,
        source: GoalSource | str = GoalSource.USER_REQUEST,
        priority: GoalPriority | str = GoalPriority.QUEUED,
        parent_id: UUID | None = None,
        due_at: datetime | None = None,
    ) -> UUID:
        async with self._pool.acquire() as conn:
            return await conn.fetchval(
                """
                SELECT create_goal(
                    $1,
                    $2,
                    $3::goal_source,
                    $4::goal_priority,
                    $5::uuid,
                    $6::timestamptz
                )
                """,
                title,
                description,
                (source.value if isinstance(source, GoalSource) else str(source)),
                (priority.value if isinstance(priority, GoalPriority) else str(priority)),
                parent_id,
                due_at,
            )

    async def queue_user_message(
        self,
        message: str,
        *,
        intent: str | None = None,
        context: dict[str, Any] | None = None,
    ) -> UUID:
        async with self._pool.acquire() as conn:
            return await conn.fetchval(
                "SELECT queue_user_message($1, $2, $3::jsonb)",
                message,
                intent,
                _to_jsonb_arg(context or {}),
            )

    async def get_ingestion_receipts(self, source_file: str, content_hashes: list[str]) -> dict[str, UUID]:
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT * FROM get_ingestion_receipts($1::text, $2::text[])",
                source_file,
                content_hashes,
            )
            out: dict[str, UUID] = {}
            for r in rows:
                try:
                    out[str(r["content_hash"])] = UUID(str(r["memory_id"]))
                except Exception:
                    continue
            return out

    async def record_ingestion_receipts(self, items: list[dict[str, Any]]) -> int:
        async with self._pool.acquire() as conn:
            import json

            return int(
                await conn.fetchval(
                    "SELECT record_ingestion_receipts($1::jsonb)",
                    json.dumps(items),
                )
                or 0
            )

    # =========================================================================
    # INTERNALS
    # =========================================================================

    async def _create_memory(
        self,
        conn: asyncpg.Connection,
        content: str,
        type: MemoryType,
        importance: float,
        emotional_valence: float,
        context: dict[str, Any] | None,
        *,
        source_attribution: dict[str, Any] | None = None,
        source_references: Any | None = None,
        trust_level: float | None = None,
    ) -> UUID:
        if type == MemoryType.EPISODIC:
            return await conn.fetchval(
                "SELECT create_episodic_memory($1::text, NULL, $2::jsonb, NULL, $3::float, CURRENT_TIMESTAMP, $4::float, $5::jsonb, $6::float)",
                content,
                _to_jsonb_arg(context),
                emotional_valence,
                importance,
                _to_jsonb_arg(source_attribution),
                trust_level,
            )
        if type == MemoryType.SEMANTIC:
            sources = source_references if source_references is not None else context
            return await conn.fetchval(
                "SELECT create_semantic_memory($1::text, 0.8::float, NULL, NULL, $2::jsonb, $3::float, $4::jsonb, $5::float)",
                content,
                _to_jsonb_arg(sources),
                importance,
                _to_jsonb_arg(source_attribution),
                trust_level,
            )
        if type == MemoryType.PROCEDURAL:
            steps = context if context is not None else {}
            return await conn.fetchval(
                "SELECT create_procedural_memory($1::text, $2::jsonb, NULL, $3::float, $4::jsonb, $5::float)",
                content,
                _to_jsonb_arg(steps),
                importance,
                _to_jsonb_arg(source_attribution),
                trust_level,
            )
        if type == MemoryType.STRATEGIC:
            return await conn.fetchval(
                "SELECT create_strategic_memory($1::text, $2::text, 0.8::float, $3::jsonb, NULL, $4::float, $5::jsonb, $6::float)",
                content,
                content,
                _to_jsonb_arg(context),
                importance,
                _to_jsonb_arg(source_attribution),
                trust_level,
            )
        raise ValueError(f"Unknown memory type: {type}")

    async def _recall_memories(
        self,
        conn: asyncpg.Connection,
        query: str,
        limit: int,
        memory_types: list[MemoryType] | None = None,
        min_importance: float = 0.0,
    ) -> list[Memory]:
        rows = await conn.fetch(
            """
            SELECT
                fr.memory_id,
                fr.content,
                fr.memory_type,
                fr.score,
                fr.source,
                m.importance,
                m.trust_level,
                m.source_attribution,
                m.created_at,
                em.emotional_valence
            FROM fast_recall($1::text, $2::int) fr
            JOIN memories m ON m.id = fr.memory_id
            LEFT JOIN episodic_memories em ON em.memory_id = fr.memory_id
            WHERE m.importance >= $3::float
            """,
            query,
            limit,
            min_importance,
        )

        memories: list[Memory] = []
        for row in rows:
            mt = MemoryType(row["memory_type"])
            if memory_types is not None and mt not in set(memory_types):
                continue
            memories.append(
                Memory(
                    id=row["memory_id"],
                    type=mt,
                    content=row["content"],
                    importance=float(row["importance"]),
                    similarity=float(row["score"]),
                    source=row["source"],
                    trust_level=float(row["trust_level"]) if row["trust_level"] is not None else None,
                    source_attribution=_coerce_json(row["source_attribution"]) if row["source_attribution"] is not None else None,
                    created_at=row["created_at"],
                    emotional_valence=row["emotional_valence"],
                )
            )
        return memories

    async def _find_partial_activations(self, conn: asyncpg.Connection, query: str) -> list[PartialActivation]:
        rows = await conn.fetch("SELECT * FROM find_partial_activations($1::text)", query)
        out: list[PartialActivation] = []
        for row in rows:
            out.append(
                PartialActivation(
                    cluster_id=row["cluster_id"],
                    cluster_name=row["cluster_name"],
                    keywords=list(row["keywords"] or []),
                    emotional_signature=_coerce_json(row["emotional_signature"]) if row["emotional_signature"] is not None else None,
                    cluster_similarity=float(row["cluster_similarity"]),
                    best_memory_similarity=float(row["best_memory_similarity"]),
                )
            )
        return out

    def _row_to_memory(self, row: asyncpg.Record) -> Memory:
        return Memory(
            id=row["id"],
            type=MemoryType(row["type"]),
            content=row["content"],
            importance=float(row["importance"]),
            trust_level=float(row["trust_level"]) if "trust_level" in row and row["trust_level"] is not None else None,
            source_attribution=_coerce_json(row["source_attribution"])
            if "source_attribution" in row and row["source_attribution"] is not None
            else None,
            created_at=row["created_at"] if "created_at" in row else None,
            emotional_valence=row["emotional_valence"] if "emotional_valence" in row else None,
        )


class CognitiveMemorySync:
    """Synchronous wrapper around CognitiveMemory for non-async call sites."""

    def __init__(self, async_client: CognitiveMemory, loop: asyncio.AbstractEventLoop):
        self._async = async_client
        self._loop = loop

    @classmethod
    def connect(cls, dsn: str, **kwargs: Any) -> "CognitiveMemorySync":
        loop = asyncio.new_event_loop()
        try:
            client = loop.run_until_complete(CognitiveMemory.create(dsn, **kwargs))
        except Exception:
            loop.close()
            raise
        return cls(client, loop)

    def close(self) -> None:
        self._loop.run_until_complete(self._async.close())
        self._loop.close()

    def hydrate(self, query: str, **kwargs: Any) -> HydratedContext:
        return self._loop.run_until_complete(self._async.hydrate(query, **kwargs))

    def recall(self, query: str, **kwargs: Any) -> RecallResult:
        return self._loop.run_until_complete(self._async.recall(query, **kwargs))

    def remember(self, content: str, **kwargs: Any) -> UUID:
        return self._loop.run_until_complete(self._async.remember(content, **kwargs))

    def remember_batch(self, memories: Iterable[MemoryInput]) -> list[UUID]:
        return self._loop.run_until_complete(self._async.remember_batch(memories))

    def remember_batch_raw(self, contents: list[str], embeddings: list[list[float]], **kwargs: Any) -> list[UUID]:
        return self._loop.run_until_complete(self._async.remember_batch_raw(contents, embeddings, **kwargs))

    def connect_memories(self, from_id: UUID, to_id: UUID, relationship: RelationshipType, **kwargs: Any) -> None:
        return self._loop.run_until_complete(self._async.connect_memories(from_id, to_id, relationship, **kwargs))

    def touch_memories(self, memory_ids: Iterable[UUID]) -> int:
        return self._loop.run_until_complete(self._async.touch_memories(memory_ids))

    def create_goal(
        self,
        title: str,
        *,
        description: str | None = None,
        source: GoalSource | str = GoalSource.USER_REQUEST,
        priority: GoalPriority | str = GoalPriority.QUEUED,
        parent_id: UUID | None = None,
        due_at: datetime | None = None,
    ) -> UUID:
        return self._loop.run_until_complete(
            self._async.create_goal(
                title,
                description=description,
                source=source,
                priority=priority,
                parent_id=parent_id,
                due_at=due_at,
            )
        )

    def queue_user_message(self, message: str, *, intent: str | None = None, context: dict[str, Any] | None = None) -> UUID:
        return self._loop.run_until_complete(self._async.queue_user_message(message, intent=intent, context=context))

    def get_ingestion_receipts(self, source_file: str, content_hashes: list[str]) -> dict[str, UUID]:
        return self._loop.run_until_complete(self._async.get_ingestion_receipts(source_file, content_hashes))

    def record_ingestion_receipts(self, items: list[dict[str, Any]]) -> int:
        return self._loop.run_until_complete(self._async.record_ingestion_receipts(items))


def format_context_for_prompt(context: HydratedContext, *, max_memories: int = 5, max_partials: int = 3) -> str:
    parts: list[str] = []

    if context.memories:
        parts.append("## Relevant Memories")
        for m in context.memories[:max_memories]:
            score = f" (score: {m.similarity:.2f})" if m.similarity is not None else ""
            trust = f", trust: {m.trust_level:.2f}" if m.trust_level is not None else ""
            src_kind = ""
            if m.source_attribution and isinstance(m.source_attribution, dict):
                kind = m.source_attribution.get("kind")
                ref = m.source_attribution.get("ref")
                if kind and ref:
                    src_kind = f", source: {kind} ({ref})"
                elif kind:
                    src_kind = f", source: {kind}"
            parts.append(f"- {m.content}{score}{trust}{src_kind}")

    if context.partial_activations:
        parts.append("\n## Vague Recollections (tip-of-tongue)")
        for pa in context.partial_activations[:max_partials]:
            keywords = ", ".join(pa.keywords[:5]) if pa.keywords else "unknown"
            parts.append(f"- Theme '{pa.cluster_name}': {keywords}")

    if context.identity:
        parts.append("\n## Identity")
        for aspect in context.identity[:3]:
            parts.append(f"- {aspect.get('aspect_type', 'unknown')}: {aspect.get('content', {})}")

    if context.worldview:
        parts.append("\n## Beliefs")
        for belief in context.worldview[:3]:
            conf = belief.get("confidence", 0)
            parts.append(f"- {belief.get('belief', '')} (confidence: {conf:.1f})")

    if context.emotional_state:
        es = context.emotional_state
        parts.append("\n## Current Emotional State")
        parts.append(f"- Feeling: {es.get('primary_emotion', 'neutral')}")
        parts.append(f"- Valence: {es.get('valence', 0):.2f}, Arousal: {es.get('arousal', 0.5):.2f}")

    if context.urgent_drives:
        parts.append("\n## Urgent Drives")
        for drive in context.urgent_drives:
            ratio = drive.get("urgency_ratio")
            if ratio is None:
                parts.append(f"- {drive.get('name')}: {drive.get('level')}")
            else:
                parts.append(f"- {drive.get('name')}: {float(ratio):.1%} urgent")

    return "\n".join(parts)


def get_personhood_prompt(kind: str) -> str:
    """
    Convenience helper for apps composing LLM prompts.

    `kind` is one of: "conversation", "heartbeat", "reflect".
    """
    if kind not in {"conversation", "heartbeat", "reflect"}:
        raise ValueError("kind must be one of: conversation, heartbeat, reflect")
    return compose_personhood_prompt(kind)  # type: ignore[arg-type]


def _coerce_json(val: Any) -> Any:
    if isinstance(val, str):
        import json

        return json.loads(val)
    return val
