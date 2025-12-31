#!/usr/bin/env python3
"""
Hexis Memory MCP Tools

Provides MCP-compatible tools for an LLM to query its memory system during conversation.
These are the function definitions and handlers that allow the model to actively
recall, search, and explore its memories.
"""

import json
import re
from datetime import datetime, timezone
from dataclasses import dataclass, asdict
from typing import Optional, Any
from enum import Enum

try:
    import psycopg2
    from psycopg2.extras import RealDictCursor
    HAS_PSYCOPG2 = True
except Exception:  # pragma: no cover
    psycopg2 = None  # type: ignore[assignment]
    RealDictCursor = None  # type: ignore[assignment]
    HAS_PSYCOPG2 = False

from core.cognitive_memory_api import (
    CognitiveMemorySync,
    GoalPriority as ApiGoalPriority,
    GoalSource as ApiGoalSource,
    MemoryType as ApiMemoryType,
)


# ============================================================================
# TOOL DEFINITIONS (OpenAI Function Calling Format)
# ============================================================================

MEMORY_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "recall",
            "description": "Search memories by semantic similarity. Use this to find memories related to a topic, concept, or question. Returns the most relevant memories based on meaning, not just keyword matching.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Natural language query describing what you want to remember. Be specific and descriptive for better results."
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum number of memories to return (default: 5, max: 20)",
                        "default": 5
                    },
                    "memory_types": {
                        "type": "array",
                        "items": {
                            "type": "string",
                            "enum": ["episodic", "semantic", "procedural", "strategic"]
                        },
                        "description": "Filter by memory types. Omit to search all types."
                    },
                    "min_importance": {
                        "type": "number",
                        "description": "Minimum importance score (0.0-1.0). Use to filter for significant memories.",
                        "default": 0.0
                    }
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "recall_recent",
            "description": "Retrieve the most recently accessed or created memories. Useful for continuing recent conversations or recalling what was just discussed.",
            "parameters": {
                "type": "object",
                "properties": {
                    "limit": {
                        "type": "integer",
                        "description": "Number of recent memories to return (default: 5, max: 20)",
                        "default": 5
                    },
                    "memory_types": {
                        "type": "array",
                        "items": {
                            "type": "string",
                            "enum": ["episodic", "semantic", "procedural", "strategic"]
                        },
                        "description": "Filter by memory types. Omit to include all types."
                    },
                    "by_access": {
                        "type": "boolean",
                        "description": "If true, sort by last accessed time. If false, sort by creation time.",
                        "default": True
                    }
                },
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "recall_episode",
            "description": "Retrieve all memories from a specific episode (a coherent sequence of related memories, like a conversation or work session). Use when you need the full context of a past interaction.",
            "parameters": {
                "type": "object",
                "properties": {
                    "episode_id": {
                        "type": "string",
                        "description": "UUID of the episode to retrieve"
                    }
                },
                "required": ["episode_id"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "explore_concept",
            "description": "Explore memories connected to a specific concept. Shows how different memories relate to an idea and what other concepts are connected.",
            "parameters": {
                "type": "object",
                "properties": {
                    "concept": {
                        "type": "string",
                        "description": "The concept to explore (e.g., 'machine learning', 'user preferences', 'project goals')"
                    },
                    "include_related": {
                        "type": "boolean",
                        "description": "If true, also return memories linked to related concepts",
                        "default": True
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum memories to return per concept",
                        "default": 5
                    }
                },
                "required": ["concept"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "explore_cluster",
            "description": "Explore a thematic cluster of memories. Clusters are automatically formed groups of related memories. Use to understand patterns and themes.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Find clusters related to this topic"
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum clusters to return",
                        "default": 3
                    }
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_procedures",
            "description": "Retrieve procedural memories (how-to knowledge) for a specific task. Returns step-by-step instructions and prerequisites.",
            "parameters": {
                "type": "object",
                "properties": {
                    "task": {
                        "type": "string",
                        "description": "The task you want to know how to do"
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum procedures to return",
                        "default": 3
                    }
                },
                "required": ["task"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_strategies",
            "description": "Retrieve strategic memories (patterns, heuristics, lessons learned) applicable to a situation. These are meta-level insights about what works.",
            "parameters": {
                "type": "object",
                "properties": {
                    "situation": {
                        "type": "string",
                        "description": "The situation or context you need strategic guidance for"
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum strategies to return",
                        "default": 3
                    }
                },
                "required": ["situation"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "list_recent_episodes",
            "description": "List recent episodes (conversation sessions, work sessions, etc.) to understand what interactions have happened.",
            "parameters": {
                "type": "object",
                "properties": {
                    "limit": {
                        "type": "integer",
                        "description": "Number of episodes to return",
                        "default": 5
                    }
                },
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "create_goal",
            "description": "Create a new goal (queued task) for the agent to pursue later. Use this for reminders, TODOs, or longer-term objectives.",
            "parameters": {
                "type": "object",
                "properties": {
                    "title": {"type": "string", "description": "Short goal title"},
                    "description": {"type": ["string", "null"], "description": "Optional longer description"},
                    "priority": {
                        "type": "string",
                        "enum": ["active", "queued", "backburner"],
                        "default": "queued",
                        "description": "Desired priority (DB may demote if at limits)"
                    },
                    "source": {
                        "type": "string",
                        "enum": ["curiosity", "user_request", "identity", "derived", "external"],
                        "default": "user_request",
                        "description": "Why this goal exists"
                    },
                    "due_at": {
                        "type": ["string", "null"],
                        "description": "Optional due timestamp in ISO8601 (e.g. 2025-01-01T12:00:00Z)"
                    }
                },
                "required": ["title"],
                "additionalProperties": False
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "queue_user_message",
            "description": "Queue a message to the user in the outbox for delivery by an external integration (worker/webhook).",
            "parameters": {
                "type": "object",
                "properties": {
                    "message": {"type": "string", "description": "Message body for the user"},
                    "intent": {"type": ["string", "null"], "description": "Optional intent/category (e.g. 'reminder', 'status', 'question')"},
                    "context": {"type": ["object", "null"], "description": "Optional JSON context payload"}
                },
                "required": ["message"],
                "additionalProperties": False
            }
        }
    }
]

_API_TOOL_NAMES = {"recall", "recall_recent", "explore_concept", "get_procedures", "get_strategies", "create_goal", "queue_user_message"}


# ============================================================================
# TOOL HANDLERS
# ============================================================================

class MemoryToolHandler:
    """Handles execution of memory tools."""
    
    def __init__(self, db_config: dict):
        self.db_config = db_config
        self.conn = None
    
    def connect(self):
        """Establish database connection."""
        if not HAS_PSYCOPG2:
            raise RuntimeError("psycopg2 is required for legacy MemoryToolHandler; use CognitiveMemory API instead.")
        if not self.conn or self.conn.closed:
            self.conn = psycopg2.connect(
                host=self.db_config.get('host', 'localhost'),
                port=self.db_config.get('port', 43815),
                dbname=self.db_config.get('dbname', 'hexis_memory'),
                user=self.db_config.get('user', 'postgres'),
                password=self.db_config.get('password', 'password')
            )
    
    def close(self):
        """Close database connection."""
        if self.conn:
            self.conn.close()
    
    def execute_tool(self, tool_name: str, arguments: dict) -> dict:
        """Execute a tool and return the result."""
        self.connect()
        
        handlers = {
            'recall': self._handle_recall,
            'recall_recent': self._handle_recall_recent,
            'recall_episode': self._handle_recall_episode,
            'explore_concept': self._handle_explore_concept,
            'explore_cluster': self._handle_explore_cluster,
            'get_procedures': self._handle_get_procedures,
            'get_strategies': self._handle_get_strategies,
            'list_recent_episodes': self._handle_list_episodes,
        }
        
        handler = handlers.get(tool_name)
        if not handler:
            return {"error": f"Unknown tool: {tool_name}"}
        
        try:
            return handler(arguments)
        except Exception as e:
            return {"error": str(e)}

    def _handle_recall(self, args: dict) -> dict:
        """Handle the recall tool - semantic memory search."""
        query = args.get('query', '')
        limit = min(args.get('limit', 5), 20)
        memory_types = args.get('memory_types')
        min_importance = args.get('min_importance', 0.0)
        
        with self.conn.cursor(cursor_factory=RealDictCursor) as cur:
            # Use the fast_recall function
            cur.execute("""
                SELECT 
                    memory_id,
                    content,
                    memory_type,
                    score,
                    source
                FROM fast_recall(%s, %s)
            """, (query, limit * 2))  # Get more, then filter
            
            results = cur.fetchall()
            
            # Filter by type and importance if specified
            if memory_types or min_importance > 0:
                cur.execute("""
                    SELECT id, importance FROM memories 
                    WHERE id = ANY(%s)
                """, ([r['memory_id'] for r in results],))
                
                importance_map = {str(row['id']): row['importance'] for row in cur.fetchall()}
                
                filtered = []
                for r in results:
                    mid = str(r['memory_id'])
                    if memory_types and r['memory_type'] not in memory_types:
                        continue
                    if importance_map.get(mid, 0) < min_importance:
                        continue
                    r['importance'] = importance_map.get(mid, 0)
                    filtered.append(r)
                results = filtered[:limit]
            
            # Update access counts
            if results:
                cur.execute("""
                    UPDATE memories 
                    SET access_count = access_count + 1, last_accessed = CURRENT_TIMESTAMP
                    WHERE id = ANY(%s)
                """, ([r['memory_id'] for r in results],))
                self.conn.commit()
        
        return {
            "memories": [dict(r) for r in results],
            "count": len(results),
            "query": query
        }
    
    def _handle_recall_recent(self, args: dict) -> dict:
        """Handle recall_recent - get recently accessed/created memories."""
        limit = min(args.get('limit', 5), 20)
        memory_types = args.get('memory_types')
        by_access = args.get('by_access', True)
        
        order_col = 'last_accessed' if by_access else 'created_at'
        
        with self.conn.cursor(cursor_factory=RealDictCursor) as cur:
            type_filter = ""
            params = [limit]
            
            if memory_types:
                type_filter = "AND type = ANY(%s)"
                params.insert(0, memory_types)
            
            cur.execute(f"""
                SELECT 
                    id as memory_id,
                    content,
                    type as memory_type,
                    importance,
                    created_at,
                    last_accessed
                FROM memories
                WHERE status = 'active' {type_filter}
                ORDER BY {order_col} DESC NULLS LAST
                LIMIT %s
            """, params)
            
            results = cur.fetchall()
        
        return {
            "memories": [dict(r) for r in results],
            "count": len(results),
            "sorted_by": order_col
        }
    
    def _handle_recall_episode(self, args: dict) -> dict:
        """Handle recall_episode - get all memories from an episode."""
        episode_id = args.get('episode_id')
        
        with self.conn.cursor(cursor_factory=RealDictCursor) as cur:
            # Get episode info
            cur.execute("""
                SELECT id, started_at, ended_at, episode_type, summary
                FROM episodes
                WHERE id = %s
            """, (episode_id,))
            
            episode = cur.fetchone()
            if not episode:
                return {"error": f"Episode not found: {episode_id}"}
            
            # Get memories in episode
            cur.execute("""
                SELECT 
                    m.id as memory_id,
                    m.content,
                    m.type as memory_type,
                    m.importance,
                    m.created_at,
                    em.sequence_order
                FROM episode_memories em
                JOIN memories m ON em.memory_id = m.id
                WHERE em.episode_id = %s
                ORDER BY em.sequence_order
            """, (episode_id,))
            
            memories = cur.fetchall()
        
        return {
            "episode": dict(episode),
            "memories": [dict(m) for m in memories],
            "count": len(memories)
        }
    
    def _handle_explore_concept(self, args: dict) -> dict:
        """Handle explore_concept - find memories linked to a concept."""
        concept = args.get('concept', '').lower().strip()
        include_related = args.get('include_related', True)
        limit = min(args.get('limit', 5), 20)
        
        with self.conn.cursor(cursor_factory=RealDictCursor) as cur:
            # Find the concept
            cur.execute("""
                SELECT id, name, description, path_text
                FROM concepts
                WHERE name ILIKE %s
                LIMIT 1
            """, (f'%{concept}%',))
            
            concept_row = cur.fetchone()
            
            if not concept_row:
                # No exact concept found, fall back to semantic search
                return self._handle_recall({'query': concept, 'limit': limit})
            
            concept_id = concept_row['id']
            
            # Get memories linked to this concept
            cur.execute("""
                SELECT 
                    m.id as memory_id,
                    m.content,
                    m.type as memory_type,
                    m.importance,
                    mc.strength as concept_strength
                FROM memory_concepts mc
                JOIN memories m ON mc.memory_id = m.id
                WHERE mc.concept_id = %s
                AND m.status = 'active'
                ORDER BY mc.strength DESC, m.importance DESC
                LIMIT %s
            """, (concept_id, limit))
            
            memories = cur.fetchall()
            
            # Get related concepts
            related_concepts = []
            if include_related:
                cur.execute("""
                    SELECT DISTINCT c2.name, COUNT(*) as shared_memories
                    FROM memory_concepts mc1
                    JOIN memory_concepts mc2 ON mc1.memory_id = mc2.memory_id
                    JOIN concepts c2 ON mc2.concept_id = c2.id
                    WHERE mc1.concept_id = %s
                    AND mc2.concept_id != %s
                    GROUP BY c2.name
                    ORDER BY shared_memories DESC
                    LIMIT 10
                """, (concept_id, concept_id))
                
                related_concepts = [dict(r) for r in cur.fetchall()]
        
        return {
            "concept": dict(concept_row),
            "memories": [dict(m) for m in memories],
            "related_concepts": related_concepts,
            "count": len(memories)
        }
    
    def _handle_explore_cluster(self, args: dict) -> dict:
        """Handle explore_cluster - find and explore thematic clusters."""
        query = args.get('query', '')
        limit = min(args.get('limit', 3), 10)
        
        with self.conn.cursor(cursor_factory=RealDictCursor) as cur:
            # Search clusters by embedding similarity
            cur.execute("""
                WITH query_embedding AS (
                    SELECT get_embedding(%s) as emb
                )
                SELECT 
                    mc.id,
                    mc.name,
                    mc.description,
                    mc.cluster_type,
                    mc.importance_score,
                    mc.keywords,
                    1 - (mc.centroid_embedding <=> (SELECT emb FROM query_embedding)) as similarity
                FROM memory_clusters mc
                WHERE mc.centroid_embedding IS NOT NULL
                ORDER BY mc.centroid_embedding <=> (SELECT emb FROM query_embedding)
                LIMIT %s
            """, (query, limit))
            
            clusters = cur.fetchall()
            
            # For each cluster, get sample memories
            result_clusters = []
            for cluster in clusters:
                cur.execute("""
                    SELECT 
                        m.id as memory_id,
                        m.content,
                        m.type as memory_type,
                        mcm.membership_strength
                    FROM memory_cluster_members mcm
                    JOIN memories m ON mcm.memory_id = m.id
                    WHERE mcm.cluster_id = %s
                    AND m.status = 'active'
                    ORDER BY mcm.membership_strength DESC
                    LIMIT 3
                """, (cluster['id'],))
                
                sample_memories = cur.fetchall()
                
                result_clusters.append({
                    **dict(cluster),
                    "sample_memories": [dict(m) for m in sample_memories]
                })
        
        return {
            "clusters": result_clusters,
            "count": len(result_clusters),
            "query": query
        }
    
    def _handle_get_procedures(self, args: dict) -> dict:
        """Handle get_procedures - find procedural knowledge."""
        task = args.get('task', '')
        limit = min(args.get('limit', 3), 10)
        
        with self.conn.cursor(cursor_factory=RealDictCursor) as cur:
            # Search procedural memories
            cur.execute("""
                WITH query_embedding AS (
                    SELECT get_embedding(%s) as emb
                )
                SELECT 
                    m.id as memory_id,
                    m.content,
                    pm.steps,
                    pm.prerequisites,
                    pm.success_rate,
                    pm.average_duration,
                    1 - (m.embedding <=> (SELECT emb FROM query_embedding)) as similarity
                FROM memories m
                JOIN procedural_memories pm ON m.id = pm.memory_id
                WHERE m.status = 'active'
                AND m.type = 'procedural'
                ORDER BY m.embedding <=> (SELECT emb FROM query_embedding)
                LIMIT %s
            """, (task, limit))
            
            procedures = cur.fetchall()
        
        return {
            "procedures": [dict(p) for p in procedures],
            "count": len(procedures),
            "task": task
        }
    
    def _handle_get_strategies(self, args: dict) -> dict:
        """Handle get_strategies - find strategic knowledge."""
        situation = args.get('situation', '')
        limit = min(args.get('limit', 3), 10)
        
        with self.conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("""
                WITH query_embedding AS (
                    SELECT get_embedding(%s) as emb
                )
                SELECT 
                    m.id as memory_id,
                    m.content,
                    sm.pattern_description,
                    sm.confidence_score,
                    sm.context_applicability,
                    sm.success_metrics,
                    1 - (m.embedding <=> (SELECT emb FROM query_embedding)) as similarity
                FROM memories m
                JOIN strategic_memories sm ON m.id = sm.memory_id
                WHERE m.status = 'active'
                AND m.type = 'strategic'
                ORDER BY m.embedding <=> (SELECT emb FROM query_embedding)
                LIMIT %s
            """, (situation, limit))
            
            strategies = cur.fetchall()
        
        return {
            "strategies": [dict(s) for s in strategies],
            "count": len(strategies),
            "situation": situation
        }
    
    def _handle_list_episodes(self, args: dict) -> dict:
        """Handle list_recent_episodes - list recent episodes."""
        limit = min(args.get('limit', 5), 20)
        
        with self.conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("""
                SELECT 
                    e.id,
                    e.started_at,
                    e.ended_at,
                    e.episode_type,
                    e.summary,
                    COUNT(em.memory_id) as memory_count
                FROM episodes e
                LEFT JOIN episode_memories em ON e.id = em.episode_id
                GROUP BY e.id
                ORDER BY e.started_at DESC
                LIMIT %s
            """, (limit,))
            
            episodes = cur.fetchall()
        
        return {
            "episodes": [dict(e) for e in episodes],
            "count": len(episodes)
        }


class ApiMemoryToolHandler:
    """Tool handler backed by CognitiveMemorySync (asyncpg)."""

    def __init__(self, db_config: dict):
        self.db_config = db_config
        self.client: CognitiveMemorySync | None = None

    def connect(self) -> None:
        if self.client is not None:
            return
        dsn = (
            f"postgresql://{self.db_config.get('user', 'postgres')}:{self.db_config.get('password', 'password')}"
            f"@{self.db_config.get('host', 'localhost')}:{int(self.db_config.get('port', 43815))}"
            f"/{self.db_config.get('dbname', 'hexis_memory')}"
        )
        self.client = CognitiveMemorySync.connect(dsn, min_size=1, max_size=5)

    def close(self) -> None:
        if self.client is not None:
            self.client.close()
            self.client = None

    def execute_tool(self, tool_name: str, arguments: dict) -> dict:
        self.connect()
        assert self.client is not None

        handlers = {
            "recall": self._handle_recall,
            "recall_recent": self._handle_recall_recent,
            "explore_concept": self._handle_explore_concept,
            "get_procedures": self._handle_get_procedures,
            "get_strategies": self._handle_get_strategies,
            "create_goal": self._handle_create_goal,
            "queue_user_message": self._handle_queue_user_message,
        }
        handler = handlers.get(tool_name)
        if not handler:
            return {"error": f"Unknown tool: {tool_name}"}
        try:
            return handler(arguments or {})
        except Exception as e:
            return {"error": str(e)}

    def _handle_recall(self, args: dict) -> dict:
        query = args.get("query", "")
        limit = min(int(args.get("limit", 5)), 20)
        memory_types = args.get("memory_types")
        min_importance = float(args.get("min_importance", 0.0) or 0.0)

        parsed_types = None
        if isinstance(memory_types, list) and memory_types:
            parsed_types = [ApiMemoryType(str(t)) for t in memory_types]

        result = self.client.recall(query, limit=limit, memory_types=parsed_types, min_importance=min_importance, include_partial=False)
        self.client.touch_memories([m.id for m in result.memories])
        memories = [
            {
                "memory_id": str(m.id),
                "content": m.content,
                "memory_type": m.type.value,
                "score": m.similarity,
                "source": m.source,
                "importance": m.importance,
                "trust_level": m.trust_level,
                "source_attribution": m.source_attribution,
            }
            for m in result.memories
        ]
        return {"memories": memories, "count": len(memories), "query": query}

    def _handle_recall_recent(self, args: dict) -> dict:
        limit = min(int(args.get("limit", 5)), 20)
        memory_types = args.get("memory_types")
        by_access = bool(args.get("by_access", True))

        # API exposes recent by created_at; "by_access" isn't supported, so we approximate by created_at.
        # If a type filter is provided, use the first type.
        mt = None
        if isinstance(memory_types, list) and memory_types:
            mt = ApiMemoryType(str(memory_types[0]))

        rows = self.client.recall_recent(limit=limit, memory_type=mt)
        if by_access:
            # Touch to keep access_count semantics similar.
            self.client.touch_memories([m.id for m in rows])
        results = [
            {
                "memory_id": str(m.id),
                "content": m.content,
                "memory_type": m.type.value,
                "importance": m.importance,
                "created_at": m.created_at.isoformat() if m.created_at else None,
                "last_accessed": None,
                "trust_level": m.trust_level,
                "source_attribution": m.source_attribution,
            }
            for m in rows
        ]
        return {"memories": results, "count": len(results)}

    def _handle_explore_concept(self, args: dict) -> dict:
        concept = str(args.get("concept", "")).strip()
        include_related = bool(args.get("include_related", True))
        limit = min(int(args.get("limit", 5)), 20)
        if not concept:
            return {"error": "Missing concept"}

        direct = self.client.find_by_concept(concept, limit=limit)
        combined: dict[str, dict] = {
            str(m.id): {
                "memory_id": str(m.id),
                "content": m.content,
                "memory_type": m.type.value,
                "importance": m.importance,
                "trust_level": m.trust_level,
                "source_attribution": m.source_attribution,
                "source": "concept",
                "score": None,
            }
            for m in direct
        }
        if include_related:
            rr = self.client.recall(concept, limit=limit, include_partial=False)
            for m in rr.memories:
                combined.setdefault(
                    str(m.id),
                    {
                        "memory_id": str(m.id),
                        "content": m.content,
                        "memory_type": m.type.value,
                        "importance": m.importance,
                        "trust_level": m.trust_level,
                        "source_attribution": m.source_attribution,
                        "source": m.source,
                        "score": m.similarity,
                    },
                )

        out = list(combined.values())[:limit]
        return {"concept": concept, "memories": out, "count": len(out)}

    def _handle_get_procedures(self, args: dict) -> dict:
        task = str(args.get("task", "")).strip()
        limit = min(int(args.get("limit", 3)), 10)
        if not task:
            return {"procedures": [], "count": 0, "task": task}
        res = self.client.recall(task, limit=limit, memory_types=[ApiMemoryType.PROCEDURAL], include_partial=False)
        return {"procedures": [{"memory_id": str(m.id), "content": m.content, "score": m.similarity} for m in res.memories], "count": len(res.memories), "task": task}

    def _handle_get_strategies(self, args: dict) -> dict:
        situation = str(args.get("situation", "")).strip()
        limit = min(int(args.get("limit", 3)), 10)
        if not situation:
            return {"strategies": [], "count": 0, "situation": situation}
        res = self.client.recall(situation, limit=limit, memory_types=[ApiMemoryType.STRATEGIC], include_partial=False)
        return {"strategies": [{"memory_id": str(m.id), "content": m.content, "score": m.similarity} for m in res.memories], "count": len(res.memories), "situation": situation}

    def _handle_create_goal(self, args: dict) -> dict:
        title = str(args.get("title", "")).strip()
        if not title:
            return {"error": "Missing title"}

        description = args.get("description")
        priority = str(args.get("priority") or ApiGoalPriority.QUEUED.value)
        source = str(args.get("source") or ApiGoalSource.USER_REQUEST.value)
        due_at_raw = args.get("due_at")

        due_at = None
        if isinstance(due_at_raw, str) and due_at_raw.strip():
            txt = due_at_raw.strip()
            if txt.endswith("Z"):
                txt = txt[:-1] + "+00:00"
            try:
                due_at = datetime.fromisoformat(txt)
            except Exception:
                due_at = None

        goal_id = self.client.create_goal(
            title,
            description=str(description) if isinstance(description, str) else None,
            source=source,
            priority=priority,
            due_at=due_at,
        )
        return {"goal_id": str(goal_id), "title": title, "priority": priority, "source": source, "due_at": due_at_raw}

    def _handle_queue_user_message(self, args: dict) -> dict:
        message = str(args.get("message", "")).strip()
        if not message:
            return {"error": "Missing message"}

        intent = args.get("intent")
        context = args.get("context")
        outbox_id = self.client.queue_user_message(
            message,
            intent=str(intent) if isinstance(intent, str) else None,
            context=context if isinstance(context, dict) else None,
        )
        return {"outbox_id": str(outbox_id), "queued": True}


# ============================================================================
# CONTEXT ENRICHMENT
# ============================================================================

class ContextEnricher:
    """
    Automatically enriches user prompts with relevant memories before
    sending to the LLM.
    """
    
    def __init__(self, db_config: dict, top_k: int = 5):
        self.db_config = db_config
        self.top_k = top_k
        self.client: CognitiveMemorySync | None = None
    
    def connect(self):
        """Establish DB connection via CognitiveMemorySync."""
        if self.client is not None:
            return
        dsn = (
            f"postgresql://{self.db_config.get('user', 'postgres')}:{self.db_config.get('password', 'password')}"
            f"@{self.db_config.get('host', 'localhost')}:{int(self.db_config.get('port', 43815))}"
            f"/{self.db_config.get('dbname', 'hexis_memory')}"
        )
        self.client = CognitiveMemorySync.connect(dsn, min_size=1, max_size=5)
    
    def enrich(self, user_message: str) -> dict:
        """
        Enrich a user message with relevant memories.
        
        Returns:
            {
                "original_message": str,
                "relevant_memories": list,
                "enriched_context": str
            }
        """
        self.connect()

        assert self.client is not None
        result = self.client.recall(user_message, limit=self.top_k, include_partial=False)
        memories = [
            {
                "memory_id": str(m.id),
                "content": m.content,
                "memory_type": m.type.value,
                "score": m.similarity,
                "source": m.source,
                "importance": m.importance,
                "trust_level": m.trust_level,
                "source_attribution": m.source_attribution,
            }
            for m in result.memories
        ]
        self.client.touch_memories([m.id for m in result.memories])
        
        # Format memories into context
        if memories:
            memory_context = self._format_memories(memories)
        else:
            memory_context = None
        
        return {
            "original_message": user_message,
            "relevant_memories": [dict(m) for m in memories],
            "enriched_context": memory_context
        }
    
    def _format_memories(self, memories: list) -> str:
        """Format memories into a context string for the LLM."""
        lines = ["[RELEVANT MEMORIES]"]
        
        for i, mem in enumerate(memories, 1):
            mem_type = mem['memory_type'].upper()
            content = mem['content']
            score = mem['score']
            
            lines.append(f"\n({i}) [{mem_type}] (relevance: {score:.2f})")
            lines.append(f"    {content}")
        
        lines.append("\n[END MEMORIES]")
        
        return "\n".join(lines)
    
    def close(self):
        """Close database connection."""
        if self.client is not None:
            self.client.close()
            self.client = None


# ============================================================================
# MEMORY FORMATION (Post-conversation)
# ============================================================================

class MemoryFormation:
    """
    Handles forming new memories from conversations.
    Called after each exchange to potentially store new information.
    """
    
    def __init__(self, db_config: dict, llm_client=None):
        self.db_config = db_config
        self.llm_client = llm_client
        self.client: CognitiveMemorySync | None = None
    
    def connect(self):
        """Establish DB connection via CognitiveMemorySync."""
        if self.client is not None:
            return
        dsn = (
            f"postgresql://{self.db_config.get('user', 'postgres')}:{self.db_config.get('password', 'password')}"
            f"@{self.db_config.get('host', 'localhost')}:{int(self.db_config.get('port', 43815))}"
            f"/{self.db_config.get('dbname', 'hexis_memory')}"
        )
        self.client = CognitiveMemorySync.connect(dsn, min_size=1, max_size=5)
    
    def should_form_memory(self, user_message: str, assistant_response: str) -> bool:
        """
        Determine if this exchange should be stored as a memory.
        Conversation turns are always worth recording as episodic memory; importance is graded in form_memory().
        """
        return True
    
    def form_memory(
        self, 
        user_message: str, 
        assistant_response: str,
        memory_type: str = 'episodic',
        importance: float = 0.5
    ) -> Optional[str]:
        """
        Form a new memory from a conversation exchange.
        
        Returns the memory ID if successful.
        """
        self.connect()
        assert self.client is not None
        
        # Create a combined memory content
        content = f"User: {user_message}\n\nAssistant: {assistant_response}"

        combined = (user_message + "\n" + assistant_response).lower()
        learning_signals = [
            "remember",
            "don't forget",
            "important",
            "note that",
            "my name is",
            "i prefer",
            "i like",
            "i don't like",
            "always",
            "never",
            "make sure",
            "keep in mind",
        ]
        if len(user_message) > 200 or len(assistant_response) > 500:
            importance = max(importance, 0.7)
        if any(signal in combined for signal in learning_signals):
            importance = max(importance, 0.8)
        importance = max(0.15, min(float(importance), 1.0))

        source_attribution = {
            "kind": "conversation",
            "ref": "conversation_turn",
            "label": "conversation turn",
            "observed_at": datetime.now(timezone.utc).isoformat(),
            "trust": 0.95,
        }
        trust_level = 0.95

        if memory_type == "semantic":
            mt = ApiMemoryType.SEMANTIC
            source_references: Any = [source_attribution]
        else:
            mt = ApiMemoryType.EPISODIC
            source_references = None

        mem_id = self.client.remember(
            content,
            type=mt,
            importance=float(importance),
            emotional_valence=0.0,
            context={"type": "conversation"},
            source_attribution=source_attribution,
            source_references=source_references,
            trust_level=trust_level if mt == ApiMemoryType.EPISODIC else None,
        )
        return str(mem_id) if mem_id else None
    
    def close(self):
        """Close database connection."""
        if self.client is not None:
            self.client.close()
            self.client = None


# ============================================================================
# EXPORTS
# ============================================================================

def get_tool_definitions() -> list:
    """Return tool definitions for function calling (conversation loop)."""
    return [t for t in MEMORY_TOOLS if t.get("function", {}).get("name") in _API_TOOL_NAMES]


def create_tool_handler(db_config: dict) -> MemoryToolHandler:
    """Create a tool handler instance (API-backed by default)."""
    return ApiMemoryToolHandler(db_config)


def create_enricher(db_config: dict, top_k: int = 5) -> ContextEnricher:
    """Create a context enricher instance."""
    return ContextEnricher(db_config, top_k)


def create_memory_formation(db_config: dict) -> MemoryFormation:
    """Create a memory formation instance."""
    return MemoryFormation(db_config)


# ============================================================================
# CROSS-PARADIGM HELPER (Vector -> SQL -> Neighborhood)
# ============================================================================

_SQL_IDENT_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _require_sql_identifier(name: str) -> str:
    if not name or not _SQL_IDENT_RE.fullmatch(name):
        raise ValueError(f"Invalid SQL identifier: {name!r}")
    return name


def cross_join_query(
    db_config: dict,
    *,
    query_text: str,
    limit: int = 10,
    table: Optional[str] = None,
    where: Optional[dict] = None,
    join_key: Optional[str] = None,
    include_neighbors: bool = True,
    neighbor_limit: int = 10,
    include_neighbor_content: bool = True,
) -> list[dict]:
    """
    Vector -> SQL -> neighborhood helper.

    - Vector: uses `search_similar_memories(query_text, limit)` in Postgres.
    - SQL: optionally filters candidates by joining against `table` with simple equality predicates.
    - Neighborhood: optionally attaches cached neighbors from `memory_neighborhoods`.
    """
    if not query_text:
        return []

    conn = psycopg2.connect(
        host=db_config.get("host", "localhost"),
        port=db_config.get("port", 43815),
        dbname=db_config.get("dbname", "hexis_memory"),
        user=db_config.get("user", "postgres"),
        password=db_config.get("password", "password"),
    )
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                "SELECT * FROM search_similar_memories(%s, %s)",
                (query_text, int(limit)),
            )
            hits = list(cur.fetchall() or [])
            if not hits:
                return []

            candidate_ids = [str(h["memory_id"]) for h in hits]

            relational_rows: dict[str, dict] = {}
            filtered_hits = hits
            if table:
                table_name = _require_sql_identifier(table)
                predicates = where or {}

                if join_key is None:
                    join_key = "id" if table_name == "memories" else "memory_id"
                join_col = _require_sql_identifier(join_key)

                clause_parts = [f"{join_col} = ANY(%s::uuid[])"]
                params: list[Any] = [candidate_ids]
                for key, val in predicates.items():
                    col = _require_sql_identifier(str(key))
                    clause_parts.append(f"{col} = %s")
                    params.append(val)

                sql = f"SELECT * FROM {table_name} WHERE " + " AND ".join(clause_parts)
                cur.execute(sql, params)
                rows = list(cur.fetchall() or [])
                for row in rows:
                    row_key = row.get(join_col)
                    if row_key is not None:
                        relational_rows[str(row_key)] = dict(row)

                filtered_hits = [h for h in hits if str(h["memory_id"]) in relational_rows]

            if not filtered_hits:
                return []

            neighbors_by_id: dict[str, list[dict]] = {}
            if include_neighbors:
                hit_ids = [str(h["memory_id"]) for h in filtered_hits]
                cur.execute(
                    """
                    SELECT memory_id, neighbors
                    FROM memory_neighborhoods
                    WHERE memory_id = ANY(%s::uuid[])
                    """,
                    (hit_ids,),
                )
                neighbor_rows = list(cur.fetchall() or [])

                neighbor_ids: set[str] = set()
                for row in neighbor_rows:
                    mem_id = str(row["memory_id"])
                    neighbors = row.get("neighbors") or {}
                    if isinstance(neighbors, str):
                        try:
                            neighbors = json.loads(neighbors)
                        except Exception:
                            neighbors = {}

                    pairs: list[tuple[str, float]] = []
                    if isinstance(neighbors, dict):
                        for k, v in neighbors.items():
                            try:
                                pairs.append((str(k), float(v)))
                            except Exception:
                                continue
                    pairs.sort(key=lambda kv: kv[1], reverse=True)

                    sliced = pairs[: max(0, int(neighbor_limit))]
                    neighbors_by_id[mem_id] = [{"memory_id": mid, "weight": w} for mid, w in sliced]
                    neighbor_ids.update(mid for mid, _ in sliced)

                neighbor_content: dict[str, dict] = {}
                if include_neighbor_content and neighbor_ids:
                    cur.execute(
                        """
                        SELECT id, type, content, importance
                        FROM memories
                        WHERE id = ANY(%s::uuid[])
                        """,
                        (list(neighbor_ids),),
                    )
                    for row in cur.fetchall() or []:
                        neighbor_content[str(row["id"])] = dict(row)

                    for mem_id, neigh_list in neighbors_by_id.items():
                        for n in neigh_list:
                            n["memory"] = neighbor_content.get(n["memory_id"])

            results: list[dict] = []
            for hit in filtered_hits:
                item = dict(hit)
                mem_id = str(hit["memory_id"])
                if relational_rows:
                    item["sql_row"] = relational_rows.get(mem_id)
                if include_neighbors:
                    item["neighbors"] = neighbors_by_id.get(mem_id, [])
                results.append(item)

            return results
    finally:
        conn.close()
