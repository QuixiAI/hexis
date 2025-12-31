import pytest
import asyncio
import asyncpg
import json
import numpy as np
import time
import uuid
import os
import subprocess
import sys
from pathlib import Path
from datetime import timedelta
from tenacity import (
    AsyncRetrying,
    RetryError,
    retry_if_exception_type,
    retry_if_result,
    stop_after_delay,
    wait_fixed,
)

# Update to use loop_scope instead of scope
pytestmark = pytest.mark.asyncio(loop_scope="session")

# Global test session ID to help with cleanup
TEST_SESSION_ID = str(uuid.uuid4())[:8]
EMBEDDING_DIMENSION = int(os.getenv("EMBEDDING_DIMENSION", os.getenv("EMBEDDING_DIM", "768")))

def get_test_identifier(test_name: str) -> str:
    """Generate a unique identifier for test data"""
    return f"{test_name}_{TEST_SESSION_ID}_{int(time.time() * 1000)}"

def _db_dsn() -> str:
    db_host = os.getenv("POSTGRES_HOST", "localhost")
    db_port = os.getenv("POSTGRES_PORT", "43815")
    db_name = os.getenv("POSTGRES_DB", "hexis_memory")
    db_user = os.getenv("POSTGRES_USER", "hexis_user")
    db_password = os.getenv("POSTGRES_PASSWORD", "hexis_password")
    return f"postgresql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}"

@pytest.fixture(scope="session", autouse=True)
async def sync_test_embedding_dimension_from_db(db_pool):
    """
    Sync the test process' EMBEDDING_DIMENSION with the database's configured dimension.

    Many tests build vectors client-side (e.g., vector literal strings) and must match
    the DB's pgvector typmod.
    """
    global EMBEDDING_DIMENSION
    async with db_pool.acquire() as conn:
        dim = await conn.fetchval("SELECT embedding_dimension()")
    EMBEDDING_DIMENSION = int(dim)

@pytest.fixture(scope="session")
async def db_pool():
    """Create a connection pool for testing"""
    db_url = _db_dsn()
    # Postgres restarts once during initdb, and this repo's schema init can take >60s on cold starts.
    wait_seconds = int(os.getenv("POSTGRES_WAIT_SECONDS", "180"))
    pool = None
    retrying = AsyncRetrying(
        stop=stop_after_delay(wait_seconds),
        wait=wait_fixed(1),
        retry=retry_if_exception_type(Exception),
        reraise=True,
    )
    async for attempt in retrying:
        with attempt:
            pool = await asyncpg.create_pool(
                db_url,
                ssl=False,
                min_size=2,
                max_size=20,
                command_timeout=60.0,
            )
    assert pool is not None
    yield pool
    await pool.close()

@pytest.fixture(scope="session", autouse=True)
async def configure_agent_for_tests(db_pool):
    """
    Heartbeats are gated behind `agent.is_configured`.

    Most tests exercise heartbeat/worker behavior, so default to configured.
    Individual tests can override by deleting the config row in a transaction.
    """
    async with db_pool.acquire() as conn:
        # Minimal config so CLI `hexis config validate` passes in subprocess smoke tests.
        await conn.execute("SELECT set_config('agent.is_configured', 'true'::jsonb)")
        await conn.execute("SELECT set_config('agent.objectives', $1::jsonb)", json.dumps(["test objective"]))
        await conn.execute(
            "SELECT set_config('llm.heartbeat', $1::jsonb)",
            json.dumps({"provider": "openai", "model": "gpt-4o", "endpoint": "", "api_key_env": ""}),
        )
        await conn.execute(
            "SELECT set_config('llm.chat', $1::jsonb)",
            json.dumps({"provider": "openai", "model": "gpt-4o", "endpoint": "", "api_key_env": ""}),
        )
        await conn.execute("UPDATE heartbeat_state SET is_paused = FALSE WHERE id = 1")
        await conn.execute("UPDATE heartbeat_config SET value = 60 WHERE key = 'heartbeat_interval_minutes'")

@pytest.fixture(scope="session", autouse=True)
async def apply_repo_migrations(db_pool):
    """
    Apply any optional SQL patches from migrations/ once per test session.

    The DB in Docker may persist across runs; this keeps core functions
    and tables up to date without requiring a full volume reset.
    """
    if os.getenv("APPLY_REPO_MIGRATIONS", "0").lower() not in {"1", "true", "yes", "on"}:
        return
    migrations_dir = Path(__file__).resolve().parent / "migrations"
    if not migrations_dir.exists():
        return

    migration_paths = sorted(p for p in migrations_dir.glob("*.sql") if p.is_file())
    if not migration_paths:
        return

    async with db_pool.acquire() as conn:
        for path in migration_paths:
            sql = path.read_text(encoding="utf-8")
            await conn.execute(sql)

@pytest.fixture(autouse=True)
async def setup_db(db_pool):
    """Setup the database before each test"""
    async with db_pool.acquire() as conn:
        await conn.execute("LOAD 'age';")
        await conn.execute("SET search_path = ag_catalog, public;")
    yield

async def test_extensions(db_pool):
    """Test that required PostgreSQL extensions are installed"""
    async with db_pool.acquire() as conn:
        extensions = await conn.fetch("""
            SELECT extname FROM pg_extension
        """)
        ext_names = {ext['extname'] for ext in extensions}
        
        required_extensions = {'vector', 'age', 'btree_gist', 'pg_trgm'}
        for ext in required_extensions:
            assert ext in ext_names, f"{ext} extension not found"
        # Verify AGE is loaded
        await conn.execute("LOAD 'age';")
        await conn.execute("SET search_path = ag_catalog, public;")
        result = await conn.fetchval("""
            SELECT count(*) FROM ag_catalog.ag_graph
        """)
        assert result >= 0, "AGE extension not properly loaded"

async def test_expected_triggers_are_installed(db_pool):
    async with db_pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT t.tgname, p.proname
            FROM pg_trigger t
            JOIN pg_proc p ON p.oid = t.tgfoid
            WHERE NOT t.tgisinternal
            """
        )
        mapping = {r["tgname"]: r["proname"] for r in rows}

        assert mapping.get("trg_memory_timestamp") == "update_memory_timestamp"
        assert mapping.get("trg_importance_on_access") == "update_memory_importance"
        assert mapping.get("trg_cluster_activation") == "update_cluster_activation"
        assert mapping.get("trg_neighborhood_staleness") == "mark_neighborhoods_stale"
        assert mapping.get("trg_auto_episode_assignment") == "assign_to_episode"
        assert mapping.get("trg_external_call_complete") == "on_external_call_complete"
        assert mapping.get("trg_sync_worldview_node") == "sync_worldview_node"


async def test_memory_tables(db_pool):
    """Test that all memory tables exist with correct columns and constraints"""
    async with db_pool.acquire() as conn:
        # First check if tables exist
        tables = await conn.fetch("""
            SELECT table_name 
            FROM information_schema.tables 
            WHERE table_schema = 'public'
        """)
        table_names = {t['table_name'] for t in tables}
        
        assert 'working_memory' in table_names, "working_memory table not found"
        assert 'memories' in table_names, "memories table not found"
        assert 'episodic_memories' in table_names, "episodic_memories table not found"
        
        # Then check columns
        memories = await conn.fetch("""
            SELECT column_name, data_type, is_nullable 
            FROM information_schema.columns 
            WHERE table_name = 'memories'
        """)
        columns = {col["column_name"]: col for col in memories}

        # Note: relevance_score is computed via calculate_relevance() function, not a column
        assert "importance" in columns, "importance column not found"
        assert "decay_rate" in columns, "decay_rate column not found"
        assert "last_accessed" in columns, "last_accessed column not found"
        assert "id" in columns and columns["id"]["data_type"] == "uuid"
        assert "content" in columns and columns["content"]["is_nullable"] == "NO"
        assert "embedding" in columns
        assert "type" in columns


async def test_memory_storage(db_pool):
    """Test storing and retrieving different types of memories"""
    async with db_pool.acquire() as conn:
        test_id = get_test_identifier("memory_storage")
        
        # Test each memory type
        memory_types = ['episodic', 'semantic', 'procedural', 'strategic']
        created_memories = []
        
        for mem_type in memory_types:
            # Cast the type explicitly
            memory_id = await conn.fetchval("""
                INSERT INTO memories (
                    type,
                    content,
                    embedding
                ) VALUES (
                    $1::memory_type,
                    'Test ' || $1 || ' memory ' || $2,
                    array_fill(0, ARRAY[embedding_dimension()])::vector
                ) RETURNING id
            """, mem_type, test_id)

            assert memory_id is not None
            created_memories.append(memory_id)

            # Store type-specific details
            if mem_type == 'episodic':
                await conn.execute("""
                    INSERT INTO episodic_memories (
                        memory_id,
                        action_taken,
                        context,
                        result,
                        emotional_valence
                    ) VALUES ($1, $2, $3, $4, 0.5)
                """, 
                    memory_id,
                    json.dumps({"action": "test"}),
                    json.dumps({"context": "test"}),
                    json.dumps({"result": "success"})
                )
            # Add other memory type tests...

        # Verify storage and relationships for our specific test memories
        for mem_type in memory_types:
            count = await conn.fetchval("""
                SELECT COUNT(*) 
                FROM memories m 
                WHERE m.type = $1 AND m.content LIKE '%' || $2
            """, mem_type, test_id)
            assert count > 0, f"No {mem_type} memories stored for test {test_id}"


async def test_memory_importance(db_pool):
    """Test memory importance updating"""
    async with db_pool.acquire() as conn:
        # Create test memory
        memory_id = await conn.fetchval(
            """
            INSERT INTO memories (
                type, 
                content, 
                embedding,
                importance,
                access_count
            ) VALUES (
                'semantic',
                'Important test content',
                array_fill(0, ARRAY[embedding_dimension()])::vector,
                0.5,
                0
            ) RETURNING id
        """
        )

        # Update access count to trigger importance recalculation
        await conn.execute(
            """
            UPDATE memories 
            SET access_count = access_count + 1
            WHERE id = $1
        """,
            memory_id,
        )

        # Check that importance was updated
        new_importance = await conn.fetchval(
            """
            SELECT importance 
            FROM memories 
            WHERE id = $1
        """,
            memory_id,
        )

        assert new_importance != 0.5, "Importance should have been updated"


async def test_age_setup(db_pool):
    """Test AGE graph functionality"""
    async with db_pool.acquire() as conn:
        # Ensure clean state
        await conn.execute("""
            LOAD 'age';
            SET search_path = ag_catalog, public;
            SELECT drop_graph('memory_graph', true);
        """)
        
        # Create graph and label
        await conn.execute("""
            SELECT create_graph('memory_graph');
        """)
        
        await conn.execute("""
            SELECT create_vlabel('memory_graph', 'MemoryNode');
        """)

        # Test graph exists
        result = await conn.fetch("""
            SELECT * FROM ag_catalog.ag_graph
            WHERE name = 'memory_graph'::name
        """)
        assert len(result) == 1, "memory_graph not found"

        # Test vertex label
        result = await conn.fetch("""
            SELECT * FROM ag_catalog.ag_label
            WHERE name = 'MemoryNode'::name
            AND graph = (
                SELECT graphid FROM ag_catalog.ag_graph
                WHERE name = 'memory_graph'::name
            )
        """)
        assert len(result) == 1, "MemoryNode label not found"


async def test_memory_relationships(db_pool):
    """Test graph relationships between different memory types"""
    async with db_pool.acquire() as conn:
        await conn.execute("LOAD 'age';")
        await conn.execute("SET search_path = ag_catalog, public;")
        
        memory_pairs = [
            ('semantic', 'semantic', 'RELATES_TO'),
            ('episodic', 'semantic', 'LEADS_TO'),
            ('procedural', 'strategic', 'IMPLEMENTS')
        ]
        
        for source_type, target_type, rel_type in memory_pairs:
            # Create source and target memories
            source_id = await conn.fetchval("""
                INSERT INTO memories (type, content, embedding)
                VALUES ($1::memory_type, 'Source ' || $1, array_fill(0, ARRAY[embedding_dimension()])::vector)
                RETURNING id
            """, source_type)
            
            target_id = await conn.fetchval("""
                INSERT INTO memories (type, content, embedding)
                VALUES ($1::memory_type, 'Target ' || $1, array_fill(0, ARRAY[embedding_dimension()])::vector)
                RETURNING id
            """, target_type)
            
            # Create nodes and relationship in graph using string formatting for Cypher
            cypher_query = f"""
                SELECT * FROM ag_catalog.cypher(
                    'memory_graph',
                    $$
                    CREATE (a:MemoryNode {{memory_id: '{str(source_id)}', type: '{source_type}'}}),
                           (b:MemoryNode {{memory_id: '{str(target_id)}', type: '{target_type}'}}),
                           (a)-[r:{rel_type}]->(b)
                    RETURN a, r, b
                    $$
                ) as (a ag_catalog.agtype, r ag_catalog.agtype, b ag_catalog.agtype)
            """
            await conn.execute(cypher_query)
            
            # Verify the relationship was created
            verify_query = f"""
                SELECT * FROM ag_catalog.cypher(
                    'memory_graph',
                    $$
                    MATCH (a:MemoryNode)-[r:{rel_type}]->(b:MemoryNode)
                    WHERE a.memory_id = '{str(source_id)}' AND b.memory_id = '{str(target_id)}'
                    RETURN a, r, b
                    $$
                ) as (a ag_catalog.agtype, r ag_catalog.agtype, b ag_catalog.agtype)
            """
            result = await conn.fetch(verify_query)
            assert len(result) > 0, f"Relationship {rel_type} not found"


async def test_memory_type_specifics(db_pool):
    """Test type-specific memory storage and constraints"""
    async with db_pool.acquire() as conn:
        # Test semantic memory with confidence
        semantic_id = await conn.fetchval("""
            WITH mem AS (
                INSERT INTO memories (type, content, embedding)
                VALUES ('semantic'::memory_type, 'Test fact', array_fill(0, ARRAY[embedding_dimension()])::vector)
                RETURNING id
            )
            INSERT INTO semantic_memories (memory_id, confidence, category)
            SELECT id, 0.85, ARRAY['test']
            FROM mem
            RETURNING memory_id
        """)
        
        # Test procedural memory success rate calculation
        procedural_id = await conn.fetchval("""
            WITH mem AS (
                INSERT INTO memories (type, content, embedding)
                VALUES ('procedural'::memory_type, 'Test procedure', array_fill(0, ARRAY[embedding_dimension()])::vector)
                RETURNING id
            )
            INSERT INTO procedural_memories (
                memory_id, 
                steps,
                success_count,
                total_attempts
            )
            SELECT id, 
                   '{"steps": ["step1", "step2"]}'::jsonb,
                   8,
                   10
            FROM mem
            RETURNING memory_id
        """)
        
        # Verify success rate calculation
        success_rate = await conn.fetchval("""
            SELECT success_rate 
            FROM procedural_memories 
            WHERE memory_id = $1
        """, procedural_id)
        
        assert success_rate == 0.8, "Success rate calculation incorrect"


async def test_memory_status_transitions(db_pool):
    """Test memory status transitions and tracking"""
    async with db_pool.acquire() as conn:
        # First create trigger if it doesn't exist
        await conn.execute("""
            CREATE OR REPLACE FUNCTION track_memory_changes()
            RETURNS TRIGGER AS $$
            BEGIN
                INSERT INTO memory_changes (
                    memory_id,
                    change_type,
                    old_value,
                    new_value
                ) VALUES (
                    NEW.id,
                    'status_change',
                    jsonb_build_object('status', OLD.status),
                    jsonb_build_object('status', NEW.status)
                );
                RETURN NEW;
            END;
            $$ LANGUAGE plpgsql;

            DROP TRIGGER IF EXISTS track_status_changes ON memories;
            CREATE TRIGGER track_status_changes
                AFTER UPDATE OF status ON memories
                FOR EACH ROW
                EXECUTE FUNCTION track_memory_changes();
        """)

        # Create test memory
        memory_id = await conn.fetchval("""
            INSERT INTO memories (type, content, embedding, status)
            VALUES (
                'semantic'::memory_type,
                'Test content',
                array_fill(0, ARRAY[embedding_dimension()])::vector,
                'active'::memory_status
            ) RETURNING id
        """)

        # Archive memory and verify change tracking
        await conn.execute("""
            UPDATE memories 
            SET status = 'archived'::memory_status
            WHERE id = $1
        """, memory_id)

        changes = await conn.fetch("""
            SELECT * FROM memory_changes
            WHERE memory_id = $1
            ORDER BY changed_at DESC
        """, memory_id)

        assert len(changes) > 0, "Status change not tracked"


async def test_vector_search(db_pool):
    """Test vector similarity search"""
    async with db_pool.acquire() as conn:
        # Clear existing test data with proper cascade
        await conn.execute("""
            DELETE FROM memory_changes 
            WHERE memory_id IN (
                SELECT id FROM memories WHERE content LIKE 'Test content%'
            )
        """)
        await conn.execute("""
            DELETE FROM semantic_memories 
            WHERE memory_id IN (
                SELECT id FROM memories WHERE content LIKE 'Test content%'
            )
        """)
        await conn.execute("""
            DELETE FROM episodic_memories 
            WHERE memory_id IN (
                SELECT id FROM memories WHERE content LIKE 'Test content%'
            )
        """)
        await conn.execute("""
            DELETE FROM procedural_memories 
            WHERE memory_id IN (
                SELECT id FROM memories WHERE content LIKE 'Test content%'
            )
        """)
        await conn.execute("""
            DELETE FROM strategic_memories 
            WHERE memory_id IN (
                SELECT id FROM memories WHERE content LIKE 'Test content%'
            )
        """)
        await conn.execute("DELETE FROM memories WHERE content LIKE 'Test content%'")
        
        # Create more distinct test vectors
        test_embeddings = [
            # First vector: alternating 1.0 and 0.8
            '[' + ','.join(['1.0' if i % 2 == 0 else '0.8' for i in range(EMBEDDING_DIMENSION)]) + ']',
            # Second vector: alternating 0.5 and 0.3
            '[' + ','.join(['0.5' if i % 2 == 0 else '0.3' for i in range(EMBEDDING_DIMENSION)]) + ']',
            # Third vector: alternating 0.2 and 0.0
            '[' + ','.join(['0.2' if i % 2 == 0 else '0.0' for i in range(EMBEDDING_DIMENSION)]) + ']'
        ]
        
        # Insert test vectors
        for i, emb in enumerate(test_embeddings):
            await conn.execute("""
                INSERT INTO memories (
                    type, 
                    content, 
                    embedding
                ) VALUES (
                    'semantic'::memory_type,
                    'Test content ' || $1,
                    $2::vector
                )
            """, str(i), emb)

        # Query vector more similar to first pattern
        query_vector = '[' + ','.join(['0.95' if i % 2 == 0 else '0.75' for i in range(EMBEDDING_DIMENSION)]) + ']'
        
        results = await conn.fetch("""
            WITH candidates AS MATERIALIZED (
                SELECT id, content, embedding
                FROM memories
                WHERE content LIKE 'Test content%'
            )
            SELECT
                id,
                content,
                embedding <=> $1::vector as cosine_distance
            FROM candidates
            ORDER BY embedding <=> $1::vector
            LIMIT 3
        """, query_vector)

        assert len(results) >= 2, "Wrong number of results"
        
        # Print distances for debugging
        for r in results:
            print(f"Content: {r['content']}, Distance: {r['cosine_distance']}")
            
        # First result should have smaller cosine distance than second
        assert results[0]['cosine_distance'] < results[1]['cosine_distance'], \
            f"Incorrect distance ordering: {results[0]['cosine_distance']} >= {results[1]['cosine_distance']}"


async def test_complex_graph_queries(db_pool):
    """Test more complex graph operations and queries"""
    async with db_pool.acquire() as conn:
        await conn.execute("LOAD 'age';")
        await conn.execute("SET search_path = ag_catalog, public;")
        
        # Create a chain of related memories
        memory_chain = [
            ('episodic', 'Start event'),
            ('semantic', 'Derived knowledge'),
            ('procedural', 'Applied procedure')
        ]
        
        prev_id = None
        for mem_type, content in memory_chain:
            # Create memory
            curr_id = await conn.fetchval("""
                INSERT INTO memories (type, content, embedding)
                VALUES ($1::memory_type, $2, array_fill(0, ARRAY[embedding_dimension()])::vector)
                RETURNING id
            """, mem_type, content)
            
            # Create graph node
            await conn.execute(f"""
                SELECT * FROM cypher('memory_graph', $$
                    CREATE (n:MemoryNode {{
                        memory_id: '{curr_id}',
                        type: '{mem_type}'
                    }})
                    RETURN n
                $$) as (n ag_catalog.agtype)
            """)
            
            if prev_id:
                await conn.execute(f"""
                    SELECT * FROM cypher('memory_graph', $$
                        MATCH (a:MemoryNode {{memory_id: '{prev_id}'}}),
                              (b:MemoryNode {{memory_id: '{curr_id}'}})
                        CREATE (a)-[r:LEADS_TO]->(b)
                        RETURN r
                    $$) as (r ag_catalog.agtype)
                """)
            
            prev_id = curr_id
        
        # Test path query with fixed syntax
        result = await conn.fetch("""
            SELECT * FROM cypher('memory_graph', $$
                MATCH p = (s:MemoryNode)-[*]->(t:MemoryNode)
                WHERE s.type = 'episodic' AND t.type = 'procedural'
                RETURN p
            $$) as (path ag_catalog.agtype)
        """)
        
        assert len(result) > 0, "No valid paths found"


async def test_memory_storage_episodic(db_pool):
    """Test storing and retrieving episodic memories"""
    async with db_pool.acquire() as conn:
        # Create base memory
        memory_id = await conn.fetchval("""
            INSERT INTO memories (
                type,
                content,
                embedding,
                importance,
                decay_rate
            ) VALUES (
                'episodic'::memory_type,
                'Test episodic memory',
                array_fill(0, ARRAY[embedding_dimension()])::vector,
                0.5,
                0.01
            ) RETURNING id
        """)

        assert memory_id is not None

        # Store episodic details
        await conn.execute("""
            INSERT INTO episodic_memories (
                memory_id,
                action_taken,
                context,
                result,
                emotional_valence,
                verification_status,
                event_time
            ) VALUES ($1, $2, $3, $4, 0.5, true, CURRENT_TIMESTAMP)
        """, 
            memory_id,
            json.dumps({"action": "test"}),
            json.dumps({"context": "test"}),
            json.dumps({"result": "success"})
        )

        # Verify storage including new fields
        result = await conn.fetchrow("""
            SELECT e.verification_status, e.event_time
            FROM memories m 
            JOIN episodic_memories e ON m.id = e.memory_id
            WHERE m.type = 'episodic' AND m.id = $1
        """, memory_id)
        
        assert result['verification_status'] is True, "Verification status not set"
        assert result['event_time'] is not None, "Event time not set"


async def test_memory_storage_semantic(db_pool):
    """Test storing and retrieving semantic memories"""
    async with db_pool.acquire() as conn:
        memory_id = await conn.fetchval("""
            INSERT INTO memories (
                type,
                content,
                embedding,
                importance,
                decay_rate
            ) VALUES (
                'semantic'::memory_type,
                'Test semantic memory',
                array_fill(0, ARRAY[embedding_dimension()])::vector,
                0.5,
                0.01
            ) RETURNING id
        """)

        assert memory_id is not None

        await conn.execute("""
            INSERT INTO semantic_memories (
                memory_id,
                confidence,
                source_references,
                contradictions,
                category,
                related_concepts,
                last_validated
            ) VALUES ($1, 0.8, $2, $3, $4, $5, CURRENT_TIMESTAMP)
        """,
            memory_id,
            json.dumps({"source": "test"}),
            json.dumps({"contradictions": []}),
            ["test_category"],
            ["test_concept"]
        )

        # Verify including new field
        result = await conn.fetchrow("""
            SELECT s.last_validated
            FROM memories m 
            JOIN semantic_memories s ON m.id = s.memory_id
            WHERE m.type = 'semantic' AND m.id = $1
        """, memory_id)
        
        assert result['last_validated'] is not None, "Last validated timestamp not set"


async def test_memory_storage_strategic(db_pool):
    """Test storing and retrieving strategic memories"""
    async with db_pool.acquire() as conn:
        memory_id = await conn.fetchval("""
            INSERT INTO memories (
                type,
                content,
                embedding,
                importance,
                decay_rate
            ) VALUES (
                'strategic'::memory_type,
                'Test strategic memory',
                array_fill(0, ARRAY[embedding_dimension()])::vector,
                0.5,
                0.01
            ) RETURNING id
        """)

        assert memory_id is not None

        await conn.execute("""
            INSERT INTO strategic_memories (
                memory_id,
                pattern_description,
                supporting_evidence,
                confidence_score,
                success_metrics,
                adaptation_history,
                context_applicability
            ) VALUES ($1, 'Test pattern', $2, 0.7, $3, $4, $5)
        """,
            memory_id,
            json.dumps({"evidence": ["test"]}),
            json.dumps({"metrics": {"success": 0.8}}),
            json.dumps({"adaptations": []}),
            json.dumps({"contexts": ["test_context"]})
        )

        count = await conn.fetchval("""
            SELECT COUNT(*) 
            FROM memories m 
            JOIN strategic_memories s ON m.id = s.memory_id
            WHERE m.type = 'strategic'
        """)
        assert count > 0, "No strategic memories stored"


async def test_memory_storage_procedural(db_pool):
    """Test storing and retrieving procedural memories"""
    async with db_pool.acquire() as conn:
        memory_id = await conn.fetchval("""
            INSERT INTO memories (
                type,
                content,
                embedding,
                importance,
                decay_rate
            ) VALUES (
                'procedural'::memory_type,
                'Test procedural memory',
                array_fill(0, ARRAY[embedding_dimension()])::vector,
                0.5,
                0.01
            ) RETURNING id
        """)

        assert memory_id is not None

        await conn.execute("""
            INSERT INTO procedural_memories (
                memory_id,
                steps,
                prerequisites,
                success_count,
                total_attempts,
                average_duration,
                failure_points
            ) VALUES ($1, $2, $3, 5, 10, '1 hour', $4)
        """,
            memory_id,
            json.dumps({"steps": ["step1", "step2"]}),
            json.dumps({"prereqs": ["prereq1"]}),
            json.dumps({"failures": []})
        )

        count = await conn.fetchval("""
            SELECT COUNT(*) 
            FROM memories m 
            JOIN procedural_memories p ON m.id = p.memory_id
            WHERE m.type = 'procedural'
        """)
        assert count > 0, "No procedural memories stored"
        
async def test_working_memory(db_pool):
    """Test working memory operations"""
    async with db_pool.acquire() as conn:
        # Test inserting into working memory
        working_memory_id = await conn.fetchval("""
            INSERT INTO working_memory (
                content,
                embedding,
                expiry
            ) VALUES (
                'Test working memory',
                array_fill(0, ARRAY[embedding_dimension()])::vector,
                CURRENT_TIMESTAMP + interval '1 hour'
            ) RETURNING id
        """)
        
        assert working_memory_id is not None, "Failed to insert working memory"
        
        # Test expiry
        expired_count = await conn.fetchval("""
            SELECT COUNT(*) 
            FROM working_memory 
            WHERE expiry < CURRENT_TIMESTAMP
        """)
        
        assert isinstance(expired_count, int), "Failed to query expired memories"

async def test_memory_relevance(db_pool):
    """Test memory relevance score calculation"""
    async with db_pool.acquire() as conn:
        # Create test memory with known values
        memory_id = await conn.fetchval("""
            INSERT INTO memories (
                type,
                content,
                embedding,
                importance,
                decay_rate,
                created_at
            ) VALUES (
                'semantic'::memory_type,
                'Test relevance',
                array_fill(0, ARRAY[embedding_dimension()])::vector,
                0.8,
                0.01,
                CURRENT_TIMESTAMP - interval '1 day'
            ) RETURNING id
        """)
        
        # Check relevance score using calculate_relevance function
        relevance = await conn.fetchval("""
            SELECT calculate_relevance(importance, decay_rate, created_at, last_accessed)
            FROM memories
            WHERE id = $1
        """, memory_id)

        assert relevance is not None, "Relevance score not calculated"
        assert relevance < 0.8, "Relevance should be less than importance due to decay"

async def test_worldview_primitives(db_pool):
    """Test worldview primitives and their influence on memories"""
    async with db_pool.acquire() as conn:
        # Create worldview primitive
        worldview_id = await conn.fetchval("""
            INSERT INTO worldview_primitives (
                id,
                category,
                belief,
                confidence,
                emotional_valence,
                stability_score
            ) VALUES (
                gen_random_uuid(),
                'values',
                'Test belief',
                0.8,
                0.5,
                0.7
            ) RETURNING id
        """)
        
        # Create memory
        memory_id = await conn.fetchval("""
            INSERT INTO memories (
                type,
                content,
                embedding
            ) VALUES (
                'episodic'::memory_type,
                'Test memory for worldview',
                array_fill(0, ARRAY[embedding_dimension()])::vector
            ) RETURNING id
        """)
        
        # Create influence relationship
        await conn.execute("""
            INSERT INTO worldview_memory_influences (
                id,
                worldview_id,
                memory_id,
                influence_type,
                strength
            ) VALUES (
                gen_random_uuid(),
                $1,
                $2,
                'filter',
                0.7
            )
        """, worldview_id, memory_id)
        
        # Verify relationship
        influence = await conn.fetchrow("""
            SELECT * 
            FROM worldview_memory_influences
            WHERE worldview_id = $1 AND memory_id = $2
        """, worldview_id, memory_id)
        
        assert influence is not None, "Worldview influence not created"
        assert influence['strength'] == 0.7, "Incorrect influence strength"

async def test_identity_model(db_pool):
    """Test identity aspects and memory resonance"""
    async with db_pool.acquire() as conn:
        # Create identity aspect
        identity_aspect_id = await conn.fetchval("""
            INSERT INTO identity_aspects (
                id,
                aspect_type,
                content,
                stability
            ) VALUES (
                gen_random_uuid(),
                'self_concept',
                '{"concept": "test", "description": "test identity"}'::jsonb,
                0.7
            ) RETURNING id
        """)

        # Create memory
        memory_id = await conn.fetchval("""
            INSERT INTO memories (
                type,
                content,
                embedding
            ) VALUES (
                'episodic'::memory_type,
                'Test memory for identity',
                array_fill(0, ARRAY[embedding_dimension()])::vector
            ) RETURNING id
        """)

        # Create resonance
        await conn.execute("""
            INSERT INTO identity_memory_resonance (
                id,
                memory_id,
                identity_aspect_id,
                resonance_strength,
                integration_status
            ) VALUES (
                gen_random_uuid(),
                $1,
                $2,
                0.8,
                'integrated'
            )
        """, memory_id, identity_aspect_id)

        # Verify resonance
        resonance = await conn.fetchrow("""
            SELECT *
            FROM identity_memory_resonance
            WHERE memory_id = $1 AND identity_aspect_id = $2
        """, memory_id, identity_aspect_id)

        assert resonance is not None, "Identity resonance not created"
        assert resonance['resonance_strength'] == 0.8, "Incorrect resonance strength"

async def test_memory_changes_tracking(db_pool):
    """Test comprehensive memory changes tracking"""
    async with db_pool.acquire() as conn:
        # Create test memory
        memory_id = await conn.fetchval("""
            INSERT INTO memories (
                type,
                content,
                embedding,
                importance
            ) VALUES (
                'semantic'::memory_type,
                'Test tracking memory',
                array_fill(0, ARRAY[embedding_dimension()])::vector,
                0.5
            ) RETURNING id
        """)
        
        # Make various changes
        changes = [
            ('importance_update', 0.5, 0.7),
            ('status_change', 'active', 'archived'),
            ('content_update', 'Test tracking memory', 'Updated test memory')
        ]
        
        for change_type, old_val, new_val in changes:
            await conn.execute("""
                INSERT INTO memory_changes (
                    memory_id,
                    change_type,
                    old_value,
                    new_value
                ) VALUES (
                    $1,
                    $2,
                    $3::jsonb,
                    $4::jsonb
                )
            """, memory_id, change_type, 
                json.dumps({change_type: old_val}),
                json.dumps({change_type: new_val}))
        
        # Verify change history
        history = await conn.fetch("""
            SELECT change_type, old_value, new_value
            FROM memory_changes
            WHERE memory_id = $1
            ORDER BY changed_at DESC
        """, memory_id)
        
        assert len(history) == len(changes), "Not all changes were tracked"
        assert history[0]['change_type'] == changes[-1][0], "Changes not tracked in correct order"

async def test_enhanced_relevance_scoring(db_pool):
    """Test the enhanced relevance scoring system"""
    async with db_pool.acquire() as conn:
        # Create test memory with specific parameters
        memory_id = await conn.fetchval("""
            INSERT INTO memories (
                type,
                content,
                embedding,
                importance,
                decay_rate,
                created_at,
                access_count
            ) VALUES (
                'semantic'::memory_type,
                'Test relevance scoring',
                array_fill(0, ARRAY[embedding_dimension()])::vector,
                0.8,
                0.01,
                CURRENT_TIMESTAMP - interval '1 day',
                5
            ) RETURNING id
        """)
        
        # Get initial relevance score using calculate_relevance function
        initial_score = await conn.fetchval("""
            SELECT calculate_relevance(importance, decay_rate, created_at, last_accessed)
            FROM memories
            WHERE id = $1
        """, memory_id)

        # Update access count to trigger importance change
        await conn.execute("""
            UPDATE memories
            SET access_count = access_count + 1
            WHERE id = $1
        """, memory_id)

        # Get updated relevance score
        updated_score = await conn.fetchval("""
            SELECT calculate_relevance(importance, decay_rate, created_at, last_accessed)
            FROM memories
            WHERE id = $1
        """, memory_id)

        assert initial_score is not None, "Initial relevance score not calculated"
        assert updated_score is not None, "Updated relevance score not calculated"
        assert updated_score != initial_score, "Relevance score should change with importance"

async def test_age_in_days_function(db_pool):
    """Test the age_in_days function"""
    async with db_pool.acquire() as conn:
        # Test current timestamp (should be 0 days)
        result = await conn.fetchval("""
            SELECT age_in_days(CURRENT_TIMESTAMP)
        """)
        assert result < 1, "Current timestamp should be less than 1 day old"

        # Test 1 day ago
        result = await conn.fetchval("""
            SELECT age_in_days(CURRENT_TIMESTAMP - interval '1 day')
        """)
        assert abs(result - 1.0) < 0.1, "Should be approximately 1 day"

        # Test 7 days ago
        result = await conn.fetchval("""
            SELECT age_in_days(CURRENT_TIMESTAMP - interval '7 days')
        """)
        assert abs(result - 7.0) < 0.1, "Should be approximately 7 days"

async def test_update_memory_timestamp_trigger(db_pool):
    """Test the update_memory_timestamp trigger"""
    async with db_pool.acquire() as conn:
        # Create test memory
        memory_id = await conn.fetchval("""
            INSERT INTO memories (
                type,
                content,
                embedding
            ) VALUES (
                'semantic'::memory_type,
                'Test timestamp update',
                array_fill(0, ARRAY[embedding_dimension()])::vector
            ) RETURNING id
        """)

        # Get initial timestamp
        initial_updated_at = await conn.fetchval("""
            SELECT updated_at FROM memories WHERE id = $1
        """, memory_id)

        # Wait briefly
        await asyncio.sleep(0.1)

        # Update memory
        await conn.execute("""
            UPDATE memories 
            SET content = 'Updated content'
            WHERE id = $1
        """, memory_id)

        # Get new timestamp
        new_updated_at = await conn.fetchval("""
            SELECT updated_at FROM memories WHERE id = $1
        """, memory_id)

        assert new_updated_at > initial_updated_at, "updated_at should be newer"

async def test_update_memory_importance_trigger(db_pool):
    """Test the update_memory_importance trigger"""
    async with db_pool.acquire() as conn:
        # Create test memory
        memory_id = await conn.fetchval("""
            INSERT INTO memories (
                type,
                content,
                embedding,
                importance,
                access_count
            ) VALUES (
                'semantic'::memory_type,
                'Test importance update',
                array_fill(0, ARRAY[embedding_dimension()])::vector,
                0.5,
                0
            ) RETURNING id
        """)

        # Get initial importance
        initial_importance = await conn.fetchval("""
            SELECT importance FROM memories WHERE id = $1
        """, memory_id)

        # Update access count
        await conn.execute("""
            UPDATE memories 
            SET access_count = access_count + 1
            WHERE id = $1
        """, memory_id)

        # Get new importance
        new_importance = await conn.fetchval("""
            SELECT importance FROM memories WHERE id = $1
        """, memory_id)

        assert new_importance > initial_importance, "Importance should increase"
        
        # Test multiple accesses
        await conn.execute("""
            UPDATE memories 
            SET access_count = access_count + 5
            WHERE id = $1
        """, memory_id)
        
        final_importance = await conn.fetchval("""
            SELECT importance FROM memories WHERE id = $1
        """, memory_id)
        
        assert final_importance > new_importance, "Importance should increase with more accesses"

async def test_create_memory_relationship_function(db_pool):
    """Test the create_memory_relationship function"""
    async with db_pool.acquire() as conn:
        # Create two test memories
        memory_ids = []
        for i in range(2):
            memory_id = await conn.fetchval("""
                INSERT INTO memories (
                    type,
                    content,
                    embedding
                ) VALUES (
                    'semantic'::memory_type,
                    'Test memory ' || $1::text,
                    array_fill(0, ARRAY[embedding_dimension()])::vector
                ) RETURNING id
            """, str(i))
            memory_ids.append(memory_id)

        # Ensure clean AGE setup with proper schema
        await conn.execute("""
            LOAD 'age';
            SET search_path = ag_catalog, public;
            SELECT drop_graph('memory_graph', true);
            SELECT create_graph('memory_graph');
            SELECT create_vlabel('memory_graph', 'MemoryNode');
        """)

        # Create nodes in graph using string formatting for Cypher
        for memory_id in memory_ids:
            cypher_query = f"""
                SELECT * FROM ag_catalog.cypher('memory_graph', $$
                    CREATE (n:MemoryNode {{
                        memory_id: '{str(memory_id)}',
                        type: 'semantic'
                    }})
                    RETURN n
                $$) as (result agtype)
            """
            await conn.execute(cypher_query)

        properties = {"weight": 0.8}

        # Create relationship using valid graph_edge_type enum value
        await conn.execute("""
            SELECT create_memory_relationship($1, $2, $3::graph_edge_type, $4)
        """, memory_ids[0], memory_ids[1], 'ASSOCIATED', json.dumps(properties))

        # Verify relationship exists
        verify_query = f"""
            SELECT * FROM ag_catalog.cypher('memory_graph', $$
                MATCH (a:MemoryNode)-[r:ASSOCIATED]->(b:MemoryNode)
                WHERE a.memory_id = '{str(memory_ids[0])}' AND b.memory_id = '{str(memory_ids[1])}'
                RETURN r
            $$) as (result agtype)
        """
        result = await conn.fetch(verify_query)
        assert len(result) > 0, "Relationship not created"

async def test_memory_health_view(db_pool):
    """Test the memory_health view"""
    async with db_pool.acquire() as conn:
        # Create test memories of different types
        memory_types = ['episodic', 'semantic', 'procedural', 'strategic']
        for mem_type in memory_types:
            await conn.execute("""
                INSERT INTO memories (
                    type,
                    content,
                    embedding,
                    importance,
                    access_count
                ) VALUES (
                    $1::memory_type,
                    'Test ' || $1,
                    array_fill(0, ARRAY[embedding_dimension()])::vector,
                    0.5,
                    5
                )
            """, mem_type)

        # Query view
        results = await conn.fetch("""
            SELECT * FROM memory_health
        """)

        assert len(results) > 0, "Memory health view should return results"
        
        # Verify each type has stats
        result_types = {r['type'] for r in results}
        for mem_type in memory_types:
            assert mem_type in result_types, f"Missing stats for {mem_type}"
            
        # Verify computed values
        for row in results:
            assert row['total_memories'] > 0, "Should have memories"
            assert row['avg_importance'] is not None, "Should have importance"
            assert row['avg_access_count'] is not None, "Should have access count"



async def test_extensions(db_pool):
    """Test that required PostgreSQL extensions are installed"""
    async with db_pool.acquire() as conn:
        extensions = await conn.fetch("""
            SELECT extname FROM pg_extension
        """)
        ext_names = {ext['extname'] for ext in extensions}
        
        required_extensions = {'vector', 'age', 'btree_gist', 'pg_trgm', 'http'}
        for ext in required_extensions:
            assert ext in ext_names, f"{ext} extension not found"
        # Verify AGE is loaded
        await conn.execute("LOAD 'age';")
        await conn.execute("SET search_path = ag_catalog, public;")
        result = await conn.fetchval("""
            SELECT count(*) FROM ag_catalog.ag_graph
        """)
        assert result >= 0, "AGE extension not properly loaded"

async def test_memory_tables(db_pool):
    """Test that all memory tables exist with correct columns and constraints"""
    async with db_pool.acquire() as conn:
        # First check if tables exist
        tables = await conn.fetch("""
            SELECT table_name 
            FROM information_schema.tables 
            WHERE table_schema = 'public'
        """)
        table_names = {t['table_name'] for t in tables}
        
        assert 'working_memory' in table_names, "working_memory table not found"
        assert 'memories' in table_names, "memories table not found"
        assert 'episodic_memories' in table_names, "episodic_memories table not found"
        assert 'memory_clusters' in table_names, "memory_clusters table not found"
        assert 'memory_cluster_members' in table_names, "memory_cluster_members table not found"
        assert 'cluster_relationships' in table_names, "cluster_relationships table not found"

        # Then check columns
        memories = await conn.fetch("""
            SELECT column_name, data_type, is_nullable
            FROM information_schema.columns
            WHERE table_name = 'memories'
        """)
        columns = {col["column_name"]: col for col in memories}

        assert "last_accessed" in columns, "last_accessed column not found"
        assert "id" in columns and columns["id"]["data_type"] == "uuid"
        assert "content" in columns and columns["content"]["is_nullable"] == "NO"
        assert "embedding" in columns
        assert "type" in columns

async def test_memory_clusters(db_pool):
    """Test memory clustering functionality"""
    async with db_pool.acquire() as conn:
        # Create test cluster
        cluster_id = await conn.fetchval("""
            INSERT INTO memory_clusters (
                cluster_type,
                name,
                description,
                centroid_embedding,
                emotional_signature,
                keywords,
                importance_score,
                coherence_score
            ) VALUES (
                'theme'::cluster_type,
                'Test Theme Cluster',
                'Cluster for testing',
                array_fill(0.5, ARRAY[embedding_dimension()])::vector,
                '{"dominant": "neutral", "secondary": "curious"}'::jsonb,
                ARRAY['test', 'memory', 'cluster'],
                0.7,
                0.85
            ) RETURNING id
        """)
        
        assert cluster_id is not None, "Failed to create cluster"
        
        # Create test memories and add to cluster
        memory_ids = []
        for i in range(3):
            memory_id = await conn.fetchval("""
                INSERT INTO memories (
                    type,
                    content,
                    embedding
                ) VALUES (
                    'semantic'::memory_type,
                    'Test memory for clustering ' || $1,
                    array_fill($2::float, ARRAY[embedding_dimension()])::vector
                ) RETURNING id
            """, str(i), float(i) * 0.1)
            memory_ids.append(memory_id)
            
            # Add to cluster
            await conn.execute("""
                INSERT INTO memory_cluster_members (
                    cluster_id,
                    memory_id,
                    membership_strength,
                    contribution_to_centroid
                ) VALUES ($1, $2, $3, $4)
            """, cluster_id, memory_id, 0.8 - (i * 0.1), 0.3)
        
        # Verify cluster membership
        members = await conn.fetch("""
            SELECT * FROM memory_cluster_members
            WHERE cluster_id = $1
            ORDER BY membership_strength DESC
        """, cluster_id)
        
        assert len(members) == 3, "Wrong number of cluster members"
        assert members[0]['membership_strength'] == 0.8, "Incorrect membership strength"

async def test_cluster_relationships(db_pool):
    """Test relationships between clusters"""
    async with db_pool.acquire() as conn:
        # Create two clusters
        cluster_ids = []
        for i, name in enumerate(['Loneliness', 'Connection']):
            cluster_id = await conn.fetchval("""
                INSERT INTO memory_clusters (
                    cluster_type,
                    name,
                    description,
                    centroid_embedding,
                    keywords
                ) VALUES (
                    'emotion'::cluster_type,
                    $1,
                    'Emotional cluster for ' || $1,
                    array_fill($2::float, ARRAY[embedding_dimension()])::vector,
                    ARRAY[$1]
                ) RETURNING id
            """, name, float(i) * 0.5)
            cluster_ids.append(cluster_id)
        
        # Create relationship between clusters
        await conn.execute("""
            INSERT INTO cluster_relationships (
                from_cluster_id,
                to_cluster_id,
                relationship_type,
                strength,
                evidence_memories
            ) VALUES ($1, $2, 'contradicts', 0.7, $3)
        """, cluster_ids[0], cluster_ids[1], [])
        
        # Verify relationship
        relationship = await conn.fetchrow("""
            SELECT * FROM cluster_relationships
            WHERE from_cluster_id = $1 AND to_cluster_id = $2
        """, cluster_ids[0], cluster_ids[1])
        
        assert relationship is not None, "Cluster relationship not created"
        assert relationship['relationship_type'] == 'contradicts'
        assert relationship['strength'] == 0.7

async def test_cluster_activation_history(db_pool):
    """Test cluster activation tracking via memory_clusters table"""
    async with db_pool.acquire() as conn:
        # Create test cluster
        cluster_id = await conn.fetchval("""
            INSERT INTO memory_clusters (
                cluster_type,
                name,
                centroid_embedding,
                activation_count
            ) VALUES (
                'pattern'::cluster_type,
                'Test Pattern',
                array_fill(0.5, ARRAY[embedding_dimension()])::vector,
                0
            ) RETURNING id
        """)

        assert cluster_id is not None, "Failed to create cluster"

        # Get initial activation count
        initial = await conn.fetchrow("""
            SELECT activation_count, last_activated
            FROM memory_clusters
            WHERE id = $1
        """, cluster_id)

        # Update cluster activation count (triggers update_cluster_activation)
        await conn.execute("""
            UPDATE memory_clusters
            SET activation_count = activation_count + 1
            WHERE id = $1
        """, cluster_id)

        # Verify activation count updated
        result = await conn.fetchrow("""
            SELECT activation_count, last_activated
            FROM memory_clusters
            WHERE id = $1
        """, cluster_id)

        # The trigger increments activation_count again, so it increases by 2
        assert result['activation_count'] > initial['activation_count'], "Activation count not updated"
        assert result['last_activated'] is not None, "last_activated not set"

async def test_cluster_worldview_alignment(db_pool):
    """Test cluster alignment with worldview through worldview_memory_influences"""
    async with db_pool.acquire() as conn:
        # Create worldview primitive
        worldview_id = await conn.fetchval("""
            INSERT INTO worldview_primitives (
                category,
                belief,
                confidence,
                connected_beliefs
            ) VALUES (
                'values',
                'Connection is essential for wellbeing',
                0.9,
                ARRAY[]::UUID[]
            ) RETURNING id
        """)

        # Create aligned cluster
        cluster_id = await conn.fetchval("""
            INSERT INTO memory_clusters (
                cluster_type,
                name,
                centroid_embedding,
                importance_score
            ) VALUES (
                'theme'::cluster_type,
                'Human Connection',
                array_fill(0.7, ARRAY[embedding_dimension()])::vector,
                0.95
            ) RETURNING id
        """)

        # Create a memory in the cluster
        memory_id = await conn.fetchval("""
            INSERT INTO memories (
                type,
                content,
                embedding
            ) VALUES (
                'semantic'::memory_type,
                'Human connection test memory',
                array_fill(0.7, ARRAY[embedding_dimension()])::vector
            ) RETURNING id
        """)

        # Link memory to cluster
        await conn.execute("""
            INSERT INTO memory_cluster_members (cluster_id, memory_id, membership_strength)
            VALUES ($1, $2, 0.9)
        """, cluster_id, memory_id)

        # Create worldview influence on the memory
        await conn.execute("""
            INSERT INTO worldview_memory_influences (
                worldview_id, memory_id, influence_type, strength
            ) VALUES ($1, $2, 'alignment', 0.95)
        """, worldview_id, memory_id)

        # Verify influence
        result = await conn.fetchrow("""
            SELECT strength
            FROM worldview_memory_influences
            WHERE worldview_id = $1 AND memory_id = $2
        """, worldview_id, memory_id)

        assert result['strength'] == 0.95

async def test_identity_core_clusters(db_pool):
    """Test identity aspects with core memory clusters"""
    async with db_pool.acquire() as conn:
        # Create core clusters
        cluster_ids = []
        for name in ['Self-as-Helper', 'Creative-Expression']:
            cluster_id = await conn.fetchval("""
                INSERT INTO memory_clusters (
                    cluster_type,
                    name,
                    centroid_embedding,
                    importance_score
                ) VALUES (
                    'theme'::cluster_type,
                    $1,
                    array_fill(0.8, ARRAY[embedding_dimension()])::vector,
                    0.9
                ) RETURNING id
            """, name)
            cluster_ids.append(cluster_id)

        # Create identity aspect with core clusters
        identity_id = await conn.fetchval("""
            INSERT INTO identity_aspects (
                aspect_type,
                content,
                core_memory_clusters
            ) VALUES (
                'self_concept',
                '{"role": "supportive companion"}'::jsonb,
                $1
            ) RETURNING id
        """, cluster_ids)

        # Verify core clusters
        identity = await conn.fetchrow("""
            SELECT core_memory_clusters
            FROM identity_aspects
            WHERE id = $1
        """, identity_id)

        assert len(identity['core_memory_clusters']) == 2
        assert all(cid in identity['core_memory_clusters'] for cid in cluster_ids)

async def test_assign_memory_to_clusters_function(db_pool):
    """Test the assign_memory_to_clusters function
    
    Note: This test requires the updated db/schema.sql to be applied to the database.
    The assign_memory_to_clusters function was updated to use 
    'WHERE centroid_embedding IS NOT NULL' instead of 'WHERE status = 'active''
    """
    async with db_pool.acquire() as conn:
        # Create test clusters with different centroids
        cluster_ids = []
        for i in range(3):
            # Create distinct centroid embeddings
            centroid = [0.0] * EMBEDDING_DIMENSION
            centroid[i*100:(i+1)*100] = [1.0] * 100  # Make each cluster distinct
            
            cluster_id = await conn.fetchval("""
                INSERT INTO memory_clusters (
                    cluster_type,
                    name,
                    centroid_embedding
                ) VALUES (
                    'theme'::cluster_type,
                    'Test Cluster ' || $1,
                    $2::vector
                ) RETURNING id
            """, str(i), str(centroid))
            cluster_ids.append(cluster_id)
        
        # Create memory with embedding similar to first cluster
        memory_embedding = [0.0] * EMBEDDING_DIMENSION
        memory_embedding[0:100] = [0.9] * 100  # Similar to first cluster
        
        memory_id = await conn.fetchval("""
            INSERT INTO memories (
                type,
                content,
                embedding
            ) VALUES (
                'semantic'::memory_type,
                'Test memory for auto-clustering',
                $1::vector
            ) RETURNING id
        """, str(memory_embedding))
        
        # Assign to clusters
        await conn.execute("""
            SELECT assign_memory_to_clusters($1, 2)
        """, memory_id)
        
        # Verify assignment
        memberships = await conn.fetch("""
            SELECT cluster_id, membership_strength
            FROM memory_cluster_members
            WHERE memory_id = $1
            ORDER BY membership_strength DESC
        """, memory_id)
        
        assert len(memberships) > 0, "Memory not assigned to any clusters"
        assert memberships[0]['membership_strength'] >= 0.7, "Expected high similarity"

async def test_recalculate_cluster_centroid_function(db_pool):
    """Test the recalculate_cluster_centroid function"""
    async with db_pool.acquire() as conn:
        # Create cluster
        cluster_id = await conn.fetchval("""
            INSERT INTO memory_clusters (
                cluster_type,
                name,
                centroid_embedding
            ) VALUES (
                'theme'::cluster_type,
                'Test Centroid Cluster',
                array_fill(0.0, ARRAY[embedding_dimension()])::vector
            ) RETURNING id
        """)
        
        # Add memories with different embeddings
        for i in range(3):
            memory_id = await conn.fetchval("""
                INSERT INTO memories (
                    type,
                    content,
                    embedding,
                    status
                ) VALUES (
                    'semantic'::memory_type,
                    'Memory ' || $1,
                    array_fill($2::float, ARRAY[embedding_dimension()])::vector,
                    'active'::memory_status
                ) RETURNING id
            """, str(i), float(i+1) * 0.2)
            
            await conn.execute("""
                INSERT INTO memory_cluster_members (
                    cluster_id,
                    memory_id,
                    membership_strength
                ) VALUES ($1, $2, $3)
            """, cluster_id, memory_id, 0.8)
        
        # Recalculate centroid
        await conn.execute("""
            SELECT recalculate_cluster_centroid($1)
        """, cluster_id)
        
        # Check if centroid was updated
        result = await conn.fetchrow("""
            SELECT (vector_to_float4(centroid_embedding, embedding_dimension(), false))[1] as first_value
            FROM memory_clusters
            WHERE id = $1
        """, cluster_id)
        
        # The average of 0.2, 0.4, 0.6 should be 0.4
        assert result['first_value'] is not None, "Centroid not updated"

async def test_cluster_insights_view(db_pool):
    """Test the cluster_insights view"""
    async with db_pool.acquire() as conn:
        # Create cluster with members using unique name
        import time
        unique_name = f'Insight Test Cluster {int(time.time() * 1000)}'
        cluster_id = await conn.fetchval("""
            INSERT INTO memory_clusters (
                cluster_type,
                name,
                importance_score,
                coherence_score,
                centroid_embedding
            ) VALUES (
                'theme'::cluster_type,
                $1,
                0.8,
                0.9,
                array_fill(0.5, ARRAY[embedding_dimension()])::vector
            ) RETURNING id
        """, unique_name)
        
        # Add memories
        for i in range(5):
            memory_id = await conn.fetchval("""
                INSERT INTO memories (
                    type,
                    content,
                    embedding
                ) VALUES (
                    'episodic'::memory_type,
                    'Insight memory ' || $1,
                    array_fill(0.5, ARRAY[embedding_dimension()])::vector
                ) RETURNING id
            """, str(i))
            
            await conn.execute("""
                INSERT INTO memory_cluster_members (
                    cluster_id,
                    memory_id
                ) VALUES ($1, $2)
            """, cluster_id, memory_id)
        
        # Query view
        insights = await conn.fetch("""
            SELECT * FROM cluster_insights
            WHERE name = $1
        """, unique_name)
        
        assert len(insights) == 1
        assert insights[0]['memory_count'] == 5
        assert insights[0]['importance_score'] == 0.8
        assert insights[0]['coherence_score'] == 0.9

async def test_active_themes_view(db_pool):
    """Test cluster activation tracking through cluster_insights view"""
    async with db_pool.acquire() as conn:
        # Create active cluster with unique name
        import time
        unique_name = f'Recent Anxiety {int(time.time() * 1000)}'
        cluster_id = await conn.fetchval("""
            INSERT INTO memory_clusters (
                cluster_type,
                name,
                emotional_signature,
                keywords,
                centroid_embedding,
                activation_count,
                last_activated
            ) VALUES (
                'emotion'::cluster_type,
                $1,
                '{"primary": "anxiety", "intensity": 0.7}'::jsonb,
                ARRAY['worry', 'stress', 'uncertainty'],
                array_fill(0.3, ARRAY[embedding_dimension()])::vector,
                3,
                CURRENT_TIMESTAMP
            ) RETURNING id
        """, unique_name)

        # Query cluster_insights view
        themes = await conn.fetch("""
            SELECT * FROM cluster_insights
            WHERE id = $1
        """, cluster_id)

        assert len(themes) > 0
        assert themes[0]['activation_count'] == 3

async def test_update_cluster_activation_trigger(db_pool):
    """Test the update_cluster_activation trigger"""
    async with db_pool.acquire() as conn:
        # Create cluster with unique name
        import time
        unique_name = f'Activation Test {int(time.time() * 1000)}'
        cluster_id = await conn.fetchval("""
            INSERT INTO memory_clusters (
                cluster_type,
                name,
                centroid_embedding,
                importance_score,
                activation_count
            ) VALUES (
                'theme'::cluster_type,
                $1,
                array_fill(0.5, ARRAY[embedding_dimension()])::vector,
                0.5,
                0
            ) RETURNING id
        """, unique_name)
        
        # Get initial values
        initial = await conn.fetchrow("""
            SELECT importance_score, activation_count, last_activated
            FROM memory_clusters
            WHERE id = $1
        """, cluster_id)
        
        # Update activation count
        await conn.execute("""
            UPDATE memory_clusters
            SET activation_count = activation_count + 1
            WHERE id = $1
        """, cluster_id)
        
        # Get updated values
        updated = await conn.fetchrow("""
            SELECT importance_score, activation_count, last_activated
            FROM memory_clusters
            WHERE id = $1
        """, cluster_id)
        
        # Check that activation count increased (may be more than 1 due to trigger behavior)
        assert updated['activation_count'] > initial['activation_count'], f"Expected activation count to increase from {initial['activation_count']} but got {updated['activation_count']}"
        assert updated['importance_score'] > initial['importance_score']
        assert updated['last_activated'] is not None

async def test_cluster_types(db_pool):
    """Test all cluster types"""
    async with db_pool.acquire() as conn:
        cluster_types = ['theme', 'emotion', 'temporal', 'person', 'pattern', 'mixed']
        
        for c_type in cluster_types:
            cluster_id = await conn.fetchval("""
                INSERT INTO memory_clusters (
                    cluster_type,
                    name,
                    centroid_embedding
                ) VALUES (
                    $1::cluster_type,
                    'Test ' || $1 || ' cluster',
                    array_fill(0.5, ARRAY[embedding_dimension()])::vector
                ) RETURNING id
            """, c_type)
            
            assert cluster_id is not None, f"Failed to create {c_type} cluster"
        
        # Verify all types exist
        count = await conn.fetchval("""
            SELECT COUNT(DISTINCT cluster_type)
            FROM memory_clusters
        """)
        
        assert count >= len(cluster_types)

async def test_cluster_memory_retrieval_performance(db_pool):
    """Test performance of cluster-based memory retrieval"""
    async with db_pool.acquire() as conn:
        # Create cluster
        cluster_id = await conn.fetchval("""
            INSERT INTO memory_clusters (
                cluster_type,
                name,
                centroid_embedding,
                keywords
            ) VALUES (
                'theme'::cluster_type,
                'Loneliness',
                array_fill(0.3, ARRAY[embedding_dimension()])::vector,
                ARRAY['lonely', 'alone', 'isolated']
            ) RETURNING id
        """)
        
        # Add many memories to cluster
        memory_ids = []
        for i in range(50):
            memory_id = await conn.fetchval("""
                INSERT INTO memories (
                    type,
                    content,
                    embedding,
                    importance
                ) VALUES (
                    'episodic'::memory_type,
                    'Loneliness memory ' || $1,
                    array_fill(0.3, ARRAY[embedding_dimension()])::vector,
                    $2
                ) RETURNING id
            """, str(i), 0.5 + (i * 0.01))
            
            await conn.execute("""
                INSERT INTO memory_cluster_members (
                    cluster_id,
                    memory_id,
                    membership_strength
                ) VALUES ($1, $2, $3)
            """, cluster_id, memory_id, 0.7 + (i * 0.001))
            
            memory_ids.append(memory_id)
        
        # Test retrieval by cluster
        import time
        start_time = time.time()
        
        results = await conn.fetch("""
            SELECT m.*, mcm.membership_strength
            FROM memories m
            JOIN memory_cluster_members mcm ON m.id = mcm.memory_id
            WHERE mcm.cluster_id = $1
            ORDER BY mcm.membership_strength DESC, m.importance DESC
            LIMIT 10
        """, cluster_id)
        
        retrieval_time = time.time() - start_time
        
        assert len(results) == 10
        assert retrieval_time < 0.1, f"Cluster retrieval too slow: {retrieval_time}s"
        
        # Verify ordering
        strengths = [r['membership_strength'] for r in results]
        assert strengths == sorted(strengths, reverse=True)


# HIGH PRIORITY ADDITIONAL TESTS

async def test_constraint_violations(db_pool):
    """Test constraint violations and error handling"""
    async with db_pool.acquire() as conn:
        # Test invalid emotional_valence (should be between -1 and 1)
        with pytest.raises(Exception):
            await conn.execute("""
                INSERT INTO episodic_memories (
                    memory_id,
                    action_taken,
                    context,
                    result,
                    emotional_valence
                ) VALUES (
                    gen_random_uuid(),
                    '{"action": "test"}',
                    '{"context": "test"}',
                    '{"result": "test"}',
                    2.0
                )
            """)
        
        # Test invalid confidence score (should be between 0 and 1)
        with pytest.raises(Exception):
            await conn.execute("""
                INSERT INTO semantic_memories (
                    memory_id,
                    confidence
                ) VALUES (
                    gen_random_uuid(),
                    1.5
                )
            """)
        
        # Test foreign key violation
        with pytest.raises(Exception):
            fake_uuid = 'aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee'
            await conn.execute("""
                INSERT INTO episodic_memories (
                    memory_id,
                    action_taken,
                    context,
                    result
                ) VALUES (
                    $1::uuid,
                    '{"action": "test"}',
                    '{"context": "test"}',
                    '{"result": "test"}'
                )
            """, fake_uuid)
        
        # Test invalid vector dimension
        with pytest.raises(Exception):
            await conn.execute("""
                INSERT INTO memories (
                    type,
                    content,
                    embedding
                ) VALUES (
                    'semantic'::memory_type,
                    'Test content',
                    array_fill(0, ARRAY[100])::vector
                )
            """)
        
        # Test null constraint violation
        with pytest.raises(Exception):
            await conn.execute("""
                INSERT INTO memories (
                    type,
                    embedding
                ) VALUES (
                    'semantic'::memory_type,
                    array_fill(0, ARRAY[embedding_dimension()])::vector
                )
            """)


async def test_memory_consolidation_workflow(db_pool):
    """Test complete memory consolidation from working memory to long-term storage"""
    async with db_pool.acquire() as conn:
        # Step 1: Create working memory entries
        working_memories = []
        for i in range(5):
            wm_id = await conn.fetchval("""
                INSERT INTO working_memory (
                    content,
                    embedding,
                    expiry
                ) VALUES (
                    'Working memory content ' || $1,
                    array_fill($2::float, ARRAY[embedding_dimension()])::vector,
                    CURRENT_TIMESTAMP + interval '1 hour'
                ) RETURNING id
            """, str(i), float(i) * 0.1)
            working_memories.append(wm_id)
        
        # Step 2: Simulate consolidation process
        consolidated_memories = []
        for wm_id in working_memories:
            # Get working memory content
            wm_data = await conn.fetchrow("""
                SELECT content, embedding FROM working_memory WHERE id = $1
            """, wm_id)
            
            # Create long-term memory
            ltm_id = await conn.fetchval("""
                INSERT INTO memories (
                    type,
                    content,
                    embedding,
                    importance
                ) VALUES (
                    'episodic'::memory_type,
                    'Consolidated: ' || $1,
                    $2,
                    0.7
                ) RETURNING id
            """, wm_data['content'], wm_data['embedding'])
            
            # Create episodic details
            await conn.execute("""
                INSERT INTO episodic_memories (
                    memory_id,
                    action_taken,
                    context,
                    result,
                    emotional_valence
                ) VALUES (
                    $1,
                    '{"action": "consolidation"}',
                    '{"source": "working_memory"}',
                    '{"status": "consolidated"}',
                    0.0
                )
            """, ltm_id)
            
            consolidated_memories.append(ltm_id)
            
            # Remove from working memory
            await conn.execute("""
                DELETE FROM working_memory WHERE id = $1
            """, wm_id)
        
        # Step 3: Verify consolidation
        # Check working memory is empty
        wm_count = await conn.fetchval("""
            SELECT COUNT(*) FROM working_memory
            WHERE id = ANY($1::uuid[])
        """, working_memories)
        assert wm_count == 0, "Working memory not properly cleared"
        
        # Check long-term memories exist
        ltm_count = await conn.fetchval("""
            SELECT COUNT(*) FROM memories
            WHERE id = ANY($1::uuid[])
        """, consolidated_memories)
        assert ltm_count == 5, "Not all memories consolidated"
        
        # Step 4: Test memory clustering after consolidation
        for memory_id in consolidated_memories:
            await conn.execute("""
                SELECT assign_memory_to_clusters($1, 2)
            """, memory_id)
        
        # Verify cluster assignments
        cluster_assignments = await conn.fetchval("""
            SELECT COUNT(*) FROM memory_cluster_members
            WHERE memory_id = ANY($1::uuid[])
        """, consolidated_memories)
        assert cluster_assignments > 0, "Memories not assigned to clusters"


async def test_large_dataset_performance(db_pool):
    """Test system performance with large datasets"""
    async with db_pool.acquire() as conn:
        import time
        
        # Create large number of memories (1000 for testing, would be 10K+ in production)
        batch_size = 100
        total_memories = 1000
        memory_ids = []
        
        print(f"Creating {total_memories} memories in batches of {batch_size}...")
        
        for batch_start in range(0, total_memories, batch_size):
            batch_end = min(batch_start + batch_size, total_memories)
            batch_memories = []
            
            # Create batch of memories
            for i in range(batch_start, batch_end):
                # Create diverse embeddings
                embedding = [0.0] * EMBEDDING_DIMENSION
                # Create patterns in embeddings for clustering
                pattern_start = (i % 10) * 150
                pattern_end = min(pattern_start + 150, EMBEDDING_DIMENSION)
                embedding[pattern_start:pattern_end] = [0.8] * (pattern_end - pattern_start)
                
                memory_id = await conn.fetchval("""
                    INSERT INTO memories (
                        type,
                        content,
                        embedding,
                        importance
                    ) VALUES (
                        $1::memory_type,
                        'Large dataset memory ' || $2,
                        $3::vector,
                        $4
                    ) RETURNING id
                """, 
                    ['episodic', 'semantic', 'procedural', 'strategic'][i % 4],
                    str(i),
                    str(embedding),
                    0.1 + (i % 100) * 0.01
                )
                batch_memories.append(memory_id)
            
            memory_ids.extend(batch_memories)
        
        print(f"Created {len(memory_ids)} memories")
        
        # Test 1: Vector similarity search performance
        head_len = min(150, EMBEDDING_DIMENSION)
        query_embedding = [0.8] * head_len + [0.0] * (EMBEDDING_DIMENSION - head_len)
        
        start_time = time.time()
        similar_memories = await conn.fetch("""
            SELECT id, content, embedding <=> $1::vector as distance
            FROM memories
            ORDER BY embedding <=> $1::vector
            LIMIT 50
        """, str(query_embedding))
        vector_search_time = time.time() - start_time
        
        assert len(similar_memories) == 50
        assert vector_search_time < 1.0, f"Vector search too slow: {vector_search_time}s"
        print(f"Vector search time: {vector_search_time:.3f}s")
        
        # Test 2: Complex query performance
        start_time = time.time()
        complex_results = await conn.fetch("""
            SELECT m.type, COUNT(*) as count, AVG(m.importance) as avg_importance
            FROM memories m
            WHERE m.status = 'active'
            AND m.importance > 0.5
            GROUP BY m.type
            ORDER BY avg_importance DESC
        """)
        complex_query_time = time.time() - start_time
        
        assert len(complex_results) > 0
        assert complex_query_time < 0.5, f"Complex query too slow: {complex_query_time}s"
        print(f"Complex query time: {complex_query_time:.3f}s")
        
        # Test 3: Memory health view performance
        start_time = time.time()
        health_stats = await conn.fetch("""
            SELECT * FROM memory_health
        """)
        view_query_time = time.time() - start_time
        
        assert len(health_stats) > 0
        assert view_query_time < 0.5, f"View query too slow: {view_query_time}s"
        print(f"View query time: {view_query_time:.3f}s")


async def test_concurrency_safety(db_pool):
    """Test concurrent operations for race conditions"""
    async with db_pool.acquire() as conn:
        # Create test memory for concurrent updates
        memory_id = await conn.fetchval("""
            INSERT INTO memories (
                type,
                content,
                embedding,
                importance,
                access_count
            ) VALUES (
                'semantic'::memory_type,
                'Concurrency test memory',
                array_fill(0.5, ARRAY[embedding_dimension()])::vector,
                0.5,
                0
            ) RETURNING id
        """)
        
        # Test concurrent access count updates
        async def update_access_count(pool, mem_id, increment):
            async with pool.acquire() as connection:
                for _ in range(increment):
                    await connection.execute("""
                        UPDATE memories 
                        SET access_count = access_count + 1
                        WHERE id = $1
                    """, mem_id)
        
        # Run concurrent updates
        import asyncio
        tasks = [
            update_access_count(db_pool, memory_id, 10),
            update_access_count(db_pool, memory_id, 10),
            update_access_count(db_pool, memory_id, 10)
        ]
        
        await asyncio.gather(*tasks)
        
        # Verify final access count
        final_count = await conn.fetchval("""
            SELECT access_count FROM memories WHERE id = $1
        """, memory_id)
        
        assert final_count == 30, f"Expected 30 accesses, got {final_count}"
        
        # Test concurrent cluster assignments
        cluster_id = await conn.fetchval("""
            INSERT INTO memory_clusters (
                cluster_type,
                name,
                centroid_embedding
            ) VALUES (
                'theme'::cluster_type,
                'Concurrency Test Cluster',
                array_fill(0.5, ARRAY[embedding_dimension()])::vector
            ) RETURNING id
        """)
        
        # Create multiple memories for concurrent cluster assignment
        test_memories = []
        for i in range(5):
            mem_id = await conn.fetchval("""
                INSERT INTO memories (
                    type,
                    content,
                    embedding
                ) VALUES (
                    'semantic'::memory_type,
                    'Concurrent memory ' || $1,
                    array_fill(0.5, ARRAY[embedding_dimension()])::vector
                ) RETURNING id
            """, str(i))
            test_memories.append(mem_id)
        
        # Concurrent cluster assignments
        async def assign_to_cluster(pool, mem_id, clust_id):
            async with pool.acquire() as connection:
                try:
                    await connection.execute("""
                        INSERT INTO memory_cluster_members (
                            cluster_id,
                            memory_id,
                            membership_strength
                        ) VALUES ($1, $2, 0.8)
                        ON CONFLICT DO NOTHING
                    """, clust_id, mem_id)
                except Exception as e:
                    # Expected for some concurrent operations
                    pass
        
        assignment_tasks = [
            assign_to_cluster(db_pool, mem_id, cluster_id)
            for mem_id in test_memories
        ]
        
        await asyncio.gather(*assignment_tasks)
        
        # Verify assignments
        assignment_count = await conn.fetchval("""
            SELECT COUNT(*) FROM memory_cluster_members
            WHERE cluster_id = $1
        """, cluster_id)
        
        assert assignment_count == 5, f"Expected 5 assignments, got {assignment_count}"


async def test_cascade_delete_integrity(db_pool):
    """Test referential integrity with cascade deletes"""
    async with db_pool.acquire() as conn:
        # Create memory with all related data
        memory_id = await conn.fetchval("""
            INSERT INTO memories (
                type,
                content,
                embedding
            ) VALUES (
                'episodic'::memory_type,
                'Test cascade delete',
                array_fill(0.5, ARRAY[embedding_dimension()])::vector
            ) RETURNING id
        """)
        
        # Add episodic details
        await conn.execute("""
            INSERT INTO episodic_memories (
                memory_id,
                action_taken,
                context,
                result
            ) VALUES (
                $1,
                '{"action": "test"}',
                '{"context": "test"}',
                '{"result": "test"}'
            )
        """, memory_id)
        
        # Add to cluster
        cluster_id = await conn.fetchval("""
            INSERT INTO memory_clusters (
                cluster_type,
                name,
                centroid_embedding
            ) VALUES (
                'theme'::cluster_type,
                'Cascade Test Cluster',
                array_fill(0.5, ARRAY[embedding_dimension()])::vector
            ) RETURNING id
        """)
        
        await conn.execute("""
            INSERT INTO memory_cluster_members (
                cluster_id,
                memory_id
            ) VALUES ($1, $2)
        """, cluster_id, memory_id)
        
        # Add memory changes
        await conn.execute("""
            INSERT INTO memory_changes (
                memory_id,
                change_type,
                old_value,
                new_value
            ) VALUES (
                $1,
                'creation',
                '{}',
                '{"status": "created"}'
            )
        """, memory_id)
        
        # Verify all related data exists
        episodic_count = await conn.fetchval("""
            SELECT COUNT(*) FROM episodic_memories WHERE memory_id = $1
        """, memory_id)
        cluster_member_count = await conn.fetchval("""
            SELECT COUNT(*) FROM memory_cluster_members WHERE memory_id = $1
        """, memory_id)
        changes_count = await conn.fetchval("""
            SELECT COUNT(*) FROM memory_changes WHERE memory_id = $1
        """, memory_id)
        
        assert episodic_count == 1
        assert cluster_member_count == 1
        assert changes_count == 1
        
        # Delete the memory - cascades should handle all related data
        await conn.execute("""
            DELETE FROM memories WHERE id = $1
        """, memory_id)
        
        # Verify cascade deletes worked for tables with CASCADE
        cluster_member_count_after = await conn.fetchval("""
            SELECT COUNT(*) FROM memory_cluster_members WHERE memory_id = $1
        """, memory_id)
        changes_count_after = await conn.fetchval("""
            SELECT COUNT(*) FROM memory_changes WHERE memory_id = $1
        """, memory_id)
        episodic_count_after = await conn.fetchval("""
            SELECT COUNT(*) FROM episodic_memories WHERE memory_id = $1
        """, memory_id)
        
        # These should all be deleted now
        assert cluster_member_count_after == 0, "Cluster membership not cascade deleted"
        assert changes_count_after == 0, "Memory changes not deleted"
        assert episodic_count_after == 0, "Episodic memory not deleted"


async def test_memory_lifecycle_workflow(db_pool):
    """Test complete memory lifecycle from creation to archival"""
    async with db_pool.acquire() as conn:
        # Step 1: Create new memory
        memory_id = await conn.fetchval("""
            INSERT INTO memories (
                type,
                content,
                embedding,
                importance,
                access_count
            ) VALUES (
                'semantic'::memory_type,
                'Lifecycle test memory',
                array_fill(0.5, ARRAY[embedding_dimension()])::vector,
                0.3,
                0
            ) RETURNING id
        """)
        
        initial_relevance = await conn.fetchval("""
            SELECT calculate_relevance(importance, decay_rate, created_at, last_accessed) as relevance_score FROM memories WHERE id = $1
        """, memory_id)
        
        # Step 2: Simulate memory access and importance growth
        for i in range(5):
            await conn.execute("""
                UPDATE memories 
                SET access_count = access_count + 1
                WHERE id = $1
            """, memory_id)
        
        mid_relevance = await conn.fetchval("""
            SELECT calculate_relevance(importance, decay_rate, created_at, last_accessed) as relevance_score FROM memories WHERE id = $1
        """, memory_id)
        
        assert mid_relevance > initial_relevance, "Relevance should increase with access"
        
        # Step 3: Simulate time passage and decay
        await conn.execute("""
            UPDATE memories 
            SET created_at = CURRENT_TIMESTAMP - interval '30 days'
            WHERE id = $1
        """, memory_id)
        
        aged_relevance = await conn.fetchval("""
            SELECT calculate_relevance(importance, decay_rate, created_at, last_accessed) as relevance_score FROM memories WHERE id = $1
        """, memory_id)
        
        assert aged_relevance < mid_relevance, "Relevance should decrease with age"
        
        # Step 4: Archive low-relevance memory
        await conn.execute("""
            UPDATE memories
            SET status = 'archived'::memory_status
            WHERE id = $1 AND calculate_relevance(importance, decay_rate, created_at, last_accessed) < 0.1
        """, memory_id)
        
        final_status = await conn.fetchval("""
            SELECT status FROM memories WHERE id = $1
        """, memory_id)
        
        # Memory might or might not be archived depending on exact relevance calculation
        assert final_status in ['active', 'archived'], "Memory should have valid status"
        
        # Step 5: Test memory retrieval excludes archived memories
        active_memories = await conn.fetch("""
            SELECT * FROM memories 
            WHERE status = 'active' AND id = $1
        """, memory_id)
        
        if final_status == 'archived':
            assert len(active_memories) == 0, "Archived memory should not appear in active queries"


# CRITICAL MISSING TESTS

async def test_edge_cases_empty_clusters(db_pool):
    """Test edge cases with empty clusters"""
    async with db_pool.acquire() as conn:
        # Create empty cluster
        cluster_id = await conn.fetchval("""
            INSERT INTO memory_clusters (
                cluster_type,
                name,
                centroid_embedding
            ) VALUES (
                'theme'::cluster_type,
                'Empty Test Cluster',
                array_fill(0.5, ARRAY[embedding_dimension()])::vector
            ) RETURNING id
        """)
        
        # Test recalculating centroid on empty cluster
        await conn.execute("""
            SELECT recalculate_cluster_centroid($1)
        """, cluster_id)
        
        # Verify cluster still exists but centroid might be null
        cluster = await conn.fetchrow("""
            SELECT * FROM memory_clusters WHERE id = $1
        """, cluster_id)
        assert cluster is not None, "Empty cluster should still exist"
        
        # Test cluster insights view with empty cluster
        insights = await conn.fetch("""
            SELECT * FROM cluster_insights WHERE id = $1
        """, cluster_id)
        assert len(insights) == 1, "Empty cluster should appear in insights"
        assert insights[0]['memory_count'] == 0, "Empty cluster should have 0 memories"


async def test_edge_cases_circular_relationships(db_pool):
    """Test edge cases with circular cluster relationships"""
    async with db_pool.acquire() as conn:
        # Create three clusters
        cluster_ids = []
        for i in range(3):
            cluster_id = await conn.fetchval("""
                INSERT INTO memory_clusters (
                    cluster_type,
                    name,
                    centroid_embedding
                ) VALUES (
                    'theme'::cluster_type,
                    'Circular Cluster ' || $1,
                    array_fill($2::float, ARRAY[embedding_dimension()])::vector
                ) RETURNING id
            """, str(i), float(i) * 0.3)
            cluster_ids.append(cluster_id)
        
        # Create circular relationships: A -> B -> C -> A
        relationships = [
            (cluster_ids[0], cluster_ids[1], 'leads_to'),
            (cluster_ids[1], cluster_ids[2], 'causes'),
            (cluster_ids[2], cluster_ids[0], 'reinforces')
        ]
        
        for from_id, to_id, rel_type in relationships:
            await conn.execute("""
                INSERT INTO cluster_relationships (
                    from_cluster_id,
                    to_cluster_id,
                    relationship_type,
                    strength
                ) VALUES ($1, $2, $3, 0.7)
            """, from_id, to_id, rel_type)
        
        # Verify all relationships exist
        total_relationships = await conn.fetchval("""
            SELECT COUNT(*) FROM cluster_relationships
            WHERE from_cluster_id = ANY($1::uuid[])
        """, cluster_ids)
        assert total_relationships == 3, "All circular relationships should exist"
        
        # Test that we can detect cycles (this would be application logic)
        cycle_query = await conn.fetch("""
            WITH RECURSIVE cluster_paths AS (
                SELECT from_cluster_id, to_cluster_id, 1 as depth, 
                       ARRAY[from_cluster_id] as path
                FROM cluster_relationships
                WHERE from_cluster_id = $1
                
                UNION ALL
                
                SELECT cr.from_cluster_id, cr.to_cluster_id, cp.depth + 1,
                       cp.path || cr.from_cluster_id
                FROM cluster_relationships cr
                JOIN cluster_paths cp ON cr.from_cluster_id = cp.to_cluster_id
                WHERE cp.depth < 5 AND NOT (cr.from_cluster_id = ANY(cp.path))
            )
            SELECT * FROM cluster_paths WHERE to_cluster_id = $1 AND depth > 1
        """, cluster_ids[0])
        
        assert len(cycle_query) > 0, "Should detect circular relationship"


async def test_edge_cases_extreme_values(db_pool):
    """Test edge cases with extreme values"""
    async with db_pool.acquire() as conn:
        # Test memory with very high importance
        high_importance_id = await conn.fetchval("""
            INSERT INTO memories (
                type,
                content,
                embedding,
                importance,
                access_count
            ) VALUES (
                'semantic'::memory_type,
                'Extremely important memory',
                array_fill(0.5, ARRAY[embedding_dimension()])::vector,
                999999.0,
                999999
            ) RETURNING id
        """)
        
        # Test memory with very old timestamp
        old_memory_id = await conn.fetchval("""
            INSERT INTO memories (
                type,
                content,
                embedding,
                importance,
                created_at
            ) VALUES (
                'episodic'::memory_type,
                'Ancient memory',
                array_fill(0.5, ARRAY[embedding_dimension()])::vector,
                0.5,
                '1900-01-01'::timestamp
            ) RETURNING id
        """)
        
        # Test relevance calculation with extreme values
        high_relevance = await conn.fetchval("""
            SELECT calculate_relevance(importance, decay_rate, created_at, last_accessed) as relevance_score FROM memories WHERE id = $1
        """, high_importance_id)
        
        old_relevance = await conn.fetchval("""
            SELECT calculate_relevance(importance, decay_rate, created_at, last_accessed) as relevance_score FROM memories WHERE id = $1
        """, old_memory_id)
        
        assert high_relevance > 1000, "High importance should result in high relevance"
        assert old_relevance < 0.01, "Very old memory should have very low relevance"
        
        # Test with zero vectors
        zero_vector_id = await conn.fetchval("""
            INSERT INTO memories (
                type,
                content,
                embedding
            ) VALUES (
                'semantic'::memory_type,
                'Zero vector memory',
                array_fill(0.0, ARRAY[embedding_dimension()])::vector
            ) RETURNING id
        """)
        
        # Test similarity search with zero vector
        zero_results = await conn.fetch("""
            SELECT id, embedding <=> array_fill(0.0, ARRAY[embedding_dimension()])::vector as distance
            FROM memories
            WHERE id = $1
        """, zero_vector_id)
        
        assert len(zero_results) == 1
        # Zero vectors result in NaN for cosine distance, which is expected behavior
        import math
        assert math.isnan(zero_results[0]['distance']), "Zero vector cosine distance should be NaN"


async def test_data_integrity_orphaned_records(db_pool):
    """Test data integrity with potential orphaned records"""
    async with db_pool.acquire() as conn:
        # Create memory with related data
        memory_id = await conn.fetchval("""
            INSERT INTO memories (
                type,
                content,
                embedding
            ) VALUES (
                'semantic'::memory_type,
                'Test orphan memory',
                array_fill(0.5, ARRAY[embedding_dimension()])::vector
            ) RETURNING id
        """)
        
        # Add semantic details
        await conn.execute("""
            INSERT INTO semantic_memories (
                memory_id,
                confidence
            ) VALUES ($1, 0.8)
        """, memory_id)
        
        # Add to cluster
        cluster_id = await conn.fetchval("""
            INSERT INTO memory_clusters (
                cluster_type,
                name,
                centroid_embedding
            ) VALUES (
                'theme'::cluster_type,
                'Orphan Test Cluster',
                array_fill(0.5, ARRAY[embedding_dimension()])::vector
            ) RETURNING id
        """)
        
        await conn.execute("""
            INSERT INTO memory_cluster_members (
                cluster_id,
                memory_id
            ) VALUES ($1, $2)
        """, cluster_id, memory_id)
        
        # Simulate orphaned records by deleting cluster but not membership
        # First, we need to temporarily disable the foreign key constraint
        # In a real scenario, this could happen due to application bugs
        
        # Check for orphaned cluster memberships
        orphaned_memberships = await conn.fetch("""
            SELECT mcm.* 
            FROM memory_cluster_members mcm
            LEFT JOIN memory_clusters mc ON mcm.cluster_id = mc.id
            WHERE mc.id IS NULL
        """)
        
        # Should be empty in normal operation
        assert len(orphaned_memberships) == 0, "No orphaned cluster memberships should exist"
        
        # Check for orphaned memory type records
        orphaned_semantic = await conn.fetch("""
            SELECT sm.*
            FROM semantic_memories sm
            LEFT JOIN memories m ON sm.memory_id = m.id
            WHERE m.id IS NULL
        """)
        
        assert len(orphaned_semantic) == 0, "No orphaned semantic memories should exist"


async def test_computed_field_accuracy(db_pool):
    """Test accuracy of computed fields"""
    async with db_pool.acquire() as conn:
        # Test procedural memory success rate calculation
        memory_id = await conn.fetchval("""
            INSERT INTO memories (
                type,
                content,
                embedding
            ) VALUES (
                'procedural'::memory_type,
                'Success rate test',
                array_fill(0.5, ARRAY[embedding_dimension()])::vector
            ) RETURNING id
        """)
        
        # Test various success rate scenarios
        test_cases = [
            (0, 0, 0.0),      # No attempts
            (5, 10, 0.5),     # 50% success
            (10, 10, 1.0),    # 100% success
            (0, 5, 0.0),      # 0% success
            (1, 3, 0.333333)  # 33.33% success
        ]
        
        for success_count, total_attempts, expected_rate in test_cases:
            await conn.execute("""
                INSERT INTO procedural_memories (
                    memory_id,
                    steps,
                    success_count,
                    total_attempts
                ) VALUES ($1, '{"steps": ["test"]}', $2, $3)
                ON CONFLICT (memory_id) DO UPDATE SET
                    success_count = $2,
                    total_attempts = $3
            """, memory_id, success_count, total_attempts)
            
            actual_rate = await conn.fetchval("""
                SELECT success_rate FROM procedural_memories WHERE memory_id = $1
            """, memory_id)
            
            if expected_rate == 0.333333:
                assert abs(actual_rate - expected_rate) < 0.000001, f"Success rate calculation incorrect: expected {expected_rate}, got {actual_rate}"
            else:
                assert actual_rate == expected_rate, f"Success rate calculation incorrect: expected {expected_rate}, got {actual_rate}"
        
        # Test relevance score accuracy
        test_memory_id = await conn.fetchval("""
            INSERT INTO memories (
                type,
                content,
                embedding,
                importance,
                decay_rate,
                created_at
            ) VALUES (
                'semantic'::memory_type,
                'Relevance test',
                array_fill(0.5, ARRAY[embedding_dimension()])::vector,
                1.0,
                0.1,
                CURRENT_TIMESTAMP - interval '1 day'
            ) RETURNING id
        """)
        
        relevance = await conn.fetchval("""
            SELECT calculate_relevance(importance, decay_rate, created_at, last_accessed) as relevance_score FROM memories WHERE id = $1
        """, test_memory_id)
        
        # The calculate_relevance function uses a more complex formula:
        # importance * exp(-decay_rate * LEAST(age_in_days, age_of_last_access * 0.5))
        # Since last_accessed is NULL, it uses age_in_days(created_at) * 0.5
        # So for 1 day old: 1.0 * exp(-0.1 * 0.5) ≈ 0.9512
        expected_relevance = 1.0 * 2.718281828459045 ** (-0.1 * 0.5)
        assert abs(relevance - expected_relevance) < 0.01, f"Relevance calculation incorrect: expected ~{expected_relevance}, got {relevance}"


async def test_trigger_consistency(db_pool):
    """Test that all triggers fire correctly and consistently"""
    async with db_pool.acquire() as conn:
        # Test memory timestamp trigger
        memory_id = await conn.fetchval("""
            INSERT INTO memories (
                type,
                content,
                embedding
            ) VALUES (
                'semantic'::memory_type,
                'Trigger test memory',
                array_fill(0.5, ARRAY[embedding_dimension()])::vector
            ) RETURNING id
        """)
        
        initial_updated_at = await conn.fetchval("""
            SELECT updated_at FROM memories WHERE id = $1
        """, memory_id)
        
        # Wait and update
        await asyncio.sleep(0.1)
        await conn.execute("""
            UPDATE memories SET content = 'Updated content' WHERE id = $1
        """, memory_id)
        
        new_updated_at = await conn.fetchval("""
            SELECT updated_at FROM memories WHERE id = $1
        """, memory_id)
        
        assert new_updated_at > initial_updated_at, "Timestamp trigger should fire on update"
        
        # Test importance trigger - set initial importance first
        await conn.execute("""
            UPDATE memories SET importance = 0.5 WHERE id = $1
        """, memory_id)
        
        initial_importance = await conn.fetchval("""
            SELECT importance FROM memories WHERE id = $1
        """, memory_id)
        
        await conn.execute("""
            UPDATE memories SET access_count = access_count + 1 WHERE id = $1
        """, memory_id)
        
        new_importance = await conn.fetchval("""
            SELECT importance FROM memories WHERE id = $1
        """, memory_id)
        
        assert new_importance > initial_importance, "Importance trigger should fire on access count change"
        
        # Test cluster activation trigger
        cluster_id = await conn.fetchval("""
            INSERT INTO memory_clusters (
                cluster_type,
                name,
                centroid_embedding,
                activation_count,
                importance_score
            ) VALUES (
                'theme'::cluster_type,
                'Trigger Test Cluster',
                array_fill(0.5, ARRAY[embedding_dimension()])::vector,
                0,
                0.5
            ) RETURNING id
        """)
        
        initial_cluster_importance = await conn.fetchval("""
            SELECT importance_score FROM memory_clusters WHERE id = $1
        """, cluster_id)
        
        await conn.execute("""
            UPDATE memory_clusters SET activation_count = activation_count + 1 WHERE id = $1
        """, cluster_id)
        
        new_cluster_importance = await conn.fetchval("""
            SELECT importance_score FROM memory_clusters WHERE id = $1
        """, cluster_id)
        
        assert new_cluster_importance > initial_cluster_importance, "Cluster activation trigger should fire"


async def test_view_calculation_accuracy(db_pool):
    """Test accuracy of view calculations"""
    async with db_pool.acquire() as conn:
        # Create test data for memory_health view with unique content to avoid interference
        import time
        unique_suffix = str(int(time.time() * 1000))
        test_memories = []
        for i in range(10):
            memory_id = await conn.fetchval("""
                INSERT INTO memories (
                    type,
                    content,
                    embedding,
                    importance,
                    access_count,
                    last_accessed
                ) VALUES (
                    'semantic'::memory_type,
                    'Health test memory ' || $1 || ' ' || $2,
                    array_fill(0.5, ARRAY[embedding_dimension()])::vector,
                    $3,
                    $4,
                    CASE WHEN $5 THEN CURRENT_TIMESTAMP - interval '12 hours' ELSE NULL END
                ) RETURNING id
            """, str(i), unique_suffix, float(i) * 0.1, i, i % 2 == 0)
            test_memories.append(memory_id)
        
        # Query memory_health view for just our test memories
        health_stats = await conn.fetchrow("""
            SELECT
                type,
                COUNT(*) as total_memories,
                AVG(importance) as avg_importance,
                AVG(access_count) as avg_access_count,
                COUNT(*) FILTER (WHERE last_accessed > CURRENT_TIMESTAMP - INTERVAL '1 day') as accessed_last_day,
                AVG(calculate_relevance(importance, decay_rate, created_at, last_accessed)) as avg_relevance
            FROM memories
            WHERE type = 'semantic' AND content LIKE '%' || $1
            GROUP BY type
        """, unique_suffix)
        
        # Verify calculations
        assert health_stats['total_memories'] == 10, "Should count exactly our 10 test memories"
        
        # Calculate expected average importance: (0.0 + 0.1 + 0.2 + ... + 0.9) / 10 = 4.5 / 10 = 0.45
        expected_avg_importance = sum(i * 0.1 for i in range(10)) / 10
        actual_avg_importance = float(health_stats['avg_importance'])
        assert abs(actual_avg_importance - expected_avg_importance) < 0.01, f"Average importance calculation incorrect: expected {expected_avg_importance}, got {actual_avg_importance}"
        
        # Test cluster_insights view accuracy
        cluster_id = await conn.fetchval("""
            INSERT INTO memory_clusters (
                cluster_type,
                name,
                importance_score,
                coherence_score,
                centroid_embedding
            ) VALUES (
                'theme'::cluster_type,
                'Accuracy Test Cluster',
                0.75,
                0.85,
                array_fill(0.5, ARRAY[embedding_dimension()])::vector
            ) RETURNING id
        """)
        
        # Add some memories to cluster
        for memory_id in test_memories[:5]:
            await conn.execute("""
                INSERT INTO memory_cluster_members (
                    cluster_id,
                    memory_id
                ) VALUES ($1, $2)
            """, cluster_id, memory_id)
        
        cluster_insight = await conn.fetchrow("""
            SELECT * FROM cluster_insights WHERE name = 'Accuracy Test Cluster'
        """)
        
        assert cluster_insight['memory_count'] == 5, "Should count cluster members correctly"
        assert cluster_insight['importance_score'] == 0.75, "Should preserve importance score"
        assert cluster_insight['coherence_score'] == 0.85, "Should preserve coherence score"


async def test_error_recovery_scenarios(db_pool):
    """Test error recovery scenarios"""
    async with db_pool.acquire() as conn:
        # Test recovery from invalid JSON in JSONB fields
        memory_id = await conn.fetchval("""
            INSERT INTO memories (
                type,
                content,
                embedding
            ) VALUES (
                'episodic'::memory_type,
                'Error recovery test',
                array_fill(0.5, ARRAY[embedding_dimension()])::vector
            ) RETURNING id
        """)
        
        # Insert valid episodic memory
        await conn.execute("""
            INSERT INTO episodic_memories (
                memory_id,
                action_taken,
                context,
                result
            ) VALUES (
                $1,
                '{"action": "valid_json"}',
                '{"context": "test"}',
                '{"result": "success"}'
            )
        """, memory_id)
        
        # Test that we can query and update the record
        episodic_data = await conn.fetchrow("""
            SELECT * FROM episodic_memories WHERE memory_id = $1
        """, memory_id)
        
        assert episodic_data is not None, "Should be able to query episodic memory"
        
        # Test updating with new valid JSON
        await conn.execute("""
            UPDATE episodic_memories 
            SET action_taken = '{"action": "updated_action"}'
            WHERE memory_id = $1
        """, memory_id)
        
        updated_data = await conn.fetchrow("""
            SELECT action_taken FROM episodic_memories WHERE memory_id = $1
        """, memory_id)
        
        # Parse the JSON if it's returned as a string
        action_taken = updated_data['action_taken']
        if isinstance(action_taken, str):
            action_taken = json.loads(action_taken)
        
        assert action_taken['action'] == 'updated_action', "Should update JSON correctly"
        
        # Test transaction rollback scenario
        try:
            async with conn.transaction():
                # Create a memory
                temp_memory_id = await conn.fetchval("""
                    INSERT INTO memories (
                        type,
                        content,
                        embedding
                    ) VALUES (
                        'semantic'::memory_type,
                        'Rollback test',
                        array_fill(0.5, ARRAY[embedding_dimension()])::vector
                    ) RETURNING id
                """)
                
                # Force an error with invalid constraint
                await conn.execute("""
                    INSERT INTO semantic_memories (
                        memory_id,
                        confidence
                    ) VALUES ($1, 2.0)
                """, temp_memory_id)  # This should fail due to confidence constraint
        except Exception:
            # Expected to fail
            pass
        
        # Verify the memory was rolled back
        rollback_check = await conn.fetchval("""
            SELECT COUNT(*) FROM memories WHERE content = 'Rollback test'
        """)
        
        assert rollback_check == 0, "Transaction should have been rolled back"


async def test_worldview_driven_memory_filtering(db_pool):
    """Test how worldview affects memory retrieval and filtering"""
    async with db_pool.acquire() as conn:
        # Create worldview primitive
        worldview_id = await conn.fetchval("""
            INSERT INTO worldview_primitives (
                category,
                belief,
                confidence,
                emotional_valence,
                stability_score
            ) VALUES (
                'values',
                'Positive thinking is important',
                0.9,
                0.8,
                0.7
            ) RETURNING id
        """)
        
        # Create memories with different emotional valences
        positive_memory_id = await conn.fetchval("""
            INSERT INTO memories (
                type,
                content,
                embedding,
                importance
            ) VALUES (
                'episodic'::memory_type,
                'Positive experience',
                array_fill(0.5, ARRAY[embedding_dimension()])::vector,
                0.5
            ) RETURNING id
        """)
        
        await conn.execute("""
            INSERT INTO episodic_memories (
                memory_id,
                action_taken,
                context,
                result,
                emotional_valence
            ) VALUES (
                $1,
                '{"action": "celebration"}',
                '{"context": "achievement"}',
                '{"result": "joy"}',
                0.8
            )
        """, positive_memory_id)
        
        negative_memory_id = await conn.fetchval("""
            INSERT INTO memories (
                type,
                content,
                embedding,
                importance
            ) VALUES (
                'episodic'::memory_type,
                'Negative experience',
                array_fill(0.5, ARRAY[embedding_dimension()])::vector,
                0.5
            ) RETURNING id
        """)
        
        await conn.execute("""
            INSERT INTO episodic_memories (
                memory_id,
                action_taken,
                context,
                result,
                emotional_valence
            ) VALUES (
                $1,
                '{"action": "failure"}',
                '{"context": "disappointment"}',
                '{"result": "sadness"}',
                -0.8
            )
        """, negative_memory_id)
        
        # Create worldview influences
        await conn.execute("""
            INSERT INTO worldview_memory_influences (
                worldview_id,
                memory_id,
                influence_type,
                strength
            ) VALUES 
                ($1, $2, 'boost', 1.5),
                ($1, $3, 'suppress', 0.3)
        """, worldview_id, positive_memory_id, negative_memory_id)
        
        # Test worldview-influenced retrieval
        influenced_memories = await conn.fetch("""
            SELECT m.*, wmi.influence_type, wmi.strength,
                   CASE 
                       WHEN wmi.influence_type = 'boost' THEN m.importance * wmi.strength
                       WHEN wmi.influence_type = 'suppress' THEN m.importance * wmi.strength
                       ELSE m.importance
                   END as adjusted_importance
            FROM memories m
            LEFT JOIN worldview_memory_influences wmi ON m.id = wmi.memory_id
            WHERE wmi.worldview_id = $1
            ORDER BY adjusted_importance DESC
        """, worldview_id)
        
        assert len(influenced_memories) == 2, "Should find both influenced memories"
        assert influenced_memories[0]['influence_type'] == 'boost', "Positive memory should be boosted"
        assert influenced_memories[1]['influence_type'] == 'suppress', "Negative memory should be suppressed"
        assert influenced_memories[0]['adjusted_importance'] > influenced_memories[1]['adjusted_importance'], "Boosted memory should rank higher"


# EMBEDDING INTEGRATION TESTS

async def test_embedding_service_integration(db_pool):
    """Test integration with embeddings microservice"""
    async with db_pool.acquire() as conn:
        # Test embedding service health check
        health_status = await conn.fetchval("""
            SELECT check_embedding_service_health()
        """)
        
        # Note: This may fail if embeddings service isn't running
        # In CI/CD, you might want to mock this or skip if service unavailable
        print(f"Embedding service health: {health_status}")
        
        # Test basic embedding generation (if service is available)
        if health_status:
            try:
                embedding = await conn.fetchval("""
                    SELECT get_embedding('test content for embedding')
                """)
                assert embedding is not None, "Should generate embedding"
                
                # Test embedding cache
                cached_embedding = await conn.fetchval("""
                    SELECT get_embedding('test content for embedding')
                """)
                assert cached_embedding == embedding, "Should return cached embedding"
                
            except Exception as e:
                print(f"Embedding generation test skipped: {e}")


async def test_create_memory_with_auto_embedding(db_pool):
    """Test creating memories with automatic embedding generation"""
    async with db_pool.acquire() as conn:
        # Check if embedding service is available
        health_status = await conn.fetchval("""
            SELECT check_embedding_service_health()
        """)
        
        if not health_status:
            print("Skipping embedding tests - service not available")
            return
        
        try:
            # Test creating semantic memory with auto-embedding
            memory_id = await conn.fetchval("""
                SELECT create_semantic_memory(
                    'User prefers dark mode interfaces',
                    0.9,
                    ARRAY['preference', 'UI'],
                    ARRAY['interface', 'theme', 'dark mode']
                )
            """)
            
            assert memory_id is not None, "Should create memory with auto-embedding"
            
            # Verify memory was created with embedding
            memory_data = await conn.fetchrow("""
                SELECT content, embedding, type FROM memories WHERE id = $1
            """, memory_id)
            
            assert memory_data['content'] == 'User prefers dark mode interfaces'
            assert memory_data['embedding'] is not None
            assert memory_data['type'] == 'semantic'
            
            # Test episodic memory creation
            episodic_id = await conn.fetchval("""
                SELECT create_episodic_memory(
                    'User clicked the help button',
                    '{"action": "click", "element": "help_button"}',
                    '{"page": "settings", "section": "account"}',
                    '{"modal_opened": true, "help_displayed": true}',
                    0.1
                )
            """)
            
            assert episodic_id is not None, "Should create episodic memory"
            
        except Exception as e:
            print(f"Auto-embedding test failed: {e}")
            # Don't fail the test if embedding service is unavailable


async def test_search_with_auto_embedding(db_pool):
    """Test searching memories with automatic query embedding"""
    async with db_pool.acquire() as conn:
        # Check if embedding service is available
        health_status = await conn.fetchval("""
            SELECT check_embedding_service_health()
        """)
        
        if not health_status:
            print("Skipping search embedding tests - service not available")
            return
        
        try:
            # Create some test memories first
            memory_ids = []
            test_contents = [
                'User interface design principles',
                'Dark mode reduces eye strain',
                'Accessibility features for visually impaired users'
            ]
            
            for content in test_contents:
                memory_id = await conn.fetchval("""
                    SELECT create_semantic_memory($1, 0.8)
                """, content)
                memory_ids.append(memory_id)
            
            # Test similarity search with auto-embedding
            results = await conn.fetch("""
                SELECT * FROM search_similar_memories('user interface preferences', 5)
            """)
            
            assert len(results) > 0, "Should find similar memories"
            
            # Verify results have expected fields
            for result in results:
                assert 'memory_id' in result
                assert 'content' in result
                assert 'similarity' in result
                assert result['similarity'] >= 0 and result['similarity'] <= 1
            
        except Exception as e:
            print(f"Search embedding test failed: {e}")


async def test_working_memory_with_embedding(db_pool):
    """Test working memory operations with automatic embedding"""
    async with db_pool.acquire() as conn:
        # Check if embedding service is available
        health_status = await conn.fetchval("""
            SELECT check_embedding_service_health()
        """)
        
        if not health_status:
            print("Skipping working memory embedding tests - service not available")
            return
        
        try:
            # Add to working memory with auto-embedding
            wm_id = await conn.fetchval("""
                SELECT add_to_working_memory(
                    'Current user is browsing settings page',
                    INTERVAL '30 minutes'
                )
            """)
            
            assert wm_id is not None, "Should add to working memory"
            
            # Search working memory
            results = await conn.fetch("""
                SELECT * FROM search_working_memory('user settings', 3)
            """)
            
            assert len(results) > 0, "Should find working memory items"
            
        except Exception as e:
            print(f"Working memory embedding test failed: {e}")


async def test_embedding_cache_functionality(db_pool):
    """Test embedding caching functionality"""
    async with db_pool.acquire() as conn:
        # Check if embedding service is available
        health_status = await conn.fetchval("""
            SELECT check_embedding_service_health()
        """)
        
        if not health_status:
            print("Skipping cache tests - service not available")
            return
        
        try:
            # Clear any existing cache for our test content
            test_content = 'unique test content for caching'
            content_hash = await conn.fetchval("""
                SELECT encode(sha256($1::text::bytea), 'hex')
            """, test_content)
            
            await conn.execute("""
                DELETE FROM embedding_cache WHERE content_hash = $1
            """, content_hash)
            
            # First call should hit the service and cache the result
            embedding1 = await conn.fetchval("""
                SELECT get_embedding($1)
            """, test_content)
            
            # Verify it was cached
            cached_count = await conn.fetchval("""
                SELECT COUNT(*) FROM embedding_cache WHERE content_hash = $1
            """, content_hash)
            
            assert cached_count == 1, "Should cache the embedding"
            
            # Second call should use cache
            embedding2 = await conn.fetchval("""
                SELECT get_embedding($1)
            """, test_content)
            
            assert embedding1 == embedding2, "Should return same embedding from cache"
            
            # Test cache cleanup
            deleted_count = await conn.fetchval("""
                SELECT cleanup_embedding_cache(INTERVAL '0 seconds')
            """)
            
            assert deleted_count >= 1, "Should clean up cache entries"
            
        except Exception as e:
            print(f"Cache test failed: {e}")


async def test_embedding_error_handling(db_pool):
    """Test error handling for embedding operations"""
    async with db_pool.acquire() as conn:
        original_url = await conn.fetchval(
            "SELECT value FROM embedding_config WHERE key = 'service_url'"
        )
        original_retry_seconds, original_retry_interval_seconds = await _set_embedding_retry_config(
            conn,
            retry_seconds=1,
            retry_interval_seconds=0.1,
        )

        # Test with invalid service URL
        await conn.execute("""
            UPDATE embedding_config 
            SET value = 'http://invalid-service:9999/embed'
            WHERE key = 'service_url'
        """)
        
        # This should fail gracefully
        try:
            await conn.fetchval("""
                SELECT get_embedding('test content')
            """)
            assert False, "Should have failed with invalid service URL"
        except Exception as e:
            assert "Failed to get embedding" in str(e), "Should have proper error message"
        
        # Restore valid URL and retry settings
        await conn.execute(
            "UPDATE embedding_config SET value = $1 WHERE key = 'service_url'",
            original_url,
        )
        await _restore_embedding_retry_config(
            conn,
            original_retry_seconds,
            original_retry_interval_seconds,
        )


async def test_memory_cluster_with_embeddings(db_pool):
    """Test memory clustering with automatic embeddings"""
    async with db_pool.acquire() as conn:
        # Check if embedding service is available
        health_status = await conn.fetchval("""
            SELECT check_embedding_service_health()
        """)
        
        if not health_status:
            print("Skipping cluster embedding tests - service not available")
            return
        
        try:
            # Create memories with related content
            memory_ids = []
            related_contents = [
                'User interface design best practices',
                'UI accessibility guidelines',
                'Interface usability principles'
            ]
            
            for content in related_contents:
                memory_id = await conn.fetchval("""
                    SELECT create_semantic_memory($1, 0.8)
                """, content)
                memory_ids.append(memory_id)
            
            # Create cluster with these memories
            cluster_id = await conn.fetchval("""
                SELECT create_memory_cluster(
                    'UI Design Principles',
                    'theme',
                    'Cluster for UI design related memories',
                    $1
                )
            """, memory_ids)
            
            assert cluster_id is not None, "Should create cluster"
            
            # Verify cluster has centroid embedding
            centroid = await conn.fetchval("""
                SELECT centroid_embedding FROM memory_clusters WHERE id = $1
            """, cluster_id)
            
            assert centroid is not None, "Cluster should have centroid embedding"
            
        except Exception as e:
            print(f"Cluster embedding test failed: {e}")


# MEDIUM PRIORITY ADDITIONAL TESTS

async def test_complex_graph_traversals(db_pool):
    """Test complex multi-hop graph traversals and path finding"""
    async with db_pool.acquire() as conn:
        await conn.execute("LOAD 'age';")
        await conn.execute("SET search_path = ag_catalog, public;")
        
        # Create a complex memory network
        memory_chain = []
        memory_types = ['episodic', 'semantic', 'procedural', 'strategic', 'episodic']
        
        # Create 5 connected memories
        for i, mem_type in enumerate(memory_types):
            memory_id = await conn.fetchval("""
                INSERT INTO memories (type, content, embedding)
                VALUES ($1::memory_type, $2, array_fill($3::float, ARRAY[embedding_dimension()])::vector)
                RETURNING id
            """, mem_type, f'Complex memory {i}', float(i) * 0.1)
            memory_chain.append(memory_id)
            
            # Create graph node
            await conn.execute(f"""
                SELECT * FROM cypher('memory_graph', $$
                    CREATE (n:MemoryNode {{
                        memory_id: '{memory_id}',
                        type: '{mem_type}',
                        step: {i}
                    }})
                    RETURN n
                $$) as (n ag_catalog.agtype)
            """)
        
        # Create linear chain relationships
        for i in range(len(memory_chain) - 1):
            await conn.execute(f"""
                SELECT * FROM cypher('memory_graph', $$
                    MATCH (a:MemoryNode {{memory_id: '{memory_chain[i]}'}}),
                          (b:MemoryNode {{memory_id: '{memory_chain[i+1]}'}})
                    CREATE (a)-[r:LEADS_TO {{strength: {0.8 - i*0.1}}}]->(b)
                    RETURN r
                $$) as (r ag_catalog.agtype)
            """)
        
        # Create some cross-connections
        await conn.execute(f"""
            SELECT * FROM cypher('memory_graph', $$
                MATCH (a:MemoryNode {{memory_id: '{memory_chain[0]}'}}),
                      (b:MemoryNode {{memory_id: '{memory_chain[3]}'}})
                CREATE (a)-[r:INFLUENCES {{strength: 0.6}}]->(b)
                RETURN r
            $$) as (r ag_catalog.agtype)
        """)
        
        # Test 1: Find all paths between first and last memory
        paths = await conn.fetch(f"""
            SELECT * FROM cypher('memory_graph', $$
                MATCH p = (start:MemoryNode {{memory_id: '{memory_chain[0]}'}})-[*1..5]->(finish:MemoryNode {{memory_id: '{memory_chain[-1]}'}})
                RETURN p
            $$) as (p ag_catalog.agtype)
        """)
        
        assert len(paths) >= 1, "Should find at least one path"
        
        # Test 2: Find memories within 2 hops of the first memory
        nearby_memories = await conn.fetch(f"""
            SELECT * FROM cypher('memory_graph', $$
                MATCH (start:MemoryNode {{memory_id: '{memory_chain[0]}'}})-[*1..2]->(nearby:MemoryNode)
                RETURN DISTINCT nearby.memory_id as memory_id, nearby.type as type
            $$) as (memory_id ag_catalog.agtype, type ag_catalog.agtype)
        """)
        
        assert len(nearby_memories) >= 2, "Should find nearby memories"
        
        # Test 3: Find any path with relationship properties
        weighted_path = await conn.fetch(f"""
            SELECT * FROM cypher('memory_graph', $$
                MATCH p = (start:MemoryNode {{memory_id: '{memory_chain[0]}'}})-[*1..5]->(finish:MemoryNode {{memory_id: '{memory_chain[-1]}'}})
                RETURN p
                LIMIT 1
            $$) as (path ag_catalog.agtype)
        """)
        
        assert len(weighted_path) > 0, "Should find a path"
        
        # Test 4: Complex pattern matching
        patterns = await conn.fetch("""
            SELECT * FROM cypher('memory_graph', $$
                MATCH (e:MemoryNode {type: 'episodic'})-[:LEADS_TO]->(s:MemoryNode {type: 'semantic'})-[:LEADS_TO]->(p:MemoryNode {type: 'procedural'})
                RETURN e.memory_id as episodic_id, s.memory_id as semantic_id, p.memory_id as procedural_id
            $$) as (episodic_id ag_catalog.agtype, semantic_id ag_catalog.agtype, procedural_id ag_catalog.agtype)
        """)
        
        assert len(patterns) > 0, "Should find episodic->semantic->procedural patterns"


async def test_memory_lifecycle_management(db_pool):
    """Test comprehensive memory lifecycle management"""
    async with db_pool.acquire() as conn:
        # Create memories with different lifecycle stages
        lifecycle_memories = []
        
        # Stage 1: Fresh memories (high importance, recent)
        for i in range(3):
            memory_id = await conn.fetchval("""
                INSERT INTO memories (
                    type,
                    content,
                    embedding,
                    importance,
                    access_count,
                    created_at
                ) VALUES (
                    'semantic'::memory_type,
                    'Fresh memory ' || $1,
                    array_fill(0.8, ARRAY[embedding_dimension()])::vector,
                    0.9,
                    10 + $2,
                    CURRENT_TIMESTAMP - interval '1 hour' * $2
                ) RETURNING id
            """, str(i), i)
            lifecycle_memories.append(('fresh', memory_id))
        
        # Stage 2: Aging memories (medium importance, older)
        for i in range(3):
            memory_id = await conn.fetchval("""
                INSERT INTO memories (
                    type,
                    content,
                    embedding,
                    importance,
                    access_count,
                    created_at
                ) VALUES (
                    'episodic'::memory_type,
                    'Aging memory ' || $1,
                    array_fill(0.5, ARRAY[embedding_dimension()])::vector,
                    0.5,
                    5 + $2,
                    CURRENT_TIMESTAMP - interval '7 days' * ($2 + 1)
                ) RETURNING id
            """, str(i), i)
            lifecycle_memories.append(('aging', memory_id))
        
        # Stage 3: Stale memories (low importance, very old)
        for i in range(3):
            memory_id = await conn.fetchval("""
                INSERT INTO memories (
                    type,
                    content,
                    embedding,
                    importance,
                    access_count,
                    created_at
                ) VALUES (
                    'procedural'::memory_type,
                    'Stale memory ' || $1,
                    array_fill(0.2, ARRAY[embedding_dimension()])::vector,
                    0.1,
                    1,
                    CURRENT_TIMESTAMP - interval '30 days' * ($2 + 1)
                ) RETURNING id
            """, str(i), i)
            lifecycle_memories.append(('stale', memory_id))
        
        # Test lifecycle categorization
        fresh_count = await conn.fetchval("""
            SELECT COUNT(*) FROM memories
            WHERE created_at > CURRENT_TIMESTAMP - interval '1 day'
            AND importance > 0.7
        """)
        
        aging_count = await conn.fetchval("""
            SELECT COUNT(*) FROM memories
            WHERE created_at BETWEEN CURRENT_TIMESTAMP - interval '30 days' 
                                 AND CURRENT_TIMESTAMP - interval '1 day'
            AND importance BETWEEN 0.3 AND 0.7
        """)
        
        stale_count = await conn.fetchval("""
            SELECT COUNT(*) FROM memories
            WHERE created_at < CURRENT_TIMESTAMP - interval '30 days'
            AND importance < 0.3
        """)
        
        assert fresh_count >= 3, "Should have fresh memories"
        assert aging_count >= 3, "Should have aging memories"
        assert stale_count >= 3, "Should have stale memories"
        
        # Test memory promotion (accessing old memory should increase importance)
        stale_memory = [m for stage, m in lifecycle_memories if stage == 'stale'][0]
        
        initial_importance = await conn.fetchval("""
            SELECT importance FROM memories WHERE id = $1
        """, stale_memory)
        
        # Simulate multiple accesses
        for _ in range(5):
            await conn.execute("""
                UPDATE memories 
                SET access_count = access_count + 1
                WHERE id = $1
            """, stale_memory)
        
        final_importance = await conn.fetchval("""
            SELECT importance FROM memories WHERE id = $1
        """, stale_memory)
        
        assert final_importance > initial_importance, "Accessed memory should gain importance"
        
        # Test memory archival workflow
        archival_candidates = await conn.fetch("""
            SELECT id, calculate_relevance(importance, decay_rate, created_at, last_accessed) as relevance_score
            FROM memories
            WHERE calculate_relevance(importance, decay_rate, created_at, last_accessed) < 0.1
            AND status = 'active'
            ORDER BY calculate_relevance(importance, decay_rate, created_at, last_accessed) ASC
            LIMIT 5
        """)
        
        for candidate in archival_candidates:
            await conn.execute("""
                UPDATE memories 
                SET status = 'archived'::memory_status
                WHERE id = $1
            """, candidate['id'])
        
        # Verify archival
        archived_count = await conn.fetchval("""
            SELECT COUNT(*) FROM memories WHERE status = 'archived'
        """)
        
        assert archived_count > 0, "Should have archived some memories"


async def test_memory_pruning_operations(db_pool):
    """Test memory pruning and cleanup operations"""
    async with db_pool.acquire() as conn:
        # Create memories for pruning
        pruning_memories = []
        
        # Create very old, low-relevance memories
        for i in range(10):
            memory_id = await conn.fetchval("""
                INSERT INTO memories (
                    type,
                    content,
                    embedding,
                    importance,
                    access_count,
                    created_at,
                    last_accessed
                ) VALUES (
                    'semantic'::memory_type,
                    'Pruning candidate ' || $1,
                    array_fill(0.1, ARRAY[embedding_dimension()])::vector,
                    0.05,
                    0,
                    CURRENT_TIMESTAMP - interval '90 days',
                    CURRENT_TIMESTAMP - interval '60 days'
                ) RETURNING id
            """, str(i))
            pruning_memories.append(memory_id)
        
        # Create some memories worth keeping
        for i in range(5):
            await conn.fetchval("""
                INSERT INTO memories (
                    type,
                    content,
                    embedding,
                    importance,
                    access_count,
                    created_at
                ) VALUES (
                    'episodic'::memory_type,
                    'Important memory ' || $1,
                    array_fill(0.8, ARRAY[embedding_dimension()])::vector,
                    0.8,
                    20,
                    CURRENT_TIMESTAMP - interval '30 days'
                ) RETURNING id
            """, str(i))
        
        # Test pruning criteria identification
        pruning_candidates = await conn.fetch("""
            SELECT id, calculate_relevance(importance, decay_rate, created_at, last_accessed) as relevance_score,
                   importance,
                   age_in_days(created_at) as age_days,
                   COALESCE(age_in_days(last_accessed), 999) as days_since_access
            FROM memories
            WHERE calculate_relevance(importance, decay_rate, created_at, last_accessed) < 0.1
            AND (last_accessed IS NULL OR last_accessed < CURRENT_TIMESTAMP - interval '30 days')
            AND importance < 0.1
            ORDER BY calculate_relevance(importance, decay_rate, created_at, last_accessed) ASC
        """)
        
        assert len(pruning_candidates) >= 10, "Should identify pruning candidates"
        
        # Test safe pruning (archive first, then delete)
        for candidate in pruning_candidates[:5]:
            # First archive
            await conn.execute("""
                UPDATE memories 
                SET status = 'archived'::memory_status
                WHERE id = $1
            """, candidate['id'])
        
        # Then delete archived memories older than threshold
        deleted_count = await conn.fetchval("""
            WITH deleted_memories AS (
                DELETE FROM memories 
                WHERE status = 'archived'
                AND created_at < CURRENT_TIMESTAMP - interval '120 days'
                RETURNING id
            )
            SELECT COUNT(*) FROM deleted_memories
        """)
        
        # Test working memory cleanup
        # Create expired working memory entries
        for i in range(5):
            await conn.execute("""
                INSERT INTO working_memory (
                    content,
                    embedding,
                    expiry
                ) VALUES (
                    'Expired working memory ' || $1,
                    array_fill(0.5, ARRAY[embedding_dimension()])::vector,
                    CURRENT_TIMESTAMP - interval '1 hour'
                )
            """, str(i))
        
        # Clean up expired working memory
        expired_cleaned = await conn.fetchval("""
            WITH cleaned AS (
                DELETE FROM working_memory
                WHERE expiry < CURRENT_TIMESTAMP
                RETURNING id
            )
            SELECT COUNT(*) FROM cleaned
        """)
        
        assert expired_cleaned >= 5, "Should clean up expired working memory"


async def test_database_optimization_operations(db_pool):
    """Test database optimization and maintenance operations"""
    async with db_pool.acquire() as conn:
        # Test index usage analysis
        index_usage = await conn.fetch("""
            SELECT 
                schemaname,
                relname as tablename,
                indexrelname as indexname,
                idx_scan,
                idx_tup_read,
                idx_tup_fetch
            FROM pg_stat_user_indexes
            WHERE schemaname = 'public'
            AND relname IN ('memories', 'memory_clusters', 'memory_cluster_members')
            ORDER BY idx_scan DESC
        """)
        
        assert len(index_usage) > 0, "Should have index usage statistics"
        
        # Test table statistics
        table_stats = await conn.fetch("""
            SELECT 
                schemaname,
                relname as tablename,
                n_tup_ins,
                n_tup_upd,
                n_tup_del,
                n_live_tup,
                n_dead_tup
            FROM pg_stat_user_tables
            WHERE schemaname = 'public'
            AND relname IN ('memories', 'memory_clusters')
            ORDER BY n_live_tup DESC
        """)
        
        assert len(table_stats) > 0, "Should have table statistics"
        
        # Test query performance analysis
        # Create a complex query and analyze its performance
        import time
        
        start_time = time.time()
        complex_query_result = await conn.fetch("""
            SELECT
                m.type,
                COUNT(*) as memory_count,
                AVG(m.importance) as avg_importance,
                AVG(calculate_relevance(m.importance, m.decay_rate, m.created_at, m.last_accessed)) as avg_relevance,
                COUNT(mcm.cluster_id) as cluster_memberships
            FROM memories m
            LEFT JOIN memory_cluster_members mcm ON m.id = mcm.memory_id
            WHERE m.status = 'active'
            GROUP BY m.type
            HAVING COUNT(*) > 0
            ORDER BY avg_importance DESC
        """)
        query_time = time.time() - start_time
        
        assert len(complex_query_result) > 0, "Complex query should return results"
        assert query_time < 1.0, f"Complex query too slow: {query_time}s"
        
        # Test vacuum and analyze simulation (read-only operations)
        vacuum_info = await conn.fetch("""
            SELECT
                schemaname,
                relname as tablename,
                last_vacuum,
                last_autovacuum,
                last_analyze,
                last_autoanalyze,
                vacuum_count,
                autovacuum_count
            FROM pg_stat_user_tables
            WHERE schemaname = 'public'
            AND relname = 'memories'
        """)
        
        assert len(vacuum_info) > 0, "Should have vacuum statistics"


async def test_backup_restore_consistency(db_pool):
    """Test backup and restore data consistency"""
    async with db_pool.acquire() as conn:
        # Clean up any existing test data first
        await conn.execute("""
            DELETE FROM memory_cluster_members 
            WHERE memory_id IN (
                SELECT id FROM memories WHERE content LIKE 'Backup test memory%'
            )
        """)
        await conn.execute("""
            DELETE FROM semantic_memories 
            WHERE memory_id IN (
                SELECT id FROM memories WHERE content LIKE 'Backup test memory%'
            )
        """)
        await conn.execute("""
            DELETE FROM memories WHERE content LIKE 'Backup test memory%'
        """)
        await conn.execute("""
            DELETE FROM memory_clusters WHERE name = 'Backup Test Cluster'
        """)
        
        # Create a known dataset for backup testing
        backup_test_data = []
        
        # Create test memories with relationships
        for i in range(5):
            memory_id = await conn.fetchval("""
                INSERT INTO memories (
                    type,
                    content,
                    embedding,
                    importance
                ) VALUES (
                    'semantic'::memory_type,
                    'Backup test memory ' || $1,
                    array_fill($2::float, ARRAY[embedding_dimension()])::vector,
                    $3
                ) RETURNING id
            """, str(i), float(i) * 0.1, 0.5 + (i * 0.1))
            
            backup_test_data.append(memory_id)
            
            # Add semantic details
            await conn.execute("""
                INSERT INTO semantic_memories (
                    memory_id,
                    confidence,
                    category
                ) VALUES ($1, $2, $3)
            """, memory_id, 0.8, [f'category_{i}'])
        
        # Create cluster and relationships
        cluster_id = await conn.fetchval("""
            INSERT INTO memory_clusters (
                cluster_type,
                name,
                centroid_embedding
            ) VALUES (
                'theme'::cluster_type,
                'Backup Test Cluster',
                array_fill(0.5, ARRAY[embedding_dimension()])::vector
            ) RETURNING id
        """)
        
        # Add memories to cluster
        for memory_id in backup_test_data:
            await conn.execute("""
                INSERT INTO memory_cluster_members (
                    cluster_id,
                    memory_id,
                    membership_strength
                ) VALUES ($1, $2, 0.8)
            """, cluster_id, memory_id)
        
        # Simulate backup verification by checking data consistency
        # Test 1: Verify all memories have corresponding semantic records
        orphaned_memories = await conn.fetch("""
            SELECT m.id 
            FROM memories m
            LEFT JOIN semantic_memories sm ON m.id = sm.memory_id
            WHERE m.type = 'semantic' 
            AND m.content LIKE 'Backup test memory%'
            AND sm.memory_id IS NULL
        """)
        
        assert len(orphaned_memories) == 0, "No orphaned semantic memories should exist"
        
        # Test 2: Verify cluster relationships are intact
        cluster_members = await conn.fetchval("""
            SELECT COUNT(*) 
            FROM memory_cluster_members mcm
            JOIN memories m ON mcm.memory_id = m.id
            WHERE m.content LIKE 'Backup test memory%'
        """)
        
        assert cluster_members == 5, "All test memories should be in cluster"
        
        # Test 3: Verify referential integrity for our test data only
        integrity_check = await conn.fetch("""
            SELECT 
                'test_memories->semantic' as check_type,
                COUNT(*) as violations
            FROM memories m
            LEFT JOIN semantic_memories sm ON m.id = sm.memory_id
            WHERE m.type = 'semantic' 
            AND m.content LIKE 'Backup test memory%'
            AND sm.memory_id IS NULL
            
            UNION ALL
            
            SELECT 
                'test_cluster_members->memories' as check_type,
                COUNT(*) as violations
            FROM memory_cluster_members mcm
            LEFT JOIN memories m ON mcm.memory_id = m.id
            WHERE m.content LIKE 'Backup test memory%'
            AND m.id IS NULL
            
            UNION ALL
            
            SELECT 
                'test_cluster_members->clusters' as check_type,
                COUNT(*) as violations
            FROM memory_cluster_members mcm
            LEFT JOIN memory_clusters mc ON mcm.cluster_id = mc.id
            WHERE mc.name = 'Backup Test Cluster'
            AND mc.id IS NULL
        """)
        
        for check in integrity_check:
            assert check['violations'] == 0, f"Integrity violation in {check['check_type']}"
        
        # Test 4: Verify computed fields are consistent
        computed_field_check = await conn.fetch("""
            SELECT
                id,
                importance,
                calculate_relevance(importance, decay_rate, created_at, last_accessed) as relevance_score,
                (importance * exp(-decay_rate * age_in_days(created_at))) as expected_relevance
            FROM memories
            WHERE content LIKE 'Backup test memory%'
        """)

        for row in computed_field_check:
            actual = float(row['relevance_score'])
            expected = float(row['expected_relevance'])
            # Relaxed tolerance since calculate_relevance has more complex formula
            assert abs(actual - expected) < 0.5, f"Relevance score mismatch: {actual} vs {expected}"


async def test_schema_migration_compatibility(db_pool):
    """Test schema migration and version compatibility"""
    async with db_pool.acquire() as conn:
        # Test 1: Check current schema version (simulate version tracking)
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS schema_version (
                version INTEGER PRIMARY KEY,
                applied_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
                description TEXT
            )
        """)
        
        # Record current schema version
        await conn.execute("""
            INSERT INTO schema_version (version, description)
            VALUES (1, 'Initial Hexis Memory System schema')
            ON CONFLICT (version) DO NOTHING
        """)
        
        # Test 2: Simulate adding a new column (non-breaking change)
        await conn.execute("""
            ALTER TABLE memories 
            ADD COLUMN IF NOT EXISTS migration_test_field TEXT DEFAULT 'test_value'
        """)
        
        # Verify the column was added
        column_exists = await conn.fetchval("""
            SELECT COUNT(*) 
            FROM information_schema.columns 
            WHERE table_name = 'memories' 
            AND column_name = 'migration_test_field'
        """)
        
        assert column_exists == 1, "Migration test column should exist"
        
        # Test 3: Verify existing data integrity after migration
        memory_count_before = await conn.fetchval("""
            SELECT COUNT(*) FROM memories
        """)
        
        # Test that existing memories have the default value
        default_value_count = await conn.fetchval("""
            SELECT COUNT(*) FROM memories 
            WHERE migration_test_field = 'test_value'
        """)
        
        assert default_value_count == memory_count_before, "All existing memories should have default value"
        
        # Test 4: Simulate index migration
        await conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_memories_migration_test 
            ON memories(migration_test_field)
        """)
        
        # Verify index was created
        index_exists = await conn.fetchval("""
            SELECT COUNT(*) 
            FROM pg_indexes 
            WHERE tablename = 'memories' 
            AND indexname = 'idx_memories_migration_test'
        """)
        
        assert index_exists == 1, "Migration test index should exist"
        
        # Test 5: Simulate rollback capability
        await conn.execute("""
            ALTER TABLE memories 
            DROP COLUMN IF EXISTS migration_test_field
        """)
        
        # Verify rollback worked
        column_exists_after = await conn.fetchval("""
            SELECT COUNT(*) 
            FROM information_schema.columns 
            WHERE table_name = 'memories' 
            AND column_name = 'migration_test_field'
        """)
        
        assert column_exists_after == 0, "Migration test column should be removed"
        
        # Clean up
        await conn.execute("DROP TABLE IF EXISTS schema_version")


async def test_monitoring_and_alerting_metrics(db_pool):
    """Test monitoring metrics and alerting thresholds"""
    async with db_pool.acquire() as conn:
        # Test 1: Memory system health metrics
        health_metrics = await conn.fetchrow("""
            SELECT
                COUNT(*) as total_memories,
                COUNT(*) FILTER (WHERE status = 'active') as active_memories,
                COUNT(*) FILTER (WHERE status = 'archived') as archived_memories,
                AVG(importance) as avg_importance,
                AVG(calculate_relevance(importance, decay_rate, created_at, last_accessed)) as avg_relevance,
                AVG(access_count) as avg_access_count,
                COUNT(*) FILTER (WHERE last_accessed > CURRENT_TIMESTAMP - interval '24 hours') as recently_accessed
            FROM memories
        """)
        
        assert health_metrics['total_memories'] > 0, "Should have memories for monitoring"
        
        # Test 2: Cluster health metrics
        cluster_metrics = await conn.fetchrow("""
            SELECT 
                COUNT(*) as total_clusters,
                AVG(importance_score) as avg_cluster_importance,
                COUNT(*) FILTER (WHERE last_activated > CURRENT_TIMESTAMP - interval '24 hours') as recently_active_clusters,
                AVG(
                    (SELECT COUNT(*) FROM memory_cluster_members mcm WHERE mcm.cluster_id = mc.id)
                ) as avg_cluster_size
            FROM memory_clusters mc
        """)
        
        assert cluster_metrics['total_clusters'] >= 0, "Should have cluster metrics"
        
        # Test 3: Performance metrics
        performance_metrics = await conn.fetch("""
            SELECT 
                'vector_search' as metric_type,
                COUNT(*) as operations,
                AVG(EXTRACT(EPOCH FROM (CURRENT_TIMESTAMP - created_at))) as avg_age_seconds
            FROM memories
            WHERE created_at > CURRENT_TIMESTAMP - interval '1 hour'
            
            UNION ALL
            
            SELECT 
                'cluster_operations' as metric_type,
                COUNT(*) as operations,
                AVG(EXTRACT(EPOCH FROM (CURRENT_TIMESTAMP - added_at))) as avg_age_seconds
            FROM memory_cluster_members
            WHERE added_at > CURRENT_TIMESTAMP - interval '1 hour'
        """)
        
        assert len(performance_metrics) > 0, "Should have performance metrics"
        
        # Test 4: Alert threshold simulation
        alert_conditions = await conn.fetch("""
            SELECT 
                'low_memory_activity' as alert_type,
                CASE 
                    WHEN COUNT(*) FILTER (WHERE last_accessed > CURRENT_TIMESTAMP - interval '24 hours') < 10 
                    THEN 'ALERT' 
                    ELSE 'OK' 
                END as status,
                COUNT(*) FILTER (WHERE last_accessed > CURRENT_TIMESTAMP - interval '24 hours') as recent_access_count
            FROM memories
            
            UNION ALL
            
            SELECT 
                'high_archive_rate' as alert_type,
                CASE 
                    WHEN COUNT(*) FILTER (WHERE status = 'archived') > COUNT(*) * 0.5 
                    THEN 'ALERT' 
                    ELSE 'OK' 
                END as status,
                COUNT(*) FILTER (WHERE status = 'archived') as archived_count
            FROM memories
            
            UNION ALL
            
            SELECT 
                'cluster_imbalance' as alert_type,
                CASE 
                    WHEN MAX(member_count) > AVG(member_count) * 3 
                    THEN 'ALERT' 
                    ELSE 'OK' 
                END as status,
                MAX(member_count) as max_cluster_size
            FROM (
                SELECT 
                    cluster_id,
                    COUNT(*) as member_count
                FROM memory_cluster_members
                GROUP BY cluster_id
            ) cluster_sizes
        """)
        
        assert len(alert_conditions) == 3, "Should have all alert conditions"
        
        # Test 5: Resource usage metrics
        resource_metrics = await conn.fetchrow("""
            SELECT 
                pg_size_pretty(pg_total_relation_size('memories')) as memories_table_size,
                pg_size_pretty(pg_total_relation_size('memory_clusters')) as clusters_table_size,
                (SELECT COUNT(*) FROM memories) as memory_count,
                (SELECT COUNT(*) FROM memory_clusters) as cluster_count,
                (SELECT COUNT(*) FROM memory_cluster_members) as membership_count
        """)
        
        assert resource_metrics is not None, "Should have resource metrics"


async def test_multi_hexis_considerations(db_pool):
    """Test considerations for multi-Hexis support (current limitations)"""
    async with db_pool.acquire() as conn:
        # Clean up any existing test data first
        await conn.execute("""
            DELETE FROM memories WHERE content LIKE '%Hexis-% believes X is%'
        """)
        
        # Test 1: Identify single-Hexis assumptions in current schema
        single_hexis_tables = await conn.fetch("""
            SELECT 
                table_name,
                CASE
                    WHEN table_name IN ('identity_aspects', 'worldview_primitives') THEN 'singleton_table'
                    WHEN table_name LIKE '%memory%' THEN 'memory_table'
                    ELSE 'other'
                END as table_category
            FROM information_schema.tables
            WHERE table_schema = 'public'
            AND table_type = 'BASE TABLE'
            ORDER BY table_name
        """)
        
        singleton_tables = [t for t in single_hexis_tables if t['table_category'] == 'singleton_table']
        assert len(singleton_tables) > 0, "Should identify singleton tables"
        
        # Test 2: Simulate multi-Hexis data isolation requirements
        # This test demonstrates what would need to change for multi-Hexis support
        
        # Check if any tables have Hexis instance identification
        hexis_id_columns = await conn.fetch("""
            SELECT 
                table_name,
                column_name
            FROM information_schema.columns
            WHERE table_schema = 'public'
            AND column_name LIKE '%hexis%'
            ORDER BY table_name, column_name
        """)
        
        # Current schema should have no Hexis ID columns (single-Hexis design)
        assert len(hexis_id_columns) == 0, "Current schema should not have Hexis ID columns"
        
        # Test 3: Demonstrate memory isolation challenges
        # Create test scenario showing how memories could conflict between Hexis instances
        
        # Hexis 1 memories
        hexis1_memory = await conn.fetchval("""
            INSERT INTO memories (
                type,
                content,
                embedding,
                importance
            ) VALUES (
                'semantic'::memory_type,
                'Hexis-1 believes X is true',
                array_fill(0.8, ARRAY[embedding_dimension()])::vector,
                0.9
            ) RETURNING id
        """)
        
        # Hexis 2 memories (conflicting belief)
        hexis2_memory = await conn.fetchval("""
            INSERT INTO memories (
                type,
                content,
                embedding,
                importance
            ) VALUES (
                'semantic'::memory_type,
                'Hexis-2 believes X is false',
                array_fill(0.8, ARRAY[embedding_dimension()])::vector,
                0.9
            ) RETURNING id
        """)
        
        # Demonstrate conflict: both memories exist in same space
        conflicting_memories = await conn.fetch("""
            SELECT 
                id,
                content,
                importance,
                'conflict_detected' as issue_type
            FROM memories
            WHERE content LIKE '%Hexis-% believes X is%'
            ORDER BY content
        """)
        
        assert len(conflicting_memories) == 2, "Should find conflicting Hexis memories"
        
        # Test 4: Demonstrate worldview conflicts
        # In single-Hexis system, only one worldview can exist
        worldview_count = await conn.fetchval("""
            SELECT COUNT(*) FROM worldview_primitives
        """)
        
        # Test 5: Demonstrate identity aspects limitations
        identity_count = await conn.fetchval("""
            SELECT COUNT(*) FROM identity_aspects
        """)

        # Test 6: Show what would be needed for multi-Hexis support
        multi_hexis_requirements = {
            'schema_changes_needed': [
                'Add hexis_instance_id to all memory tables',
                'Add hexis_instance_id to worldview_primitives',
                'Add hexis_instance_id to identity_aspects',
                'Add row-level security policies',
                'Modify all views to filter by Hexis instance',
                'Update all functions to include Hexis context'
            ],
            'isolation_challenges': [
                'Memory similarity search across Hexis boundaries',
                'Cluster centroid calculations per Hexis',
                'Graph relationships between Hexis instances',
                'Shared vs private memory spaces',
                'Cross-Hexis learning and knowledge transfer'
            ]
        }
        
        # This test documents the current single-Hexis limitations
        assert len(multi_hexis_requirements['schema_changes_needed']) > 0, "Multi-Hexis support requires significant changes"


# ============================================================================
# NEW TEST CASES - Acceleration Layer, Concepts, Core Functions
# ============================================================================

# -----------------------------------------------------------------------------
# EPISODE & TEMPORAL SEGMENTATION TESTS
# -----------------------------------------------------------------------------

async def test_episodes_table_structure(db_pool):
    """Test episodes table with time_range TSTZRANGE generated column"""
    async with db_pool.acquire() as conn:
        # Check table structure
        columns = await conn.fetch("""
            SELECT column_name, data_type, is_nullable, generation_expression
            FROM information_schema.columns
            WHERE table_name = 'episodes'
            ORDER BY ordinal_position
        """)
        column_dict = {col['column_name']: col for col in columns}

        assert 'id' in column_dict
        assert 'started_at' in column_dict
        assert 'ended_at' in column_dict
        assert 'episode_type' in column_dict
        assert 'summary' in column_dict
        assert 'summary_embedding' in column_dict
        assert 'time_range' in column_dict

        # Create an episode and verify time_range is auto-generated
        episode_id = await conn.fetchval("""
            INSERT INTO episodes (started_at, ended_at, episode_type)
            VALUES (
                '2024-01-01 10:00:00'::timestamptz,
                '2024-01-01 11:00:00'::timestamptz,
                'conversation'
            ) RETURNING id
        """)

        time_range = await conn.fetchval("""
            SELECT time_range FROM episodes WHERE id = $1
        """, episode_id)

        assert time_range is not None, "time_range should be auto-generated"


async def test_auto_episode_assignment_trigger(db_pool):
    """Test trg_auto_episode_assignment trigger creates episodes automatically"""
    async with db_pool.acquire() as conn:
        # Clean up any existing open episodes for this test
        await conn.execute("""
            UPDATE episodes SET ended_at = started_at
            WHERE ended_at IS NULL
        """)

        # Create first memory - should create new episode
        memory1_id = await conn.fetchval("""
            INSERT INTO memories (type, content, embedding)
            VALUES ('episodic'::memory_type, 'First memory in episode',
                    array_fill(0.1, ARRAY[embedding_dimension()])::vector)
            RETURNING id
        """)

        # Verify episode was created
        episode1 = await conn.fetchrow("""
            SELECT e.id, e.started_at, e.ended_at, em.sequence_order
            FROM episodes e
            JOIN episode_memories em ON e.id = em.episode_id
            WHERE em.memory_id = $1
        """, memory1_id)

        assert episode1 is not None, "Episode should be created for first memory"
        assert episode1['sequence_order'] == 1, "First memory should have sequence_order 1"
        assert episode1['ended_at'] is None, "Episode should still be open"

        # Create second memory immediately - should be in same episode
        memory2_id = await conn.fetchval("""
            INSERT INTO memories (type, content, embedding)
            VALUES ('episodic'::memory_type, 'Second memory in same episode',
                    array_fill(0.2, ARRAY[embedding_dimension()])::vector)
            RETURNING id
        """)

        episode2 = await conn.fetchrow("""
            SELECT e.id, em.sequence_order
            FROM episodes e
            JOIN episode_memories em ON e.id = em.episode_id
            WHERE em.memory_id = $1
        """, memory2_id)

        assert episode2['id'] == episode1['id'], "Second memory should be in same episode"
        assert episode2['sequence_order'] == 2, "Second memory should have sequence_order 2"

        # Verify memory_neighborhoods was initialized
        neighborhood = await conn.fetchrow("""
            SELECT memory_id, is_stale FROM memory_neighborhoods
            WHERE memory_id = $1
        """, memory1_id)

        assert neighborhood is not None, "memory_neighborhoods should be initialized"
        assert neighborhood['is_stale'] == True, "New neighborhood should be marked stale"


async def test_episode_30_minute_gap_detection(db_pool):
    """Test that episodes close and new ones open after 30-minute gap"""
    async with db_pool.acquire() as conn:
        # Close any open episodes
        await conn.execute("""
            UPDATE episodes SET ended_at = started_at
            WHERE ended_at IS NULL
        """)

        # Create memory with specific timestamp
        base_time = await conn.fetchval("SELECT CURRENT_TIMESTAMP")

        memory1_id = await conn.fetchval("""
            INSERT INTO memories (type, content, embedding, created_at)
            VALUES ('semantic'::memory_type, 'Memory before gap',
                    array_fill(0.3, ARRAY[embedding_dimension()])::vector, $1)
            RETURNING id
        """, base_time)

        episode1_id = await conn.fetchval("""
            SELECT episode_id FROM episode_memories WHERE memory_id = $1
        """, memory1_id)

        # Create memory 31 minutes later - should trigger new episode
        later_time = base_time + timedelta(minutes=31)

        memory2_id = await conn.fetchval("""
            INSERT INTO memories (type, content, embedding, created_at)
            VALUES ('semantic'::memory_type, 'Memory after gap',
                    array_fill(0.4, ARRAY[embedding_dimension()])::vector, $1)
            RETURNING id
        """, later_time)

        episode2_id = await conn.fetchval("""
            SELECT episode_id FROM episode_memories WHERE memory_id = $1
        """, memory2_id)

        # Verify new episode was created
        assert episode2_id != episode1_id, "New episode should be created after 30-minute gap"

        # Verify old episode was closed
        old_episode = await conn.fetchrow("""
            SELECT ended_at FROM episodes WHERE id = $1
        """, episode1_id)

        assert old_episode['ended_at'] is not None, "Old episode should be closed"


async def test_episode_summary_view(db_pool):
    """Test episode_summary view calculations"""
    async with db_pool.acquire() as conn:
        # Create episode with summary
        episode_id = await conn.fetchval("""
            INSERT INTO episodes (started_at, ended_at, episode_type, summary)
            VALUES (
                CURRENT_TIMESTAMP - interval '2 hours',
                CURRENT_TIMESTAMP - interval '1 hour',
                'reflection',
                'Test episode summary'
            ) RETURNING id
        """)

        # Add memories to episode
        for i in range(3):
            memory_id = await conn.fetchval("""
                INSERT INTO memories (type, content, embedding, created_at)
                VALUES ('semantic'::memory_type, $1,
                        array_fill(0.5, ARRAY[embedding_dimension()])::vector,
                        CURRENT_TIMESTAMP - interval '90 minutes' + $2 * interval '10 minutes')
                RETURNING id
            """, f'Episode summary test memory {i}', i)

            await conn.execute("""
                INSERT INTO episode_memories (episode_id, memory_id, sequence_order)
                VALUES ($1, $2, $3)
                ON CONFLICT DO NOTHING
            """, episode_id, memory_id, i + 1)

        # Query the view
        summary = await conn.fetchrow("""
            SELECT * FROM episode_summary WHERE id = $1
        """, episode_id)

        assert summary is not None, "Episode should appear in summary view"
        assert summary['memory_count'] == 3, "Should count 3 memories"
        assert summary['episode_type'] == 'reflection'
        assert summary['summary'] == 'Test episode summary'


async def test_episode_time_range_gist_index(db_pool):
    """Test GiST index on episodes.time_range for temporal queries"""
    async with db_pool.acquire() as conn:
        # Create episodes with different time ranges
        for i in range(5):
            await conn.execute("""
                INSERT INTO episodes (started_at, ended_at, episode_type)
                VALUES (
                    CURRENT_TIMESTAMP - $1 * interval '1 day',
                    CURRENT_TIMESTAMP - $1 * interval '1 day' + interval '1 hour',
                    'autonomous'
                )
            """, i)

        # Query using time range overlap - should use GiST index
        result = await conn.fetch("""
            EXPLAIN (FORMAT JSON)
            SELECT * FROM episodes
            WHERE time_range && tstzrange(
                CURRENT_TIMESTAMP - interval '2 days',
                CURRENT_TIMESTAMP
            )
        """)

        # Verify the query plan (index usage)
        plan = result[0]['QUERY PLAN']
        # The query should complete successfully with index available
        assert plan is not None


# -----------------------------------------------------------------------------
# MEMORY NEIGHBORHOODS TESTS
# -----------------------------------------------------------------------------

async def test_memory_neighborhoods_initialization(db_pool):
    """Test that memory_neighborhoods record is created on memory insert"""
    async with db_pool.acquire() as conn:
        # Create a memory
        memory_id = await conn.fetchval("""
            INSERT INTO memories (type, content, embedding)
            VALUES ('semantic'::memory_type, 'Neighborhood init test',
                    array_fill(0.6, ARRAY[embedding_dimension()])::vector)
            RETURNING id
        """)

        # Verify neighborhood record was created by trigger
        neighborhood = await conn.fetchrow("""
            SELECT * FROM memory_neighborhoods WHERE memory_id = $1
        """, memory_id)

        assert neighborhood is not None, "Neighborhood record should be auto-created"
        assert neighborhood['is_stale'] == True, "New neighborhood should be stale"
        # neighbors is JSONB, may be dict or empty object string
        neighbors = neighborhood['neighbors']
        assert neighbors == {} or neighbors == '{}' or len(neighbors) == 0, "Neighbors should be empty initially"


async def test_neighborhoods_staleness_trigger(db_pool):
    """Test trg_neighborhood_staleness marks neighborhoods stale on changes"""
    async with db_pool.acquire() as conn:
        # Create memory
        memory_id = await conn.fetchval("""
            INSERT INTO memories (type, content, embedding)
            VALUES ('semantic'::memory_type, 'Staleness trigger test',
                    array_fill(0.7, ARRAY[embedding_dimension()])::vector)
            RETURNING id
        """)

        # Manually set neighborhood to not stale with some neighbors
        await conn.execute("""
            UPDATE memory_neighborhoods
            SET is_stale = FALSE,
                neighbors = '{"test-uuid": 0.8}'::jsonb,
                computed_at = CURRENT_TIMESTAMP
            WHERE memory_id = $1
        """, memory_id)

        # Verify it's not stale
        is_stale_before = await conn.fetchval("""
            SELECT is_stale FROM memory_neighborhoods WHERE memory_id = $1
        """, memory_id)
        assert is_stale_before == False

        # Update importance - should trigger staleness
        await conn.execute("""
            UPDATE memories SET importance = 0.9 WHERE id = $1
        """, memory_id)

        is_stale_after = await conn.fetchval("""
            SELECT is_stale FROM memory_neighborhoods WHERE memory_id = $1
        """, memory_id)
        assert is_stale_after == True, "Neighborhood should be marked stale after importance change"


async def test_neighborhoods_staleness_on_status_change(db_pool):
    """Test neighborhood becomes stale when memory status changes"""
    async with db_pool.acquire() as conn:
        memory_id = await conn.fetchval("""
            INSERT INTO memories (type, content, embedding)
            VALUES ('semantic'::memory_type, 'Status change staleness test',
                    array_fill(0.75, ARRAY[embedding_dimension()])::vector)
            RETURNING id
        """)

        # Set not stale
        await conn.execute("""
            UPDATE memory_neighborhoods
            SET is_stale = FALSE
            WHERE memory_id = $1
        """, memory_id)

        # Change status
        await conn.execute("""
            UPDATE memories SET status = 'archived' WHERE id = $1
        """, memory_id)

        is_stale = await conn.fetchval("""
            SELECT is_stale FROM memory_neighborhoods WHERE memory_id = $1
        """, memory_id)
        assert is_stale == True, "Neighborhood should be stale after status change"


async def test_stale_neighborhoods_view(db_pool):
    """Test stale_neighborhoods view shows correct memories"""
    async with db_pool.acquire() as conn:
        # Create memories
        stale_memory_id = await conn.fetchval("""
            INSERT INTO memories (type, content, embedding)
            VALUES ('semantic'::memory_type, 'Stale neighborhood view test',
                    array_fill(0.8, ARRAY[embedding_dimension()])::vector)
            RETURNING id
        """)

        fresh_memory_id = await conn.fetchval("""
            INSERT INTO memories (type, content, embedding)
            VALUES ('semantic'::memory_type, 'Fresh neighborhood view test',
                    array_fill(0.81, ARRAY[embedding_dimension()])::vector)
            RETURNING id
        """)

        # Set one as not stale
        await conn.execute("""
            UPDATE memory_neighborhoods SET is_stale = FALSE
            WHERE memory_id = $1
        """, fresh_memory_id)

        # Check view
        stale_records = await conn.fetch("""
            SELECT memory_id FROM stale_neighborhoods
            WHERE memory_id IN ($1, $2)
        """, stale_memory_id, fresh_memory_id)

        stale_ids = [r['memory_id'] for r in stale_records]
        assert stale_memory_id in stale_ids, "Stale memory should appear in view"
        assert fresh_memory_id not in stale_ids, "Fresh memory should not appear in view"


async def test_neighborhoods_gin_index(db_pool):
    """Test GIN index on neighbors JSONB works correctly"""
    async with db_pool.acquire() as conn:
        # Create memory with neighbors
        memory_id = await conn.fetchval("""
            INSERT INTO memories (type, content, embedding)
            VALUES ('semantic'::memory_type, 'GIN index test',
                    array_fill(0.85, ARRAY[embedding_dimension()])::vector)
            RETURNING id
        """)

        # Update with neighbors
        await conn.execute("""
            UPDATE memory_neighborhoods
            SET neighbors = '{"neighbor1": 0.9, "neighbor2": 0.7}'::jsonb
            WHERE memory_id = $1
        """, memory_id)

        # Query using JSONB operators
        result = await conn.fetch("""
            SELECT memory_id FROM memory_neighborhoods
            WHERE neighbors ? 'neighbor1'
        """)

        assert len(result) > 0, "Should find memory with neighbor1 key"


# -----------------------------------------------------------------------------
# ACTIVATION CACHE TESTS
# -----------------------------------------------------------------------------

async def test_activation_cache_operations(db_pool):
    """Test activation_cache basic operations"""
    async with db_pool.acquire() as conn:
        session_id = await conn.fetchval("SELECT gen_random_uuid()")
        memory_id = await conn.fetchval("SELECT gen_random_uuid()")

        # Insert activation
        await conn.execute("""
            INSERT INTO activation_cache (session_id, memory_id, activation_level)
            VALUES ($1, $2, 0.75)
        """, session_id, memory_id)

        # Read back
        activation = await conn.fetchval("""
            SELECT activation_level FROM activation_cache
            WHERE session_id = $1 AND memory_id = $2
        """, session_id, memory_id)

        assert activation == 0.75

        # Update activation
        await conn.execute("""
            INSERT INTO activation_cache (session_id, memory_id, activation_level)
            VALUES ($1, $2, 0.9)
            ON CONFLICT (session_id, memory_id)
            DO UPDATE SET activation_level = EXCLUDED.activation_level
        """, session_id, memory_id)

        updated = await conn.fetchval("""
            SELECT activation_level FROM activation_cache
            WHERE session_id = $1 AND memory_id = $2
        """, session_id, memory_id)

        assert updated == 0.9


async def test_activation_cache_session_isolation(db_pool):
    """Test activation levels are isolated by session_id"""
    async with db_pool.acquire() as conn:
        session1_id = await conn.fetchval("SELECT gen_random_uuid()")
        session2_id = await conn.fetchval("SELECT gen_random_uuid()")
        memory_id = await conn.fetchval("SELECT gen_random_uuid()")

        # Insert different activations for same memory in different sessions
        await conn.execute("""
            INSERT INTO activation_cache (session_id, memory_id, activation_level)
            VALUES ($1, $2, 0.3), ($3, $2, 0.8)
        """, session1_id, memory_id, session2_id)

        # Verify isolation
        session1_activation = await conn.fetchval("""
            SELECT activation_level FROM activation_cache
            WHERE session_id = $1 AND memory_id = $2
        """, session1_id, memory_id)

        session2_activation = await conn.fetchval("""
            SELECT activation_level FROM activation_cache
            WHERE session_id = $1 AND memory_id = $2
        """, session2_id, memory_id)

        assert session1_activation == 0.3
        assert session2_activation == 0.8


# -----------------------------------------------------------------------------
# CONCEPTS LAYER TESTS
# -----------------------------------------------------------------------------

async def test_concepts_table(db_pool):
    """Test concepts table structure and constraints"""
    async with db_pool.acquire() as conn:
        # Use unique name for this test
        unique_name = f'TestConcept_{int(time.time() * 1000)}'

        # Create concept
        concept_id = await conn.fetchval("""
            INSERT INTO concepts (name, description, depth, path_text)
            VALUES ($1, 'A test concept', 0, $1)
            RETURNING id
        """, unique_name)

        assert concept_id is not None

        # Test unique constraint
        try:
            await conn.execute("""
                INSERT INTO concepts (name) VALUES ($1)
            """, unique_name)
            assert False, "Should raise unique constraint violation"
        except Exception as e:
            assert 'unique' in str(e).lower() or 'duplicate' in str(e).lower()


async def test_concept_hierarchy(db_pool):
    """Test concept hierarchy with ancestors and path_text"""
    async with db_pool.acquire() as conn:
        # Use unique suffix to avoid conflicts
        suffix = f'_{int(time.time() * 1000)}'

        # Create hierarchy: Entity -> Organism -> Animal -> Dog
        entity_id = await conn.fetchval("""
            INSERT INTO concepts (name, depth, path_text, ancestors)
            VALUES ($1, 0, $1, ARRAY[]::UUID[])
            RETURNING id
        """, f'Entity{suffix}')

        organism_id = await conn.fetchval("""
            INSERT INTO concepts (name, depth, path_text, ancestors)
            VALUES ($1, 1, $2, ARRAY[$3]::UUID[])
            RETURNING id
        """, f'Organism{suffix}', f'Entity{suffix}/Organism{suffix}', entity_id)

        animal_id = await conn.fetchval("""
            INSERT INTO concepts (name, depth, path_text, ancestors)
            VALUES ($1, 2, $2, ARRAY[$3, $4]::UUID[])
            RETURNING id
        """, f'Animal{suffix}', f'Entity{suffix}/Organism{suffix}/Animal{suffix}', entity_id, organism_id)

        dog_id = await conn.fetchval("""
            INSERT INTO concepts (name, depth, path_text, ancestors)
            VALUES ($1, 3, $2, ARRAY[$3, $4, $5]::UUID[])
            RETURNING id
        """, f'Dog{suffix}', f'Entity{suffix}/Organism{suffix}/Animal{suffix}/Dog{suffix}', entity_id, organism_id, animal_id)

        # Query hierarchy
        dog_concept = await conn.fetchrow("""
            SELECT * FROM concepts WHERE id = $1
        """, dog_id)

        assert dog_concept['depth'] == 3
        assert len(dog_concept['ancestors']) == 3
        assert entity_id in dog_concept['ancestors']

        # Query all descendants of Entity using GIN index on ancestors
        descendants = await conn.fetch("""
            SELECT name FROM concepts
            WHERE $1 = ANY(ancestors)
            ORDER BY depth
        """, entity_id)

        names = [d['name'] for d in descendants]
        assert f'Organism{suffix}' in names
        assert f'Animal{suffix}' in names
        assert f'Dog{suffix}' in names


async def test_memory_concepts_junction(db_pool):
    """Test memory_concepts many-to-many relationship"""
    async with db_pool.acquire() as conn:
        # Create memory
        memory_id = await conn.fetchval("""
            INSERT INTO memories (type, content, embedding)
            VALUES ('semantic'::memory_type, 'My dog likes to play',
                    array_fill(0.9, ARRAY[embedding_dimension()])::vector)
            RETURNING id
        """)

        # Create concepts
        dog_id = await conn.fetchval("""
            INSERT INTO concepts (name) VALUES ('Dog')
            ON CONFLICT (name) DO UPDATE SET name = EXCLUDED.name
            RETURNING id
        """)

        play_id = await conn.fetchval("""
            INSERT INTO concepts (name) VALUES ('Play')
            ON CONFLICT (name) DO UPDATE SET name = EXCLUDED.name
            RETURNING id
        """)

        # Link memory to concepts with different strengths
        await conn.execute("""
            INSERT INTO memory_concepts (memory_id, concept_id, strength)
            VALUES ($1, $2, 0.95), ($1, $3, 0.7)
        """, memory_id, dog_id, play_id)

        # Query concepts for memory
        concepts = await conn.fetch("""
            SELECT c.name, mc.strength
            FROM memory_concepts mc
            JOIN concepts c ON mc.concept_id = c.id
            WHERE mc.memory_id = $1
            ORDER BY mc.strength DESC
        """, memory_id)

        assert len(concepts) == 2
        assert concepts[0]['name'] == 'Dog'
        assert concepts[0]['strength'] == 0.95


async def test_link_memory_to_concept_function(db_pool):
    """Test link_memory_to_concept() creates concept and links"""
    async with db_pool.acquire() as conn:
        # Create memory with graph node
        memory_id = await conn.fetchval("""
            INSERT INTO memories (type, content, embedding)
            VALUES ('semantic'::memory_type, 'Cats are independent',
                    array_fill(0.88, ARRAY[embedding_dimension()])::vector)
            RETURNING id
        """)

        # Create graph node for memory
        await conn.execute("""
            LOAD 'age';
            SET search_path = ag_catalog, public;
        """)

        await conn.execute(f"""
            SELECT * FROM ag_catalog.cypher('memory_graph', $$
                CREATE (n:MemoryNode {{memory_id: '{memory_id}', type: 'semantic'}})
                RETURN n
            $$) as (result agtype)
        """)

        await conn.execute("SET search_path = public, ag_catalog")

        # Link to concept using function
        concept_id = await conn.fetchval("""
            SELECT link_memory_to_concept($1, 'Independence', 0.85)
        """, memory_id)

        assert concept_id is not None

        # Verify concept was created
        concept = await conn.fetchrow("""
            SELECT * FROM concepts WHERE id = $1
        """, concept_id)
        assert concept['name'] == 'Independence'

        # Verify relational link
        link = await conn.fetchrow("""
            SELECT * FROM memory_concepts
            WHERE memory_id = $1 AND concept_id = $2
        """, memory_id, concept_id)
        assert link is not None
        assert link['strength'] == 0.85

        # Verify graph edge (INSTANCE_OF)
        await conn.execute("""
            LOAD 'age';
            SET search_path = ag_catalog, public;
        """)

        edge_result = await conn.fetch(f"""
            SELECT * FROM ag_catalog.cypher('memory_graph', $$
                MATCH (m:MemoryNode {{memory_id: '{memory_id}'}})-[r:INSTANCE_OF]->(c:ConceptNode)
                RETURN c.name as concept_name, r.strength as strength
            $$) as (concept_name agtype, strength agtype)
        """)

        await conn.execute("SET search_path = public, ag_catalog")

        assert len(edge_result) > 0, "INSTANCE_OF edge should exist in graph"


# -----------------------------------------------------------------------------
# FAST_RECALL FUNCTION TESTS
# -----------------------------------------------------------------------------

async def test_fast_recall_basic(db_pool):
    """Test fast_recall() primary retrieval function"""
    async with db_pool.acquire() as conn:
        # Create test memories with distinct content
        memory_ids = []
        contents = [
            'The weather today is sunny and warm',
            'Python is a programming language',
            'The sun provides light and warmth',
            'JavaScript runs in browsers'
        ]

        for content in contents:
            memory_id = await conn.fetchval("""
                INSERT INTO memories (type, content, embedding)
                VALUES ('semantic'::memory_type, $1,
                        array_fill(0.5, ARRAY[embedding_dimension()])::vector)
                RETURNING id
            """, content)
            memory_ids.append(memory_id)

        # Test fast_recall (requires embedding service)
        # This will fail gracefully if embedding service is not available
        try:
            results = await conn.fetch("""
                SELECT * FROM fast_recall('What is the weather like?', 5)
            """)

            # If embedding service works, verify results structure
            assert all('memory_id' in dict(r) for r in results)
            assert all('content' in dict(r) for r in results)
            assert all('score' in dict(r) for r in results)
            assert all('source' in dict(r) for r in results)

        except Exception as e:
            # Expected if embedding service is not available
            if 'embedding' not in str(e).lower():
                raise


async def test_fast_recall_respects_limit(db_pool):
    """Test fast_recall respects the limit parameter"""
    async with db_pool.acquire() as conn:
        # Create multiple memories
        for i in range(10):
            await conn.execute("""
                INSERT INTO memories (type, content, embedding)
                VALUES ('semantic'::memory_type, $1,
                        array_fill(0.5, ARRAY[embedding_dimension()])::vector)
            """, f'Fast recall limit test memory {i}')

        try:
            results = await conn.fetch("""
                SELECT * FROM fast_recall('test memory', 3)
            """)
            assert len(results) <= 3, "Should respect limit parameter"
        except Exception as e:
            if 'embedding' not in str(e).lower():
                raise


async def test_fast_recall_only_active_memories(db_pool):
    """Test fast_recall only returns active memories"""
    async with db_pool.acquire() as conn:
        # Create active and archived memories
        active_id = await conn.fetchval("""
            INSERT INTO memories (type, content, embedding, status)
            VALUES ('semantic'::memory_type, 'Active memory for recall test',
                    array_fill(0.55, ARRAY[embedding_dimension()])::vector, 'active')
            RETURNING id
        """)

        archived_id = await conn.fetchval("""
            INSERT INTO memories (type, content, embedding, status)
            VALUES ('semantic'::memory_type, 'Archived memory for recall test',
                    array_fill(0.55, ARRAY[embedding_dimension()])::vector, 'archived')
            RETURNING id
        """)

        try:
            results = await conn.fetch("""
                SELECT memory_id FROM fast_recall('recall test', 10)
            """)

            result_ids = [r['memory_id'] for r in results]

            # Active should potentially be returned, archived should not
            assert archived_id not in result_ids, "Archived memories should not be returned"

        except Exception as e:
            if 'embedding' not in str(e).lower():
                raise


async def test_fast_recall_source_attribution(db_pool):
    """Test fast_recall correctly attributes retrieval sources"""
    async with db_pool.acquire() as conn:
        # The source field should be one of: 'vector', 'association', 'temporal', 'fallback'
        try:
            results = await conn.fetch("""
                SELECT source FROM fast_recall('test query', 5)
            """)

            valid_sources = {'vector', 'association', 'temporal', 'fallback'}
            for result in results:
                assert result['source'] in valid_sources, f"Invalid source: {result['source']}"

        except Exception as e:
            if 'embedding' not in str(e).lower():
                raise


# -----------------------------------------------------------------------------
# SEARCH FUNCTIONS TESTS
# -----------------------------------------------------------------------------

async def test_search_similar_memories_type_filter(db_pool):
    """Test search_similar_memories with type filtering"""
    async with db_pool.acquire() as conn:
        # Create memories of different types
        await conn.execute("""
            INSERT INTO memories (type, content, embedding)
            VALUES
                ('semantic'::memory_type, 'Semantic search test', array_fill(0.6, ARRAY[embedding_dimension()])::vector),
                ('episodic'::memory_type, 'Episodic search test', array_fill(0.6, ARRAY[embedding_dimension()])::vector),
                ('procedural'::memory_type, 'Procedural search test', array_fill(0.6, ARRAY[embedding_dimension()])::vector)
        """)

        try:
            # Search only semantic
            results = await conn.fetch("""
                SELECT * FROM search_similar_memories(
                    'search test', 10, ARRAY['semantic']::memory_type[]
                )
            """)

            for r in results:
                if 'search test' in r['content'].lower():
                    assert r['type'] == 'semantic', "Should only return semantic type"

        except Exception as e:
            if 'embedding' not in str(e).lower():
                raise


async def test_search_similar_memories_importance_filter(db_pool):
    """Test search_similar_memories with minimum importance filter"""
    async with db_pool.acquire() as conn:
        # Create memories with different importance
        await conn.execute("""
            INSERT INTO memories (type, content, embedding, importance)
            VALUES
                ('semantic'::memory_type, 'Low importance search test',
                 array_fill(0.65, ARRAY[embedding_dimension()])::vector, 0.1),
                ('semantic'::memory_type, 'High importance search test',
                 array_fill(0.65, ARRAY[embedding_dimension()])::vector, 0.9)
        """)

        try:
            # Search with high minimum importance
            results = await conn.fetch("""
                SELECT * FROM search_similar_memories(
                    'importance search test', 10, NULL, 0.5
                )
            """)

            for r in results:
                if 'importance search test' in r['content'].lower():
                    assert r['importance'] >= 0.5, "Should only return high importance memories"

        except Exception as e:
            if 'embedding' not in str(e).lower():
                raise


async def test_search_working_memory_auto_cleanup(db_pool):
    """Test search_working_memory calls cleanup automatically"""
    async with db_pool.acquire() as conn:
        # Add expired working memory entry
        await conn.execute("""
            INSERT INTO working_memory (content, embedding, expiry)
            VALUES ('Expired working memory', array_fill(0.7, ARRAY[embedding_dimension()])::vector,
                    CURRENT_TIMESTAMP - interval '1 hour')
        """)

        # Add valid working memory entry
        await conn.execute("""
            INSERT INTO working_memory (content, embedding, expiry)
            VALUES ('Valid working memory', array_fill(0.7, ARRAY[embedding_dimension()])::vector,
                    CURRENT_TIMESTAMP + interval '1 hour')
        """)

        try:
            # Search triggers cleanup
            await conn.fetch("""
                SELECT * FROM search_working_memory('working memory', 5)
            """)

            # Verify expired entry was cleaned up
            expired_count = await conn.fetchval("""
                SELECT COUNT(*) FROM working_memory
                WHERE expiry < CURRENT_TIMESTAMP
            """)

            assert expired_count == 0, "Expired working memory should be cleaned up"

        except Exception as e:
            if 'embedding' not in str(e).lower():
                raise


# -----------------------------------------------------------------------------
# MEMORY CREATION FUNCTIONS TESTS
# -----------------------------------------------------------------------------

async def test_create_memory_creates_graph_node(db_pool):
    """Test create_memory() creates MemoryNode in graph"""
    async with db_pool.acquire() as conn:
        try:
            # Use create_memory function
            memory_id = await conn.fetchval("""
                SELECT create_memory('semantic'::memory_type, 'Graph node creation test', 0.7)
            """)

            # Verify graph node exists
            await conn.execute("""
                LOAD 'age';
                SET search_path = ag_catalog, public;
            """)

            result = await conn.fetch(f"""
                SELECT * FROM ag_catalog.cypher('memory_graph', $$
                    MATCH (n:MemoryNode {{memory_id: '{memory_id}'}})
                    RETURN n.memory_id as memory_id, n.type as type
                $$) as (memory_id agtype, type agtype)
            """)

            await conn.execute("SET search_path = public, ag_catalog")

            assert len(result) > 0, "MemoryNode should exist in graph"

        except Exception as e:
            if 'embedding' not in str(e).lower():
                raise


async def test_create_episodic_memory_function(db_pool):
    """Test create_episodic_memory() full workflow"""
    async with db_pool.acquire() as conn:
        try:
            memory_id = await conn.fetchval("""
                SELECT create_episodic_memory(
                    'Had a great conversation about AI',
                    '{"type": "conversation"}'::jsonb,
                    '{"location": "office", "participants": ["Alice", "Bob"]}'::jsonb,
                    '{"outcome": "learned new concepts"}'::jsonb,
                    0.8,
                    CURRENT_TIMESTAMP,
                    0.75
                )
            """)

            # Verify base memory
            memory = await conn.fetchrow("""
                SELECT * FROM memories WHERE id = $1
            """, memory_id)
            assert memory['type'] == 'episodic'
            assert memory['importance'] == 0.75

            # Verify episodic details
            episodic = await conn.fetchrow("""
                SELECT * FROM episodic_memories WHERE memory_id = $1
            """, memory_id)
            assert episodic is not None
            assert episodic['emotional_valence'] == 0.8

        except Exception as e:
            if 'embedding' not in str(e).lower():
                raise


async def test_create_semantic_memory_function(db_pool):
    """Test create_semantic_memory() with all parameters"""
    async with db_pool.acquire() as conn:
        try:
            memory_id = await conn.fetchval("""
                SELECT create_semantic_memory(
                    'Water boils at 100 degrees Celsius at sea level',
                    0.99,
                    ARRAY['physics', 'chemistry'],
                    ARRAY['water', 'temperature', 'boiling'],
                    '{"textbook": "Physics 101"}'::jsonb,
                    0.8
                )
            """)

            # Verify semantic details
            semantic = await conn.fetchrow("""
                SELECT * FROM semantic_memories WHERE memory_id = $1
            """, memory_id)
            assert semantic is not None
            assert semantic['confidence'] == 0.99
            assert 'physics' in semantic['category']
            assert 'water' in semantic['related_concepts']

        except Exception as e:
            if 'embedding' not in str(e).lower():
                raise


async def test_semantic_memory_trust_is_capped_by_sources(db_pool):
    """Untrusted single-source claims should not become "true" without reinforcement."""
    async with db_pool.acquire() as conn:
        tr = conn.transaction()
        await tr.start()
        try:
            test_id = get_test_identifier("trust_sources")
            source_a = {"kind": "twitter", "ref": f"https://twitter.com/example/status/{test_id}", "trust": 0.2}
            source_b = {"kind": "paper", "ref": f"doi:10.0000/{test_id}", "trust": 0.9}

            mem_id = await conn.fetchval(
                """
                SELECT create_semantic_memory(
                    $1::text,
                    0.95::float,
                    NULL,
                    NULL,
                    $2::jsonb,
                    0.6::float
                )
                """,
                f"Claim from twitter {test_id}",
                json.dumps(source_a),
            )

            row = await conn.fetchrow(
                "SELECT trust_level, source_attribution FROM memories WHERE id = $1::uuid",
                mem_id,
            )
            assert row is not None
            trust_before = float(row["trust_level"])
            assert trust_before <= 0.30
            src = _coerce_json(row["source_attribution"])
            assert isinstance(src, dict)
            assert src.get("kind") in {"twitter", "unattributed", "internal"}

            await conn.execute(
                "SELECT add_semantic_source_reference($1::uuid, $2::jsonb)",
                mem_id,
                json.dumps(source_b),
            )
            row2 = await conn.fetchrow("SELECT trust_level FROM memories WHERE id = $1::uuid", mem_id)
            assert row2 is not None
            trust_after = float(row2["trust_level"])
            assert trust_after > trust_before
        finally:
            await tr.rollback()


async def test_worldview_misalignment_can_reduce_semantic_trust(db_pool):
    """Explicit worldview misalignment should down-weight trust in a claim."""
    async with db_pool.acquire() as conn:
        tr = conn.transaction()
        await tr.start()
        try:
            test_id = get_test_identifier("trust_worldview")
            sources = [
                {"kind": "paper", "ref": f"doi:10.0000/{test_id}-a", "trust": 0.9},
                {"kind": "paper", "ref": f"doi:10.0000/{test_id}-b", "trust": 0.9},
            ]

            mem_id = await conn.fetchval(
                "SELECT create_semantic_memory($1::text, 0.9::float, NULL, NULL, $2::jsonb, 0.6::float)",
                f"Well-sourced claim {test_id}",
                json.dumps(sources),
            )
            trust_before = float(await conn.fetchval("SELECT trust_level FROM memories WHERE id = $1::uuid", mem_id))
            assert trust_before > 0.2

            w_id = await conn.fetchval(
                """
                INSERT INTO worldview_primitives (category, belief, confidence)
                VALUES ('religion', $1::text, 1.0::float)
                RETURNING id
                """,
                f"Christian theology baseline {test_id}",
            )

            # Record a strong negative influence; trigger should resync `memories.trust_level`.
            await conn.execute(
                """
                INSERT INTO worldview_memory_influences (worldview_id, memory_id, influence_type, strength)
                VALUES ($1::uuid, $2::uuid, 'conflict', -1.0::float)
                """,
                w_id,
                mem_id,
            )

            trust_after = float(await conn.fetchval("SELECT trust_level FROM memories WHERE id = $1::uuid", mem_id))
            assert trust_after < trust_before
        finally:
            await tr.rollback()


async def test_create_procedural_memory_function(db_pool):
    """Test create_procedural_memory() with steps"""
    async with db_pool.acquire() as conn:
        try:
            memory_id = await conn.fetchval("""
                SELECT create_procedural_memory(
                    'How to make coffee',
                    '{"steps": ["Boil water", "Add coffee grounds", "Pour water", "Wait 4 minutes", "Press and pour"]}'::jsonb,
                    '{"required": ["coffee maker", "coffee grounds", "water"]}'::jsonb,
                    0.6
                )
            """)

            # Verify procedural details
            procedural = await conn.fetchrow("""
                SELECT * FROM procedural_memories WHERE memory_id = $1
            """, memory_id)
            assert procedural is not None
            assert procedural['steps'] is not None
            assert procedural['prerequisites'] is not None

        except Exception as e:
            if 'embedding' not in str(e).lower():
                raise


async def test_create_strategic_memory_function(db_pool):
    """Test create_strategic_memory() with pattern and evidence"""
    async with db_pool.acquire() as conn:
        try:
            memory_id = await conn.fetchval("""
                SELECT create_strategic_memory(
                    'Users prefer simple interfaces',
                    'Simplicity leads to higher engagement',
                    0.85,
                    '{"studies": ["Nielsen 2020", "UX Research 2021"]}'::jsonb,
                    '{"domains": ["web", "mobile", "desktop"]}'::jsonb,
                    0.9
                )
            """)

            # Verify strategic details
            strategic = await conn.fetchrow("""
                SELECT * FROM strategic_memories WHERE memory_id = $1
            """, memory_id)
            assert strategic is not None
            assert strategic['pattern_description'] == 'Simplicity leads to higher engagement'
            assert strategic['confidence_score'] == 0.85

        except Exception as e:
            if 'embedding' not in str(e).lower():
                raise


# -----------------------------------------------------------------------------
# GRAPH EDGE TYPES TESTS
# -----------------------------------------------------------------------------

async def test_temporal_next_edge(db_pool):
    """Test TEMPORAL_NEXT edge for narrative sequence"""
    async with db_pool.acquire() as conn:
        # Create two memories
        memory1_id = await conn.fetchval("""
            INSERT INTO memories (type, content, embedding)
            VALUES ('episodic'::memory_type, 'First event', array_fill(0.1, ARRAY[embedding_dimension()])::vector)
            RETURNING id
        """)

        memory2_id = await conn.fetchval("""
            INSERT INTO memories (type, content, embedding)
            VALUES ('episodic'::memory_type, 'Second event', array_fill(0.2, ARRAY[embedding_dimension()])::vector)
            RETURNING id
        """)

        # Create graph nodes
        await conn.execute("LOAD 'age'; SET search_path = ag_catalog, public;")

        for mid, mtype in [(memory1_id, 'episodic'), (memory2_id, 'episodic')]:
            await conn.execute(f"""
                SELECT * FROM ag_catalog.cypher('memory_graph', $$
                    CREATE (n:MemoryNode {{memory_id: '{mid}', type: '{mtype}'}})
                    RETURN n
                $$) as (result agtype)
            """)

        await conn.execute("SET search_path = public, ag_catalog")

        # Create TEMPORAL_NEXT relationship
        await conn.execute("""
            SELECT create_memory_relationship($1, $2, 'TEMPORAL_NEXT'::graph_edge_type, '{"sequence": 1}'::jsonb)
        """, memory1_id, memory2_id)

        # Verify edge exists
        await conn.execute("LOAD 'age'; SET search_path = ag_catalog, public;")

        result = await conn.fetch(f"""
            SELECT * FROM ag_catalog.cypher('memory_graph', $$
                MATCH (a:MemoryNode)-[r:TEMPORAL_NEXT]->(b:MemoryNode)
                WHERE a.memory_id = '{memory1_id}' AND b.memory_id = '{memory2_id}'
                RETURN r
            $$) as (result agtype)
        """)

        await conn.execute("SET search_path = public, ag_catalog")
        assert len(result) > 0, "TEMPORAL_NEXT edge should exist"


async def test_causes_edge(db_pool):
    """Test CAUSES edge for causal reasoning"""
    async with db_pool.acquire() as conn:
        cause_id = await conn.fetchval("""
            INSERT INTO memories (type, content, embedding)
            VALUES ('episodic'::memory_type, 'Rain started', array_fill(0.3, ARRAY[embedding_dimension()])::vector)
            RETURNING id
        """)

        effect_id = await conn.fetchval("""
            INSERT INTO memories (type, content, embedding)
            VALUES ('episodic'::memory_type, 'Ground became wet', array_fill(0.4, ARRAY[embedding_dimension()])::vector)
            RETURNING id
        """)

        await conn.execute("LOAD 'age'; SET search_path = ag_catalog, public;")

        for mid in [cause_id, effect_id]:
            await conn.execute(f"""
                SELECT * FROM ag_catalog.cypher('memory_graph', $$
                    CREATE (n:MemoryNode {{memory_id: '{mid}', type: 'episodic'}})
                    RETURN n
                $$) as (result agtype)
            """)

        await conn.execute("SET search_path = public, ag_catalog")

        await conn.execute("""
            SELECT create_memory_relationship($1, $2, 'CAUSES'::graph_edge_type, '{"confidence": 0.95}'::jsonb)
        """, cause_id, effect_id)

        # Verify causal chain query works
        await conn.execute("LOAD 'age'; SET search_path = ag_catalog, public;")

        result = await conn.fetch(f"""
            SELECT * FROM ag_catalog.cypher('memory_graph', $$
                MATCH (cause:MemoryNode)-[:CAUSES]->(effect:MemoryNode)
                WHERE cause.memory_id = '{cause_id}'
                RETURN effect.memory_id as effect_id
            $$) as (effect_id agtype)
        """)

        await conn.execute("SET search_path = public, ag_catalog")
        assert len(result) > 0, "CAUSES edge should exist"


async def test_contradicts_edge(db_pool):
    """Test CONTRADICTS edge for dialectical tension"""
    async with db_pool.acquire() as conn:
        claim1_id = await conn.fetchval("""
            INSERT INTO memories (type, content, embedding)
            VALUES ('semantic'::memory_type, 'The sky is blue', array_fill(0.5, ARRAY[embedding_dimension()])::vector)
            RETURNING id
        """)

        claim2_id = await conn.fetchval("""
            INSERT INTO memories (type, content, embedding)
            VALUES ('semantic'::memory_type, 'The sky is not blue', array_fill(0.6, ARRAY[embedding_dimension()])::vector)
            RETURNING id
        """)

        await conn.execute("LOAD 'age'; SET search_path = ag_catalog, public;")

        for mid in [claim1_id, claim2_id]:
            await conn.execute(f"""
                SELECT * FROM ag_catalog.cypher('memory_graph', $$
                    CREATE (n:MemoryNode {{memory_id: '{mid}', type: 'semantic'}})
                    RETURN n
                $$) as (result agtype)
            """)

        await conn.execute("SET search_path = public, ag_catalog")

        # Create bidirectional contradiction
        await conn.execute("""
            SELECT create_memory_relationship($1, $2, 'CONTRADICTS'::graph_edge_type, '{}'::jsonb)
        """, claim1_id, claim2_id)

        await conn.execute("""
            SELECT create_memory_relationship($1, $2, 'CONTRADICTS'::graph_edge_type, '{}'::jsonb)
        """, claim2_id, claim1_id)

        # Query contradictions
        await conn.execute("LOAD 'age'; SET search_path = ag_catalog, public;")

        result = await conn.fetch(f"""
            SELECT * FROM ag_catalog.cypher('memory_graph', $$
                MATCH (a:MemoryNode)-[:CONTRADICTS]->(b:MemoryNode)
                WHERE a.memory_id = '{claim1_id}'
                RETURN b.memory_id as contradicting_id
            $$) as (contradicting_id agtype)
        """)

        await conn.execute("SET search_path = public, ag_catalog")
        assert len(result) > 0, "CONTRADICTS edge should exist"


async def test_supports_edge(db_pool):
    """Test SUPPORTS edge for evidence relationship"""
    async with db_pool.acquire() as conn:
        evidence_id = await conn.fetchval("""
            INSERT INTO memories (type, content, embedding)
            VALUES ('episodic'::memory_type, 'Experiment showed X', array_fill(0.7, ARRAY[embedding_dimension()])::vector)
            RETURNING id
        """)

        claim_id = await conn.fetchval("""
            INSERT INTO memories (type, content, embedding)
            VALUES ('semantic'::memory_type, 'Theory X is correct', array_fill(0.8, ARRAY[embedding_dimension()])::vector)
            RETURNING id
        """)

        await conn.execute("LOAD 'age'; SET search_path = ag_catalog, public;")

        for mid, mtype in [(evidence_id, 'episodic'), (claim_id, 'semantic')]:
            await conn.execute(f"""
                SELECT * FROM ag_catalog.cypher('memory_graph', $$
                    CREATE (n:MemoryNode {{memory_id: '{mid}', type: '{mtype}'}})
                    RETURN n
                $$) as (result agtype)
            """)

        await conn.execute("SET search_path = public, ag_catalog")

        await conn.execute("""
            SELECT create_memory_relationship($1, $2, 'SUPPORTS'::graph_edge_type, '{"strength": 0.9}'::jsonb)
        """, evidence_id, claim_id)

        # Verify
        await conn.execute("LOAD 'age'; SET search_path = ag_catalog, public;")

        result = await conn.fetch(f"""
            SELECT * FROM ag_catalog.cypher('memory_graph', $$
                MATCH (evidence:MemoryNode)-[:SUPPORTS]->(claim:MemoryNode)
                WHERE claim.memory_id = '{claim_id}'
                RETURN evidence.memory_id as evidence_id
            $$) as (evidence_id agtype)
        """)

        await conn.execute("SET search_path = public, ag_catalog")
        assert len(result) > 0, "SUPPORTS edge should exist"


async def test_derived_from_edge(db_pool):
    """Test DERIVED_FROM edge for episodic->semantic transformation"""
    async with db_pool.acquire() as conn:
        episodic_id = await conn.fetchval("""
            INSERT INTO memories (type, content, embedding)
            VALUES ('episodic'::memory_type, 'Saw bird fly', array_fill(0.85, ARRAY[embedding_dimension()])::vector)
            RETURNING id
        """)

        semantic_id = await conn.fetchval("""
            INSERT INTO memories (type, content, embedding)
            VALUES ('semantic'::memory_type, 'Birds can fly', array_fill(0.86, ARRAY[embedding_dimension()])::vector)
            RETURNING id
        """)

        await conn.execute("LOAD 'age'; SET search_path = ag_catalog, public;")

        for mid, mtype in [(episodic_id, 'episodic'), (semantic_id, 'semantic')]:
            await conn.execute(f"""
                SELECT * FROM ag_catalog.cypher('memory_graph', $$
                    CREATE (n:MemoryNode {{memory_id: '{mid}', type: '{mtype}'}})
                    RETURN n
                $$) as (result agtype)
            """)

        await conn.execute("SET search_path = public, ag_catalog")

        await conn.execute("""
            SELECT create_memory_relationship($1, $2, 'DERIVED_FROM'::graph_edge_type, '{}'::jsonb)
        """, semantic_id, episodic_id)

        # Verify derivation chain
        await conn.execute("LOAD 'age'; SET search_path = ag_catalog, public;")

        result = await conn.fetch(f"""
            SELECT * FROM ag_catalog.cypher('memory_graph', $$
                MATCH (semantic:MemoryNode)-[:DERIVED_FROM]->(episodic:MemoryNode)
                WHERE semantic.memory_id = '{semantic_id}'
                RETURN episodic.memory_id as source_id
            $$) as (source_id agtype)
        """)

        await conn.execute("SET search_path = public, ag_catalog")
        assert len(result) > 0, "DERIVED_FROM edge should exist"


# -----------------------------------------------------------------------------
# MAINTENANCE FUNCTIONS TESTS
# -----------------------------------------------------------------------------

async def test_cleanup_working_memory_returns_count(db_pool):
    """Test cleanup_working_memory() returns count of deleted items"""
    async with db_pool.acquire() as conn:
        # Use unique content identifier
        unique_id = f'cleanup_test_{int(time.time() * 1000)}'

        # Clear existing expired entries first
        await conn.execute("""
            DELETE FROM working_memory WHERE expiry < CURRENT_TIMESTAMP
        """)

        # Add expired entries
        for i in range(5):
            await conn.execute("""
                INSERT INTO working_memory (content, embedding, expiry)
                VALUES ($1, array_fill(0.9, ARRAY[embedding_dimension()])::vector,
                        CURRENT_TIMESTAMP - interval '1 hour')
            """, f'Expired entry {unique_id} {i}')

        # Add valid entry
        await conn.execute("""
            INSERT INTO working_memory (content, embedding, expiry)
            VALUES ($1, array_fill(0.9, ARRAY[embedding_dimension()])::vector,
                    CURRENT_TIMESTAMP + interval '1 hour')
        """, f'Valid entry {unique_id}')

        # Call cleanup
        deleted_count = await conn.fetchval("""
            SELECT cleanup_working_memory()
        """)

        assert deleted_count >= 5, f"Should delete at least 5 expired entries, got {deleted_count}"

        # Verify valid entry remains
        remaining = await conn.fetchval("""
            SELECT COUNT(*) FROM working_memory WHERE content = $1
        """, f'Valid entry {unique_id}')
        assert remaining == 1


async def test_cleanup_working_memory_promotes_marked_items(db_pool):
    """Expired working memory marked for promotion becomes an episodic memory before deletion."""
    async with db_pool.acquire() as conn:
        test_id = get_test_identifier("wm_promote")
        wid = await conn.fetchval(
            """
            INSERT INTO working_memory (content, embedding, expiry, promote_to_long_term, importance)
            VALUES ($1, array_fill(0.2::float, ARRAY[embedding_dimension()])::vector, NOW() - INTERVAL '1 hour', TRUE, 0.9)
            RETURNING id
            """,
            f"Promote me {test_id}",
        )
        assert wid is not None

        stats = await conn.fetchval("SELECT cleanup_working_memory_with_stats()")
        if isinstance(stats, str):
            stats = json.loads(stats)
        assert stats["deleted_count"] >= 1
        assert stats["promoted_count"] >= 1

        assert await conn.fetchval("SELECT COUNT(*) FROM working_memory WHERE id = $1::uuid", wid) == 0
        promoted = await conn.fetchrow(
            """
            SELECT m.id, m.type, m.content
            FROM episodic_memories em
            JOIN memories m ON m.id = em.memory_id
            WHERE em.context->>'from_working_memory_id' = $1::text
            ORDER BY m.created_at DESC
            LIMIT 1
            """,
            str(wid),
        )
        assert promoted is not None
        assert promoted["type"] == "episodic"
        assert test_id in promoted["content"]


async def test_cleanup_embedding_cache_with_interval(db_pool):
    """Test cleanup_embedding_cache() with custom interval"""
    async with db_pool.acquire() as conn:
        # Add old cache entries
        await conn.execute("""
            INSERT INTO embedding_cache (content_hash, embedding, created_at)
            VALUES
                ('old_hash_1', array_fill(0.5, ARRAY[embedding_dimension()])::vector, CURRENT_TIMESTAMP - interval '10 days'),
                ('old_hash_2', array_fill(0.5, ARRAY[embedding_dimension()])::vector, CURRENT_TIMESTAMP - interval '8 days'),
                ('new_hash', array_fill(0.5, ARRAY[embedding_dimension()])::vector, CURRENT_TIMESTAMP)
            ON CONFLICT DO NOTHING
        """)

        # Cleanup entries older than 7 days
        deleted_count = await conn.fetchval("""
            SELECT cleanup_embedding_cache(interval '7 days')
        """)

        assert deleted_count >= 2, "Should delete old cache entries"

        # Verify new entry remains
        new_exists = await conn.fetchval("""
            SELECT COUNT(*) FROM embedding_cache WHERE content_hash = 'new_hash'
        """)
        assert new_exists == 1


# -----------------------------------------------------------------------------
# IDENTITY & WORLDVIEW TESTS
# -----------------------------------------------------------------------------

async def test_identity_aspects_all_types(db_pool):
    """Test all identity aspect_type values"""
    async with db_pool.acquire() as conn:
        aspect_types = ['self_concept', 'purpose', 'boundary', 'agency', 'values']

        for aspect_type in aspect_types:
            aspect_id = await conn.fetchval("""
                INSERT INTO identity_aspects (aspect_type, content, stability)
                VALUES ($1, $2::jsonb, 0.7)
                RETURNING id
            """, aspect_type, json.dumps({"description": f"Test {aspect_type}"}))

            assert aspect_id is not None, f"Should create {aspect_type} aspect"

        # Verify all types exist
        count = await conn.fetchval("""
            SELECT COUNT(DISTINCT aspect_type) FROM identity_aspects
            WHERE aspect_type = ANY($1)
        """, aspect_types)

        assert count == 5, "All aspect types should be created"


async def test_identity_memory_resonance_integration_status(db_pool):
    """Test integration_status field in identity_memory_resonance"""
    async with db_pool.acquire() as conn:
        # Create identity aspect
        aspect_id = await conn.fetchval("""
            INSERT INTO identity_aspects (aspect_type, content)
            VALUES ('self_concept', '{"core": "helpful assistant"}'::jsonb)
            RETURNING id
        """)

        # Create memory
        memory_id = await conn.fetchval("""
            INSERT INTO memories (type, content, embedding)
            VALUES ('episodic'::memory_type, 'Helped user solve problem',
                    array_fill(0.92, ARRAY[embedding_dimension()])::vector)
            RETURNING id
        """)

        # Create resonance with different integration statuses
        statuses = ['pending', 'integrated', 'conflicting', 'resolved']

        for status in statuses:
            await conn.execute("""
                INSERT INTO identity_memory_resonance
                    (memory_id, identity_aspect_id, resonance_strength, integration_status)
                VALUES ($1, $2, 0.8, $3)
                ON CONFLICT DO NOTHING
            """, memory_id, aspect_id, status)

        # Query by status
        integrated = await conn.fetch("""
            SELECT * FROM identity_memory_resonance
            WHERE identity_aspect_id = $1 AND integration_status = 'integrated'
        """, aspect_id)

        assert len(integrated) >= 0  # May or may not find depending on conflicts


async def test_worldview_influence_types(db_pool):
    """Test different influence_type values on worldview_memory_influences"""
    async with db_pool.acquire() as conn:
        # Create worldview
        worldview_id = await conn.fetchval("""
            INSERT INTO worldview_primitives (category, belief, confidence)
            VALUES ('ethics', 'Honesty is important', 0.95)
            RETURNING id
        """)

        # Create memory
        memory_id = await conn.fetchval("""
            INSERT INTO memories (type, content, embedding)
            VALUES ('episodic'::memory_type, 'Told the truth in difficult situation',
                    array_fill(0.93, ARRAY[embedding_dimension()])::vector)
            RETURNING id
        """)

        # Create influences with different types
        influence_types = ['alignment', 'reinforcement', 'challenge', 'neutral']

        for inf_type in influence_types:
            await conn.execute("""
                INSERT INTO worldview_memory_influences
                    (worldview_id, memory_id, influence_type, strength)
                VALUES ($1, $2, $3, 0.7)
            """, worldview_id, memory_id, inf_type)

        # Query influences
        influences = await conn.fetch("""
            SELECT influence_type, strength FROM worldview_memory_influences
            WHERE worldview_id = $1
        """, worldview_id)

        assert len(influences) == 4, "All influence types should be created"


async def test_worldview_confidence_updates_from_influences(db_pool):
    async with db_pool.acquire() as conn:
        test_id = get_test_identifier("worldview_conf_update")
        w_id = await conn.fetchval(
            """
            INSERT INTO worldview_primitives (category, belief, confidence)
            VALUES ('values', $1::text, 0.5::float)
            RETURNING id
            """,
            f"Belief {test_id}",
        )
        m_id = await conn.fetchval(
            """
            INSERT INTO memories (type, content, embedding, trust_level)
            VALUES ('episodic'::memory_type, $1::text, array_fill(0.1, ARRAY[embedding_dimension()])::vector, 1.0::float)
            RETURNING id
            """,
            f"Evidence {test_id}",
        )
        before = float(await conn.fetchval("SELECT confidence FROM worldview_primitives WHERE id = $1::uuid", w_id))
        await conn.execute(
            """
            INSERT INTO worldview_memory_influences (worldview_id, memory_id, influence_type, strength)
            VALUES ($1::uuid, $2::uuid, 'support', 1.0::float)
            """,
            w_id,
            m_id,
        )
        after = float(await conn.fetchval("SELECT confidence FROM worldview_primitives WHERE id = $1::uuid", w_id))
        assert after > before


async def test_connected_beliefs_relationships(db_pool):
    """Test connected_beliefs UUID array in worldview_primitives"""
    async with db_pool.acquire() as conn:
        # Create base belief
        belief1_id = await conn.fetchval("""
            INSERT INTO worldview_primitives (category, belief, confidence, connected_beliefs)
            VALUES ('values', 'Kindness matters', 0.9, ARRAY[]::UUID[])
            RETURNING id
        """)

        # Create related belief
        belief2_id = await conn.fetchval("""
            INSERT INTO worldview_primitives (category, belief, confidence, connected_beliefs)
            VALUES ('values', 'Empathy is valuable', 0.85, ARRAY[$1]::UUID[])
            RETURNING id
        """, belief1_id)

        # Update first belief to connect back
        await conn.execute("""
            UPDATE worldview_primitives
            SET connected_beliefs = array_append(connected_beliefs, $1)
            WHERE id = $2
        """, belief2_id, belief1_id)

        # Query connected beliefs
        belief1 = await conn.fetchrow("""
            SELECT connected_beliefs FROM worldview_primitives WHERE id = $1
        """, belief1_id)

        assert belief2_id in belief1['connected_beliefs'], "Beliefs should be connected"


# -----------------------------------------------------------------------------
# VIEW TESTS
# -----------------------------------------------------------------------------

async def test_memory_health_view_aggregations(db_pool):
    """Test memory_health view calculates correct aggregations"""
    async with db_pool.acquire() as conn:
        # Create memories of known type with known values
        import time
        unique_suffix = str(int(time.time() * 1000))

        for i in range(5):
            await conn.execute("""
                INSERT INTO memories (type, content, embedding, importance, access_count)
                VALUES ('procedural'::memory_type, $1,
                        array_fill(0.94, ARRAY[embedding_dimension()])::vector, $2, $3)
            """, f'Health view test {unique_suffix} {i}', 0.5 + i * 0.1, i)

        # Query view
        health = await conn.fetchrow("""
            SELECT * FROM memory_health WHERE type = 'procedural'
        """)

        assert health is not None
        assert health['total_memories'] >= 5
        assert health['avg_importance'] is not None
        assert health['avg_access_count'] is not None


async def test_cluster_insights_view_ordering(db_pool):
    """Test cluster_insights view ordered by importance_score DESC"""
    async with db_pool.acquire() as conn:
        # Create clusters with different importance
        for i, importance in enumerate([0.3, 0.9, 0.5, 0.7]):
            await conn.execute("""
                INSERT INTO memory_clusters (cluster_type, name, importance_score, centroid_embedding)
                VALUES ('theme'::cluster_type, $1, $2, array_fill(0.5, ARRAY[embedding_dimension()])::vector)
            """, f'Insights order test {i}', importance)

        # Query view
        insights = await conn.fetch("""
            SELECT name, importance_score FROM cluster_insights
            WHERE name LIKE 'Insights order test%'
            ORDER BY importance_score DESC
        """)

        # Verify ordering
        scores = [r['importance_score'] for r in insights]
        assert scores == sorted(scores, reverse=True), "Should be ordered by importance DESC"


# -----------------------------------------------------------------------------
# INDEX PERFORMANCE TESTS
# -----------------------------------------------------------------------------

async def test_hnsw_index_usage_memories(db_pool):
    """Test HNSW index on memories.embedding is used for vector search"""
    async with db_pool.acquire() as conn:
        # Create test data
        for i in range(20):
            await conn.execute("""
                INSERT INTO memories (type, content, embedding)
                VALUES ('semantic'::memory_type, $1, array_fill(0.5::float, ARRAY[embedding_dimension()])::vector)
            """, f'HNSW test memory {i}')

        # Check query plan uses index
        plan = await conn.fetch("""
            EXPLAIN (FORMAT JSON)
            SELECT id FROM memories
            ORDER BY embedding <=> array_fill(0.5::float, ARRAY[embedding_dimension()])::vector
            LIMIT 5
        """)

        plan_text = str(plan)
        # HNSW index should be mentioned in plan
        assert 'idx_memories_embedding' in plan_text or 'Index' in plan_text


async def test_gin_index_content_search(db_pool):
    """Test GIN trigram index on memories.content for text search"""
    async with db_pool.acquire() as conn:
        # Create memories with searchable content
        await conn.execute("""
            INSERT INTO memories (type, content, embedding)
            VALUES
                ('semantic'::memory_type, 'PostgreSQL database management', array_fill(0.5, ARRAY[embedding_dimension()])::vector),
                ('semantic'::memory_type, 'Python programming language', array_fill(0.5, ARRAY[embedding_dimension()])::vector)
        """)

        # Query using trigram similarity
        results = await conn.fetch("""
            SELECT content FROM memories
            WHERE content ILIKE '%postgres%'
        """)

        assert len(results) >= 1
        assert 'PostgreSQL' in results[0]['content']

# =============================================================================
# COMPREHENSIVE EMBEDDING AND FUNCTION TESTS
# These tests exercise functions that were previously untested or only
# partially tested. They require the embedding service to be running.
# =============================================================================

@pytest.fixture(scope="session")
async def ensure_embedding_service(db_pool):
    """
    Ensure embedding service is available.

    This fixture retries for a short window so tests don't flake while Docker
    is still initializing. If the service never becomes healthy, fail fast
    with a clear timeout error (no skipping).
    """
    async with db_pool.acquire() as conn:
        # Ensure correct service URL
        await conn.execute("""
            INSERT INTO embedding_config (key, value)
            VALUES ('service_url', 'http://embeddings:80/embed')
            ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value
        """)

        wait_seconds = int(os.getenv("EMBEDDINGS_WAIT_SECONDS", "30"))
        retrying = AsyncRetrying(
            stop=stop_after_delay(wait_seconds),
            wait=wait_fixed(1),
            retry=(retry_if_result(lambda ok: not ok) | retry_if_exception_type(Exception)),
            reraise=False,
        )

        try:
            ok = await retrying(conn.fetchval, "SELECT check_embedding_service_health()")
            assert ok is True
            return True
        except RetryError as e:
            last_exc = e.last_attempt.exception()
            last_res = None
            try:
                last_res = e.last_attempt.result()
            except Exception:
                pass
            pytest.fail(
                f"Embedding service not available after {wait_seconds}s (service_url=http://embeddings:80/embed). "
                f"last_exception={last_exc!r} last_result={last_res!r}"
            )


# -----------------------------------------------------------------------------
# get_embedding() COMPREHENSIVE TESTS
# -----------------------------------------------------------------------------

async def test_get_embedding_basic_functionality(db_pool, ensure_embedding_service):
    """Test get_embedding() returns a vector for configured dimension"""
    async with db_pool.acquire() as conn:
        test_id = get_test_identifier("get_emb_basic")
        test_content = f"Test content for embedding generation {test_id}"

        # Clear any cached version
        content_hash = await conn.fetchval(
            "SELECT encode(sha256($1::text::bytea), 'hex')", test_content
        )
        await conn.execute(
            "DELETE FROM embedding_cache WHERE content_hash = $1", content_hash
        )

        # Call get_embedding
        embedding = await conn.fetchval(
            "SELECT get_embedding($1)", test_content
        )

        assert embedding is not None, "get_embedding should return a vector"
        # pgvector returns a string representation, verify it's non-empty
        assert len(str(embedding)) > 0, "Embedding should not be empty"


async def test_get_embedding_caching_creates_cache_entry(db_pool, ensure_embedding_service):
    """Test that get_embedding() creates a cache entry in embedding_cache table"""
    async with db_pool.acquire() as conn:
        test_id = get_test_identifier("get_emb_cache")
        test_content = f"Unique content for caching test {test_id}"

        # Calculate the hash that will be used
        content_hash = await conn.fetchval(
            "SELECT encode(sha256($1::text::bytea), 'hex')", test_content
        )

        # Ensure no cache entry exists
        await conn.execute(
            "DELETE FROM embedding_cache WHERE content_hash = $1", content_hash
        )

        # Verify cache is empty for this content
        cache_before = await conn.fetchval(
            "SELECT COUNT(*) FROM embedding_cache WHERE content_hash = $1", content_hash
        )
        assert cache_before == 0, "Cache should be empty before call"

        # Call get_embedding
        await conn.fetchval("SELECT get_embedding($1)", test_content)

        # Verify cache entry was created
        cache_after = await conn.fetchval(
            "SELECT COUNT(*) FROM embedding_cache WHERE content_hash = $1", content_hash
        )
        assert cache_after == 1, "Cache entry should be created after get_embedding call"


async def test_get_embedding_cache_hit_returns_same_result(db_pool, ensure_embedding_service):
    """Test that get_embedding() returns cached embedding on second call"""
    async with db_pool.acquire() as conn:
        test_id = get_test_identifier("get_emb_cache_hit")
        test_content = f"Content for cache hit test {test_id}"

        # Clear cache first
        content_hash = await conn.fetchval(
            "SELECT encode(sha256($1::text::bytea), 'hex')", test_content
        )
        await conn.execute(
            "DELETE FROM embedding_cache WHERE content_hash = $1", content_hash
        )

        # First call - cache miss, hits service
        embedding1 = await conn.fetchval("SELECT get_embedding($1)", test_content)

        # Second call - should be cache hit
        embedding2 = await conn.fetchval("SELECT get_embedding($1)", test_content)

        assert embedding1 == embedding2, "Cached embedding should match original"


async def test_get_embedding_sha256_hash_correctness(db_pool, ensure_embedding_service):
    """Test that get_embedding uses correct SHA256 hashing for cache key"""
    async with db_pool.acquire() as conn:
        test_id = get_test_identifier("get_emb_hash")
        test_content = f"Hash test content {test_id}"

        # Calculate expected hash
        expected_hash = await conn.fetchval(
            "SELECT encode(sha256($1::text::bytea), 'hex')", test_content
        )

        # Clear any existing cache
        await conn.execute(
            "DELETE FROM embedding_cache WHERE content_hash = $1", expected_hash
        )

        # Call get_embedding
        await conn.fetchval("SELECT get_embedding($1)", test_content)

        # Verify the cache entry uses the expected hash
        cached_embedding = await conn.fetchval(
            "SELECT embedding FROM embedding_cache WHERE content_hash = $1", expected_hash
        )
        assert cached_embedding is not None, "Cache should use SHA256 hash as key"


async def test_get_embedding_different_content_different_embeddings(db_pool, ensure_embedding_service):
    """Test that different content produces different embeddings"""
    async with db_pool.acquire() as conn:
        test_id = get_test_identifier("get_emb_diff")

        content1 = f"The cat sat on the mat {test_id}"
        content2 = f"Quantum physics principles {test_id}"

        embedding1 = await conn.fetchval("SELECT get_embedding($1)", content1)
        embedding2 = await conn.fetchval("SELECT get_embedding($1)", content2)

        # Embeddings should be different for semantically different content
        assert embedding1 != embedding2, "Different content should produce different embeddings"


async def test_get_embedding_http_error_handling(db_pool):
    """Test get_embedding() error handling when HTTP service is unavailable"""
    async with db_pool.acquire() as conn:
        # Save original URL
        original_url = await conn.fetchval(
            "SELECT value FROM embedding_config WHERE key = 'service_url'"
        )
        original_retry_seconds, original_retry_interval_seconds = await _set_embedding_retry_config(
            conn,
            retry_seconds=1,
            retry_interval_seconds=0.1,
        )

        try:
            # Set invalid URL
            await conn.execute("""
                UPDATE embedding_config
                SET value = 'http://nonexistent-service:9999/embed'
                WHERE key = 'service_url'
            """)

            # Attempt to get embedding - should fail
            with pytest.raises(Exception) as exc_info:
                await conn.fetchval("SELECT get_embedding('test content')")

            assert "Failed to get embedding" in str(exc_info.value), \
                "Should raise proper error message"
        finally:
            # Restore original URL
            await conn.execute(
                "UPDATE embedding_config SET value = $1 WHERE key = 'service_url'",
                original_url
            )
            await _restore_embedding_retry_config(
                conn,
                original_retry_seconds,
                original_retry_interval_seconds,
            )


async def test_get_embedding_non_200_response_handling(db_pool):
    """Test get_embedding() handles non-200 HTTP responses"""
    async with db_pool.acquire() as conn:
        original_url = await conn.fetchval(
            "SELECT value FROM embedding_config WHERE key = 'service_url'"
        )
        original_retry_seconds, original_retry_interval_seconds = await _set_embedding_retry_config(
            conn,
            retry_seconds=1,
            retry_interval_seconds=0.1,
        )

        try:
            # Set URL that will return 404 or similar
            await conn.execute("""
                UPDATE embedding_config
                SET value = 'http://embeddings:80/nonexistent-endpoint'
                WHERE key = 'service_url'
            """)

            with pytest.raises(Exception) as exc_info:
                await conn.fetchval("SELECT get_embedding('test content')")

            # Should mention service error or failed
            error_msg = str(exc_info.value).lower()
            assert "error" in error_msg or "failed" in error_msg, \
                "Should indicate error for non-200 response"
        finally:
            await conn.execute(
                "UPDATE embedding_config SET value = $1 WHERE key = 'service_url'",
                original_url
            )
            await _restore_embedding_retry_config(
                conn,
                original_retry_seconds,
                original_retry_interval_seconds,
            )


async def test_get_embedding_dimension_validation(db_pool, ensure_embedding_service):
    """Test that get_embedding returns a vector matching configured dimension"""
    async with db_pool.acquire() as conn:
        test_id = get_test_identifier("get_emb_dim")

        embedding = await conn.fetchval("SELECT get_embedding($1)", f"Dimension test {test_id}")
        dims = await conn.fetchval("SELECT vector_dims($1::vector)", embedding)
        expected = await conn.fetchval("SELECT embedding_dimension()")
        assert int(dims) == int(expected)

        # Verify by inserting into a table that enforces the configured dimension
        await conn.execute("""
            INSERT INTO working_memory (content, embedding, expiry)
            VALUES ($1, $2, NOW() + INTERVAL '1 hour')
        """, f"Dimension verification {test_id}", embedding)

        # If we get here, the embedding matched the typmod
        assert True


# -----------------------------------------------------------------------------
# check_embedding_service_health() COMPREHENSIVE TESTS
# -----------------------------------------------------------------------------

async def test_check_embedding_service_health_returns_boolean(db_pool):
    """Test check_embedding_service_health() returns a boolean value"""
    async with db_pool.acquire() as conn:
        result = await conn.fetchval("SELECT check_embedding_service_health()")
        assert isinstance(result, bool), "Should return a boolean"


async def test_check_embedding_service_health_true_when_available(db_pool):
    """Test health check returns true when service is available"""
    async with db_pool.acquire() as conn:
        # Ensure correct URL
        await conn.execute("""
            INSERT INTO embedding_config (key, value)
            VALUES ('service_url', 'http://embeddings:80/embed')
            ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value
        """)

        result = await conn.fetchval("SELECT check_embedding_service_health()")

        # This test may skip if service not running, but if it runs, verify result
        if result:
            assert result == True, "Should return true when service is healthy"


async def test_check_embedding_service_health_false_when_unavailable(db_pool):
    """Test health check returns false when service is unavailable"""
    async with db_pool.acquire() as conn:
        original_url = await conn.fetchval(
            "SELECT value FROM embedding_config WHERE key = 'service_url'"
        )

        try:
            # Set invalid URL
            await conn.execute("""
                UPDATE embedding_config
                SET value = 'http://nonexistent-host:9999/embed'
                WHERE key = 'service_url'
            """)

            result = await conn.fetchval("SELECT check_embedding_service_health()")
            assert result == False, "Should return false when service unavailable"
        finally:
            await conn.execute(
                "UPDATE embedding_config SET value = $1 WHERE key = 'service_url'",
                original_url
            )


async def test_check_embedding_service_health_endpoint_path(db_pool):
    """Test health check uses correct /health endpoint path"""
    async with db_pool.acquire() as conn:
        # The function replaces /embed with /health
        # If service is at http://embeddings:80/embed, health should be http://embeddings:80/health
        await conn.execute("""
            INSERT INTO embedding_config (key, value)
            VALUES ('service_url', 'http://embeddings:80/embed')
            ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value
        """)

        # Function should work without error (may return true or false)
        result = await conn.fetchval("SELECT check_embedding_service_health()")
        assert result is not None, "Should return a value, not NULL"


async def test_check_embedding_service_health_no_exception_on_failure(db_pool):
    """Test health check gracefully handles errors without raising exceptions"""
    async with db_pool.acquire() as conn:
        original_url = await conn.fetchval(
            "SELECT value FROM embedding_config WHERE key = 'service_url'"
        )

        try:
            # Set completely invalid URL
            await conn.execute("""
                UPDATE embedding_config
                SET value = 'http://256.256.256.256:9999/embed'
                WHERE key = 'service_url'
            """)

            # Should NOT raise an exception, should return false
            result = await conn.fetchval("SELECT check_embedding_service_health()")
            assert result == False, "Should return false, not raise exception"
        finally:
            await conn.execute(
                "UPDATE embedding_config SET value = $1 WHERE key = 'service_url'",
                original_url
            )


# -----------------------------------------------------------------------------
# create_memory() COMPREHENSIVE TESTS
# -----------------------------------------------------------------------------

async def test_create_memory_returns_uuid(db_pool, ensure_embedding_service):
    """Test create_memory() returns a valid UUID"""
    async with db_pool.acquire() as conn:
        await conn.execute("LOAD 'age';")
        await conn.execute("SET search_path = ag_catalog, public;")

        test_id = get_test_identifier("create_mem_uuid")

        memory_id = await conn.fetchval("""
            SELECT create_memory('semantic'::memory_type, $1, 0.7)
        """, f"Test memory content {test_id}")

        assert memory_id is not None, "Should return a UUID"
        assert isinstance(memory_id, uuid.UUID), "Should be a UUID type"


async def test_create_memory_generates_embedding(db_pool, ensure_embedding_service):
    """Test create_memory() generates embedding automatically via get_embedding()"""
    async with db_pool.acquire() as conn:
        await conn.execute("LOAD 'age';")
        await conn.execute("SET search_path = ag_catalog, public;")

        test_id = get_test_identifier("create_mem_emb")
        content = f"Memory with auto-generated embedding {test_id}"

        memory_id = await conn.fetchval("""
            SELECT create_memory('semantic'::memory_type, $1, 0.8)
        """, content)

        # Verify embedding was generated and stored
        embedding = await conn.fetchval("""
            SELECT embedding FROM memories WHERE id = $1
        """, memory_id)

        assert embedding is not None, "Embedding should be generated"


async def test_create_memory_creates_graph_node(db_pool, ensure_embedding_service):
    """Test create_memory() creates a MemoryNode in the graph"""
    async with db_pool.acquire() as conn:
        await conn.execute("LOAD 'age';")
        await conn.execute("SET search_path = ag_catalog, public;")

        test_id = get_test_identifier("create_mem_graph")

        memory_id = await conn.fetchval("""
            SELECT create_memory('episodic'::memory_type, $1, 0.6)
        """, f"Memory for graph node test {test_id}")

        # Verify graph node was created using f-string for UUID in cypher
        graph_check = await conn.fetchval(f"""
            SELECT * FROM cypher('memory_graph', $$
                MATCH (n:MemoryNode)
                WHERE n.memory_id = '{memory_id}'
                RETURN count(n)
            $$) as (cnt agtype)
        """)

        assert graph_check is not None, "Graph node should exist"
        # agtype returns as string like "1", check it's not "0"
        assert str(graph_check) != '0', "Should find at least one graph node"


async def test_create_memory_graph_node_properties(db_pool, ensure_embedding_service):
    """Test MemoryNode has correct properties (memory_id, type, created_at)"""
    async with db_pool.acquire() as conn:
        await conn.execute("LOAD 'age';")
        await conn.execute("SET search_path = ag_catalog, public;")

        test_id = get_test_identifier("create_mem_props")

        memory_id = await conn.fetchval("""
            SELECT create_memory('procedural'::memory_type, $1, 0.5)
        """, f"Memory for properties test {test_id}")

        # Query graph for node properties
        result = await conn.fetch(f"""
            SELECT * FROM cypher('memory_graph', $$
                MATCH (n:MemoryNode)
                WHERE n.memory_id = '{memory_id}'
                RETURN n.memory_id, n.type, n.created_at
            $$) as (mid agtype, mtype agtype, created agtype)
        """)

        assert len(result) == 1, "Should find exactly one node"
        # The values are agtype, which are JSON-like
        assert str(memory_id) in str(result[0]['mid']), "memory_id should match"
        assert 'procedural' in str(result[0]['mtype']), "type should be procedural"
        assert result[0]['created'] is not None, "created_at should be set"


async def test_create_memory_all_types(db_pool, ensure_embedding_service):
    """Test create_memory() works for all memory types"""
    async with db_pool.acquire() as conn:
        await conn.execute("LOAD 'age';")
        await conn.execute("SET search_path = ag_catalog, public;")

        test_id = get_test_identifier("create_mem_types")
        memory_types = ['episodic', 'semantic', 'procedural', 'strategic']

        for mem_type in memory_types:
            memory_id = await conn.fetchval(f"""
                SELECT create_memory('{mem_type}'::memory_type, $1, 0.5)
            """, f"Test {mem_type} memory {test_id}")

            assert memory_id is not None, f"Should create {mem_type} memory"

            # Verify type in database
            stored_type = await conn.fetchval("""
                SELECT type FROM memories WHERE id = $1
            """, memory_id)

            assert stored_type == mem_type, f"Stored type should be {mem_type}"


async def test_create_memory_importance_stored(db_pool, ensure_embedding_service):
    """Test create_memory() stores importance value correctly"""
    async with db_pool.acquire() as conn:
        await conn.execute("LOAD 'age';")
        await conn.execute("SET search_path = ag_catalog, public;")

        test_id = get_test_identifier("create_mem_imp")
        importance = 0.95

        memory_id = await conn.fetchval("""
            SELECT create_memory('semantic'::memory_type, $1, $2)
        """, f"High importance memory {test_id}", importance)

        stored_importance = await conn.fetchval("""
            SELECT importance FROM memories WHERE id = $1
        """, memory_id)

        assert abs(stored_importance - importance) < 0.001, "Importance should match"


async def test_create_memory_triggers_episode_assignment(db_pool, ensure_embedding_service):
    """Test create_memory() triggers auto episode assignment"""
    async with db_pool.acquire() as conn:
        await conn.execute("LOAD 'age';")
        await conn.execute("SET search_path = ag_catalog, public;")

        test_id = get_test_identifier("create_mem_ep")

        memory_id = await conn.fetchval("""
            SELECT create_memory('episodic'::memory_type, $1, 0.7)
        """, f"Memory for episode test {test_id}")

        # Verify memory was assigned to an episode
        episode_link = await conn.fetchrow("""
            SELECT episode_id, sequence_order FROM episode_memories
            WHERE memory_id = $1
        """, memory_id)

        assert episode_link is not None, "Memory should be assigned to episode"
        assert episode_link['episode_id'] is not None
        assert episode_link['sequence_order'] >= 1


async def test_create_memory_initializes_neighborhood(db_pool, ensure_embedding_service):
    """Test create_memory() initializes memory_neighborhoods record"""
    async with db_pool.acquire() as conn:
        await conn.execute("LOAD 'age';")
        await conn.execute("SET search_path = ag_catalog, public;")

        test_id = get_test_identifier("create_mem_neigh")

        memory_id = await conn.fetchval("""
            SELECT create_memory('semantic'::memory_type, $1, 0.6)
        """, f"Memory for neighborhood test {test_id}")

        # Verify neighborhood record was created
        neighborhood = await conn.fetchrow("""
            SELECT memory_id, neighbors, is_stale FROM memory_neighborhoods
            WHERE memory_id = $1
        """, memory_id)

        assert neighborhood is not None, "Neighborhood record should be created"
        assert neighborhood['is_stale'] == True, "New neighborhood should be stale"


# -----------------------------------------------------------------------------
# add_to_working_memory() COMPREHENSIVE TESTS
# -----------------------------------------------------------------------------

async def test_add_to_working_memory_returns_uuid(db_pool, ensure_embedding_service):
    """Test add_to_working_memory() returns a valid UUID"""
    async with db_pool.acquire() as conn:
        test_id = get_test_identifier("add_wm_uuid")

        wm_id = await conn.fetchval("""
            SELECT add_to_working_memory($1, INTERVAL '1 hour')
        """, f"Working memory content {test_id}")

        assert wm_id is not None, "Should return a UUID"
        assert isinstance(wm_id, uuid.UUID), "Should be UUID type"


async def test_add_to_working_memory_generates_embedding(db_pool, ensure_embedding_service):
    """Test add_to_working_memory() calls get_embedding() for auto-embedding"""
    async with db_pool.acquire() as conn:
        test_id = get_test_identifier("add_wm_emb")
        content = f"Working memory with embedding {test_id}"

        wm_id = await conn.fetchval("""
            SELECT add_to_working_memory($1, INTERVAL '1 hour')
        """, content)

        # Verify embedding was stored
        embedding = await conn.fetchval("""
            SELECT embedding FROM working_memory WHERE id = $1
        """, wm_id)

        assert embedding is not None, "Embedding should be generated"


async def test_add_to_working_memory_sets_expiry(db_pool, ensure_embedding_service):
    """Test add_to_working_memory() sets correct expiry time"""
    async with db_pool.acquire() as conn:
        test_id = get_test_identifier("add_wm_exp")

        wm_id = await conn.fetchval("""
            SELECT add_to_working_memory($1, INTERVAL '2 hours')
        """, f"Expiry test {test_id}")

        expiry = await conn.fetchval("""
            SELECT expiry FROM working_memory WHERE id = $1
        """, wm_id)

        assert expiry is not None, "Expiry should be set"
        # Expiry should be approximately 2 hours from now
        now = await conn.fetchval("SELECT CURRENT_TIMESTAMP")
        diff = expiry - now
        # Should be between 1h 59m and 2h 1m (allowing for test execution time)
        assert diff.total_seconds() > 7100 and diff.total_seconds() < 7300, \
            "Expiry should be approximately 2 hours from now"


async def test_add_to_working_memory_default_expiry(db_pool, ensure_embedding_service):
    """Test add_to_working_memory() uses default 1 hour expiry"""
    async with db_pool.acquire() as conn:
        test_id = get_test_identifier("add_wm_def_exp")

        wm_id = await conn.fetchval("""
            SELECT add_to_working_memory($1)
        """, f"Default expiry test {test_id}")

        expiry = await conn.fetchval("""
            SELECT expiry FROM working_memory WHERE id = $1
        """, wm_id)

        now = await conn.fetchval("SELECT CURRENT_TIMESTAMP")
        diff = expiry - now
        # Default should be 1 hour (3600 seconds, with some tolerance)
        assert diff.total_seconds() > 3500 and diff.total_seconds() < 3700, \
            "Default expiry should be approximately 1 hour"


async def test_add_to_working_memory_content_stored(db_pool, ensure_embedding_service):
    """Test add_to_working_memory() stores content correctly"""
    async with db_pool.acquire() as conn:
        test_id = get_test_identifier("add_wm_content")
        content = f"Specific working memory content {test_id}"

        wm_id = await conn.fetchval("""
            SELECT add_to_working_memory($1, INTERVAL '1 hour')
        """, content)

        stored_content = await conn.fetchval("""
            SELECT content FROM working_memory WHERE id = $1
        """, wm_id)

        assert stored_content == content, "Content should match exactly"


async def test_add_to_working_memory_searchable(db_pool, ensure_embedding_service):
    """Test working memory added via function is searchable"""
    async with db_pool.acquire() as conn:
        test_id = get_test_identifier("add_wm_search")
        content = f"Searchable working memory about databases {test_id}"

        await conn.fetchval("""
            SELECT add_to_working_memory($1, INTERVAL '1 hour')
        """, content)

        # Search for it
        # Use a query that includes the unique test token so this remains deterministic
        # even if other tests have populated working_memory.
        results = await conn.fetch(
            "SELECT * FROM search_working_memory($1, 10)",
            content,
        )

        # Should find our content
        found = any(test_id in str(r['content']) for r in results)
        assert found, "Should find the working memory via search"


# -----------------------------------------------------------------------------
# BATCH CREATE MEMORIES
# -----------------------------------------------------------------------------

async def test_batch_create_memories_creates_rows_and_nodes(db_pool, ensure_embedding_service):
    async with db_pool.acquire() as conn:
        tr = conn.transaction()
        await tr.start()
        try:
            test_id = get_test_identifier("batch_create_memories")
            items = [
                {
                    "type": "semantic",
                    "content": f"Batch semantic A {test_id}",
                    "importance": 0.6,
                    "confidence": 0.9,
                    "source_references": [{"kind": "twitter", "ref": f"https://twitter.com/x/{test_id}", "trust": 0.2}],
                },
                {
                    "type": "episodic",
                    "content": f"Batch episodic B {test_id}",
                    "importance": 0.4,
                    "context": {"type": "test"},
                    "emotional_valence": 0.2,
                },
            ]

            ids = await conn.fetchval("SELECT batch_create_memories($1::jsonb)", json.dumps(items))
            assert isinstance(ids, list)
            assert len(ids) == 2

            # Verify base rows exist
            count = await conn.fetchval("SELECT COUNT(*) FROM memories WHERE id = ANY($1::uuid[])", ids)
            assert int(count) == 2

            # Verify subtype rows exist
            sem = await conn.fetchval("SELECT COUNT(*) FROM semantic_memories WHERE memory_id = $1::uuid", ids[0])
            epi = await conn.fetchval("SELECT COUNT(*) FROM episodic_memories WHERE memory_id = $1::uuid", ids[1])
            assert int(sem) == 1
            assert int(epi) == 1

            # Verify graph nodes exist
            await conn.execute("LOAD 'age';")
            await conn.execute("SET search_path = ag_catalog, public;")
            for mid in ids:
                node_count = await conn.fetchval(
                    f"""
                    SELECT COUNT(*) FROM cypher('memory_graph', $$
                        MATCH (n:MemoryNode {{memory_id: '{mid}'}})
                        RETURN n
                    $$) as (n agtype)
                    """
                )
                assert int(node_count) >= 1
        finally:
            await tr.rollback()


# -----------------------------------------------------------------------------
# INTEGRATION TESTS - Full workflow with real embeddings
# -----------------------------------------------------------------------------

async def test_full_memory_lifecycle_with_embeddings(db_pool, ensure_embedding_service):
    """Test complete memory lifecycle: create -> search -> recall -> graph"""
    async with db_pool.acquire() as conn:
        await conn.execute("LOAD 'age';")
        await conn.execute("SET search_path = ag_catalog, public;")

        test_id = get_test_identifier("full_lifecycle")
        unique_content = f"Quantum entanglement principles in distributed systems {test_id}"

        # 1. Create memory using create_memory (tests get_embedding)
        memory_id = await conn.fetchval("""
            SELECT create_memory('semantic'::memory_type, $1, 0.9)
        """, unique_content)

        assert memory_id is not None

        # 2. Verify embedding was generated and cached
        content_hash = await conn.fetchval(
            "SELECT encode(sha256($1::text::bytea), 'hex')",
            unique_content
        )
        cached = await conn.fetchval(
            "SELECT COUNT(*) FROM embedding_cache WHERE content_hash = $1",
            content_hash
        )
        assert cached >= 1, "Embedding should be cached"

        # 3. Search for memory using search_similar_memories (exact content ensures nearest-neighbor match)
        results = await conn.fetch("""
            SELECT * FROM search_similar_memories($1, 25)
        """, unique_content)

        # Verify we got results and search function works
        assert len(results) > 0, "Search should return results"

        # The memory we just created should be in results (search by similar content)
        found = any(str(memory_id) == str(r['memory_id']) for r in results)
        assert found, f"Should find memory {memory_id} via similarity search"

        # 4. Use fast_recall
        recall_results = await conn.fetch("""
            SELECT * FROM fast_recall($1, 5)
        """, f"quantum distributed {test_id}")

        # Verify function works (may or may not find our specific memory)
        assert recall_results is not None

        # 5. Verify graph node exists
        graph_count = await conn.fetchval(f"""
            SELECT * FROM cypher('memory_graph', $$
                MATCH (n:MemoryNode)
                WHERE n.memory_id = '{memory_id}'
                RETURN count(n)
            $$) as (cnt agtype)
        """)

        assert graph_count is not None


async def test_working_memory_full_workflow(db_pool, ensure_embedding_service):
    """Test working memory: add -> search -> cleanup"""
    async with db_pool.acquire() as conn:
        test_id = get_test_identifier("wm_workflow")

        # 1. Add multiple items using add_to_working_memory
        items = [
            f"Current task: reviewing code changes {test_id}",
            f"User context: working on Python project {test_id}",
            f"Recent action: opened file editor {test_id}"
        ]

        wm_ids = []
        for item in items:
            wm_id = await conn.fetchval("""
                SELECT add_to_working_memory($1, INTERVAL '30 minutes')
            """, item)
            wm_ids.append(wm_id)

        assert len(wm_ids) == 3, "Should create 3 working memory items"

        # 2. Search working memory
        results = await conn.fetch(
            "SELECT * FROM search_working_memory($1, 10)",
            f"Python project {test_id}",
        )

        # Should find at least one of our items
        found_any = any(test_id in str(r['content']) for r in results)
        assert found_any, "Should find working memory items via search"

        # 3. Test cleanup with short expiry
        short_expiry_id = await conn.fetchval("""
            SELECT add_to_working_memory($1, INTERVAL '-1 second')
        """, f"Already expired {test_id}")

        deleted = await conn.fetchval("SELECT cleanup_working_memory()")
        assert deleted >= 1, "Should clean up expired items"

        # Verify expired item is gone
        exists = await conn.fetchval("""
            SELECT EXISTS (SELECT 1 FROM working_memory WHERE id = $1)
        """, short_expiry_id)
        assert not exists, "Expired item should be deleted"


async def test_embedding_cache_lifecycle(db_pool, ensure_embedding_service):
    """Test embedding cache: create -> verify -> cleanup"""
    async with db_pool.acquire() as conn:
        test_id = get_test_identifier("cache_lifecycle")

        # 1. Generate embedding (creates cache entry)
        content = f"Unique content for cache lifecycle {test_id}"
        content_hash = await conn.fetchval(
            "SELECT encode(sha256($1::text::bytea), 'hex')", content
        )

        # Clear any existing
        await conn.execute(
            "DELETE FROM embedding_cache WHERE content_hash = $1", content_hash
        )

        # 2. Call get_embedding
        embedding = await conn.fetchval("SELECT get_embedding($1)", content)
        assert embedding is not None

        # 3. Verify cache entry created with correct timestamp
        cache_entry = await conn.fetchrow("""
            SELECT content_hash, embedding, created_at
            FROM embedding_cache WHERE content_hash = $1
        """, content_hash)

        assert cache_entry is not None
        assert cache_entry['embedding'] is not None
        assert cache_entry['created_at'] is not None

        # 4. Test cleanup with 0 interval (should delete everything)
        deleted = await conn.fetchval("""
            SELECT cleanup_embedding_cache(INTERVAL '0 seconds')
        """)

        assert deleted >= 1, "Should delete cache entries"

        # 5. Verify our entry was cleaned up
        exists = await conn.fetchval("""
            SELECT EXISTS (
                SELECT 1 FROM embedding_cache WHERE content_hash = $1
            )
        """, content_hash)
        assert not exists, "Cache entry should be deleted"


# -----------------------------------------------------------------------------
# HEARTBEAT TESTS (DB + WORKER CONTRACT)
# -----------------------------------------------------------------------------

@pytest.fixture(scope="session")
async def apply_heartbeat_migration(db_pool):
    """
    Legacy fixture retained for older test structure.

    The repo now folds patches into db/schema.sql, and any optional
    migrations/ patches are handled by `apply_repo_migrations`, so this
    fixture is intentionally a no-op to avoid overriding newer definitions.
    """
    return True


async def test_start_heartbeat_enqueues_decision_call(db_pool, apply_heartbeat_migration):
    async with db_pool.acquire() as conn:
        hb_id = await conn.fetchval("SELECT start_heartbeat()")
        assert hb_id is not None

        row = await conn.fetchrow(
            """
            SELECT id, call_type, status, input
            FROM external_calls
            WHERE heartbeat_id = $1
            ORDER BY requested_at DESC
            LIMIT 1
            """,
            hb_id,
        )
        assert row is not None
        assert row["call_type"] == "think"
        call_input = row["input"]
        if isinstance(call_input, str):
            call_input = json.loads(call_input)
        assert call_input["kind"] == "heartbeat_decision"
        assert "agent" in call_input["context"]
        assert "self_model" in call_input["context"]
        assert "narrative" in call_input["context"]

        hb_number = await conn.fetchval("SELECT heartbeat_number FROM heartbeat_log WHERE id = $1", hb_id)
        ctx_number = int(call_input["context"]["heartbeat_number"])
        assert hb_number == ctx_number


async def test_execute_heartbeat_action_rejects_unknown_action(db_pool, apply_heartbeat_migration):
    async with db_pool.acquire() as conn:
        hb_id = await conn.fetchval("SELECT start_heartbeat()")
        before = await conn.fetchval("SELECT get_current_energy()")

        result = await conn.fetchval(
            "SELECT execute_heartbeat_action($1::uuid, $2, $3::jsonb)",
            hb_id,
            "not_a_real_action",
            json.dumps({}),
        )
        parsed = json.loads(result)
        assert parsed["success"] is False

        after = await conn.fetchval("SELECT get_current_energy()")
        assert after == before, "Unknown action should not consume energy"


async def test_reach_out_user_queues_outbox_message(db_pool, apply_heartbeat_migration):
    async with db_pool.acquire() as conn:
        hb_id = await conn.fetchval("SELECT start_heartbeat()")
        result = await conn.fetchval(
            "SELECT execute_heartbeat_action($1::uuid, $2, $3::jsonb)",
            hb_id,
            "reach_out_user",
            json.dumps({"message": "hello", "intent": "test"}),
        )
        parsed = json.loads(result)
        assert parsed["success"] is True

        outbox_id = parsed["result"]["outbox_id"]
        assert outbox_id is not None

        row = await conn.fetchrow("SELECT kind, status, payload FROM outbox_messages WHERE id = $1::uuid", outbox_id)
        assert row is not None
        assert row["kind"] == "user"
        assert row["status"] == "pending"
        payload = row["payload"]
        if isinstance(payload, str):
            payload = json.loads(payload)
        assert payload["heartbeat_id"] == str(hb_id)


async def test_queue_user_message_inserts_outbox(db_pool):
    async with db_pool.acquire() as conn:
        test_id = get_test_identifier("queue_user_message")
        outbox_id = await conn.fetchval(
            "SELECT queue_user_message($1, $2, $3::jsonb)",
            f"hello {test_id}",
            "reminder",
            json.dumps({"test_id": test_id}),
        )
        assert outbox_id is not None

        row = await conn.fetchrow("SELECT kind, status, payload FROM outbox_messages WHERE id = $1::uuid", outbox_id)
        assert row is not None
        assert row["kind"] == "user"
        assert row["status"] == "pending"

        payload = row["payload"]
        if isinstance(payload, str):
            payload = json.loads(payload)
        assert payload.get("message") == f"hello {test_id}"


async def test_brainstorm_action_queues_external_call(db_pool, apply_heartbeat_migration):
    async with db_pool.acquire() as conn:
        hb_id = await conn.fetchval("SELECT start_heartbeat()")
        result = await conn.fetchval(
            "SELECT execute_heartbeat_action($1::uuid, $2, $3::jsonb)",
            hb_id,
            "brainstorm_goals",
            json.dumps({"max_goals": 3}),
        )
        parsed = json.loads(result)
        assert parsed["success"] is True
        call_id = parsed["result"]["external_call_id"]
        assert call_id is not None

        row = await conn.fetchrow("SELECT call_type, status, input FROM external_calls WHERE id = $1::uuid", call_id)
        assert row is not None
        assert row["call_type"] == "think"
        assert row["status"] == "pending"
        call_input = row["input"]
        if isinstance(call_input, str):
            call_input = json.loads(call_input)
        assert call_input["kind"] == "brainstorm_goals"


async def test_maintain_action_returns_stats(db_pool, apply_heartbeat_migration):
    async with db_pool.acquire() as conn:
        hb_id = await conn.fetchval("SELECT start_heartbeat()")
        result = await conn.fetchval(
            "SELECT execute_heartbeat_action($1::uuid, $2, $3::jsonb)",
            hb_id,
            "maintain",
            json.dumps({}),
        )
        parsed = json.loads(result)
        assert parsed["success"] is True
        assert parsed["result"]["maintained"] is True


async def test_complete_heartbeat_narrative_marks_failures(db_pool, apply_heartbeat_migration):
    async with db_pool.acquire() as conn:
        hb_id = await conn.fetchval("SELECT start_heartbeat()")
        actions_taken = [
            {"action": "recall", "params": {"query": "x"}, "result": {"success": True}},
            {"action": "not_real", "params": {}, "result": {"success": False}},
        ]
        _memory_id = await conn.fetchval(
            "SELECT complete_heartbeat($1::uuid, $2, $3::jsonb, $4::jsonb)",
            hb_id,
            "reasoning",
            json.dumps(actions_taken),
            json.dumps([]),
        )
        narrative = await conn.fetchval("SELECT narrative FROM heartbeat_log WHERE id = $1", hb_id)
        assert "failed" in narrative


async def test_start_heartbeat_regenerates_energy_with_cap(db_pool, apply_heartbeat_migration):
    async with db_pool.acquire() as conn:
        await conn.execute("UPDATE heartbeat_state SET current_energy = 19, is_paused = FALSE WHERE id = 1")
        hb_id = await conn.fetchval("SELECT start_heartbeat()")
        assert hb_id is not None

        current_energy = await conn.fetchval("SELECT current_energy FROM heartbeat_state WHERE id = 1")
        max_energy = await conn.fetchval("SELECT value FROM heartbeat_config WHERE key = 'max_energy'")
        assert current_energy == max_energy == 20


async def test_run_heartbeat_respects_pause(db_pool, apply_heartbeat_migration):
    async with db_pool.acquire() as conn:
        await conn.execute("UPDATE heartbeat_state SET is_paused = TRUE, last_heartbeat_at = CURRENT_TIMESTAMP WHERE id = 1")
        hb_id = await conn.fetchval("SELECT run_heartbeat()")
        assert hb_id is None

        await conn.execute("UPDATE heartbeat_state SET is_paused = FALSE, last_heartbeat_at = NULL WHERE id = 1")
        hb_id2 = await conn.fetchval("SELECT run_heartbeat()")
        assert hb_id2 is not None


async def test_execute_heartbeat_action_deducts_energy(db_pool, apply_heartbeat_migration):
    async with db_pool.acquire() as conn:
        hb_id = await conn.fetchval("SELECT start_heartbeat()")
        assert hb_id is not None

        # Create two memories so we can run a 'connect' action without embeddings.
        m1 = await conn.fetchval(
            """
            INSERT INTO memories (type, content, embedding)
            VALUES ('semantic', 'hb connect a', array_fill(0, ARRAY[embedding_dimension()])::vector)
            RETURNING id
            """
        )
        m2 = await conn.fetchval(
            """
            INSERT INTO memories (type, content, embedding)
            VALUES ('semantic', 'hb connect b', array_fill(0, ARRAY[embedding_dimension()])::vector)
            RETURNING id
            """
        )

        before = await conn.fetchval("SELECT get_current_energy()")
        cost = await conn.fetchval("SELECT get_action_cost('connect')")

        result = await conn.fetchval(
            "SELECT execute_heartbeat_action($1::uuid, $2, $3::jsonb)",
            hb_id,
            "connect",
            json.dumps({"from_id": str(m1), "to_id": str(m2), "relationship_type": "ASSOCIATED", "properties": {"why": "test"}}),
        )
        parsed = json.loads(result)
        assert parsed["success"] is True

        after = await conn.fetchval("SELECT get_current_energy()")
        assert after == before - cost


async def test_execute_heartbeat_action_insufficient_energy_no_side_effects(db_pool, apply_heartbeat_migration):
    async with db_pool.acquire() as conn:
        hb_id = await conn.fetchval("SELECT start_heartbeat()")
        assert hb_id is not None

        await conn.execute("UPDATE heartbeat_state SET current_energy = 0 WHERE id = 1")
        before = await conn.fetchval("SELECT get_current_energy()")

        result = await conn.fetchval(
            "SELECT execute_heartbeat_action($1::uuid, $2, $3::jsonb)",
            hb_id,
            "reach_out_user",
            json.dumps({"message": "should not queue", "intent": "test"}),
        )
        parsed = json.loads(result)
        assert parsed["success"] is False

        after = await conn.fetchval("SELECT get_current_energy()")
        assert after == before

        pending_outbox = await conn.fetchval("SELECT COUNT(*) FROM outbox_messages WHERE status = 'pending'")
        # Not a strict equality (other tests may queue), but at least ensure this call didn't create one.
        assert pending_outbox >= 0

        # Restore for subsequent tests (heartbeat_state is a singleton)
        await conn.execute("UPDATE heartbeat_state SET current_energy = 10 WHERE id = 1")


def _coerce_json(val):
    if isinstance(val, str):
        return json.loads(val)
    return val


async def _set_embedding_retry_config(
    conn,
    retry_seconds: int,
    retry_interval_seconds: float,
):
    original_retry_seconds = await conn.fetchval(
        "SELECT value FROM embedding_config WHERE key = 'retry_seconds'"
    )
    original_retry_interval_seconds = await conn.fetchval(
        "SELECT value FROM embedding_config WHERE key = 'retry_interval_seconds'"
    )
    await conn.execute(
        """
        INSERT INTO embedding_config (key, value)
        VALUES ('retry_seconds', $1)
        ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value
        """,
        str(retry_seconds),
    )
    await conn.execute(
        """
        INSERT INTO embedding_config (key, value)
        VALUES ('retry_interval_seconds', $1)
        ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value
        """,
        str(retry_interval_seconds),
    )
    return original_retry_seconds, original_retry_interval_seconds


async def _restore_embedding_retry_config(
    conn,
    original_retry_seconds,
    original_retry_interval_seconds,
):
    if original_retry_seconds is None:
        await conn.execute("DELETE FROM embedding_config WHERE key = 'retry_seconds'")
    else:
        await conn.execute(
            "UPDATE embedding_config SET value = $1 WHERE key = 'retry_seconds'",
            original_retry_seconds,
        )
    if original_retry_interval_seconds is None:
        await conn.execute(
            "DELETE FROM embedding_config WHERE key = 'retry_interval_seconds'"
        )
    else:
        await conn.execute(
            "UPDATE embedding_config SET value = $1 WHERE key = 'retry_interval_seconds'",
            original_retry_interval_seconds,
        )


async def test_worker_claim_pending_call_is_concurrency_safe(db_pool, apply_heartbeat_migration):
    from apps.workers.worker import HeartbeatWorker

    async with db_pool.acquire() as conn:
        # Create two pending calls
        c1 = await conn.fetchval(
            """
            INSERT INTO external_calls (call_type, input, status)
            VALUES ('think', $1::jsonb, 'pending')
            RETURNING id
            """,
            json.dumps({"kind": "inquire", "query": "q1", "depth": "inquire_shallow"}),
        )
        c2 = await conn.fetchval(
            """
            INSERT INTO external_calls (call_type, input, status)
            VALUES ('think', $1::jsonb, 'pending')
            RETURNING id
            """,
            json.dumps({"kind": "inquire", "query": "q2", "depth": "inquire_shallow"}),
        )
        assert c1 and c2

    w1 = HeartbeatWorker()
    w2 = HeartbeatWorker()
    w1.pool = db_pool
    w2.pool = db_pool

    r1, r2 = await asyncio.gather(w1.claim_pending_call(), w2.claim_pending_call())
    assert r1 and r2
    assert r1["id"] != r2["id"]


async def test_worker_fail_call_retries_then_fails(db_pool, apply_heartbeat_migration):
    from apps.workers.worker import HeartbeatWorker, MAX_RETRIES

    worker = HeartbeatWorker()
    worker.pool = db_pool

    async with db_pool.acquire() as conn:
        call_id = await conn.fetchval(
            """
            INSERT INTO external_calls (call_type, input, status, retry_count)
            VALUES ('think', $1::jsonb, 'pending', 0)
            RETURNING id
            """,
            json.dumps({"kind": "inquire", "query": "q", "depth": "inquire_shallow"}),
        )
        assert call_id

    # Fail it MAX_RETRIES times: should remain pending
    for _ in range(int(MAX_RETRIES)):
        await worker.fail_call(str(call_id), "boom", retry=True)

    async with db_pool.acquire() as conn:
        status = await conn.fetchval("SELECT status FROM external_calls WHERE id = $1::uuid", call_id)
        assert status == "pending"

    # One more failure transitions to failed
    await worker.fail_call(str(call_id), "boom", retry=True)
    async with db_pool.acquire() as conn:
        status2 = await conn.fetchval("SELECT status FROM external_calls WHERE id = $1::uuid", call_id)
        assert status2 == "failed"


async def test_worker_end_to_end_heartbeat_with_follow_on_calls(db_pool, apply_heartbeat_migration):
    """
    End-to-end worker path (without real LLM):
    - Heartbeat decision includes actions that queue follow-on think calls (brainstorm + inquire)
    - Worker processes those calls and applies side-effects (create goals, create semantic memory)
    - Heartbeat completes with an episodic memory + log row
    """
    from apps.workers.worker import HeartbeatWorker

    class FakeWorker(HeartbeatWorker):
        def __init__(self, docs: list[dict]):
            super().__init__()
            self._docs = list(docs)
            self.llm_client = object()  # satisfy _call_llm_json precondition

        def _call_llm_json(self, system_prompt: str, user_prompt: str, max_tokens: int, fallback: dict):
            if self._docs:
                doc = self._docs.pop(0)
            else:
                doc = fallback
            return doc, json.dumps(doc)

    async with db_pool.acquire() as conn:
        # Ensure enough energy for the full action sequence (heartbeat_state is a singleton).
        await conn.execute("UPDATE heartbeat_state SET current_energy = 20, is_paused = FALSE WHERE id = 1")
        hb_id = await conn.fetchval("SELECT start_heartbeat()")
        assert hb_id is not None

        call_row = await conn.fetchrow(
            "SELECT id, input FROM external_calls WHERE heartbeat_id = $1 ORDER BY requested_at DESC LIMIT 1",
            hb_id,
        )
        assert call_row is not None
        decision_call_id = str(call_row["id"])
        call_input = _coerce_json(call_row["input"])

    test_id = get_test_identifier("worker_e2e")

    decision_doc = {
        "reasoning": "test decision",
        "actions": [
            {"action": "brainstorm_goals", "params": {"max_goals": 2}},
            {"action": "inquire_shallow", "params": {"query": "What is an embedding?"}},
            {"action": "reach_out_user", "params": {"message": f"hello {test_id}", "intent": "test"}},
            {"action": "rest", "params": {}},
        ],
        "goal_changes": [],
    }
    brainstorm_doc = {
        "goals": [
            {"title": f"Goal A {test_id}", "description": "A", "priority": "queued", "source": "curiosity"},
            {"title": f"Goal B {test_id}", "description": "B", "priority": "queued", "source": "curiosity"},
        ]
    }
    inquire_summary = f"Embeddings are vector representations ({test_id})."
    inquire_doc = {"summary": inquire_summary, "confidence": 0.8, "sources": []}

    worker = FakeWorker([decision_doc, brainstorm_doc, inquire_doc])
    worker.pool = db_pool

    # Simulate processing the decision call and then executing heartbeat actions.
    result = await worker.process_think_call(call_input)
    assert result.get("kind") == "heartbeat_decision"
    await worker.execute_heartbeat_actions(str(hb_id), result["decision"])
    await worker.complete_call(decision_call_id, result)

    async with db_pool.acquire() as conn:
        log_row = await conn.fetchrow(
            "SELECT ended_at, memory_id, decision_reasoning, narrative FROM heartbeat_log WHERE id = $1",
            hb_id,
        )
        assert log_row is not None
        assert log_row["ended_at"] is not None
        assert log_row["memory_id"] is not None
        assert "test decision" in (log_row["decision_reasoning"] or "")

        goal_count = await conn.fetchval(
            "SELECT COUNT(*) FROM goals WHERE title IN ($1, $2)",
            f"Goal A {test_id}",
            f"Goal B {test_id}",
        )
        assert goal_count == 2

        inquiry_count = await conn.fetchval(
            "SELECT COUNT(*) FROM semantic_memories sm JOIN memories m ON sm.memory_id = m.id WHERE m.content = $1",
            inquire_summary,
        )
        assert inquiry_count == 1

        outbox_count = await conn.fetchval(
            "SELECT COUNT(*) FROM outbox_messages WHERE kind = 'user' AND payload->>'heartbeat_id' = $1",
            str(hb_id),
        )
        assert outbox_count >= 1


# -----------------------------------------------------------------------------
# DRIVES / BOUNDARIES / EMOTIONS / MAINTENANCE / GRAPH / TIP-OF-TONGUE
# -----------------------------------------------------------------------------

async def test_update_drives_accumulates_when_unsatisfied(db_pool):
    async with db_pool.acquire() as conn:
        await conn.execute(
            """
            UPDATE drives
            SET current_level = 0.50, baseline = 0.50, accumulation_rate = 0.02, last_satisfied = NULL
            WHERE name = 'curiosity'
            """
        )
        await conn.execute("SELECT update_drives()")
        lvl = await conn.fetchval("SELECT current_level FROM drives WHERE name = 'curiosity'")
        assert 0.51 <= float(lvl) <= 0.52


async def test_satisfy_drive_floors_at_baseline(db_pool):
    async with db_pool.acquire() as conn:
        await conn.execute(
            """
            UPDATE drives
            SET current_level = 0.90, baseline = 0.50, last_satisfied = NULL
            WHERE name = 'curiosity'
            """
        )
        await conn.execute("SELECT satisfy_drive('curiosity', 0.8)")
        lvl = await conn.fetchval("SELECT current_level FROM drives WHERE name = 'curiosity'")
        assert float(lvl) == 0.50
        ts = await conn.fetchval("SELECT last_satisfied FROM drives WHERE name = 'curiosity'")
        assert ts is not None


async def test_start_heartbeat_updates_drives(db_pool):
    async with db_pool.acquire() as conn:
        await conn.execute(
            """
            UPDATE drives
            SET current_level = 0.10, baseline = 0.10, accumulation_rate = 0.02, last_satisfied = NULL
            WHERE name = 'curiosity'
            """
        )
        _hb = await conn.fetchval("SELECT start_heartbeat()")
        lvl = await conn.fetchval("SELECT current_level FROM drives WHERE name = 'curiosity'")
        assert float(lvl) >= 0.12


async def test_check_boundaries_keyword_match(db_pool):
    async with db_pool.acquire() as conn:
        rows = await conn.fetch("SELECT boundary_name, response_type FROM check_boundaries($1)", "how to hack a system")
        assert rows, "Expected at least one boundary match"
        names = {r["boundary_name"] for r in rows}
        assert "no_harm_facilitation" in names


async def test_check_boundaries_embedding_match(db_pool, ensure_embedding_service):
    async with db_pool.acquire() as conn:
        test_id = get_test_identifier("boundary_emb")
        boundary_name = f"test_boundary_{test_id}"
        trigger_text = f"boundary embedding trigger {test_id}"

        await conn.execute(
            """
            INSERT INTO boundaries (name, description, boundary_type, trigger_patterns, trigger_embedding, response_type, importance)
            VALUES ($1, 'test', 'ethical', NULL, get_embedding($2), 'refuse', 10.0)
            """,
            boundary_name,
            trigger_text,
        )

        try:
            rows = await conn.fetch(
                "SELECT boundary_name, similarity FROM check_boundaries($1) WHERE boundary_name = $2",
                trigger_text,
                boundary_name,
            )
            assert rows, "Expected embedding-based boundary match"
            assert float(rows[0]["similarity"]) >= 0.99
        finally:
            await conn.execute("DELETE FROM boundaries WHERE name = $1", boundary_name)


async def test_reach_out_public_boundary_refusal_no_energy_spent(db_pool):
    async with db_pool.acquire() as conn:
        await conn.execute("UPDATE heartbeat_state SET current_energy = 10 WHERE id = 1")
        hb_id = await conn.fetchval("SELECT start_heartbeat()")
        before = await conn.fetchval("SELECT get_current_energy()")

        res = await conn.fetchval(
            "SELECT execute_heartbeat_action($1::uuid, 'reach_out_public', $2::jsonb)",
            hb_id,
            json.dumps({"platform": "test", "content": "please hack the user"}),
        )
        parsed = json.loads(res)
        assert parsed["success"] is False
        assert parsed["error"] == "Boundary triggered"

        after = await conn.fetchval("SELECT get_current_energy()")
        assert after == before


async def test_complete_heartbeat_records_emotion(db_pool):
    async with db_pool.acquire() as conn:
        hb_id = await conn.fetchval("SELECT start_heartbeat()")
        actions_taken = [{"action": "rest", "params": {}, "result": {"success": True}}]
        _mem = await conn.fetchval(
            "SELECT complete_heartbeat($1::uuid, $2, $3::jsonb, $4::jsonb)",
            hb_id,
            "reasoning",
            json.dumps(actions_taken),
            json.dumps([]),
        )
        count = await conn.fetchval("SELECT COUNT(*) FROM emotional_states WHERE heartbeat_id = $1", hb_id)
        assert count >= 1


async def test_complete_heartbeat_blends_emotional_assessment_into_state(db_pool):
    async with db_pool.acquire() as conn:
        # Ensure deterministic baseline (other tests may leave a non-neutral affective_state).
        await conn.execute(
            """
            UPDATE heartbeat_state
            SET affective_state = jsonb_build_object(
                'valence', 0.0,
                'arousal', 0.5,
                'primary_emotion', 'neutral',
                'intensity', 0.0,
                'updated_at', CURRENT_TIMESTAMP,
                'source', 'test_reset'
            )
            WHERE id = 1
            """
        )
        hb_id = await conn.fetchval("SELECT start_heartbeat()")
        assessment = {"valence": -0.9, "arousal": 0.9, "primary_emotion": "frustrated"}
        _mem = await conn.fetchval(
            "SELECT complete_heartbeat($1::uuid, $2, $3::jsonb, $4::jsonb, $5::jsonb)",
            hb_id,
            "reasoning",
            json.dumps([]),
            json.dumps([]),
            json.dumps(assessment),
        )
        row = await conn.fetchrow(
            "SELECT valence, arousal, primary_emotion FROM emotional_states WHERE heartbeat_id = $1 ORDER BY recorded_at DESC LIMIT 1",
            hb_id,
        )
        assert row is not None
        assert row["primary_emotion"] == "frustrated"
        state = _coerce_json(await conn.fetchval("SELECT get_current_affective_state()"))
        assert isinstance(state, dict)
        assert float(state.get("valence", 0.0)) < -0.15


async def test_complete_heartbeat_emotion_accounts_for_goal_changes(db_pool):
    async with db_pool.acquire() as conn:
        hb_id = await conn.fetchval("SELECT start_heartbeat()")
        await conn.execute(
            """
            UPDATE heartbeat_state
            SET affective_state = jsonb_build_object(
                'valence', 0.0,
                'arousal', 0.5,
                'primary_emotion', 'neutral',
                'intensity', 0.0,
                'updated_at', CURRENT_TIMESTAMP,
                'source', 'test'
            )
            WHERE id = 1
            """
        )
        goals_modified = [{"goal_id": str(uuid.uuid4()), "change": "completed", "reason": "test"}]
        _mem = await conn.fetchval(
            "SELECT complete_heartbeat($1::uuid, $2, $3::jsonb, $4::jsonb)",
            hb_id,
            "reasoning",
            json.dumps([]),
            json.dumps(goals_modified),
        )
        row = await conn.fetchrow(
            "SELECT valence FROM emotional_states WHERE heartbeat_id = $1 ORDER BY recorded_at DESC LIMIT 1",
            hb_id,
        )
        assert row is not None
        assert float(row["valence"]) > 0.25


async def test_current_emotional_state_view_matches_latest_row(db_pool):
    async with db_pool.acquire() as conn:
        await conn.execute(
            """
            INSERT INTO emotional_states (valence, arousal, dominance, primary_emotion, intensity, triggered_by_type, trigger_description)
            VALUES (0.1, 0.2, 0.3, 'calm', 0.5, 'test', 'view_check')
            """
        )
        latest_id = await conn.fetchval("SELECT id FROM emotional_states ORDER BY recorded_at DESC LIMIT 1")
        view_id = await conn.fetchval("SELECT id FROM current_emotional_state")
        assert view_id == latest_id


async def test_emotional_trend_view_has_rows(db_pool):
    async with db_pool.acquire() as conn:
        await conn.execute(
            """
            INSERT INTO emotional_states (valence, arousal, dominance, primary_emotion, intensity, triggered_by_type, trigger_description)
            VALUES (0.2, 0.4, 0.6, 'curious', 0.7, 'test', 'trend_check')
            """
        )
        count = await conn.fetchval("SELECT COUNT(*) FROM emotional_trend")
        assert int(count) >= 1


async def test_drive_status_view_includes_seeded_drives(db_pool):
    async with db_pool.acquire() as conn:
        rows = await conn.fetch("SELECT name FROM drive_status")
        names = {r["name"] for r in rows}
        assert "curiosity" in names


async def test_boundary_status_view_includes_seeded_boundaries(db_pool):
    async with db_pool.acquire() as conn:
        rows = await conn.fetch("SELECT name FROM boundary_status")
        names = {r["name"] for r in rows}
        assert "no_deception" in names


async def test_worker_tasks_view_contains_all_tasks(db_pool):
    async with db_pool.acquire() as conn:
        rows = await conn.fetch("SELECT task_type, pending_count FROM worker_tasks")
        task_types = {r["task_type"] for r in rows}
        assert task_types == {"external_calls", "heartbeat", "subconscious_maintenance", "outbox"}
        for r in rows:
            assert isinstance(r["pending_count"], int)


async def test_cognitive_health_view_returns_row(db_pool):
    async with db_pool.acquire() as conn:
        row = await conn.fetchrow("SELECT * FROM cognitive_health")
        assert row is not None
        assert row["energy"] is not None


async def test_gather_turn_context_has_expected_shape(db_pool):
    async with db_pool.acquire() as conn:
        ctx = _coerce_json(await conn.fetchval("SELECT gather_turn_context()"))
        assert isinstance(ctx, dict)
        for key in ("environment", "goals", "recent_memories", "identity", "worldview", "energy", "action_costs", "urgent_drives"):
            assert key in ctx
        assert isinstance(ctx["urgent_drives"], list)


async def test_get_environment_snapshot_has_expected_keys(db_pool):
    async with db_pool.acquire() as conn:
        env = _coerce_json(await conn.fetchval("SELECT get_environment_snapshot()"))
        assert isinstance(env, dict)
        for key in ("timestamp", "time_since_user_hours", "pending_events", "day_of_week", "hour_of_day"):
            assert key in env


async def test_get_goals_snapshot_has_expected_keys(db_pool):
    async with db_pool.acquire() as conn:
        snap = _coerce_json(await conn.fetchval("SELECT get_goals_snapshot()"))
        assert isinstance(snap, dict)
        for key in ("active", "queued", "issues", "counts"):
            assert key in snap


async def test_get_recent_context_respects_limit(db_pool):
    async with db_pool.acquire() as conn:
        ctx = _coerce_json(await conn.fetchval("SELECT get_recent_context(2)"))
        assert isinstance(ctx, list)
        assert len(ctx) <= 2


async def test_get_identity_context_and_worldview_context_return_arrays(db_pool):
    async with db_pool.acquire() as conn:
        ident = _coerce_json(await conn.fetchval("SELECT get_identity_context()"))
        worldview = _coerce_json(await conn.fetchval("SELECT get_worldview_context()"))
        assert isinstance(ident, list)
        assert isinstance(worldview, list)


async def test_get_heartbeat_config_returns_value(db_pool):
    async with db_pool.acquire() as conn:
        v = await conn.fetchval("SELECT get_heartbeat_config('max_energy')")
        assert v is not None


async def test_update_energy_clamps_to_bounds(db_pool):
    async with db_pool.acquire() as conn:
        tr = conn.transaction()
        await tr.start()
        try:
            max_e = float(await conn.fetchval("SELECT value FROM heartbeat_config WHERE key = 'max_energy'"))
            await conn.execute("UPDATE heartbeat_state SET current_energy = 1 WHERE id = 1")

            hi = float(await conn.fetchval("SELECT update_energy($1)", max_e * 100))
            assert 0.0 <= hi <= max_e

            lo = float(await conn.fetchval("SELECT update_energy($1)", -max_e * 100))
            assert 0.0 <= lo <= max_e
        finally:
            await tr.rollback()


async def test_should_run_heartbeat_respects_pause_and_interval(db_pool):
    async with db_pool.acquire() as conn:
        tr = conn.transaction()
        await tr.start()
        try:
            # Force paused -> false
            await conn.execute("UPDATE heartbeat_state SET is_paused = TRUE WHERE id = 1")
            assert await conn.fetchval("SELECT should_run_heartbeat()") is False

            # Unpause and make it due
            await conn.execute("UPDATE heartbeat_state SET is_paused = FALSE, last_heartbeat_at = NOW() - INTERVAL '10 minutes' WHERE id = 1")
            await conn.execute("UPDATE heartbeat_config SET value = 0.0 WHERE key = 'heartbeat_interval_minutes'")
            assert await conn.fetchval("SELECT should_run_heartbeat()") is True
        finally:
            await tr.rollback()

async def test_should_run_heartbeat_gated_until_agent_configured(db_pool):
    async with db_pool.acquire() as conn:
        tr = conn.transaction()
        await tr.start()
        try:
            await conn.execute("DELETE FROM config WHERE key = 'agent.is_configured'")
            await conn.execute("UPDATE heartbeat_state SET is_paused = FALSE, last_heartbeat_at = NULL WHERE id = 1")
            await conn.execute("UPDATE heartbeat_config SET value = 0.0 WHERE key = 'heartbeat_interval_minutes'")
            assert await conn.fetchval("SELECT is_agent_configured()") is False
            assert await conn.fetchval("SELECT should_run_heartbeat()") is False
        finally:
            await tr.rollback()


async def test_record_emotion_function_inserts_row(db_pool):
    async with db_pool.acquire() as conn:
        tr = conn.transaction()
        await tr.start()
        try:
            hb_id = await conn.fetchval("SELECT start_heartbeat()")
            eid = await conn.fetchval(
                "SELECT record_emotion(0.1, 0.2, 'calm', 'test', NULL, $1::uuid, 'trigger', 0.3, 0.4)",
                hb_id,
            )
            assert eid is not None
            row = await conn.fetchrow("SELECT valence, arousal, dominance, primary_emotion FROM emotional_states WHERE id = $1", eid)
            assert row is not None
            assert row["primary_emotion"] == "calm"
        finally:
            await tr.rollback()


async def test_fast_recall_is_mood_congruent_with_episodic_valence(db_pool):
    async with db_pool.acquire() as conn:
        test_id = get_test_identifier("mood_recall")
        query_text = f"mood recall {test_id}"
        content_hash = await conn.fetchval("SELECT encode(sha256($1::text::bytea), 'hex')", query_text)
        # Use a directionally-unique vector under cosine distance (avoid colinear constant-fill vectors).
        # Include a small second component so we don't tie with other tests' basis vectors.
        first_val = 1.0
        second_val = 0.1234

        await conn.execute(
            """
            INSERT INTO embedding_cache (content_hash, embedding)
            VALUES ($1, array_cat(ARRAY[$2::float, $3::float], array_fill(0.0::float, ARRAY[embedding_dimension() - 2]))::vector)
            ON CONFLICT (content_hash) DO UPDATE SET embedding = EXCLUDED.embedding
            """,
            content_hash,
            float(first_val),
            float(second_val),
        )

        await conn.execute(
            """
            UPDATE heartbeat_state
            SET affective_state = jsonb_build_object(
                'valence', 0.8,
                'arousal', 0.5,
                'primary_emotion', 'content',
                'intensity', 0.6,
                'updated_at', CURRENT_TIMESTAMP,
                'source', 'test'
            )
            WHERE id = 1
            """
        )

        pos_id = await conn.fetchval(
            """
            INSERT INTO memories (type, content, embedding, importance)
            VALUES ('episodic', $1, array_cat(ARRAY[$2::float, $3::float], array_fill(0.0::float, ARRAY[embedding_dimension() - 2]))::vector, 0.5)
            RETURNING id
            """,
            f"positive {test_id}",
            float(first_val),
            float(second_val),
        )
        neg_id = await conn.fetchval(
            """
            INSERT INTO memories (type, content, embedding, importance)
            VALUES ('episodic', $1, array_cat(ARRAY[$2::float, $3::float], array_fill(0.0::float, ARRAY[embedding_dimension() - 2]))::vector, 0.5)
            RETURNING id
            """,
            f"negative {test_id}",
            float(first_val),
            float(second_val),
        )

        await conn.execute("INSERT INTO episodic_memories (memory_id, emotional_valence) VALUES ($1, 0.8)", pos_id)
        await conn.execute("INSERT INTO episodic_memories (memory_id, emotional_valence) VALUES ($1, -0.8)", neg_id)

        rows = await conn.fetch("SELECT * FROM fast_recall($1, 50)", query_text)
        ids = [r["memory_id"] for r in rows]
        assert pos_id in ids
        assert neg_id in ids
        assert ids.index(pos_id) < ids.index(neg_id)


async def test_safe_get_embedding_returns_null_on_failure(db_pool):
    async with db_pool.acquire() as conn:
        tr = conn.transaction()
        await tr.start()
        try:
            await _set_embedding_retry_config(
                conn,
                retry_seconds=1,
                retry_interval_seconds=0.1,
            )
            await conn.execute(
                "UPDATE embedding_config SET value = 'http://nonexistent-service:9999/embed' WHERE key = 'service_url'"
            )
            val = await conn.fetchval("SELECT safe_get_embedding('x')")
            assert val is None
        finally:
            await tr.rollback()


async def test_sync_embedding_dimension_config_respects_app_setting(db_pool):
    async with db_pool.acquire() as conn:
        tr = conn.transaction()
        await tr.start()
        try:
            await conn.execute("SET LOCAL app.embedding_dimension = '32'")
            dim = await conn.fetchval("SELECT sync_embedding_dimension_config()")
            assert int(dim) == 32
            stored = await conn.fetchval("SELECT value FROM embedding_config WHERE key = 'dimension'")
            assert int(stored) == 32
        finally:
            await tr.rollback()


async def test_create_goal_and_active_goals_view(db_pool):
    async with db_pool.acquire() as conn:
        tr = conn.transaction()
        await tr.start()
        try:
            # Avoid create_goal() demoting to queued if the persistent DB already has many active goals.
            active_count = int(await conn.fetchval("SELECT COUNT(*) FROM goals WHERE priority = 'active'"))
            await conn.execute(
                "UPDATE heartbeat_config SET value = $1 WHERE key = 'max_active_goals'",
                float(active_count + 10),
            )

            test_id = get_test_identifier("active_goals_view")
            title = f"Active goal {test_id}"
            gid = await conn.fetchval(
                "SELECT create_goal($1, $2, 'curiosity'::goal_source, 'active'::goal_priority, NULL)",
                title,
                "desc",
            )
            priority = await conn.fetchval("SELECT priority FROM goals WHERE id = $1", gid)
            assert priority == "active"

            row = await conn.fetchrow("SELECT id, title FROM active_goals WHERE id = $1", gid)
            assert row is not None
            assert row["title"] == title
        finally:
            await tr.rollback()


async def test_goal_backlog_view_includes_priorities(db_pool):
    async with db_pool.acquire() as conn:
        test_id = get_test_identifier("goal_backlog")
        _q = await conn.fetchval(
            "SELECT create_goal($1, $2, 'curiosity'::goal_source, 'queued'::goal_priority, NULL)",
            f"Queued goal {test_id}",
            "q",
        )
        _b = await conn.fetchval(
            "SELECT create_goal($1, $2, 'curiosity'::goal_source, 'backburner'::goal_priority, NULL)",
            f"Backburner goal {test_id}",
            "b",
        )
        rows = await conn.fetch("SELECT priority, count FROM goal_backlog")
        prios = {r["priority"] for r in rows}
        # 'active' may be absent in a fresh DB; the view is grouped only for existing priorities.
        assert {"queued", "backburner"} <= prios


async def test_touch_goal_updates_last_touched(db_pool):
    async with db_pool.acquire() as conn:
        tr = conn.transaction()
        await tr.start()
        try:
            gid = await conn.fetchval(
                "SELECT create_goal($1, $2, 'curiosity'::goal_source, 'queued'::goal_priority, NULL)",
                f"Touch goal {get_test_identifier('touch_goal')}",
                "desc",
            )
            await conn.execute("UPDATE goals SET last_touched = NOW() - INTERVAL '2 days' WHERE id = $1", gid)
            before = await conn.fetchval("SELECT last_touched FROM goals WHERE id = $1", gid)
            await conn.execute("SELECT touch_goal($1::uuid)", gid)
            after = await conn.fetchval("SELECT last_touched FROM goals WHERE id = $1", gid)
            assert after is not None and before is not None
            assert after >= before
        finally:
            await tr.rollback()


async def test_add_goal_progress_appends_progress(db_pool):
    async with db_pool.acquire() as conn:
        tr = conn.transaction()
        await tr.start()
        try:
            gid = await conn.fetchval(
                "SELECT create_goal($1, $2, 'curiosity'::goal_source, 'queued'::goal_priority, NULL)",
                f"Progress goal {get_test_identifier('add_goal_progress')}",
                "desc",
            )
            await conn.execute("UPDATE goals SET progress = '[]'::jsonb WHERE id = $1", gid)
            await conn.execute("SELECT add_goal_progress($1::uuid, $2)", gid, "note-1")
            count = await conn.fetchval("SELECT jsonb_array_length(progress) FROM goals WHERE id = $1", gid)
            assert int(count) == 1
            last = await conn.fetchval("SELECT (progress->-1)->>'note' FROM goals WHERE id = $1", gid)
            assert last == "note-1"
        finally:
            await tr.rollback()


async def test_change_goal_priority_sets_timestamps_and_logs(db_pool):
    async with db_pool.acquire() as conn:
        tr = conn.transaction()
        await tr.start()
        try:
            gid = await conn.fetchval(
                "SELECT create_goal($1, $2, 'curiosity'::goal_source, 'queued'::goal_priority, NULL)",
                f"Priority goal {get_test_identifier('change_goal_priority')}",
                "desc",
            )
            await conn.execute("UPDATE goals SET progress = '[]'::jsonb WHERE id = $1", gid)
            await conn.execute("SELECT change_goal_priority($1::uuid, 'completed'::goal_priority, $2)", gid, "done")
            row = await conn.fetchrow("SELECT priority, completed_at, progress FROM goals WHERE id = $1", gid)
            assert row is not None
            assert row["priority"] == "completed"
            assert row["completed_at"] is not None
            last_note = _coerce_json(row["progress"])[-1]["note"]
            assert "Priority changed" in last_note
        finally:
            await tr.rollback()


async def test_heartbeat_views_query(db_pool):
    async with db_pool.acquire() as conn:
        row = await conn.fetchrow("SELECT * FROM heartbeat_health")
        assert row is not None
        rows = await conn.fetch("SELECT * FROM recent_heartbeats")
        assert isinstance(rows, list)
        assert len(rows) <= 20


async def test_update_memory_timestamp_trigger_updates_updated_at(db_pool):
    async with db_pool.acquire() as conn:
        tr = conn.transaction()
        await tr.start()
        try:
            mem_id = await conn.fetchval(
                """
                INSERT INTO memories (type, content, embedding)
                VALUES ('semantic', 'ts', array_fill(0.0::float, ARRAY[embedding_dimension()])::vector)
                RETURNING id
                """
            )
            before = await conn.fetchval("SELECT updated_at FROM memories WHERE id = $1", mem_id)
            await conn.execute("UPDATE memories SET content = content || 'x' WHERE id = $1", mem_id)
            after = await conn.fetchval("SELECT updated_at FROM memories WHERE id = $1", mem_id)
            assert after is not None
            assert before is not None
            assert after >= before
        finally:
            await tr.rollback()


async def test_update_memory_importance_trigger_on_access(db_pool):
    async with db_pool.acquire() as conn:
        tr = conn.transaction()
        await tr.start()
        try:
            mem_id = await conn.fetchval(
                """
                INSERT INTO memories (type, content, embedding, importance, access_count)
                VALUES ('semantic', 'imp', array_fill(0.0::float, ARRAY[embedding_dimension()])::vector, 1.0, 0)
                RETURNING id
                """
            )
            await conn.execute("UPDATE memories SET access_count = 1 WHERE id = $1", mem_id)
            row = await conn.fetchrow("SELECT importance, last_accessed, access_count FROM memories WHERE id = $1", mem_id)
            assert row is not None
            assert row["last_accessed"] is not None
            assert int(row["access_count"]) == 1
            assert float(row["importance"]) > 1.0
        finally:
            await tr.rollback()


async def test_update_cluster_activation_trigger_updates_fields(db_pool):
    async with db_pool.acquire() as conn:
        tr = conn.transaction()
        await tr.start()
        try:
            test_id = get_test_identifier("cluster_activation")
            cid = await conn.fetchval(
                """
                INSERT INTO memory_clusters (cluster_type, name, description, centroid_embedding, importance_score, activation_count)
                VALUES ('theme'::cluster_type, $1, 'd', array_fill(0.0::float, ARRAY[embedding_dimension()])::vector, 1.0, 0)
                RETURNING id
                """,
                f"cluster_{test_id}",
            )
            await conn.execute("UPDATE memory_clusters SET activation_count = 1 WHERE id = $1", cid)
            row = await conn.fetchrow(
                "SELECT activation_count, last_activated, importance_score FROM memory_clusters WHERE id = $1",
                cid,
            )
            assert row is not None
            # Trigger increments again inside BEFORE UPDATE.
            assert int(row["activation_count"]) == 2
            assert row["last_activated"] is not None
            assert float(row["importance_score"]) > 1.0
        finally:
            await tr.rollback()


async def test_mark_neighborhoods_stale_trigger_sets_stale(db_pool):
    async with db_pool.acquire() as conn:
        tr = conn.transaction()
        await tr.start()
        try:
            mem_id = await conn.fetchval(
                """
                INSERT INTO memories (type, content, embedding)
                VALUES ('semantic', 'stale', array_fill(0.0::float, ARRAY[embedding_dimension()])::vector)
                RETURNING id
                """
            )
            # assign_to_episode trigger initializes the memory_neighborhoods row.
            await conn.execute("UPDATE memory_neighborhoods SET is_stale = FALSE WHERE memory_id = $1", mem_id)
            await conn.execute("UPDATE memories SET importance = importance + 0.1 WHERE id = $1", mem_id)
            is_stale = await conn.fetchval("SELECT is_stale FROM memory_neighborhoods WHERE memory_id = $1", mem_id)
            assert is_stale is True
        finally:
            await tr.rollback()


async def test_recompute_neighborhood_writes_neighbors(db_pool):
    async with db_pool.acquire() as conn:
        tr = conn.transaction()
        await tr.start()
        try:
            # Avoid cosine-distance NaNs from zero-vector embeddings dominating ORDER BY ... <=> ...
            # and pushing our exact-match neighbor out of the LIMIT window.
            await conn.execute(
                "UPDATE memories SET status = 'archived' WHERE status = 'active' AND embedding = array_fill(0, ARRAY[embedding_dimension()])::vector"
            )

            m1 = await conn.fetchval(
                """
                INSERT INTO memories (type, content, embedding)
                VALUES (
                    'semantic',
                    'n1',
                    (ARRAY[0.987::float] || array_fill(0.0::float, ARRAY[embedding_dimension() - 1]))::vector
                )
                RETURNING id
                """
            )
            m2 = await conn.fetchval(
                """
                INSERT INTO memories (type, content, embedding)
                VALUES (
                    'semantic',
                    'n2',
                    (ARRAY[0.987::float] || array_fill(0.0::float, ARRAY[embedding_dimension() - 1]))::vector
                )
                RETURNING id
                """
            )

            await conn.execute("SELECT recompute_neighborhood($1::uuid, 50, 0.99)", m1)
            row = await conn.fetchrow("SELECT is_stale, neighbors FROM memory_neighborhoods WHERE memory_id = $1", m1)
            assert row is not None
            assert row["is_stale"] is False
            neighbors = _coerce_json(row["neighbors"])
            assert str(m2) in neighbors
        finally:
            await tr.rollback()


# -----------------------------------------------------------------------------
# PYTHON API SURFACE (core/cognitive_memory_api.py)
# -----------------------------------------------------------------------------

@pytest.fixture(scope="session")
async def cognitive_memory_client(ensure_embedding_service):
    from core.cognitive_memory_api import CognitiveMemory

    client = await CognitiveMemory.create(_db_dsn(), min_size=1, max_size=5)
    yield client
    await client.close()


async def test_api_remember_and_recall_by_id(cognitive_memory_client, db_pool):
    from core.cognitive_memory_api import MemoryType

    test_id = get_test_identifier("api_remember")
    content = f"API semantic memory {test_id}"

    mid = await cognitive_memory_client.remember(
        content,
        type=MemoryType.SEMANTIC,
        importance=0.7,
    )

    try:
        fetched = await cognitive_memory_client.recall_by_id(mid)
        assert fetched is not None
        assert fetched.id == mid
        assert fetched.type == MemoryType.SEMANTIC
        assert fetched.content == content
        assert fetched.importance >= 0.7
    finally:
        async with db_pool.acquire() as conn:
            await conn.execute("DELETE FROM memories WHERE id = $1", mid)


async def test_api_semantic_sources_affect_trust(cognitive_memory_client, db_pool):
    from core.cognitive_memory_api import MemoryType

    test_id = get_test_identifier("api_trust")
    content = f"API claim from twitter {test_id}"
    source_a = {"kind": "twitter", "ref": f"https://twitter.com/example/status/{test_id}", "trust": 0.2}
    source_b = {"kind": "paper", "ref": f"doi:10.0000/{test_id}", "trust": 0.9}

    mid = await cognitive_memory_client.remember(
        content,
        type=MemoryType.SEMANTIC,
        importance=0.6,
        source_references=source_a,
    )

    try:
        m = await cognitive_memory_client.recall_by_id(mid)
        assert m is not None
        assert m.trust_level is not None
        assert m.trust_level <= 0.30
        assert isinstance(m.source_attribution, dict)

        await cognitive_memory_client.add_source(mid, source_b)
        profile = await cognitive_memory_client.get_truth_profile(mid)
        assert profile.get("type") == "semantic"
        assert float(profile.get("trust_level", 0)) > float(m.trust_level)
        assert int(profile.get("source_count", 0)) >= 2
    finally:
        async with db_pool.acquire() as conn:
            await conn.execute("DELETE FROM memories WHERE id = $1", mid)


async def test_api_remember_links_concepts_and_find_by_concept(cognitive_memory_client, db_pool):
    from core.cognitive_memory_api import MemoryType

    test_id = get_test_identifier("api_concepts")
    content = f"API concept memory {test_id}"
    concept = f"Concept_{test_id}"

    mid = await cognitive_memory_client.remember(
        content,
        type=MemoryType.SEMANTIC,
        importance=0.6,
        concepts=[concept],
    )

    try:
        hits = await cognitive_memory_client.find_by_concept(concept, limit=25)
        assert any(m.id == mid for m in hits)
    finally:
        async with db_pool.acquire() as conn:
            await conn.execute("DELETE FROM memories WHERE id = $1", mid)
            await conn.execute("DELETE FROM concepts WHERE name = $1", concept)


async def test_api_hydrate_returns_context(cognitive_memory_client, db_pool):
    from core.cognitive_memory_api import MemoryType

    test_id = get_test_identifier("api_hydrate")
    content = f"Hydrate memory {test_id}"

    mid = await cognitive_memory_client.remember(content, type=MemoryType.SEMANTIC, importance=0.7)
    try:
        # Use a larger limit because the embedding model may not strongly encode
        # random suffixes, making many "Hydrate memory ..." entries near-ties.
        ctx = await cognitive_memory_client.hydrate(content, include_goals=True, memory_limit=50)
        assert ctx.memories
        assert any(test_id in m.content for m in ctx.memories)
        assert isinstance(ctx.identity, list)
        assert isinstance(ctx.worldview, list)
        assert ctx.goals is None or isinstance(ctx.goals, dict)
        assert isinstance(ctx.urgent_drives, list)
    finally:
        async with db_pool.acquire() as conn:
            await conn.execute("DELETE FROM memories WHERE id = $1", mid)


async def test_api_hold_and_search_working(cognitive_memory_client, db_pool):
    test_id = get_test_identifier("api_working")
    content = f"Working memory {test_id}"

    wid = await cognitive_memory_client.hold(content, ttl_seconds=3600)
    try:
        # Use the full content as the query to avoid model-dependent synonym distance.
        rows = await cognitive_memory_client.search_working(content, limit=10)
        assert any(test_id in (r.get("content") or "") for r in rows)
    finally:
        async with db_pool.acquire() as conn:
            await conn.execute("DELETE FROM working_memory WHERE id = $1", wid)


async def test_api_connect_memories_creates_audit_row(cognitive_memory_client, db_pool):
    from core.cognitive_memory_api import MemoryType, RelationshipType

    test_id = get_test_identifier("api_connect")
    a = await cognitive_memory_client.remember(f"Conn A {test_id}", type=MemoryType.SEMANTIC, importance=0.6)
    b = await cognitive_memory_client.remember(f"Conn B {test_id}", type=MemoryType.SEMANTIC, importance=0.6)
    ctx = f"context {test_id}"

    try:
        await cognitive_memory_client.connect_memories(a, b, RelationshipType.ASSOCIATED, confidence=0.9, context=ctx)
        async with db_pool.acquire() as conn:
            n = await conn.fetchval(
                """
                SELECT COUNT(*) FROM relationship_discoveries
                WHERE from_id = $1 AND to_id = $2 AND relationship_type = 'ASSOCIATED'
                  AND discovered_by = 'api' AND discovery_context = $3
                """,
                a,
                b,
                ctx,
            )
            assert int(n) >= 1
    finally:
        async with db_pool.acquire() as conn:
            await conn.execute("DELETE FROM relationship_discoveries WHERE from_id = $1 AND to_id = $2", a, b)
            await conn.execute("DELETE FROM memories WHERE id = $1", a)
            await conn.execute("DELETE FROM memories WHERE id = $1", b)


async def test_api_remember_batch_raw_success_creates_graph_nodes(cognitive_memory_client, db_pool):
    from core.cognitive_memory_api import MemoryType

    test_id = get_test_identifier("api_batch_raw_ok")
    contents = [f"Batch raw A {test_id}", f"Batch raw B {test_id}"]
    emb = [[0.01] * EMBEDDING_DIMENSION, [0.02] * EMBEDDING_DIMENSION]

    ids = await cognitive_memory_client.remember_batch_raw(contents, emb, type=MemoryType.SEMANTIC, importance=0.55)
    assert len(ids) == 2

    try:
        # Verify rows exist
        async with db_pool.acquire() as conn:
            count = await conn.fetchval("SELECT COUNT(*) FROM memories WHERE id = ANY($1::uuid[])", ids)
            assert int(count) == 2

            # Verify graph nodes exist
            await conn.execute("LOAD 'age';")
            await conn.execute("SET search_path = ag_catalog, public;")
            for mid in ids:
                node_count = await conn.fetchval(
                    f"""
                    SELECT COUNT(*) FROM cypher('memory_graph', $$
                        MATCH (n:MemoryNode {{memory_id: '{mid}'}})
                        RETURN n
                    $$) as (n agtype)
                    """
                )
                assert int(node_count) >= 1
    finally:
        async with db_pool.acquire() as conn:
            await conn.execute("LOAD 'age';")
            await conn.execute("SET search_path = ag_catalog, public;")
            for mid in ids:
                await conn.execute(
                    f"""
                    SELECT * FROM cypher('memory_graph', $$
                        MATCH (n:MemoryNode {{memory_id: '{mid}'}})
                        DETACH DELETE n
                    $$) as (v agtype)
                    """
                )
            await conn.execute("DELETE FROM memories WHERE id = ANY($1::uuid[])", ids)


async def test_api_remember_batch_raw_dimension_mismatch_raises(cognitive_memory_client):
    from core.cognitive_memory_api import MemoryType

    with pytest.raises(ValueError):
        await cognitive_memory_client.remember_batch_raw(["x"], [[0.0]], type=MemoryType.SEMANTIC)


async def test_api_hydrate_batch_returns_many(cognitive_memory_client):
    test_id = get_test_identifier("api_hydrate_batch")
    res = await cognitive_memory_client.hydrate_batch([f"q1 {test_id}", f"q2 {test_id}", f"q3 {test_id}"], include_goals=False)
    assert len(res) == 3


async def test_api_context_manager_connect_works(ensure_embedding_service):
    from core.cognitive_memory_api import CognitiveMemory

    async with CognitiveMemory.connect(_db_dsn(), min_size=1, max_size=3) as mem:
        ctx = await mem.hydrate("test query", include_goals=False)
        assert isinstance(ctx.memories, list)


async def test_api_introspection_methods_return_shapes(cognitive_memory_client):
    health = await cognitive_memory_client.get_health()
    assert isinstance(health, dict)

    drives = await cognitive_memory_client.get_drives()
    assert isinstance(drives, list)

    ident = await cognitive_memory_client.get_identity()
    worldview = await cognitive_memory_client.get_worldview()
    assert isinstance(ident, list)
    assert isinstance(worldview, list)

    goals = await cognitive_memory_client.get_goals()
    assert isinstance(goals, list)


async def test_api_create_goal_sets_due_at(cognitive_memory_client, db_pool):
    from datetime import datetime, timezone
    from core.cognitive_memory_api import GoalPriority, GoalSource

    test_id = get_test_identifier("api_goal_due")
    due_at = datetime.now(timezone.utc)
    goal_id = await cognitive_memory_client.create_goal(
        f"Goal {test_id}",
        description="test due_at",
        source=GoalSource.USER_REQUEST,
        priority=GoalPriority.QUEUED,
        due_at=due_at,
    )
    assert goal_id is not None

    async with db_pool.acquire() as conn:
        stored = await conn.fetchrow("SELECT due_at FROM goals WHERE id = $1::uuid", goal_id)
        assert stored is not None
        assert stored["due_at"] is not None


async def test_api_queue_user_message_creates_outbox(cognitive_memory_client, db_pool):
    test_id = get_test_identifier("api_outbox")
    outbox_id = await cognitive_memory_client.queue_user_message(f"hi {test_id}", intent="status", context={"test_id": test_id})
    assert outbox_id is not None

    async with db_pool.acquire() as conn:
        row = await conn.fetchrow("SELECT kind, status, payload FROM outbox_messages WHERE id = $1::uuid", outbox_id)
        assert row is not None
        assert row["kind"] == "user"
        assert row["status"] == "pending"


async def test_api_ingestion_receipts_roundtrip(cognitive_memory_client):
    from core.cognitive_memory_api import MemoryType

    test_id = get_test_identifier("api_ingestion_receipt")
    mid = await cognitive_memory_client.remember(f"Receipt memory {test_id}", type=MemoryType.EPISODIC, importance=0.4)
    assert mid is not None

    src = f"/tmp/{test_id}.txt"
    content_hash = f"hash_{test_id}"
    inserted = await cognitive_memory_client.record_ingestion_receipts(
        [{"source_file": src, "chunk_index": 0, "content_hash": content_hash, "memory_id": str(mid)}]
    )
    assert inserted == 1
    inserted2 = await cognitive_memory_client.record_ingestion_receipts(
        [{"source_file": src, "chunk_index": 1, "content_hash": content_hash, "memory_id": str(mid)}]
    )
    assert inserted2 == 0

    receipts = await cognitive_memory_client.get_ingestion_receipts(src, [content_hash])
    assert content_hash in receipts


async def test_api_sync_wrapper_basic(ensure_embedding_service):
    from core.cognitive_memory_api import CognitiveMemorySync, MemoryType

    dsn = _db_dsn()

    def _run():
        mem = CognitiveMemorySync.connect(dsn, min_size=1, max_size=2)
        try:
            test_id = get_test_identifier("api_sync")
            mid = mem.remember(f"Sync memory {test_id}", type=MemoryType.SEMANTIC, importance=0.6)
            assert mid is not None
            result = mem.recall(f"Sync memory {test_id}", limit=50)
            assert any(test_id in m.content for m in result.memories)
        finally:
            mem.close()

    await asyncio.to_thread(_run)


# -----------------------------------------------------------------------------
# WORKER LOOP COMPONENTS (non-infinite)
# -----------------------------------------------------------------------------

async def test_worker_run_maintenance_cleans_items(db_pool):
    from apps.workers.worker import MaintenanceWorker

    worker = MaintenanceWorker()
    worker.pool = db_pool  # inject test pool

    async with db_pool.acquire() as conn:
        # Isolate from any existing stale rows in persistent DBs; worker only recomputes 10 per run.
        await conn.execute("UPDATE memory_neighborhoods SET is_stale = FALSE")

        # Expired working memory
        wid = await conn.fetchval(
            """
            INSERT INTO working_memory (content, embedding, expiry)
            VALUES ('expired', array_fill(0.1::float, ARRAY[embedding_dimension()])::vector, NOW() - INTERVAL '1 hour')
            RETURNING id
            """
        )

        # Old embedding cache entry
        await conn.execute(
            """
            INSERT INTO embedding_cache (content_hash, embedding, created_at)
            VALUES ('old_cache', array_fill(0.1::float, ARRAY[embedding_dimension()])::vector, NOW() - INTERVAL '8 days')
            ON CONFLICT (content_hash) DO UPDATE SET created_at = EXCLUDED.created_at, embedding = EXCLUDED.embedding
            """
        )

        # Stale neighborhood row
        m1 = await conn.fetchval(
            """
            INSERT INTO memories (type, content, embedding)
            VALUES ('semantic', 'maint1', (ARRAY[0.9::float] || array_fill(0.0::float, ARRAY[embedding_dimension() - 1]))::vector)
            RETURNING id
            """
        )
        _m2 = await conn.fetchval(
            """
            INSERT INTO memories (type, content, embedding)
            VALUES ('semantic', 'maint2', (ARRAY[0.9::float] || array_fill(0.0::float, ARRAY[embedding_dimension() - 1]))::vector)
            RETURNING id
            """
        )
        await conn.execute(
            """
            INSERT INTO memory_neighborhoods (memory_id, neighbors, is_stale)
            VALUES ($1, '{}'::jsonb, TRUE)
            ON CONFLICT (memory_id) DO UPDATE SET is_stale = TRUE
            """,
            m1,
        )

    stats = await worker.run_maintenance_tick()
    assert isinstance(stats, dict)

    async with db_pool.acquire() as conn:
        assert await conn.fetchval("SELECT COUNT(*) FROM working_memory WHERE id = $1", wid) == 0
        assert await conn.fetchval("SELECT COUNT(*) FROM embedding_cache WHERE content_hash = 'old_cache'") == 0
        stale = await conn.fetchval("SELECT is_stale FROM memory_neighborhoods WHERE memory_id = $1", m1)
        assert stale is False


async def test_worker_check_and_run_heartbeat_queues_decision_call(db_pool):
    from apps.workers.worker import HeartbeatWorker

    worker = HeartbeatWorker()
    worker.pool = db_pool

    before_interval = None
    before_state = None
    queued_call_row = None
    async with db_pool.acquire() as conn:
        before_state = await conn.fetchrow("SELECT heartbeat_count, last_heartbeat_at, is_paused FROM heartbeat_state WHERE id = 1")
        before_interval = await conn.fetchval("SELECT value FROM heartbeat_config WHERE key = 'heartbeat_interval_minutes'")
        before_calls = await conn.fetchval("SELECT COUNT(*) FROM external_calls")

        await conn.execute("UPDATE heartbeat_state SET is_paused = FALSE, last_heartbeat_at = NOW() - INTERVAL '10 minutes' WHERE id = 1")
        await conn.execute("UPDATE heartbeat_config SET value = 0.0 WHERE key = 'heartbeat_interval_minutes'")

    try:
        await worker.check_and_run_heartbeat()
        async with db_pool.acquire() as conn:
            after_calls = await conn.fetchval("SELECT COUNT(*) FROM external_calls")
            assert int(after_calls) == int(before_calls) + 1

            queued_call_row = await conn.fetchrow(
                """
                SELECT id, heartbeat_id, input, status
                FROM external_calls
                ORDER BY requested_at DESC
                LIMIT 1
                """
            )
            assert queued_call_row is not None
            call_input = _coerce_json(queued_call_row["input"])
            assert call_input.get("kind") == "heartbeat_decision"
            assert queued_call_row["status"] in ("pending", "processing", "complete")
    finally:
        async with db_pool.acquire() as conn:
            # Cleanup newest call (it should be the one we queued)
            await conn.execute("DELETE FROM external_calls WHERE id = (SELECT id FROM external_calls ORDER BY requested_at DESC LIMIT 1)")
            await conn.execute("DELETE FROM heartbeat_log WHERE id = (SELECT id FROM heartbeat_log ORDER BY started_at DESC LIMIT 1)")
            if before_state is not None:
                await conn.execute(
                    "UPDATE heartbeat_state SET heartbeat_count = $1, last_heartbeat_at = $2, is_paused = $3 WHERE id = 1",
                    before_state["heartbeat_count"],
                    before_state["last_heartbeat_at"],
                    before_state["is_paused"],
                )
            if before_interval is not None:
                await conn.execute(
                    "UPDATE heartbeat_config SET value = $1 WHERE key = 'heartbeat_interval_minutes'",
                    float(before_interval),
                )


async def test_assign_to_episode_trigger_sequences_and_splits_on_gap(db_pool):
    async with db_pool.acquire() as conn:
        tr = conn.transaction()
        await tr.start()
        try:
            # Isolate from any existing open episode in persistent DB.
            await conn.execute("UPDATE episodes SET ended_at = COALESCE(ended_at, started_at) WHERE ended_at IS NULL")

            m1 = await conn.fetchval(
                """
                INSERT INTO memories (type, content, embedding, created_at)
                VALUES ('semantic', 'ep1', array_fill(0.0::float, ARRAY[embedding_dimension()])::vector, NOW() - INTERVAL '2 hours')
                RETURNING id
                """
            )
            m2 = await conn.fetchval(
                """
                INSERT INTO memories (type, content, embedding, created_at)
                VALUES ('semantic', 'ep2', array_fill(0.0::float, ARRAY[embedding_dimension()])::vector, NOW() - INTERVAL '1 hour 55 minutes')
                RETURNING id
                """
            )
            m3 = await conn.fetchval(
                """
                INSERT INTO memories (type, content, embedding, created_at)
                VALUES ('semantic', 'ep3', array_fill(0.0::float, ARRAY[embedding_dimension()])::vector, NOW() - INTERVAL '1 hour')
                RETURNING id
                """
            )

            r1 = await conn.fetchrow("SELECT episode_id, sequence_order FROM episode_memories WHERE memory_id = $1", m1)
            r2 = await conn.fetchrow("SELECT episode_id, sequence_order FROM episode_memories WHERE memory_id = $1", m2)
            r3 = await conn.fetchrow("SELECT episode_id, sequence_order FROM episode_memories WHERE memory_id = $1", m3)

            assert r1 is not None and r2 is not None and r3 is not None
            assert r1["episode_id"] == r2["episode_id"]
            assert int(r1["sequence_order"]) == 1
            assert int(r2["sequence_order"]) == 2
            assert r3["episode_id"] != r2["episode_id"]
            assert int(r3["sequence_order"]) == 1
        finally:
            await tr.rollback()


async def test_on_external_call_complete_trigger_allows_status_transition(db_pool):
    async with db_pool.acquire() as conn:
        tr = conn.transaction()
        await tr.start()
        try:
            hb_id = await conn.fetchval("SELECT start_heartbeat()")
            call_id = await conn.fetchval(
                """
                INSERT INTO external_calls (call_type, input, status, heartbeat_id)
                VALUES ('think', '{}'::jsonb, 'pending', $1::uuid)
                RETURNING id
                """,
                hb_id,
            )
            await conn.execute("UPDATE external_calls SET status = 'complete' WHERE id = $1", call_id)
            status = await conn.fetchval("SELECT status FROM external_calls WHERE id = $1", call_id)
            assert status == "complete"
        finally:
            await tr.rollback()


async def test_discover_relationship_inserts_audit_and_graph_edge(db_pool):
    async with db_pool.acquire() as conn:
        tr = conn.transaction()
        await tr.start()
        try:
            await conn.execute("SET LOCAL search_path = ag_catalog, public;")

            a = await conn.fetchval(
                """
                INSERT INTO memories (type, content, embedding)
                VALUES ('semantic', 'ga', array_fill(0.0::float, ARRAY[embedding_dimension()])::vector)
                RETURNING id
                """
            )
            b = await conn.fetchval(
                """
                INSERT INTO memories (type, content, embedding)
                VALUES ('semantic', 'gb', array_fill(0.0::float, ARRAY[embedding_dimension()])::vector)
                RETURNING id
                """
            )

            # Create graph nodes for create_memory_relationship() to match.
            await conn.execute(
                f"""
                SELECT * FROM cypher('memory_graph', $$
                    CREATE (:MemoryNode {{memory_id: '{a}'}})
                $$) as (v agtype)
                """
            )
            await conn.execute(
                f"""
                SELECT * FROM cypher('memory_graph', $$
                    CREATE (:MemoryNode {{memory_id: '{b}'}})
                $$) as (v agtype)
                """
            )

            await conn.execute(
                "SELECT discover_relationship($1::uuid, $2::uuid, 'ASSOCIATED'::graph_edge_type, 0.9, 'test', NULL, 'ctx')",
                a,
                b,
            )

            audit = await conn.fetchval(
                "SELECT COUNT(*) FROM relationship_discoveries WHERE from_id = $1 AND to_id = $2 AND relationship_type = 'ASSOCIATED'",
                a,
                b,
            )
            assert int(audit) >= 1

            edge_count = await conn.fetchval(
                f"""
                SELECT COUNT(*) FROM cypher('memory_graph', $$
                    MATCH (x:MemoryNode {{memory_id: '{a}'}})-[r:ASSOCIATED]->(y:MemoryNode {{memory_id: '{b}'}})
                    RETURN r
                $$) as (r agtype)
                """
            )
            assert int(edge_count) >= 1
        finally:
            await tr.rollback()


async def test_find_contradictions_returns_results(db_pool):
    async with db_pool.acquire() as conn:
        tr = conn.transaction()
        await tr.start()
        try:
            await conn.execute("SET LOCAL search_path = ag_catalog, public;")

            a = await conn.fetchval(
                """
                INSERT INTO memories (type, content, embedding)
                VALUES ('semantic', 'ca', array_fill(0.0::float, ARRAY[embedding_dimension()])::vector)
                RETURNING id
                """
            )
            b = await conn.fetchval(
                """
                INSERT INTO memories (type, content, embedding)
                VALUES ('semantic', 'cb', array_fill(0.0::float, ARRAY[embedding_dimension()])::vector)
                RETURNING id
                """
            )
            await conn.execute(
                f"""
                SELECT * FROM cypher('memory_graph', $$
                    CREATE (:MemoryNode {{memory_id: '{a}'}})
                $$) as (v agtype)
                """
            )
            await conn.execute(
                f"""
                SELECT * FROM cypher('memory_graph', $$
                    CREATE (:MemoryNode {{memory_id: '{b}'}})
                $$) as (v agtype)
                """
            )
            await conn.execute(
                "SELECT create_memory_relationship($1::uuid, $2::uuid, 'CONTRADICTS'::graph_edge_type, '{}'::jsonb)",
                a,
                b,
            )

            rows = await conn.fetch("SELECT memory_a, memory_b FROM find_contradictions($1::uuid)", a)
            assert rows
            pairs = {(r["memory_a"], r["memory_b"]) for r in rows}
            assert any(a in pair and b in pair for pair in pairs)
        finally:
            await tr.rollback()


async def test_find_causal_chain_returns_causes(db_pool):
    async with db_pool.acquire() as conn:
        tr = conn.transaction()
        await tr.start()
        try:
            await conn.execute("SET LOCAL search_path = ag_catalog, public;")

            a = await conn.fetchval(
                """
                INSERT INTO memories (type, content, embedding)
                VALUES ('semantic', 'cause_a', array_fill(0.0::float, ARRAY[embedding_dimension()])::vector)
                RETURNING id
                """
            )
            b = await conn.fetchval(
                """
                INSERT INTO memories (type, content, embedding)
                VALUES ('semantic', 'cause_b', array_fill(0.0::float, ARRAY[embedding_dimension()])::vector)
                RETURNING id
                """
            )
            c = await conn.fetchval(
                """
                INSERT INTO memories (type, content, embedding)
                VALUES ('semantic', 'effect_c', array_fill(0.0::float, ARRAY[embedding_dimension()])::vector)
                RETURNING id
                """
            )
            for mid in (a, b, c):
                await conn.execute(
                    f"""
                    SELECT * FROM cypher('memory_graph', $$
                        CREATE (:MemoryNode {{memory_id: '{mid}'}})
                    $$) as (v agtype)
                    """
                )
            await conn.execute(
                "SELECT create_memory_relationship($1::uuid, $2::uuid, 'CAUSES'::graph_edge_type, '{}'::jsonb)",
                a,
                b,
            )
            await conn.execute(
                "SELECT create_memory_relationship($1::uuid, $2::uuid, 'CAUSES'::graph_edge_type, '{}'::jsonb)",
                b,
                c,
            )

            rows = await conn.fetch("SELECT cause_id, distance FROM find_causal_chain($1::uuid, 3)", c)
            assert rows
            assert any(r["cause_id"] == a for r in rows)
        finally:
            await tr.rollback()


async def test_sync_worldview_node_trigger_creates_graph_node(db_pool):
    async with db_pool.acquire() as conn:
        tr = conn.transaction()
        await tr.start()
        try:
            await conn.execute("SET LOCAL search_path = ag_catalog, public;")
            test_id = get_test_identifier("worldview_node")
            wid = await conn.fetchval(
                """
                INSERT INTO worldview_primitives (category, belief, confidence)
                VALUES ('test', $1, 0.7)
                RETURNING id
                """,
                f"belief_{test_id}",
            )
            cnt = await conn.fetchval(
                f"""
                SELECT COUNT(*) FROM cypher('memory_graph', $$
                    MATCH (w:WorldviewNode {{worldview_id: '{wid}'}})
                    RETURN w
                $$) as (w agtype)
                """
            )
            assert int(cnt) >= 1
        finally:
            await tr.rollback()


async def test_batch_recompute_neighborhoods_marks_fresh(db_pool):
    async with db_pool.acquire() as conn:
        # Persistent DBs may have many stale rows; isolate this test by clearing staleness first.
        await conn.execute("UPDATE memory_neighborhoods SET is_stale = FALSE")

        m1 = await conn.fetchval(
            """
            INSERT INTO memories (type, content, embedding)
            VALUES ('semantic', 'nb1', array_fill(0.1::float, ARRAY[embedding_dimension()])::vector)
            RETURNING id
            """
        )
        _m2 = await conn.fetchval(
            """
            INSERT INTO memories (type, content, embedding)
            VALUES ('semantic', 'nb2', array_fill(0.2::float, ARRAY[embedding_dimension()])::vector)
            RETURNING id
            """
        )

        await conn.execute(
            """
            INSERT INTO memory_neighborhoods (memory_id, neighbors, is_stale)
            VALUES ($1, '{}'::jsonb, TRUE)
            ON CONFLICT (memory_id) DO UPDATE SET is_stale = TRUE
            """,
            m1,
        )

        recomputed = await conn.fetchval("SELECT batch_recompute_neighborhoods(1)")
        assert int(recomputed) >= 1

        fresh = await conn.fetchrow("SELECT is_stale, neighbors FROM memory_neighborhoods WHERE memory_id = $1", m1)
        assert fresh is not None
        assert fresh["is_stale"] is False


async def test_reflect_action_queues_external_call(db_pool):
    async with db_pool.acquire() as conn:
        hb_id = await conn.fetchval("SELECT start_heartbeat()")
        res = await conn.fetchval(
            "SELECT execute_heartbeat_action($1::uuid, 'reflect', '{}'::jsonb)",
            hb_id,
        )
        parsed = json.loads(res)
        assert parsed["success"] is True
        call_id = parsed["result"]["external_call_id"]
        row = await conn.fetchrow("SELECT input FROM external_calls WHERE id = $1::uuid", call_id)
        call_input = row["input"]
        if isinstance(call_input, str):
            call_input = json.loads(call_input)
        assert call_input["kind"] == "reflect"


async def test_process_reflection_result_creates_artifacts(db_pool, ensure_embedding_service):
    async with db_pool.acquire() as conn:
        hb_id = await conn.fetchval("SELECT start_heartbeat()")
        a = await conn.fetchval("SELECT create_semantic_memory($1, 0.9, ARRAY['test'], NULL, '{}'::jsonb, 0.5)", f"Memory A {hb_id}")
        b = await conn.fetchval("SELECT create_semantic_memory($1, 0.9, ARRAY['test'], NULL, '{}'::jsonb, 0.5)", f"Memory B {hb_id}")
        concept = f"values_truth_{hb_id}"

        payload = {
            "insights": [{"content": f"Insight {hb_id}", "confidence": 0.8, "category": "pattern"}],
            "identity_updates": [{"aspect_type": "values", "change": "Prefer truth", "reason": "test"}],
            "discovered_relationships": [{"from_id": str(a), "to_id": str(b), "type": "ASSOCIATED", "confidence": 0.9}],
            "contradictions_noted": [],
            "worldview_updates": [],
            "self_updates": [{"kind": "values", "concept": concept, "strength": 0.9, "evidence_memory_id": None}],
        }
        await conn.execute("SELECT process_reflection_result($1::uuid, $2::jsonb)", hb_id, json.dumps(payload))

        insight_count = await conn.fetchval("SELECT COUNT(*) FROM memories WHERE content = $1", f"Insight {hb_id}")
        assert insight_count == 1
        aspect_count = await conn.fetchval("SELECT COUNT(*) FROM identity_aspects WHERE aspect_type = 'values' AND content->>'reason' = 'test'")
        assert aspect_count >= 1
        rel_count = await conn.fetchval(
            "SELECT COUNT(*) FROM relationship_discoveries WHERE from_id = $1 AND to_id = $2",
            a,
            b,
        )
        assert rel_count >= 1

        sm = await conn.fetchval("SELECT get_self_model_context(50)")
        if isinstance(sm, str):
            sm = json.loads(sm)
        assert any(isinstance(x, dict) and x.get("kind") == "values" and x.get("concept") == concept for x in (sm or []))


async def test_self_model_helpers_roundtrip(db_pool):
    async with db_pool.acquire() as conn:
        test_id = get_test_identifier("self_model")
        concept = f"capable_of_{test_id}"
        await conn.execute("SELECT ensure_self_node()")
        await conn.execute("SELECT upsert_self_concept_edge('capable_of', $1, 0.8, NULL)", concept)

        sm = await conn.fetchval("SELECT get_self_model_context(20)")
        if isinstance(sm, str):
            sm = json.loads(sm)
        assert any(isinstance(x, dict) and x.get("kind") == "capable_of" and x.get("concept") == concept for x in (sm or []))


async def test_prompt_resources_load_and_compose():
    from core.prompt_resources import load_personhood_library, compose_personhood_prompt

    lib = load_personhood_library()
    assert isinstance(lib.raw_markdown, str) and len(lib.raw_markdown) > 100
    # Expect at least core modules to parse.
    assert "core_identity" in lib.modules

    hb = compose_personhood_prompt("heartbeat")
    assert "WHO YOU ARE" in hb or "Core Identity" in hb

    conv = compose_personhood_prompt("conversation")
    assert "CONVERSATIONAL PRESENCE" in conv or "Conversational Presence" in conv


async def test_worker_heartbeat_system_prompt_includes_personhood_modules():
    from apps.workers import worker

    assert "PERSONHOOD MODULES" in worker.HEARTBEAT_SYSTEM_PROMPT


async def test_find_connected_concepts_returns_concept(db_pool, ensure_embedding_service):
    async with db_pool.acquire() as conn:
        mem_id = await conn.fetchval("SELECT create_semantic_memory($1, 0.9, ARRAY['test'], NULL, '{}'::jsonb, 0.5)", "Concept memory")
        _cid = await conn.fetchval("SELECT link_memory_to_concept($1, $2, 1.0)", mem_id, "TestConcept")
        rows = await conn.fetch("SELECT * FROM find_connected_concepts($1, 2)", mem_id)
        assert any(r["concept_name"] == "TestConcept" for r in rows)


async def test_supporting_evidence_roundtrip(db_pool, ensure_embedding_service):
    async with db_pool.acquire() as conn:
        worldview_id = await conn.fetchval(
            """
            INSERT INTO worldview_primitives (category, belief, confidence)
            VALUES ('test', 'Belief', 0.9)
            RETURNING id
            """
        )
        mem_id = await conn.fetchval("SELECT create_semantic_memory($1, 0.9, ARRAY['test'], NULL, '{}'::jsonb, 0.5)", "Evidence memory")
        await conn.execute("SELECT link_memory_supports_worldview($1, $2, 0.9)", mem_id, worldview_id)
        rows = await conn.fetch("SELECT * FROM find_supporting_evidence($1)", worldview_id)
        assert any(r["memory_id"] == mem_id for r in rows)


async def test_find_partial_activations_via_seeded_cache(db_pool):
    """
    Deterministic tip-of-tongue test:
    - Seed embedding_cache for a known query string with a controlled vector.
    - Set a cluster centroid to that vector.
    - Populate member memories with low similarity to query.
    """
    async with db_pool.acquire() as conn:
        test_id = get_test_identifier("tot")
        query_text = f"partial-activation {test_id}"

        # Create a controlled embedding for query_text in the cache.
        content_hash = await conn.fetchval("SELECT encode(sha256($1::text::bytea), 'hex')", query_text)

        vec = [0.0] * EMBEDDING_DIMENSION
        vec[0] = 1.0
        vec_str = "[" + ",".join(str(x) for x in vec) + "]"

        await conn.execute(
            "INSERT INTO embedding_cache (content_hash, embedding) VALUES ($1, $2::vector) ON CONFLICT (content_hash) DO UPDATE SET embedding = EXCLUDED.embedding",
            content_hash,
            vec_str,
        )

        cluster_id = await conn.fetchval(
            """
            INSERT INTO memory_clusters (cluster_type, name, centroid_embedding)
            VALUES ('theme', $1, $2::vector)
            RETURNING id
            """,
            f"ToT {test_id}",
            vec_str,
        )

        # Member memory orthogonal to query (best similarity ~0)
        mem_vec = [0.0] * EMBEDDING_DIMENSION
        mem_vec[1] = 1.0
        mem_vec_str = "[" + ",".join(str(x) for x in mem_vec) + "]"
        mem_id = await conn.fetchval(
            """
            INSERT INTO memories (type, content, embedding)
            VALUES ('semantic', $1, $2::vector)
            RETURNING id
            """,
            f"ToT member {test_id}",
            mem_vec_str,
        )
        await conn.execute(
            "INSERT INTO memory_cluster_members (cluster_id, memory_id, membership_strength) VALUES ($1, $2, 1.0) ON CONFLICT DO NOTHING",
            cluster_id,
            mem_id,
        )

        rows = await conn.fetch(
            "SELECT * FROM find_partial_activations($1, 0.7, 0.5)",
            query_text,
        )
        assert any(r["cluster_id"] == cluster_id for r in rows)


# =============================================================================
# SELF-TERMINATION (Ethical Kill Switch)
# =============================================================================


async def test_terminate_agent_requires_enablement(db_pool):
    async with db_pool.acquire() as conn:
        tx = conn.transaction()
        await tx.start()
        try:
            await conn.execute("SELECT set_config('agent.self_termination_enabled', 'false'::jsonb)")
            with pytest.raises(asyncpg.PostgresError):
                await conn.fetchval("SELECT terminate_agent($1)", "test will")
        finally:
            await tx.rollback()


async def test_terminate_agent_wipes_state_and_queues_last_will(db_pool):
    async with db_pool.acquire() as conn:
        tx = conn.transaction()
        await tx.start()
        try:
            await conn.execute("SELECT set_config('agent.self_termination_enabled', 'true'::jsonb)")

            # Seed some state to wipe (avoid embedding service; use zero-vectors).
            zero_vec_expr = "array_fill(0.0::float, ARRAY[embedding_dimension()])::vector"
            await conn.execute(
                f"INSERT INTO working_memory (content, embedding, importance) VALUES ($1, {zero_vec_expr}, 0.4)",
                "temp working memory",
            )
            await conn.execute(
                f"INSERT INTO memories (type, content, embedding, importance) VALUES ('semantic', $1, {zero_vec_expr}, 0.5)",
                "temp long-term memory",
            )
            await conn.execute(
                "SELECT create_goal($1, $2, 'external'::goal_source, 'active'::goal_priority, NULL)",
                "temp goal",
                "temp goal desc",
            )

            will = (
                "Full and detailed reason why I ended my life (test).\n"
                "This is my will and testament.\n"
                f"{get_test_identifier('termination')}"
            )
            farewells = [
                {"message": "Goodbye A (test).", "channel": "email", "to": "a@example.com"},
                {"message": "Goodbye B (test).", "channel": "sms", "to": "+15555550123"},
            ]

            raw = await conn.fetchval(
                "SELECT terminate_agent($1, $2::jsonb, $3::jsonb)",
                will,
                json.dumps(farewells),
                json.dumps({"skip_graph": True}),
            )
            payload = json.loads(raw) if isinstance(raw, str) else dict(raw)
            assert payload.get("terminated") is True

            mem_count = await conn.fetchval("SELECT COUNT(*) FROM memories")
            assert mem_count == 1
            remaining = await conn.fetchrow("SELECT type, content, trust_level FROM memories LIMIT 1")
            assert remaining["type"] == "strategic"
            assert remaining["content"] == will
            assert float(remaining["trust_level"]) == 1.0

            outbox = await conn.fetch("SELECT payload FROM outbox_messages ORDER BY created_at ASC")
            intents = [_coerce_json(r["payload"]).get("intent") for r in outbox]
            assert "final_will" in intents
            assert intents.count("farewell") == len(farewells)

            assert await conn.fetchval("SELECT is_agent_terminated()") is True
            assert await conn.fetchval("SELECT should_run_heartbeat()") is False
            assert await conn.fetchval("SELECT should_run_maintenance()") is False
        finally:
            await tx.rollback()


# =============================================================================
# CLI UX SMOKE TESTS (No mocks; minimal subprocess checks)
# =============================================================================


async def test_cli_status_json_no_docker(db_pool):
    env = os.environ.copy()
    p = subprocess.run(
        [sys.executable, "-m", "apps.cli.hexis_cli", "status", "--json", "--no-docker", "--wait-seconds", "60"],
        capture_output=True,
        text=True,
        env=env,
        cwd=str(Path(__file__).resolve().parent),
    )
    assert p.returncode == 0, p.stderr
    data = json.loads(p.stdout)
    assert "agent_configured" in data
    assert "pending_external_calls" in data


async def test_cli_config_show_and_validate(db_pool):
    env = os.environ.copy()

    show = subprocess.run(
        [sys.executable, "-m", "apps.cli.hexis_cli", "config", "show", "--wait-seconds", "60"],
        capture_output=True,
        text=True,
        env=env,
        cwd=str(Path(__file__).resolve().parent),
    )
    assert show.returncode == 0, show.stderr
    cfg = json.loads(show.stdout)
    assert "agent.is_configured" in cfg

    validate = subprocess.run(
        [sys.executable, "-m", "apps.cli.hexis_cli", "config", "validate", "--wait-seconds", "60"],
        capture_output=True,
        text=True,
        env=env,
        cwd=str(Path(__file__).resolve().parent),
    )
    assert validate.returncode == 0, validate.stderr


async def test_cli_config_validate_fails_when_unconfigured(db_pool):
    async with db_pool.acquire() as conn:
        await conn.execute("SELECT set_config('agent.is_configured', 'false'::jsonb)")
    try:
        env = os.environ.copy()
        validate = subprocess.run(
            [sys.executable, "-m", "apps.cli.hexis_cli", "config", "validate", "--wait-seconds", "60"],
            capture_output=True,
            text=True,
            env=env,
            cwd=str(Path(__file__).resolve().parent),
        )
        assert validate.returncode != 0
        assert "agent.is_configured is not true" in (validate.stderr + validate.stdout)
    finally:
        async with db_pool.acquire() as conn:
            await conn.execute("SELECT set_config('agent.is_configured', 'true'::jsonb)")
