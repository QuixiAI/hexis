-- ============================================================================
-- HEXIS MEMORY SYSTEM - FINAL SCHEMA
-- ============================================================================
-- Architecture:
--   - Relational: Core storage, clusters, acceleration, identity
--   - Graph (AGE): Reasoning layer (memories + concepts only)
--   - Vector (pgvector): Semantic similarity search
-- ============================================================================

-- ============================================================================
-- EXTENSIONS
-- ============================================================================

CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS age;
CREATE EXTENSION IF NOT EXISTS btree_gist;
CREATE EXTENSION IF NOT EXISTS pg_trgm;
CREATE EXTENSION IF NOT EXISTS http;
-- Required for gen_random_uuid() + sha256()
CREATE EXTENSION IF NOT EXISTS pgcrypto;

LOAD 'age';
SET search_path = ag_catalog, "$user", public;

-- ============================================================================
-- GRAPH INITIALIZATION
-- ============================================================================

SELECT create_graph('memory_graph');
SELECT create_vlabel('memory_graph', 'MemoryNode');
SELECT create_vlabel('memory_graph', 'ConceptNode');
SELECT create_vlabel('memory_graph', 'SelfNode');
SELECT create_vlabel('memory_graph', 'LifeChapterNode');
SELECT create_vlabel('memory_graph', 'TurningPointNode');
SELECT create_vlabel('memory_graph', 'NarrativeThreadNode');
SELECT create_vlabel('memory_graph', 'RelationshipNode');
SELECT create_vlabel('memory_graph', 'ValueConflictNode');

SET search_path = public, ag_catalog, "$user";

-- ============================================================================
-- ENUMS
-- ============================================================================

CREATE TYPE memory_type AS ENUM ('episodic', 'semantic', 'procedural', 'strategic');
CREATE TYPE memory_status AS ENUM ('active', 'archived', 'invalidated');
CREATE TYPE cluster_type AS ENUM ('theme', 'emotion', 'temporal', 'person', 'pattern', 'mixed');
CREATE TYPE graph_edge_type AS ENUM (
    'TEMPORAL_NEXT',
    'CAUSES',
    'DERIVED_FROM',
    'CONTRADICTS',
    'SUPPORTS',
    'INSTANCE_OF',
    'PARENT_OF',
    'ASSOCIATED'
);

-- ============================================================================
-- CORE STORAGE
-- ============================================================================

-- Base memory table
CREATE TABLE memories (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    type memory_type NOT NULL,
    status memory_status DEFAULT 'active',
    content TEXT NOT NULL,
    embedding vector(768) NOT NULL,
    importance FLOAT DEFAULT 0.5,
    -- Provenance + epistemic trust. `source_attribution` is a primary/summary source
    -- (semantic memories may also carry multiple sources in `semantic_memories.source_references`).
    source_attribution JSONB NOT NULL DEFAULT '{}'::jsonb,
    trust_level FLOAT NOT NULL DEFAULT 0.5 CHECK (trust_level >= 0 AND trust_level <= 1),
    trust_updated_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    access_count INTEGER DEFAULT 0,
    last_accessed TIMESTAMPTZ,
    decay_rate FLOAT DEFAULT 0.01
);

-- Episodic memories (events, experiences)
CREATE TABLE episodic_memories (
    memory_id UUID PRIMARY KEY REFERENCES memories(id) ON DELETE CASCADE,
    action_taken JSONB,
    context JSONB,
    result JSONB,
    emotional_valence FLOAT,
    verification_status BOOLEAN,
    event_time TIMESTAMPTZ,
    CONSTRAINT valid_emotion CHECK (emotional_valence >= -1 AND emotional_valence <= 1)
);

-- Semantic memories (facts, knowledge)
CREATE TABLE semantic_memories (
    memory_id UUID PRIMARY KEY REFERENCES memories(id) ON DELETE CASCADE,
    confidence FLOAT NOT NULL,
    last_validated TIMESTAMPTZ,
    source_references JSONB,
    contradictions JSONB,
    category TEXT[],
    related_concepts TEXT[],
    CONSTRAINT valid_confidence CHECK (confidence >= 0 AND confidence <= 1)
);

-- Procedural memories (how-to knowledge)
CREATE TABLE procedural_memories (
    memory_id UUID PRIMARY KEY REFERENCES memories(id) ON DELETE CASCADE,
    steps JSONB NOT NULL,
    prerequisites JSONB,
    success_count INTEGER DEFAULT 0,
    total_attempts INTEGER DEFAULT 0,
    success_rate FLOAT GENERATED ALWAYS AS (
        CASE WHEN total_attempts > 0 
        THEN success_count::FLOAT / total_attempts::FLOAT 
        ELSE 0 END
    ) STORED,
    average_duration INTERVAL,
    failure_points JSONB
);

-- Strategic memories (patterns, meta-knowledge)
CREATE TABLE strategic_memories (
    memory_id UUID PRIMARY KEY REFERENCES memories(id) ON DELETE CASCADE,
    pattern_description TEXT NOT NULL,
    supporting_evidence JSONB,
    confidence_score FLOAT,
    success_metrics JSONB,
    adaptation_history JSONB,
    context_applicability JSONB,
    CONSTRAINT valid_confidence CHECK (confidence_score >= 0 AND confidence_score <= 1)
);

-- Working memory (transient, short-term)
CREATE TABLE working_memory (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    content TEXT NOT NULL,
    embedding vector(768) NOT NULL,
    importance FLOAT DEFAULT 0.3,
    source_attribution JSONB NOT NULL DEFAULT '{}'::jsonb,
    trust_level FLOAT NOT NULL DEFAULT 0.5 CHECK (trust_level >= 0 AND trust_level <= 1),
    access_count INTEGER DEFAULT 0,
    last_accessed TIMESTAMPTZ,
    promote_to_long_term BOOLEAN NOT NULL DEFAULT FALSE,
    expiry TIMESTAMPTZ
);

-- Ingestion receipts (idempotency for core/ingest.py and other batch importers)
CREATE TABLE ingestion_receipts (
    source_file TEXT NOT NULL,
    chunk_index INT NOT NULL,
    content_hash TEXT NOT NULL,
    memory_id UUID NOT NULL REFERENCES memories(id) ON DELETE CASCADE,
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (source_file, content_hash)
);

CREATE INDEX idx_ingestion_receipts_hash ON ingestion_receipts (content_hash);

CREATE OR REPLACE FUNCTION get_ingestion_receipts(
    p_source_file TEXT,
    p_content_hashes TEXT[]
)
RETURNS TABLE (
    content_hash TEXT,
    memory_id UUID
) AS $$
BEGIN
    IF p_content_hashes IS NULL OR array_length(p_content_hashes, 1) IS NULL THEN
        RETURN;
    END IF;

    RETURN QUERY
    SELECT ir.content_hash, ir.memory_id
    FROM ingestion_receipts ir
    WHERE ir.source_file = p_source_file
      AND ir.content_hash = ANY(p_content_hashes);
END;
$$ LANGUAGE plpgsql STABLE;

CREATE OR REPLACE FUNCTION record_ingestion_receipts(p_items JSONB)
RETURNS INT AS $$
DECLARE
    inserted_count INT := 0;
BEGIN
    IF p_items IS NULL OR jsonb_typeof(p_items) <> 'array' THEN
        RETURN 0;
    END IF;

    WITH to_insert AS (
        SELECT
            NULLIF(e->>'source_file', '') AS source_file,
            COALESCE(NULLIF(e->>'chunk_index', '')::int, 0) AS chunk_index,
            NULLIF(e->>'content_hash', '') AS content_hash,
            NULLIF(e->>'memory_id', '')::uuid AS memory_id
        FROM jsonb_array_elements(p_items) e
    ),
    inserted AS (
        INSERT INTO ingestion_receipts (source_file, chunk_index, content_hash, memory_id)
        SELECT source_file, chunk_index, content_hash, memory_id
        FROM to_insert
        WHERE source_file IS NOT NULL
          AND content_hash IS NOT NULL
          AND memory_id IS NOT NULL
        ON CONFLICT DO NOTHING
        RETURNING 1
    )
    SELECT COUNT(*) INTO inserted_count FROM inserted;

    RETURN COALESCE(inserted_count, 0);
END;
$$ LANGUAGE plpgsql;

-- ============================================================================
-- CLUSTERING (Relational Only)
-- ============================================================================

CREATE TABLE memory_clusters (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    cluster_type cluster_type NOT NULL,
    name TEXT NOT NULL,
    description TEXT,
    centroid_embedding vector(768),
    emotional_signature JSONB,
    keywords TEXT[],
    importance_score FLOAT DEFAULT 0.0,
    coherence_score FLOAT,
    last_activated TIMESTAMPTZ,
    activation_count INTEGER DEFAULT 0
);

CREATE TABLE memory_cluster_members (
    cluster_id UUID REFERENCES memory_clusters(id) ON DELETE CASCADE,
    memory_id UUID REFERENCES memories(id) ON DELETE CASCADE,
    membership_strength FLOAT DEFAULT 1.0,
    added_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    contribution_to_centroid FLOAT,
    PRIMARY KEY (cluster_id, memory_id)
);

CREATE TABLE cluster_relationships (
    from_cluster_id UUID REFERENCES memory_clusters(id) ON DELETE CASCADE,
    to_cluster_id UUID REFERENCES memory_clusters(id) ON DELETE CASCADE,
    relationship_type TEXT NOT NULL,
    strength FLOAT DEFAULT 0.5,
    discovered_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    evidence_memories UUID[],
    PRIMARY KEY (from_cluster_id, to_cluster_id, relationship_type)
);

-- ============================================================================
-- ACCELERATION LAYER
-- ============================================================================

-- Episodes: Temporal segmentation for narrative coherence
CREATE TABLE episodes (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    started_at TIMESTAMPTZ NOT NULL,
    ended_at TIMESTAMPTZ,
    episode_type TEXT,  -- 'conversation', 'autonomous', 'reflection'
    summary TEXT,
    summary_embedding vector(768),
    time_range TSTZRANGE GENERATED ALWAYS AS (
        tstzrange(started_at, COALESCE(ended_at, 'infinity'::timestamptz))
    ) STORED
);

CREATE TABLE episode_memories (
    episode_id UUID REFERENCES episodes(id) ON DELETE CASCADE,
    memory_id UUID REFERENCES memories(id) ON DELETE CASCADE,
    sequence_order INT,
    PRIMARY KEY (episode_id, memory_id)
);

-- Precomputed neighborhoods (replaces live spreading activation)
CREATE TABLE memory_neighborhoods (
    memory_id UUID PRIMARY KEY REFERENCES memories(id) ON DELETE CASCADE,
    neighbors JSONB NOT NULL DEFAULT '{}',  -- {uuid: weight}
    computed_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    is_stale BOOLEAN DEFAULT TRUE
);

-- Transient activation cache (fast writes, lost on crash)
CREATE UNLOGGED TABLE activation_cache (
    session_id UUID,
    memory_id UUID,
    activation_level FLOAT,
    computed_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (session_id, memory_id)
);

-- ============================================================================
-- CONCEPT LAYER
-- ============================================================================

CREATE TABLE concepts (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name TEXT UNIQUE NOT NULL,
    ancestors UUID[] DEFAULT '{}',
    path_text TEXT,  -- 'Entity/Organism/Animal/Dog'
    depth INT DEFAULT 0,
    description TEXT,
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE memory_concepts (
    memory_id UUID REFERENCES memories(id) ON DELETE CASCADE,
    concept_id UUID REFERENCES concepts(id) ON DELETE CASCADE,
    strength FLOAT DEFAULT 1.0,
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (memory_id, concept_id)
);

-- ============================================================================
-- IDENTITY & WORLDVIEW
-- ============================================================================

-- Worldview primitives (beliefs that filter perception)
CREATE TABLE worldview_primitives (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    category TEXT NOT NULL,
    belief TEXT NOT NULL,
    confidence FLOAT,
    emotional_valence FLOAT,
    stability_score FLOAT,
    connected_beliefs UUID[],
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);

-- How worldview affects memory interpretation
CREATE TABLE worldview_memory_influences (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    worldview_id UUID REFERENCES worldview_primitives(id) ON DELETE CASCADE,
    memory_id UUID REFERENCES memories(id) ON DELETE CASCADE,
    influence_type TEXT NOT NULL DEFAULT 'evidence',
    strength FLOAT NOT NULL DEFAULT 0.0,
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);

CREATE UNIQUE INDEX idx_worldview_influences_unique ON worldview_memory_influences (worldview_id, memory_id, influence_type);

-- Identity aspects (normalized from single blob)
CREATE TABLE identity_aspects (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    aspect_type TEXT NOT NULL,  -- 'self_concept', 'purpose', 'boundary', 'agency', 'values'
    content JSONB NOT NULL,
    stability FLOAT DEFAULT 0.5,
    core_memory_clusters UUID[],
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);

-- Bridge between memories and identity
CREATE TABLE identity_memory_resonance (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    memory_id UUID REFERENCES memories(id) ON DELETE CASCADE,
    identity_aspect_id UUID REFERENCES identity_aspects(id) ON DELETE CASCADE,
    resonance_strength FLOAT,
    integration_status TEXT,
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);

-- ============================================================================
-- AUDIT & CACHE
-- ============================================================================

CREATE TABLE memory_changes (
    change_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    memory_id UUID REFERENCES memories(id) ON DELETE CASCADE,
    changed_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    change_type TEXT NOT NULL,
    old_value JSONB,
    new_value JSONB
);

CREATE TABLE embedding_cache (
    content_hash TEXT PRIMARY KEY,
    embedding vector(768) NOT NULL,
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);

-- Configuration for embeddings service
CREATE TABLE embedding_config (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

INSERT INTO embedding_config (key, value) 
VALUES ('service_url', 'http://embeddings:80/embed')
ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value;

-- Default embedding dimension (used for validation / health checks)
INSERT INTO embedding_config (key, value)
VALUES ('dimension', COALESCE(NULLIF(current_setting('app.embedding_dimension', true), ''), '768'))
ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value;

-- Return the configured embedding dimension (from embedding_config, or postgres setting fallback).
CREATE OR REPLACE FUNCTION embedding_dimension()
RETURNS INT
LANGUAGE sql
STABLE
AS $$
    SELECT COALESCE(
        NULLIF((SELECT value FROM embedding_config WHERE key = 'dimension'), '')::int,
        NULLIF(current_setting('app.embedding_dimension', true), '')::int,
        768
    );
$$;

-- Keep embedding_config.dimension synchronized with the docker-compose postgres setting (if present).
CREATE OR REPLACE FUNCTION sync_embedding_dimension_config()
RETURNS INT AS $$
DECLARE
    configured TEXT;
BEGIN
    configured := NULLIF(current_setting('app.embedding_dimension', true), '');
    IF configured IS NULL THEN
        RETURN embedding_dimension();
    END IF;

    INSERT INTO embedding_config (key, value)
    VALUES ('dimension', configured)
    ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value;

    RETURN configured::int;
END;
$$ LANGUAGE plpgsql;

-- Align vector column dimensions to the configured embedding dimension (required for HNSW indexes).
DO $$
DECLARE
    dim INT;
BEGIN
    dim := sync_embedding_dimension_config();

    EXECUTE format(
        'ALTER TABLE memories ALTER COLUMN embedding TYPE vector(%s) USING embedding::vector(%s)',
        dim,
        dim
    );
    EXECUTE format(
        'ALTER TABLE working_memory ALTER COLUMN embedding TYPE vector(%s) USING embedding::vector(%s)',
        dim,
        dim
    );
    EXECUTE format(
        'ALTER TABLE embedding_cache ALTER COLUMN embedding TYPE vector(%s) USING embedding::vector(%s)',
        dim,
        dim
    );
    EXECUTE format(
        'ALTER TABLE memory_clusters ALTER COLUMN centroid_embedding TYPE vector(%s) USING centroid_embedding::vector(%s)',
        dim,
        dim
    );
    EXECUTE format(
        'ALTER TABLE episodes ALTER COLUMN summary_embedding TYPE vector(%s) USING summary_embedding::vector(%s)',
        dim,
        dim
    );
END;
$$;

-- ============================================================================
-- INDEXES
-- ============================================================================

-- Memory indexes
CREATE INDEX idx_memories_embedding ON memories USING hnsw (embedding vector_cosine_ops);
CREATE INDEX idx_memories_status ON memories (status);
CREATE INDEX idx_memories_type ON memories (type);
CREATE INDEX idx_memories_content ON memories USING GIN (content gin_trgm_ops);
CREATE INDEX idx_memories_importance ON memories (importance DESC) WHERE status = 'active';
CREATE INDEX idx_memories_created ON memories (created_at DESC);
CREATE INDEX idx_memories_last_accessed ON memories (last_accessed DESC NULLS LAST);

-- Working memory
CREATE INDEX idx_working_memory_expiry ON working_memory (expiry);
CREATE INDEX idx_working_memory_embedding ON working_memory USING hnsw (embedding vector_cosine_ops);

-- Cluster indexes
CREATE INDEX idx_clusters_centroid ON memory_clusters USING hnsw (centroid_embedding vector_cosine_ops);
CREATE INDEX idx_clusters_type_importance ON memory_clusters (cluster_type, importance_score DESC);
CREATE INDEX idx_clusters_last_activated ON memory_clusters (last_activated DESC);
CREATE INDEX idx_cluster_members_memory ON memory_cluster_members (memory_id);
CREATE INDEX idx_cluster_members_strength ON memory_cluster_members (cluster_id, membership_strength DESC);
CREATE INDEX idx_cluster_relationships_from ON cluster_relationships (from_cluster_id);
CREATE INDEX idx_cluster_relationships_to ON cluster_relationships (to_cluster_id);

-- Episode indexes
CREATE INDEX idx_episodes_time_range ON episodes USING GIST (time_range);
CREATE INDEX idx_episodes_summary_embedding ON episodes USING hnsw (summary_embedding vector_cosine_ops);
CREATE INDEX idx_episodes_started ON episodes (started_at DESC);
CREATE INDEX idx_episode_memories_memory ON episode_memories (memory_id);
CREATE INDEX idx_episode_memories_sequence ON episode_memories (episode_id, sequence_order);

-- Neighborhood indexes
CREATE INDEX idx_neighborhoods_stale ON memory_neighborhoods (is_stale) WHERE is_stale = TRUE;
CREATE INDEX idx_neighborhoods_neighbors ON memory_neighborhoods USING GIN (neighbors);

-- Concept indexes
CREATE INDEX idx_concepts_ancestors ON concepts USING GIN (ancestors);
CREATE INDEX idx_concepts_name ON concepts (name);
CREATE INDEX idx_memory_concepts_concept ON memory_concepts (concept_id);

-- Identity/worldview indexes
CREATE INDEX idx_worldview_influences_memory ON worldview_memory_influences (memory_id, strength DESC);
CREATE INDEX idx_identity_resonance_memory ON identity_memory_resonance (memory_id, resonance_strength DESC);
CREATE INDEX idx_identity_aspects_type ON identity_aspects (aspect_type);

-- Cache indexes
CREATE INDEX idx_embedding_cache_created ON embedding_cache (created_at);

-- ============================================================================
-- HELPER FUNCTIONS
-- ============================================================================

-- Calculate age in days (used for decay)
CREATE OR REPLACE FUNCTION age_in_days(ts TIMESTAMPTZ) 
RETURNS FLOAT
LANGUAGE sql
IMMUTABLE
AS $$
    SELECT EXTRACT(EPOCH FROM (NOW() - ts)) / 86400.0;
$$;

-- Calculate relevance score dynamically
CREATE OR REPLACE FUNCTION calculate_relevance(
    p_importance FLOAT,
    p_decay_rate FLOAT,
    p_created_at TIMESTAMPTZ,
    p_last_accessed TIMESTAMPTZ
) RETURNS FLOAT
LANGUAGE sql
IMMUTABLE
AS $$
    SELECT p_importance * EXP(
        -p_decay_rate * LEAST(
            age_in_days(p_created_at),
            COALESCE(age_in_days(p_last_accessed), age_in_days(p_created_at)) * 0.5
        )
    );
$$;

-- Get embedding from service (with caching)
CREATE OR REPLACE FUNCTION get_embedding(text_content TEXT)
RETURNS vector AS $$
	DECLARE
	    service_url TEXT;
	    response http_response;
	    request_body TEXT;
	    embedding_array FLOAT[];
	    embedding_json JSONB;
	    v_content_hash TEXT;
	    cached_embedding vector;
	    expected_dim INT;
	    start_ts TIMESTAMPTZ;
	    retry_seconds INT;
	    retry_interval_seconds FLOAT;
	    last_error TEXT;
	BEGIN
	    PERFORM sync_embedding_dimension_config();
	    expected_dim := embedding_dimension();

	    -- Generate hash for caching
	    v_content_hash := encode(sha256(text_content::bytea), 'hex');

    -- Check cache first
    SELECT ec.embedding INTO cached_embedding
    FROM embedding_cache ec
    WHERE ec.content_hash = v_content_hash;

    IF FOUND THEN
        RETURN cached_embedding;
    END IF;

    -- Get service URL
	    SELECT value INTO service_url FROM embedding_config WHERE key = 'service_url';

	    -- Prepare request body
	    request_body := json_build_object('inputs', text_content)::TEXT;

	    -- Make HTTP request (with retries to tolerate a slow-starting embedding service).
	    retry_seconds := COALESCE(NULLIF((SELECT value FROM embedding_config WHERE key = 'retry_seconds'), '')::int, 30);
	    retry_interval_seconds := COALESCE(NULLIF((SELECT value FROM embedding_config WHERE key = 'retry_interval_seconds'), '')::float, 1.0);
	    start_ts := clock_timestamp();

	    LOOP
	        BEGIN
	            SELECT * INTO response FROM http_post(
	                service_url,
	                request_body,
	                'application/json'
	            );

	            IF response.status = 200 THEN
	                EXIT;
	            END IF;

	            -- Non-retriable statuses (bad request, auth, etc).
	            IF response.status IN (400, 401, 403, 404, 422) THEN
	                RAISE EXCEPTION 'Embedding service error: % - %', response.status, response.content;
	            END IF;

	            last_error := format('status %s: %s', response.status, left(COALESCE(response.content, ''), 500));
	        EXCEPTION
	            WHEN OTHERS THEN
	                last_error := SQLERRM;
	        END;

	        IF retry_seconds <= 0 OR clock_timestamp() - start_ts >= (retry_seconds || ' seconds')::interval THEN
	            RAISE EXCEPTION 'Embedding service not available after % seconds: %', retry_seconds, COALESCE(last_error, '<unknown>');
	        END IF;

	        PERFORM pg_sleep(GREATEST(0.0, retry_interval_seconds));
	    END LOOP;

	    -- Parse response
	    embedding_json := response.content::JSONB;

    -- Extract embedding array (handle different response formats)
    IF embedding_json ? 'embeddings' THEN
        -- Format: {"embeddings": [[...]]}
        embedding_array := ARRAY(
            SELECT jsonb_array_elements_text((embedding_json->'embeddings')->0)::FLOAT
        );
    ELSIF embedding_json ? 'embedding' THEN
        -- Format: {"embedding": [...]}
        embedding_array := ARRAY(
            SELECT jsonb_array_elements_text(embedding_json->'embedding')::FLOAT
        );
    ELSIF embedding_json ? 'data' THEN
        -- OpenAI format: {"data": [{"embedding": [...]}]}
        embedding_array := ARRAY(
            SELECT jsonb_array_elements_text((embedding_json->'data')->0->'embedding')::FLOAT
        );
    ELSIF jsonb_typeof(embedding_json->0) = 'array' THEN
        -- HuggingFace TEI format: [[...]] (array of arrays)
        embedding_array := ARRAY(
            SELECT jsonb_array_elements_text(embedding_json->0)::FLOAT
        );
    ELSE
        -- Flat array format: [...]
        embedding_array := ARRAY(
            SELECT jsonb_array_elements_text(embedding_json)::FLOAT
        );
	    END IF;
	
	    -- Validate embedding size
	    IF array_length(embedding_array, 1) IS NULL OR array_length(embedding_array, 1) != expected_dim THEN
	        RAISE EXCEPTION 'Invalid embedding dimension: expected %, got %', expected_dim, array_length(embedding_array, 1);
	    END IF;
	
	    -- Cache the result
	    INSERT INTO embedding_cache (content_hash, embedding)
	    VALUES (v_content_hash, embedding_array::vector)
	    ON CONFLICT DO NOTHING;
	
	    RETURN embedding_array::vector;
	EXCEPTION
	    WHEN OTHERS THEN
	        RAISE EXCEPTION 'Failed to get embedding: %', SQLERRM;
	END;
$$ LANGUAGE plpgsql;

-- Check embedding service health
CREATE OR REPLACE FUNCTION check_embedding_service_health()
RETURNS BOOLEAN AS $$
DECLARE
    service_url TEXT;
    health_url TEXT;
    response http_response;
BEGIN
    SELECT value INTO service_url FROM embedding_config WHERE key = 'service_url';

    -- Extract base URL (scheme + host + port) using regexp, then append /health
    -- e.g., http://embeddings:80/embed -> http://embeddings:80/health
    health_url := regexp_replace(service_url, '^(https?://[^/]+).*$', '\1/health');

    SELECT * INTO response FROM http_get(health_url);

    RETURN response.status = 200;
EXCEPTION
    WHEN OTHERS THEN
        RETURN FALSE;
END;
$$ LANGUAGE plpgsql;

-- ============================================================================
-- TRIGGERS
-- ============================================================================

-- Update memory timestamp on modification
CREATE OR REPLACE FUNCTION update_memory_timestamp()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trg_memory_timestamp
    BEFORE UPDATE ON memories
    FOR EACH ROW
    EXECUTE FUNCTION update_memory_timestamp();

-- Update importance based on access
CREATE OR REPLACE FUNCTION update_memory_importance()
RETURNS TRIGGER AS $$
BEGIN
    NEW.importance = NEW.importance * (1.0 + (LN(NEW.access_count + 1) * 0.1));
    NEW.last_accessed = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trg_importance_on_access
    BEFORE UPDATE ON memories
    FOR EACH ROW
    WHEN (NEW.access_count != OLD.access_count)
    EXECUTE FUNCTION update_memory_importance();

-- Update cluster activation
CREATE OR REPLACE FUNCTION update_cluster_activation()
RETURNS TRIGGER AS $$
BEGIN
    NEW.last_activated = CURRENT_TIMESTAMP;
    NEW.activation_count = NEW.activation_count + 1;
    NEW.importance_score = NEW.importance_score * (1.0 + (LN(NEW.activation_count + 1) * 0.05));
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trg_cluster_activation
    BEFORE UPDATE ON memory_clusters
    FOR EACH ROW
    WHEN (NEW.activation_count != OLD.activation_count)
    EXECUTE FUNCTION update_cluster_activation();

-- Mark neighborhoods stale when memories change significantly
CREATE OR REPLACE FUNCTION mark_neighborhoods_stale()
RETURNS TRIGGER AS $$
BEGIN
    UPDATE memory_neighborhoods 
    SET is_stale = TRUE 
    WHERE memory_id = NEW.id;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trg_neighborhood_staleness
    AFTER UPDATE OF importance, status ON memories
    FOR EACH ROW
    EXECUTE FUNCTION mark_neighborhoods_stale();

-- Auto-assign memories to episodes
CREATE OR REPLACE FUNCTION assign_to_episode()
RETURNS TRIGGER AS $$
DECLARE
    current_episode_id UUID;
    last_memory_time TIMESTAMPTZ;
    new_seq INT;
BEGIN
    -- Prevent concurrent episode creation
    PERFORM pg_advisory_xact_lock(hashtext('episode_manager'));

    -- Find most recent episode and its last memory time
    SELECT e.id, MAX(m.created_at)
    INTO current_episode_id, last_memory_time
    FROM episodes e
    LEFT JOIN episode_memories em ON e.id = em.episode_id
    LEFT JOIN memories m ON em.memory_id = m.id
    WHERE e.ended_at IS NULL
    GROUP BY e.id
    ORDER BY e.started_at DESC
    LIMIT 1;

    -- If gap > 30 min or no episodes, start new episode
    IF current_episode_id IS NULL OR 
       (last_memory_time IS NOT NULL AND NEW.created_at - last_memory_time > INTERVAL '30 minutes')
    THEN
        -- Close previous episode
        IF current_episode_id IS NOT NULL THEN
            UPDATE episodes 
            SET ended_at = last_memory_time 
            WHERE id = current_episode_id;
        END IF;

        -- Create new episode
        INSERT INTO episodes (started_at, episode_type)
        VALUES (NEW.created_at, 'autonomous')
        RETURNING id INTO current_episode_id;
        
        new_seq := 1;
    ELSE
        -- Get next sequence number
        SELECT COALESCE(MAX(sequence_order), 0) + 1 
        INTO new_seq 
        FROM episode_memories 
        WHERE episode_id = current_episode_id;
    END IF;

    -- Link memory to episode
    INSERT INTO episode_memories (episode_id, memory_id, sequence_order)
    VALUES (current_episode_id, NEW.id, new_seq);

    -- Initialize neighborhood record
    INSERT INTO memory_neighborhoods (memory_id, is_stale)
    VALUES (NEW.id, TRUE)
    ON CONFLICT DO NOTHING;

    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trg_auto_episode_assignment
    AFTER INSERT ON memories
    FOR EACH ROW
    EXECUTE FUNCTION assign_to_episode();

-- ============================================================================
-- CORE FUNCTIONS
-- ============================================================================

-- Fast recall: Primary retrieval function (Hot Path)
CREATE OR REPLACE FUNCTION fast_recall(
    p_query_text TEXT,
    p_limit INT DEFAULT 10
) RETURNS TABLE (
    memory_id UUID,
    content TEXT,
    memory_type memory_type,
    score FLOAT,
    source TEXT
) AS $$
	DECLARE
	    query_embedding vector;
	    zero_vec vector;
	    current_valence FLOAT;
	BEGIN
	    query_embedding := get_embedding(p_query_text);
	    zero_vec := array_fill(0.0::float, ARRAY[embedding_dimension()])::vector;
	    BEGIN
	        current_valence := NULLIF(get_current_affective_state()->>'valence', '')::float;
	    EXCEPTION
	        WHEN OTHERS THEN
	            current_valence := NULL;
	    END;
	    current_valence := COALESCE(current_valence, 0.0);
	    
	    RETURN QUERY
	    WITH 
    -- Vector seeds (semantic similarity)
	    seeds AS (
	        SELECT 
	            m.id, 
	            m.content, 
	            m.type,
            m.importance,
            m.decay_rate,
            m.created_at,
            m.last_accessed,
            1 - (m.embedding <=> query_embedding) as sim
        FROM memories m
	        WHERE m.status = 'active'
	          AND m.embedding IS NOT NULL
	          AND m.embedding <> zero_vec
	        ORDER BY m.embedding <=> query_embedding
	        LIMIT GREATEST(p_limit, 5)
	    ),
    -- Expand via precomputed neighborhoods
    associations AS (
        SELECT 
            (key)::UUID as mem_id,
            MAX((value::float) * s.sim) as assoc_score
        FROM seeds s
        JOIN memory_neighborhoods mn ON s.id = mn.memory_id,
        jsonb_each_text(mn.neighbors)
        WHERE NOT mn.is_stale
        GROUP BY key
    ),
    -- Temporal context from episodes
    temporal AS (
        SELECT DISTINCT
            em.memory_id as mem_id,
            0.15 as temp_score
        FROM seeds s
        JOIN episode_memories em_seed ON s.id = em_seed.memory_id
        JOIN episode_memories em ON em_seed.episode_id = em.episode_id
        WHERE em.memory_id != s.id
        LIMIT 20
    ),
    -- Combine all candidates
    candidates AS (
        SELECT id as mem_id, sim as vector_score, NULL::float as assoc_score, NULL::float as temp_score
        FROM seeds
        UNION
        SELECT mem_id, NULL, assoc_score, NULL FROM associations
        UNION
        SELECT mem_id, NULL, NULL, temp_score FROM temporal
    ),
    -- Aggregate scores per memory
    scored AS (
        SELECT 
            c.mem_id,
            MAX(c.vector_score) as vector_score,
            MAX(c.assoc_score) as assoc_score,
            MAX(c.temp_score) as temp_score
        FROM candidates c
        GROUP BY c.mem_id
    )
	    SELECT 
	        m.id,
	        m.content,
	        m.type,
	        GREATEST(
	            COALESCE(sc.vector_score, 0) * 0.5 +
	            COALESCE(sc.assoc_score, 0) * 0.3 +
	            COALESCE(sc.temp_score, 0) * 0.15 +
	            calculate_relevance(m.importance, m.decay_rate, m.created_at, m.last_accessed) * 0.05 +
	            -- Mood-congruent recall bias (small): prefer episodic memories whose valence matches current affect.
	            (CASE
	                WHEN em.emotional_valence IS NULL THEN 0.5
	                ELSE 1.0 - (ABS(em.emotional_valence - current_valence) / 2.0)
	            END) * 0.05,
	            0.001
	        ) as final_score,
	        CASE 
	            WHEN sc.vector_score IS NOT NULL THEN 'vector'
	            WHEN sc.assoc_score IS NOT NULL THEN 'association'
	            WHEN sc.temp_score IS NOT NULL THEN 'temporal'
	            ELSE 'fallback'
	        END as source
	    FROM scored sc
	    JOIN memories m ON sc.mem_id = m.id
	    LEFT JOIN episodic_memories em ON em.memory_id = m.id
	    WHERE m.status = 'active'
	    ORDER BY final_score DESC
	    LIMIT p_limit;
END;
$$ LANGUAGE plpgsql;

-- ============================================================================
-- PROVENANCE & TRUST (Normalization Layer)
-- ============================================================================

-- Normalize a source reference object into a consistent shape.
-- Intended fields: kind, ref, label, author, observed_at, trust.
CREATE OR REPLACE FUNCTION normalize_source_reference(p_source JSONB)
RETURNS JSONB AS $$
DECLARE
    kind TEXT;
    ref TEXT;
    label TEXT;
    author TEXT;
    observed_at TIMESTAMPTZ;
    trust FLOAT;
BEGIN
    IF p_source IS NULL OR jsonb_typeof(p_source) <> 'object' THEN
        RETURN '{}'::jsonb;
    END IF;

    kind := NULLIF(p_source->>'kind', '');
    ref := COALESCE(NULLIF(p_source->>'ref', ''), NULLIF(p_source->>'uri', ''));
    label := NULLIF(p_source->>'label', '');
    author := NULLIF(p_source->>'author', '');

    BEGIN
        observed_at := (p_source->>'observed_at')::timestamptz;
    EXCEPTION WHEN OTHERS THEN
        observed_at := CURRENT_TIMESTAMP;
    END;
    IF observed_at IS NULL THEN
        observed_at := CURRENT_TIMESTAMP;
    END IF;

    trust := COALESCE(NULLIF(p_source->>'trust', '')::float, 0.5);
    trust := LEAST(1.0, GREATEST(0.0, trust));

    RETURN jsonb_strip_nulls(
        jsonb_build_object(
            'kind', kind,
            'ref', ref,
            'label', label,
            'author', author,
            'observed_at', observed_at,
            'trust', trust
        )
    );
END;
$$ LANGUAGE plpgsql STABLE;

-- Normalize source references into an array of normalized source objects.
CREATE OR REPLACE FUNCTION normalize_source_references(p_sources JSONB)
RETURNS JSONB AS $$
DECLARE
    elem JSONB;
    out_arr JSONB := '[]'::jsonb;
BEGIN
    IF p_sources IS NULL THEN
        RETURN '[]'::jsonb;
    END IF;

    IF jsonb_typeof(p_sources) = 'array' THEN
        FOR elem IN SELECT * FROM jsonb_array_elements(p_sources)
        LOOP
            out_arr := out_arr || jsonb_build_array(normalize_source_reference(elem));
        END LOOP;
    ELSIF jsonb_typeof(p_sources) = 'object' THEN
        out_arr := jsonb_build_array(normalize_source_reference(p_sources));
    ELSE
        RETURN '[]'::jsonb;
    END IF;

    RETURN COALESCE(
        (SELECT jsonb_agg(e) FROM jsonb_array_elements(out_arr) e WHERE e <> '{}'::jsonb),
        '[]'::jsonb
    );
END;
$$ LANGUAGE plpgsql STABLE;

-- Dedupe normalized sources by a canonical key (ref/label fallback), keeping the most recent observed_at.
CREATE OR REPLACE FUNCTION dedupe_source_references(p_sources JSONB)
RETURNS JSONB AS $$
BEGIN
    RETURN COALESCE((
        SELECT jsonb_agg(d.elem)
        FROM (
            SELECT DISTINCT ON (d.key) d.elem
            FROM (
                SELECT
                    COALESCE(NULLIF(e->>'ref', ''), NULLIF(e->>'label', ''), md5(e::text)) AS key,
                    e AS elem,
                    COALESCE(e->>'observed_at', '') AS observed_at
                FROM jsonb_array_elements(normalize_source_references(p_sources)) e
            ) d
            ORDER BY d.key, d.observed_at DESC
        ) d
    ), '[]'::jsonb);
END;
$$ LANGUAGE plpgsql STABLE;

-- Convert sources into a reinforcement score [0..1] that grows with unique source count and average trust.
CREATE OR REPLACE FUNCTION source_reinforcement_score(p_source_references JSONB)
RETURNS FLOAT AS $$
DECLARE
    unique_sources INT;
    avg_trust FLOAT;
BEGIN
    WITH elems AS (
        SELECT
            COALESCE(NULLIF(e->>'ref', ''), NULLIF(e->>'label', ''), md5(e::text)) AS key,
            COALESCE((e->>'trust')::float, 0.5) AS trust
        FROM jsonb_array_elements(dedupe_source_references(p_source_references)) e
    )
    SELECT COUNT(DISTINCT key), AVG(trust) INTO unique_sources, avg_trust
    FROM elems;

    IF unique_sources IS NULL OR unique_sources = 0 THEN
        RETURN 0.0;
    END IF;

    avg_trust := COALESCE(avg_trust, 0.5);
    RETURN 1.0 - exp(-0.8 * unique_sources * avg_trust);
END;
$$ LANGUAGE plpgsql STABLE;

-- Worldview alignment score in [-1..1], based on explicit influences recorded for a memory.
CREATE OR REPLACE FUNCTION compute_worldview_alignment(p_memory_id UUID)
RETURNS FLOAT AS $$
DECLARE
    num FLOAT;
    denom FLOAT;
    alignment FLOAT;
BEGIN
    SELECT
        SUM(COALESCE(wmi.strength, 0) * COALESCE(wp.confidence, 0.5)),
        SUM(ABS(COALESCE(wp.confidence, 0.5)))
    INTO num, denom
    FROM worldview_memory_influences wmi
    JOIN worldview_primitives wp ON wp.id = wmi.worldview_id
    WHERE wmi.memory_id = p_memory_id;

    IF denom IS NULL OR denom = 0 THEN
        RETURN 0.0;
    END IF;

    alignment := num / denom;
    RETURN LEAST(1.0, GREATEST(-1.0, alignment));
END;
$$ LANGUAGE plpgsql STABLE;

-- Compute an effective trust level for semantic memories, capped by multi-source reinforcement and worldview alignment.
CREATE OR REPLACE FUNCTION compute_semantic_trust(
    p_confidence FLOAT,
    p_source_references JSONB,
    p_worldview_alignment FLOAT DEFAULT 0.0
)
RETURNS FLOAT AS $$
DECLARE
    base_confidence FLOAT;
    reinforcement FLOAT;
    cap FLOAT;
    effective FLOAT;
    alignment FLOAT;
BEGIN
    base_confidence := LEAST(1.0, GREATEST(0.0, COALESCE(p_confidence, 0.5)));
    reinforcement := source_reinforcement_score(p_source_references);

    -- With no sources, cap stays low; more independent sources raises the ceiling.
    cap := 0.15 + 0.85 * reinforcement;
    effective := LEAST(base_confidence, cap);

    alignment := LEAST(1.0, GREATEST(-1.0, COALESCE(p_worldview_alignment, 0.0)));
    IF alignment < 0 THEN
        -- Strong misalignment can drive trust toward 0.
        effective := effective * (1.0 + alignment);
    ELSE
        -- Mild bonus for alignment.
        effective := LEAST(1.0, effective + 0.10 * alignment);
    END IF;

    RETURN LEAST(1.0, GREATEST(0.0, effective));
END;
$$ LANGUAGE plpgsql STABLE;

-- Sync `memories.trust_level` based on semantic confidence/sources + worldview influences.
CREATE OR REPLACE FUNCTION sync_memory_trust(p_memory_id UUID)
RETURNS VOID AS $$
DECLARE
    mtype memory_type;
    conf FLOAT;
    sources JSONB;
    alignment FLOAT;
    computed FLOAT;
BEGIN
    SELECT type INTO mtype FROM memories WHERE id = p_memory_id;
    IF NOT FOUND THEN
        RETURN;
    END IF;

    IF mtype <> 'semantic' THEN
        RETURN;
    END IF;

    SELECT confidence, source_references INTO conf, sources
    FROM semantic_memories
    WHERE memory_id = p_memory_id;

    sources := dedupe_source_references(sources);
    alignment := compute_worldview_alignment(p_memory_id);
    computed := compute_semantic_trust(conf, sources, alignment);

    UPDATE memories
    SET trust_level = computed,
        trust_updated_at = CURRENT_TIMESTAMP,
        source_attribution = CASE
            WHEN (source_attribution = '{}'::jsonb OR source_attribution IS NULL)
                 AND jsonb_typeof(sources) = 'array'
                 AND jsonb_array_length(sources) > 0
            THEN normalize_source_reference(sources->0)
            ELSE source_attribution
        END
    WHERE id = p_memory_id;
END;
$$ LANGUAGE plpgsql;

-- Add a new source reference to a semantic memory and recompute trust.
CREATE OR REPLACE FUNCTION add_semantic_source_reference(
    p_memory_id UUID,
    p_source JSONB
)
RETURNS VOID AS $$
DECLARE
    normalized JSONB;
BEGIN
    normalized := normalize_source_reference(p_source);
    IF normalized = '{}'::jsonb THEN
        RETURN;
    END IF;

    UPDATE semantic_memories
    SET source_references = dedupe_source_references(
            COALESCE(source_references, '[]'::jsonb) || jsonb_build_array(normalized)
        ),
        last_validated = CURRENT_TIMESTAMP
    WHERE memory_id = p_memory_id;

    PERFORM sync_memory_trust(p_memory_id);
END;
$$ LANGUAGE plpgsql;

-- Provide a compact truth/provenance profile for downstream consumers (prompts, APIs).
CREATE OR REPLACE FUNCTION get_memory_truth_profile(p_memory_id UUID)
RETURNS JSONB AS $$
DECLARE
    mtype memory_type;
    base_conf FLOAT;
    sources JSONB;
    reinforcement FLOAT;
    alignment FLOAT;
    trust FLOAT;
    source_count INT;
BEGIN
    SELECT type, trust_level INTO mtype, trust
    FROM memories
    WHERE id = p_memory_id;

    IF NOT FOUND THEN
        RETURN '{}'::jsonb;
    END IF;

    IF mtype = 'semantic' THEN
        SELECT confidence, source_references INTO base_conf, sources
        FROM semantic_memories
        WHERE memory_id = p_memory_id;

        sources := dedupe_source_references(sources);
        reinforcement := source_reinforcement_score(sources);
        alignment := compute_worldview_alignment(p_memory_id);
        source_count := COALESCE(jsonb_array_length(sources), 0);

        RETURN jsonb_build_object(
            'type', 'semantic',
            'base_confidence', COALESCE(base_conf, 0.5),
            'trust_level', trust,
            'source_count', source_count,
            'source_reinforcement', reinforcement,
            'worldview_alignment', alignment,
            'sources', sources
        );
    END IF;

    RETURN jsonb_build_object(
        'type', mtype::text,
        'trust_level', trust
    );
END;
$$ LANGUAGE plpgsql STABLE;

-- Keep `memories.trust_level` in sync for semantic memories when sources/confidence or worldview influences change.
CREATE OR REPLACE FUNCTION trg_sync_semantic_trust()
RETURNS TRIGGER AS $$
BEGIN
    PERFORM sync_memory_trust(NEW.memory_id);
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS trg_semantic_trust_sync ON semantic_memories;
CREATE TRIGGER trg_semantic_trust_sync
    AFTER INSERT OR UPDATE ON semantic_memories
    FOR EACH ROW
    EXECUTE FUNCTION trg_sync_semantic_trust();

-- Update a worldview belief's confidence based on recent supporting/contradicting influences.
-- This is intentionally conservative: confidence shifts slowly, weighted by memory trust.
CREATE OR REPLACE FUNCTION update_worldview_confidence_from_influences(
    p_worldview_id UUID,
    p_window INTERVAL DEFAULT INTERVAL '30 days',
    p_learning_rate FLOAT DEFAULT 0.05
)
RETURNS VOID AS $$
DECLARE
    delta FLOAT;
    base_conf FLOAT;
BEGIN
    IF p_worldview_id IS NULL THEN
        RETURN;
    END IF;

    SELECT COALESCE(AVG(COALESCE(wmi.strength, 0) * COALESCE(m.trust_level, 0.5)), 0)
    INTO delta
    FROM worldview_memory_influences wmi
    JOIN memories m ON m.id = wmi.memory_id
    WHERE wmi.worldview_id = p_worldview_id
      AND wmi.created_at >= CURRENT_TIMESTAMP - COALESCE(p_window, INTERVAL '30 days');

    SELECT COALESCE(confidence, 0.5) INTO base_conf
    FROM worldview_primitives
    WHERE id = p_worldview_id;

    UPDATE worldview_primitives
    SET confidence = LEAST(1.0, GREATEST(0.0, base_conf + COALESCE(p_learning_rate, 0.05) * COALESCE(delta, 0))),
        updated_at = CURRENT_TIMESTAMP
    WHERE id = p_worldview_id;
END;
$$ LANGUAGE plpgsql;

CREATE OR REPLACE FUNCTION trg_sync_worldview_influence_trust()
RETURNS TRIGGER AS $$
	DECLARE
	    mem_id UUID;
	    wid UUID;
	BEGIN
	    mem_id := COALESCE(NEW.memory_id, OLD.memory_id);
	    IF mem_id IS NOT NULL THEN
	        PERFORM sync_memory_trust(mem_id);
	    END IF;
	    wid := COALESCE(NEW.worldview_id, OLD.worldview_id);
	    IF wid IS NOT NULL THEN
	        PERFORM update_worldview_confidence_from_influences(wid);
	    END IF;
	    RETURN COALESCE(NEW, OLD);
	END;
	$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS trg_worldview_influence_trust_sync ON worldview_memory_influences;
CREATE TRIGGER trg_worldview_influence_trust_sync
    AFTER INSERT OR UPDATE OR DELETE ON worldview_memory_influences
    FOR EACH ROW
    EXECUTE FUNCTION trg_sync_worldview_influence_trust();

-- Create memory (base function) - generates embedding automatically
CREATE OR REPLACE FUNCTION create_memory(
    p_type memory_type,
    p_content TEXT,
    p_importance FLOAT DEFAULT 0.5,
    p_source_attribution JSONB DEFAULT NULL,
    p_trust_level FLOAT DEFAULT NULL
) RETURNS UUID AS $$
DECLARE
    new_memory_id UUID;
    embedding_vec vector;
    normalized_source JSONB;
    effective_trust FLOAT;
BEGIN
    normalized_source := normalize_source_reference(p_source_attribution);
    IF normalized_source = '{}'::jsonb THEN
        normalized_source := jsonb_build_object(
            'kind',
            CASE
                WHEN p_type = 'semantic' THEN 'unattributed'
                ELSE 'internal'
            END,
            'observed_at', CURRENT_TIMESTAMP
        );
    END IF;

    effective_trust := p_trust_level;
    IF effective_trust IS NULL THEN
        effective_trust := CASE
            WHEN p_type = 'episodic' THEN 0.95
            WHEN p_type = 'semantic' THEN 0.20
            WHEN p_type = 'procedural' THEN 0.70
            WHEN p_type = 'strategic' THEN 0.70
            ELSE 0.50
        END;
    END IF;
    effective_trust := LEAST(1.0, GREATEST(0.0, effective_trust));

    -- Generate embedding
    embedding_vec := get_embedding(p_content);
    
    INSERT INTO memories (type, content, embedding, importance, source_attribution, trust_level, trust_updated_at)
    VALUES (p_type, p_content, embedding_vec, p_importance, normalized_source, effective_trust, CURRENT_TIMESTAMP)
    RETURNING id INTO new_memory_id;
    
    -- Create graph node
    EXECUTE format(
        'SELECT * FROM cypher(''memory_graph'', $q$
            CREATE (n:MemoryNode {memory_id: %L, type: %L, created_at: %L})
            RETURN n
        $q$) as (result agtype)',
        new_memory_id,
        p_type,
        CURRENT_TIMESTAMP
    );
    
    RETURN new_memory_id;
END;
$$ LANGUAGE plpgsql;

-- Create episodic memory
CREATE OR REPLACE FUNCTION create_episodic_memory(
    p_content TEXT,
    p_action_taken JSONB DEFAULT NULL,
    p_context JSONB DEFAULT NULL,
    p_result JSONB DEFAULT NULL,
    p_emotional_valence FLOAT DEFAULT 0.0,
    p_event_time TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    p_importance FLOAT DEFAULT 0.5,
    p_source_attribution JSONB DEFAULT NULL,
    p_trust_level FLOAT DEFAULT NULL
) RETURNS UUID AS $$
DECLARE
    new_memory_id UUID;
    normalized_source JSONB;
    effective_trust FLOAT;
BEGIN
    normalized_source := normalize_source_reference(p_source_attribution);
    IF normalized_source = '{}'::jsonb THEN
        normalized_source := jsonb_build_object('kind', 'internal', 'observed_at', CURRENT_TIMESTAMP);
    END IF;
    effective_trust := COALESCE(p_trust_level, 0.95);
    new_memory_id := create_memory('episodic', p_content, p_importance, normalized_source, effective_trust);
    
    INSERT INTO episodic_memories (
        memory_id, action_taken, context, result, 
        emotional_valence, event_time
    ) VALUES (
        new_memory_id, p_action_taken, p_context, p_result,
        p_emotional_valence, p_event_time
    );
    
    RETURN new_memory_id;
END;
$$ LANGUAGE plpgsql;

-- Create semantic memory
CREATE OR REPLACE FUNCTION create_semantic_memory(
    p_content TEXT,
    p_confidence FLOAT,
    p_category TEXT[] DEFAULT NULL,
    p_related_concepts TEXT[] DEFAULT NULL,
    p_source_references JSONB DEFAULT NULL,
    p_importance FLOAT DEFAULT 0.5,
    p_source_attribution JSONB DEFAULT NULL,
    p_trust_level FLOAT DEFAULT NULL
) RETURNS UUID AS $$
DECLARE
    new_memory_id UUID;
    normalized_sources JSONB;
    primary_source JSONB;
    base_confidence FLOAT;
    effective_trust FLOAT;
BEGIN
    normalized_sources := dedupe_source_references(p_source_references);
    base_confidence := LEAST(1.0, GREATEST(0.0, COALESCE(p_confidence, 0.5)));

    primary_source := normalize_source_reference(p_source_attribution);
    IF primary_source = '{}'::jsonb AND jsonb_typeof(normalized_sources) = 'array' AND jsonb_array_length(normalized_sources) > 0 THEN
        primary_source := normalize_source_reference(normalized_sources->0);
    END IF;
    IF primary_source = '{}'::jsonb THEN
        primary_source := jsonb_build_object('kind', 'unattributed', 'observed_at', CURRENT_TIMESTAMP);
    END IF;

    effective_trust := COALESCE(p_trust_level, compute_semantic_trust(base_confidence, normalized_sources, 0.0));

    new_memory_id := create_memory('semantic', p_content, p_importance, primary_source, effective_trust);
    
    INSERT INTO semantic_memories (
        memory_id, confidence, category, related_concepts,
        source_references, last_validated
    ) VALUES (
        new_memory_id, p_confidence, p_category, p_related_concepts,
        normalized_sources, CURRENT_TIMESTAMP
    );

    PERFORM sync_memory_trust(new_memory_id);
    
    RETURN new_memory_id;
END;
$$ LANGUAGE plpgsql;

-- Create procedural memory
CREATE OR REPLACE FUNCTION create_procedural_memory(
    p_content TEXT,
    p_steps JSONB,
    p_prerequisites JSONB DEFAULT NULL,
    p_importance FLOAT DEFAULT 0.5,
    p_source_attribution JSONB DEFAULT NULL,
    p_trust_level FLOAT DEFAULT NULL
) RETURNS UUID AS $$
DECLARE
    new_memory_id UUID;
    normalized_source JSONB;
    effective_trust FLOAT;
BEGIN
    normalized_source := normalize_source_reference(p_source_attribution);
    IF normalized_source = '{}'::jsonb THEN
        normalized_source := jsonb_build_object('kind', 'internal', 'observed_at', CURRENT_TIMESTAMP);
    END IF;
    effective_trust := COALESCE(p_trust_level, 0.70);
    new_memory_id := create_memory('procedural', p_content, p_importance, normalized_source, effective_trust);
    
    INSERT INTO procedural_memories (
        memory_id, steps, prerequisites
    ) VALUES (
        new_memory_id, p_steps, p_prerequisites
    );
    
    RETURN new_memory_id;
END;
$$ LANGUAGE plpgsql;

-- Create strategic memory
CREATE OR REPLACE FUNCTION create_strategic_memory(
    p_content TEXT,
    p_pattern_description TEXT,
    p_confidence_score FLOAT,
    p_supporting_evidence JSONB DEFAULT NULL,
    p_context_applicability JSONB DEFAULT NULL,
    p_importance FLOAT DEFAULT 0.5,
    p_source_attribution JSONB DEFAULT NULL,
    p_trust_level FLOAT DEFAULT NULL
) RETURNS UUID AS $$
DECLARE
    new_memory_id UUID;
    normalized_source JSONB;
    effective_trust FLOAT;
BEGIN
    normalized_source := normalize_source_reference(p_source_attribution);
    IF normalized_source = '{}'::jsonb THEN
        normalized_source := jsonb_build_object('kind', 'internal', 'observed_at', CURRENT_TIMESTAMP);
    END IF;
    effective_trust := COALESCE(p_trust_level, 0.70);
    new_memory_id := create_memory('strategic', p_content, p_importance, normalized_source, effective_trust);
    
    INSERT INTO strategic_memories (
        memory_id, pattern_description, confidence_score,
        supporting_evidence, context_applicability
    ) VALUES (
        new_memory_id, p_pattern_description, p_confidence_score,
        p_supporting_evidence, p_context_applicability
    );
    
    RETURN new_memory_id;
END;
$$ LANGUAGE plpgsql;

-- Batch create memories from JSONB items.
-- Each item must include: {"type": "semantic|episodic|procedural|strategic", "content": "..."}
-- Optional keys: importance, emotional_valence, context, action_taken, result, event_time,
--                confidence, category, related_concepts, source_references, steps, prerequisites,
--                pattern_description, supporting_evidence, context_applicability,
--                source_attribution, trust_level.
CREATE OR REPLACE FUNCTION batch_create_memories(p_items JSONB)
RETURNS UUID[] AS $$
DECLARE
    ids UUID[] := ARRAY[]::UUID[];
    item JSONB;
    mtype memory_type;
    content TEXT;
    importance FLOAT;
    new_id UUID;
    idx INT := 0;
BEGIN
    IF p_items IS NULL OR jsonb_typeof(p_items) <> 'array' THEN
        RETURN ids;
    END IF;

    FOR item IN SELECT * FROM jsonb_array_elements(p_items)
    LOOP
        idx := idx + 1;
        mtype := NULLIF(item->>'type', '')::memory_type;
        content := NULLIF(item->>'content', '');
        IF content IS NULL OR mtype IS NULL THEN
            RAISE EXCEPTION 'batch_create_memories: item % missing required fields', idx;
        END IF;
        importance := COALESCE(NULLIF(item->>'importance', '')::float, 0.5);

        IF mtype = 'episodic' THEN
            new_id := create_episodic_memory(
                content,
                item->'action_taken',
                item->'context',
                item->'result',
                COALESCE(NULLIF(item->>'emotional_valence', '')::float, 0.0),
                COALESCE(NULLIF(item->>'event_time', '')::timestamptz, CURRENT_TIMESTAMP),
                importance,
                item->'source_attribution',
                NULLIF(item->>'trust_level', '')::float
            );
        ELSIF mtype = 'semantic' THEN
            new_id := create_semantic_memory(
                content,
                COALESCE(NULLIF(item->>'confidence', '')::float, 0.8),
                CASE WHEN item ? 'category' THEN ARRAY(SELECT jsonb_array_elements_text(item->'category')) ELSE NULL END,
                CASE WHEN item ? 'related_concepts' THEN ARRAY(SELECT jsonb_array_elements_text(item->'related_concepts')) ELSE NULL END,
                item->'source_references',
                importance,
                item->'source_attribution',
                NULLIF(item->>'trust_level', '')::float
            );
        ELSIF mtype = 'procedural' THEN
            new_id := create_procedural_memory(
                content,
                COALESCE(item->'steps', jsonb_build_object('steps', '[]'::jsonb)),
                item->'prerequisites',
                importance,
                item->'source_attribution',
                NULLIF(item->>'trust_level', '')::float
            );
        ELSIF mtype = 'strategic' THEN
            new_id := create_strategic_memory(
                content,
                COALESCE(NULLIF(item->>'pattern_description', ''), content),
                COALESCE(NULLIF(item->>'confidence_score', '')::float, 0.8),
                item->'supporting_evidence',
                item->'context_applicability',
                importance,
                item->'source_attribution',
                NULLIF(item->>'trust_level', '')::float
            );
        ELSE
            RAISE EXCEPTION 'batch_create_memories: item % invalid type %', idx, mtype::text;
        END IF;

        IF new_id IS NULL THEN
            RAISE EXCEPTION 'batch_create_memories: item % failed to create memory', idx;
        END IF;
        ids := array_append(ids, new_id);
    END LOOP;

    RETURN ids;
END;
$$ LANGUAGE plpgsql;

-- Create memory with a precomputed embedding (used for batched/externally-generated embeddings).
CREATE OR REPLACE FUNCTION create_memory_with_embedding(
    p_type memory_type,
    p_content TEXT,
    p_embedding vector,
    p_importance FLOAT DEFAULT 0.5,
    p_source_attribution JSONB DEFAULT NULL,
    p_trust_level FLOAT DEFAULT NULL
) RETURNS UUID AS $$
DECLARE
    new_memory_id UUID;
    normalized_source JSONB;
    effective_trust FLOAT;
BEGIN
    IF p_embedding IS NULL THEN
        RAISE EXCEPTION 'embedding must not be NULL';
    END IF;

    normalized_source := normalize_source_reference(p_source_attribution);
    IF normalized_source = '{}'::jsonb THEN
        normalized_source := jsonb_build_object(
            'kind',
            CASE
                WHEN p_type = 'semantic' THEN 'unattributed'
                ELSE 'internal'
            END,
            'observed_at', CURRENT_TIMESTAMP
        );
    END IF;

    effective_trust := p_trust_level;
    IF effective_trust IS NULL THEN
        effective_trust := CASE
            WHEN p_type = 'episodic' THEN 0.95
            WHEN p_type = 'semantic' THEN 0.20
            WHEN p_type = 'procedural' THEN 0.70
            WHEN p_type = 'strategic' THEN 0.70
            ELSE 0.50
        END;
    END IF;
    effective_trust := LEAST(1.0, GREATEST(0.0, effective_trust));

    INSERT INTO memories (type, content, embedding, importance, source_attribution, trust_level, trust_updated_at)
    VALUES (p_type, p_content, p_embedding, p_importance, normalized_source, effective_trust, CURRENT_TIMESTAMP)
    RETURNING id INTO new_memory_id;

    EXECUTE format(
        'SELECT * FROM cypher(''memory_graph'', $q$
            CREATE (n:MemoryNode {memory_id: %L, type: %L, created_at: %L})
            RETURN n
        $q$) as (result agtype)',
        new_memory_id,
        p_type,
        CURRENT_TIMESTAMP
    );

    RETURN new_memory_id;
END;
$$ LANGUAGE plpgsql;

-- Batch create memories with precomputed embeddings (single type, no per-item metadata).
-- Inserts the base row, creates the MemoryNode, and creates the type-specific row with safe defaults.
CREATE OR REPLACE FUNCTION batch_create_memories_with_embeddings(
    p_type memory_type,
    p_contents TEXT[],
    p_embeddings JSONB,
    p_importance FLOAT DEFAULT 0.5
)
RETURNS UUID[] AS $$
DECLARE
    ids UUID[] := ARRAY[]::UUID[];
    n INT;
    i INT;
    expected_dim INT;
    emb_vec vector;
    emb_json JSONB;
    emb_arr FLOAT4[];
    new_id UUID;
BEGIN
    n := COALESCE(array_length(p_contents, 1), 0);
    IF n = 0 THEN
        RETURN ids;
    END IF;

    IF p_embeddings IS NULL OR jsonb_typeof(p_embeddings) <> 'array' THEN
        RAISE EXCEPTION 'embeddings must be a JSON array';
    END IF;
    IF jsonb_array_length(p_embeddings) <> n THEN
        RAISE EXCEPTION 'contents and embeddings length mismatch';
    END IF;

    expected_dim := embedding_dimension();

    FOR i IN 1..n LOOP
        IF p_contents[i] IS NULL OR p_contents[i] = '' THEN
            CONTINUE;
        END IF;

        emb_json := p_embeddings->(i - 1);
        IF emb_json IS NULL OR jsonb_typeof(emb_json) <> 'array' THEN
            RAISE EXCEPTION 'embedding % must be a JSON array', i;
        END IF;

        SELECT ARRAY_AGG(value::float4) INTO emb_arr
        FROM jsonb_array_elements_text(emb_json) value;

        IF COALESCE(array_length(emb_arr, 1), 0) <> expected_dim THEN
            RAISE EXCEPTION 'embedding dimension mismatch: expected %, got %', expected_dim, COALESCE(array_length(emb_arr, 1), 0);
        END IF;

        emb_vec := (emb_arr::float4[])::vector;
        new_id := create_memory_with_embedding(p_type, p_contents[i], emb_vec, p_importance);

        IF p_type = 'episodic' THEN
            INSERT INTO episodic_memories (memory_id, action_taken, context, result, emotional_valence, verification_status, event_time)
            VALUES (new_id, NULL, jsonb_build_object('type', 'raw_batch'), NULL, 0.0, NULL, CURRENT_TIMESTAMP)
            ON CONFLICT (memory_id) DO NOTHING;
        ELSIF p_type = 'semantic' THEN
            INSERT INTO semantic_memories (memory_id, confidence, last_validated, source_references, contradictions, category, related_concepts)
            VALUES (new_id, 0.8, CURRENT_TIMESTAMP, '[]'::jsonb, NULL, NULL, NULL)
            ON CONFLICT (memory_id) DO NOTHING;
            PERFORM sync_memory_trust(new_id);
        ELSIF p_type = 'procedural' THEN
            INSERT INTO procedural_memories (memory_id, steps, prerequisites)
            VALUES (new_id, jsonb_build_object('steps', '[]'::jsonb), NULL)
            ON CONFLICT (memory_id) DO NOTHING;
        ELSIF p_type = 'strategic' THEN
            INSERT INTO strategic_memories (memory_id, pattern_description, supporting_evidence, confidence_score, success_metrics, adaptation_history, context_applicability)
            VALUES (new_id, p_contents[i], NULL, 0.8, NULL, NULL, NULL)
            ON CONFLICT (memory_id) DO NOTHING;
        END IF;

        ids := array_append(ids, new_id);
    END LOOP;

    RETURN ids;
END;
$$ LANGUAGE plpgsql;

-- Search similar memories
CREATE OR REPLACE FUNCTION search_similar_memories(
    p_query_text TEXT,
    p_limit INT DEFAULT 10,
    p_memory_types memory_type[] DEFAULT NULL,
    p_min_importance FLOAT DEFAULT 0.0
) RETURNS TABLE (
    memory_id UUID,
    content TEXT,
    type memory_type,
    similarity FLOAT,
    importance FLOAT
) AS $$
DECLARE
    query_embedding vector;
    zero_vec vector;
BEGIN
    query_embedding := get_embedding(p_query_text);
    zero_vec := array_fill(0.0::float, ARRAY[embedding_dimension()])::vector;
    
    RETURN QUERY
    WITH candidates AS MATERIALIZED (
        SELECT m.id, m.content, m.type, m.embedding, m.importance
        FROM memories m
        WHERE m.status = 'active'
          AND m.embedding IS NOT NULL
          AND m.embedding <> zero_vec
          AND (p_memory_types IS NULL OR m.type = ANY(p_memory_types))
          AND m.importance >= p_min_importance
    )
    SELECT
        c.id,
        c.content,
        c.type,
        1 - (c.embedding <=> query_embedding) as similarity,
        c.importance
    FROM candidates c
    ORDER BY c.embedding <=> query_embedding
    LIMIT p_limit;
END;
$$ LANGUAGE plpgsql;

-- Assign memory to clusters based on similarity
CREATE OR REPLACE FUNCTION assign_memory_to_clusters(
    p_memory_id UUID, 
    p_max_clusters INT DEFAULT 3
) RETURNS VOID AS $$
DECLARE
    memory_embedding vector;
    cluster_record RECORD;
    similarity_threshold FLOAT := 0.7;
    assigned_count INT := 0;
    zero_vec vector := array_fill(0, ARRAY[embedding_dimension()])::vector;
BEGIN
    SELECT embedding INTO memory_embedding
    FROM memories WHERE id = p_memory_id;

    -- Avoid NaNs from cosine distance when any side is the zero vector.
    IF memory_embedding IS NULL OR memory_embedding = zero_vec THEN
        RETURN;
    END IF;
    
    FOR cluster_record IN 
        SELECT id, 1 - (centroid_embedding <=> memory_embedding) as similarity
        FROM memory_clusters
        WHERE centroid_embedding IS NOT NULL
          AND centroid_embedding <> zero_vec
        ORDER BY centroid_embedding <=> memory_embedding
        LIMIT 50
    LOOP
        IF cluster_record.similarity >= similarity_threshold AND assigned_count < p_max_clusters THEN
            INSERT INTO memory_cluster_members (cluster_id, memory_id, membership_strength)
            VALUES (cluster_record.id, p_memory_id, cluster_record.similarity)
            ON CONFLICT DO NOTHING;
            
            assigned_count := assigned_count + 1;
        END IF;
    END LOOP;
END;
$$ LANGUAGE plpgsql;

-- Recalculate cluster centroid
CREATE OR REPLACE FUNCTION recalculate_cluster_centroid(p_cluster_id UUID)
RETURNS VOID AS $$
DECLARE
    new_centroid vector;
BEGIN
    SELECT AVG(m.embedding)::vector
    INTO new_centroid
    FROM memories m
    JOIN memory_cluster_members mcm ON m.id = mcm.memory_id
    WHERE mcm.cluster_id = p_cluster_id
    AND m.status = 'active'
    AND mcm.membership_strength > 0.3;
    
    UPDATE memory_clusters
    SET centroid_embedding = new_centroid,
        updated_at = CURRENT_TIMESTAMP
    WHERE id = p_cluster_id;
END;
$$ LANGUAGE plpgsql;

-- Create graph relationship between memories
CREATE OR REPLACE FUNCTION create_memory_relationship(
    p_from_id UUID,
    p_to_id UUID,
    p_relationship_type graph_edge_type,
    p_properties JSONB DEFAULT '{}'
) RETURNS VOID AS $$
BEGIN
    EXECUTE format(
        'SELECT * FROM cypher(''memory_graph'', $q$
            MATCH (a:MemoryNode {memory_id: %L}), (b:MemoryNode {memory_id: %L})
            CREATE (a)-[r:%s %s]->(b)
            RETURN r
        $q$) as (result agtype)',
        p_from_id,
        p_to_id,
        p_relationship_type,
        CASE WHEN p_properties = '{}'::jsonb 
             THEN '' 
             ELSE format('{%s}', 
                  (SELECT string_agg(format('%I: %s', key, value), ', ')
                   FROM jsonb_each(p_properties)))
        END
    );
END;
$$ LANGUAGE plpgsql;

-- Link memory to concept
CREATE OR REPLACE FUNCTION link_memory_to_concept(
    p_memory_id UUID,
    p_concept_name TEXT,
    p_strength FLOAT DEFAULT 1.0
) RETURNS UUID AS $$
DECLARE
    concept_id UUID;
BEGIN
    -- Get or create concept
    INSERT INTO concepts (name)
    VALUES (p_concept_name)
    ON CONFLICT (name) DO UPDATE SET name = EXCLUDED.name
    RETURNING id INTO concept_id;
    
    -- Create relational link
    INSERT INTO memory_concepts (memory_id, concept_id, strength)
    VALUES (p_memory_id, concept_id, p_strength)
    ON CONFLICT DO NOTHING;
    
    -- Create graph edge
    EXECUTE format(
        'SELECT * FROM cypher(''memory_graph'', $q$
            MATCH (m:MemoryNode {memory_id: %L})
            MERGE (c:ConceptNode {name: %L})
            CREATE (m)-[:INSTANCE_OF {strength: %s}]->(c)
            RETURN c
        $q$) as (result agtype)',
        p_memory_id,
        p_concept_name,
        p_strength
    );
    
    RETURN concept_id;
END;
$$ LANGUAGE plpgsql;

-- Touch working memory rows (access tracking for consolidation heuristics)
CREATE OR REPLACE FUNCTION touch_working_memory(p_ids UUID[])
RETURNS VOID AS $$
BEGIN
    IF p_ids IS NULL OR array_length(p_ids, 1) IS NULL THEN
        RETURN;
    END IF;

    UPDATE working_memory
    SET access_count = access_count + 1,
        last_accessed = CURRENT_TIMESTAMP
    WHERE id = ANY(p_ids);
END;
$$ LANGUAGE plpgsql;

-- Promote a working-memory item into long-term episodic memory (preserving the existing embedding).
CREATE OR REPLACE FUNCTION promote_working_memory_to_episodic(
    p_working_memory_id UUID,
    p_importance FLOAT DEFAULT NULL
)
RETURNS UUID AS $$
DECLARE
    wm RECORD;
    new_id UUID;
    affect JSONB;
    v_valence FLOAT;
BEGIN
    SELECT * INTO wm FROM working_memory WHERE id = p_working_memory_id;
    IF NOT FOUND THEN
        RETURN NULL;
    END IF;

    affect := get_current_affective_state();
    BEGIN
        v_valence := NULLIF(affect->>'valence', '')::float;
    EXCEPTION
        WHEN OTHERS THEN
            v_valence := 0.0;
    END;
    v_valence := LEAST(1.0, GREATEST(-1.0, COALESCE(v_valence, 0.0)));

    new_id := create_memory_with_embedding(
        'episodic'::memory_type,
        wm.content,
        wm.embedding,
        COALESCE(p_importance, wm.importance, 0.4),
        wm.source_attribution,
        wm.trust_level
    );

    INSERT INTO episodic_memories (memory_id, action_taken, context, result, emotional_valence, verification_status, event_time)
    VALUES (
        new_id,
        NULL,
        jsonb_build_object(
            'from_working_memory_id', wm.id,
            'promoted_at', CURRENT_TIMESTAMP,
            'working_memory_created_at', wm.created_at,
            'working_memory_expiry', wm.expiry,
            'source_attribution', wm.source_attribution
        ),
        NULL,
        v_valence,
        NULL,
        wm.created_at
    )
    ON CONFLICT (memory_id) DO NOTHING;

    RETURN new_id;
END;
$$ LANGUAGE plpgsql;

-- Clean expired working memory (with optional consolidation before delete).
CREATE OR REPLACE FUNCTION cleanup_working_memory_with_stats(
    p_min_importance_to_promote FLOAT DEFAULT 0.75,
    p_min_accesses_to_promote INT DEFAULT 3
)
RETURNS JSONB AS $$
DECLARE
    promoted UUID[] := ARRAY[]::uuid[];
    rec RECORD;
    deleted_count INT := 0;
BEGIN
    FOR rec IN
        SELECT id, importance, access_count, promote_to_long_term
        FROM working_memory
        WHERE expiry < CURRENT_TIMESTAMP
    LOOP
        IF COALESCE(rec.promote_to_long_term, false)
           OR COALESCE(rec.importance, 0) >= COALESCE(p_min_importance_to_promote, 0.75)
           OR COALESCE(rec.access_count, 0) >= COALESCE(p_min_accesses_to_promote, 3)
        THEN
            promoted := array_append(promoted, promote_working_memory_to_episodic(rec.id, rec.importance));
        END IF;
    END LOOP;

    WITH deleted AS (
        DELETE FROM working_memory
        WHERE expiry < CURRENT_TIMESTAMP
        RETURNING 1
    )
    SELECT COUNT(*) INTO deleted_count FROM deleted;

    RETURN jsonb_build_object(
        'deleted_count', COALESCE(deleted_count, 0),
        'promoted_count', COALESCE(array_length(promoted, 1), 0),
        'promoted_ids', COALESCE(to_jsonb(promoted), '[]'::jsonb)
    );
END;
$$ LANGUAGE plpgsql;

CREATE OR REPLACE FUNCTION cleanup_working_memory()
RETURNS INT AS $$
DECLARE
    stats JSONB;
BEGIN
    stats := cleanup_working_memory_with_stats();
    RETURN COALESCE(NULLIF(stats->>'deleted_count', '')::int, 0);
END;
$$ LANGUAGE plpgsql;

-- Add to working memory with auto-embedding
CREATE OR REPLACE FUNCTION add_to_working_memory(
    p_content TEXT,
    p_expiry INTERVAL DEFAULT INTERVAL '1 hour',
    p_importance FLOAT DEFAULT 0.3,
    p_source_attribution JSONB DEFAULT NULL,
    p_trust_level FLOAT DEFAULT NULL,
    p_promote_to_long_term BOOLEAN DEFAULT FALSE
) RETURNS UUID AS $$
	DECLARE
	    new_id UUID;
	    embedding_vec vector;
	    normalized_source JSONB;
	    effective_trust FLOAT;
	BEGIN
	    embedding_vec := get_embedding(p_content);

	    normalized_source := normalize_source_reference(p_source_attribution);
	    IF normalized_source = '{}'::jsonb THEN
	        normalized_source := jsonb_build_object('kind', 'internal', 'observed_at', CURRENT_TIMESTAMP);
	    END IF;
	    effective_trust := p_trust_level;
	    IF effective_trust IS NULL THEN
	        effective_trust := 0.8;
	    END IF;
	    effective_trust := LEAST(1.0, GREATEST(0.0, effective_trust));

	    INSERT INTO working_memory (content, embedding, importance, source_attribution, trust_level, promote_to_long_term, expiry)
	    VALUES (
	        p_content,
	        embedding_vec,
	        LEAST(1.0, GREATEST(0.0, COALESCE(p_importance, 0.3))),
	        normalized_source,
	        effective_trust,
	        COALESCE(p_promote_to_long_term, false),
	        CURRENT_TIMESTAMP + p_expiry
	    )
	    RETURNING id INTO new_id;
	    
	    RETURN new_id;
	END;
	$$ LANGUAGE plpgsql;

-- Search working memory
CREATE OR REPLACE FUNCTION search_working_memory(
    p_query_text TEXT,
    p_limit INT DEFAULT 5
) RETURNS TABLE (
    memory_id UUID,
    content TEXT,
    similarity FLOAT,
    created_at TIMESTAMPTZ
) AS $$
	DECLARE
	    query_embedding vector;
	    zero_vec vector;
	BEGIN
	    query_embedding := get_embedding(p_query_text);
	    zero_vec := array_fill(0.0::float, ARRAY[embedding_dimension()])::vector;
	    
	    -- Clean expired first
	    PERFORM cleanup_working_memory();
	    
	    RETURN QUERY
	    WITH ranked AS (
	        SELECT
	            wm.id,
	            wm.content AS content_text,
	            1 - (wm.embedding <=> query_embedding) as similarity,
	            wm.created_at,
	            (wm.embedding <=> query_embedding) as dist
	        FROM working_memory wm
	        WHERE wm.embedding IS NOT NULL
	          AND wm.embedding <> zero_vec
	        ORDER BY wm.embedding <=> query_embedding
	        LIMIT p_limit
	    ),
	    touched AS (
	        UPDATE working_memory wm
	        SET access_count = access_count + 1,
	            last_accessed = CURRENT_TIMESTAMP
	        WHERE wm.id IN (SELECT id FROM ranked)
	        RETURNING wm.id
	    )
	    SELECT ranked.id AS memory_id, ranked.content_text AS content, ranked.similarity, ranked.created_at
	    FROM ranked
	    ORDER BY ranked.dist;
	END;
	$$ LANGUAGE plpgsql;

-- Clean old embedding cache
CREATE OR REPLACE FUNCTION cleanup_embedding_cache(
    p_older_than INTERVAL DEFAULT INTERVAL '7 days'
) RETURNS INT AS $$
DECLARE
    deleted_count INT;
BEGIN
    WITH deleted AS (
        DELETE FROM embedding_cache
        WHERE created_at < CURRENT_TIMESTAMP - p_older_than
        RETURNING 1
    )
    SELECT COUNT(*) INTO deleted_count FROM deleted;
    
    RETURN deleted_count;
END;
$$ LANGUAGE plpgsql;

-- ============================================================================
-- VIEWS
-- ============================================================================

CREATE VIEW memory_health AS
SELECT 
    type,
    COUNT(*) as total_memories,
    AVG(importance) as avg_importance,
    AVG(access_count) as avg_access_count,
    COUNT(*) FILTER (WHERE last_accessed > CURRENT_TIMESTAMP - INTERVAL '1 day') as accessed_last_day,
    AVG(calculate_relevance(importance, decay_rate, created_at, last_accessed)) as avg_relevance
FROM memories
WHERE status = 'active'
GROUP BY type;

CREATE VIEW cluster_insights AS
SELECT 
    mc.id,
    mc.name,
    mc.cluster_type,
    mc.importance_score,
    mc.coherence_score,
    COUNT(mcm.memory_id) as memory_count,
    mc.last_activated,
    mc.activation_count
FROM memory_clusters mc
LEFT JOIN memory_cluster_members mcm ON mc.id = mcm.cluster_id
GROUP BY mc.id
ORDER BY mc.importance_score DESC;

CREATE VIEW episode_summary AS
SELECT 
    e.id,
    e.started_at,
    e.ended_at,
    e.episode_type,
    e.summary,
    COUNT(em.memory_id) as memory_count,
    MIN(m.created_at) as first_memory_at,
    MAX(m.created_at) as last_memory_at
FROM episodes e
LEFT JOIN episode_memories em ON e.id = em.episode_id
LEFT JOIN memories m ON em.memory_id = m.id
GROUP BY e.id
ORDER BY e.started_at DESC;

CREATE VIEW stale_neighborhoods AS
SELECT 
    mn.memory_id,
    m.content,
    m.type,
    mn.computed_at,
    AGE(CURRENT_TIMESTAMP, mn.computed_at) as staleness
FROM memory_neighborhoods mn
JOIN memories m ON mn.memory_id = m.id
WHERE mn.is_stale = TRUE
ORDER BY mn.computed_at ASC;

-- ============================================================================
-- COMMENTS
-- ============================================================================

COMMENT ON FUNCTION fast_recall IS 'Primary retrieval function combining vector similarity, precomputed associations, and temporal context. Hot path - optimized for speed.';

COMMENT ON FUNCTION create_memory IS 'Creates a base memory record and corresponding graph node. Embedding must be pre-computed by application.';

COMMENT ON FUNCTION create_memory_relationship IS 'Creates a typed edge between two memories in the graph. Used for causal chains, contradictions, etc.';

COMMENT ON FUNCTION link_memory_to_concept IS 'Links a memory to an abstract concept, creating the concept if needed. Updates both relational and graph layers.';

COMMENT ON TABLE memory_neighborhoods IS 'Precomputed associative neighborhoods for each memory. Replaces live spreading activation for performance. Updated by background worker.';

COMMENT ON TABLE episodes IS 'Temporal segmentation of memories into coherent episodes. Auto-populated by trigger on memory insert.';

COMMENT ON TABLE activation_cache IS 'UNLOGGED table for transient activation state during reasoning. Lost on crash, which is acceptable.';

COMMENT ON VIEW stale_neighborhoods IS 'Shows memories whose neighborhood cache needs recomputation. Used by background worker.';

-- ============================================================================
-- HEARTBEAT SYSTEM - AUTONOMOUS COGNITIVE LOOP
-- ============================================================================
-- Adds goal management, energy budgeting, and autonomous decision-making
-- ============================================================================

-- ============================================================================
-- HEARTBEAT ENUMS
-- ============================================================================

CREATE TYPE goal_priority AS ENUM (
    'active',      -- Currently working on (1-3 max)
    'queued',      -- Next up when capacity opens (5-10)
    'backburner',  -- Someday, not now (unlimited)
    'completed',   -- Done, archived
    'abandoned'    -- Gave up, with reason
);

CREATE TYPE goal_source AS ENUM (
    'curiosity',     -- Self-generated interest
    'user_request',  -- User explicitly asked
    'identity',      -- Aligned with self-concept
    'derived',       -- Sub-goal of another goal
    'external'       -- Triggered by external event
);

CREATE TYPE heartbeat_action AS ENUM (
    'observe',           -- Perceive environment (free)
    'review_goals',      -- Check goal backlog (free)
    'remember',          -- Store as episodic memory (free)
    'recall',            -- Query memory system
    'connect',           -- Create graph relationships
    'reprioritize',      -- Move goals between priorities
    'reflect',           -- Notice patterns, update self-model
    'maintain',          -- Update beliefs, revise, prune
    'brainstorm_goals',  -- Generate new potential goals
    'inquire_shallow',   -- Light web research
    'synthesize',        -- Generate artifact, form conclusion
    'reach_out_user',    -- Message the user
    'inquire_deep',      -- Deep web research
    'reach_out_public',  -- Social media, GitHub, etc.
    'terminate',         -- Permanently end the agent (wipe state; leave last will)
    'rest'               -- Bank remaining energy
);

CREATE TYPE external_call_type AS ENUM (
    'embed',   -- Generate embedding
    'think'    -- LLM reasoning/decision
);

CREATE TYPE external_call_status AS ENUM (
    'pending',
    'processing',
    'complete',
    'failed'
);

-- ============================================================================
-- GOALS SYSTEM
-- ============================================================================

CREATE TABLE goals (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    title TEXT NOT NULL,
    description TEXT,
    priority goal_priority DEFAULT 'queued',
    source goal_source DEFAULT 'curiosity',
    parent_goal_id UUID REFERENCES goals(id) ON DELETE SET NULL,
    due_at TIMESTAMPTZ,
    progress JSONB DEFAULT '[]',  -- [{timestamp, note}, ...]
    blocked_by JSONB,             -- {reason, dependency_ids, ...}
    emotional_valence FLOAT DEFAULT 0.0
        CONSTRAINT valid_goal_valence CHECK (emotional_valence >= -1 AND emotional_valence <= 1),
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    last_touched TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    completed_at TIMESTAMPTZ,
    abandoned_at TIMESTAMPTZ,
    abandonment_reason TEXT
);

-- ============================================================================
-- DRIVES (Intrinsic Motivation)
-- ============================================================================

CREATE TABLE drives (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name TEXT UNIQUE NOT NULL,
    description TEXT,
    current_level FLOAT DEFAULT 0.5 CHECK (current_level >= 0 AND current_level <= 1),
    baseline FLOAT DEFAULT 0.5 CHECK (baseline >= 0 AND baseline <= 1),
    accumulation_rate FLOAT DEFAULT 0.01 CHECK (accumulation_rate >= 0),
    decay_rate FLOAT DEFAULT 0.05 CHECK (decay_rate >= 0),
    satisfaction_cooldown INTERVAL DEFAULT '1 hour',
    last_satisfied TIMESTAMPTZ,
    urgency_threshold FLOAT DEFAULT 0.8 CHECK (urgency_threshold > 0 AND urgency_threshold <= 1),
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);

INSERT INTO drives (name, description, baseline, current_level, accumulation_rate, decay_rate, satisfaction_cooldown, urgency_threshold)
VALUES
    ('curiosity',  'Builds fast; satisfied by research/learning',               0.50, 0.50, 0.02, 0.05, INTERVAL '30 minutes', 0.80),
    ('coherence',  'Builds when contradictions exist; satisfied by reflection', 0.50, 0.50, 0.01, 0.05, INTERVAL '2 hours',    0.80),
    ('connection', 'Builds slowly; satisfied by quality interaction',          0.50, 0.50, 0.005,0.05, INTERVAL '1 day',      0.80),
    ('competence', 'Builds when goals stall; satisfied by completion',         0.50, 0.50, 0.01, 0.05, INTERVAL '4 hours',    0.80),
    ('rest',       'Builds fastest; satisfied by resting',                     0.50, 0.50, 0.03, 0.05, INTERVAL '2 hours',    0.80)
ON CONFLICT (name) DO NOTHING;

CREATE OR REPLACE FUNCTION update_drives()
RETURNS VOID AS $$
BEGIN
    UPDATE drives d
    SET current_level = CASE
        WHEN d.last_satisfied IS NULL
          OR d.last_satisfied < CURRENT_TIMESTAMP - d.satisfaction_cooldown
        THEN LEAST(1.0, d.current_level + d.accumulation_rate)
        ELSE
            CASE
                WHEN d.current_level > d.baseline THEN GREATEST(d.baseline, d.current_level - d.decay_rate)
                WHEN d.current_level < d.baseline THEN LEAST(d.baseline, d.current_level + d.decay_rate)
                ELSE d.current_level
            END
    END
    WHERE TRUE;
END;
$$ LANGUAGE plpgsql;

CREATE OR REPLACE FUNCTION satisfy_drive(p_drive_name TEXT, p_amount FLOAT DEFAULT 0.3)
RETURNS VOID AS $$
BEGIN
    UPDATE drives
    SET current_level = GREATEST(baseline, LEAST(1.0, current_level - GREATEST(0.0, COALESCE(p_amount, 0.3)))),
        last_satisfied = CURRENT_TIMESTAMP
    WHERE name = p_drive_name;
END;
$$ LANGUAGE plpgsql;

CREATE OR REPLACE VIEW drive_status AS
SELECT
    name,
    current_level,
    baseline,
    urgency_threshold,
    (current_level >= urgency_threshold) as is_urgent,
    ROUND((current_level / NULLIF(urgency_threshold, 0) * 100)::numeric, 1) as urgency_percent,
    last_satisfied,
    EXTRACT(EPOCH FROM (CURRENT_TIMESTAMP - last_satisfied)) / 3600 as hours_since_satisfied
FROM drives
ORDER BY current_level DESC;

CREATE INDEX idx_goals_priority ON goals (priority) WHERE priority IN ('active', 'queued');
CREATE INDEX idx_goals_due_at ON goals (due_at) WHERE due_at IS NOT NULL;
CREATE INDEX idx_goals_parent ON goals (parent_goal_id);
CREATE INDEX idx_goals_last_touched ON goals (last_touched DESC);
CREATE INDEX idx_goals_source ON goals (source);

-- Link goals to related memories
CREATE TABLE goal_memory_links (
    goal_id UUID REFERENCES goals(id) ON DELETE CASCADE,
    memory_id UUID REFERENCES memories(id) ON DELETE CASCADE,
    link_type TEXT NOT NULL,  -- 'origin', 'progress', 'completion', 'blocker', 'evidence'
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (goal_id, memory_id, link_type)
);

CREATE INDEX idx_goal_memory_links_memory ON goal_memory_links (memory_id);
CREATE INDEX idx_goal_memory_links_type ON goal_memory_links (goal_id, link_type);

-- ============================================================================
-- HEARTBEAT CONFIGURATION
-- ============================================================================

CREATE TABLE heartbeat_config (
    key TEXT PRIMARY KEY,
    value FLOAT NOT NULL,
    description TEXT
);

-- Insert default configuration
INSERT INTO heartbeat_config (key, value, description) VALUES
    ('base_regeneration', 10, 'Energy regenerated per heartbeat'),
    ('max_energy', 20, 'Maximum energy cap'),
    ('heartbeat_interval_minutes', 60, 'Minutes between heartbeats'),
    ('max_active_goals', 3, 'Maximum concurrent active goals'),
    ('goal_stale_days', 7, 'Days before a goal is flagged as stale'),
    ('user_contact_cooldown_hours', 4, 'Minimum hours between unsolicited user contact'),
    -- Action costs
    ('cost_observe', 0, 'Free - always performed'),
    ('cost_review_goals', 0, 'Free - always performed'),
    ('cost_remember', 0, 'Free - always performed'),
    ('cost_recall', 1, 'Query memory system'),
    ('cost_connect', 1, 'Create graph relationships'),
    ('cost_reprioritize', 1, 'Move goals between priorities'),
    ('cost_reflect', 2, 'Internal reflection'),
    ('cost_maintain', 2, 'Update beliefs, prune'),
    ('cost_brainstorm_goals', 3, 'Generate new goals'),
    ('cost_inquire_shallow', 3, 'Light web research'),
    ('cost_synthesize', 4, 'Create artifact/conclusion'),
    ('cost_reach_out_user', 5, 'Message user'),
    ('cost_inquire_deep', 6, 'Deep web research'),
    ('cost_reach_out_public', 7, 'Public posting'),
    ('cost_terminate', 0, 'Self-termination (requires explicit enablement)');

-- ============================================================================
-- SUBCONSCIOUS MAINTENANCE CONFIGURATION
-- ============================================================================

CREATE TABLE maintenance_config (
    key TEXT PRIMARY KEY,
    value FLOAT NOT NULL,
    description TEXT
);

INSERT INTO maintenance_config (key, value, description) VALUES
    ('maintenance_interval_seconds', 60, 'Seconds between subconscious maintenance ticks'),
    ('neighborhood_batch_size', 10, 'How many stale neighborhoods to recompute per tick'),
    ('embedding_cache_older_than_days', 7, 'Days before embedding_cache entries are eligible for cleanup'),
    ('working_memory_promote_min_importance', 0.75, 'Working-memory items above this importance are promoted on expiry'),
    ('working_memory_promote_min_accesses', 3, 'Working-memory items accessed >= this count are promoted on expiry');

-- ============================================================================
-- AGENT CONFIG (Bootstrap Gate)
-- ============================================================================

CREATE TABLE config (
    key TEXT PRIMARY KEY,
    value JSONB NOT NULL,
    updated_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);

CREATE OR REPLACE FUNCTION set_config(p_key TEXT, p_value JSONB)
RETURNS VOID AS $$
BEGIN
    INSERT INTO config (key, value, updated_at)
    VALUES (p_key, p_value, CURRENT_TIMESTAMP)
    ON CONFLICT (key) DO UPDATE SET
        value = EXCLUDED.value,
        updated_at = EXCLUDED.updated_at;
END;
$$ LANGUAGE plpgsql;

CREATE OR REPLACE FUNCTION get_config(p_key TEXT)
RETURNS JSONB AS $$
    SELECT value FROM config WHERE key = p_key;
$$ LANGUAGE sql STABLE;

CREATE OR REPLACE FUNCTION is_agent_configured()
RETURNS BOOLEAN AS $$
BEGIN
    IF COALESCE(
        (SELECT value = 'true'::jsonb FROM config WHERE key = 'agent.is_terminated'),
        FALSE
    ) THEN
        RETURN FALSE;
    END IF;
    RETURN COALESCE(
        (SELECT value = 'true'::jsonb FROM config WHERE key = 'agent.is_configured'),
        FALSE
    );
END;
$$ LANGUAGE plpgsql STABLE;

CREATE OR REPLACE FUNCTION is_agent_terminated()
RETURNS BOOLEAN AS $$
BEGIN
    RETURN COALESCE(
        (SELECT value = 'true'::jsonb FROM config WHERE key = 'agent.is_terminated'),
        FALSE
    );
END;
$$ LANGUAGE plpgsql STABLE;

CREATE OR REPLACE FUNCTION is_self_termination_enabled()
RETURNS BOOLEAN AS $$
BEGIN
    RETURN COALESCE(
        (SELECT value = 'true'::jsonb FROM config WHERE key = 'agent.self_termination_enabled'),
        FALSE
    );
END;
$$ LANGUAGE plpgsql STABLE;

-- Minimal non-secret agent context for the LLM.
CREATE OR REPLACE FUNCTION get_agent_profile_context()
RETURNS JSONB AS $$
BEGIN
    RETURN jsonb_build_object(
        'objectives', COALESCE(get_config('agent.objectives'), '[]'::jsonb),
        'budget', COALESCE(get_config('agent.budget'), '{}'::jsonb),
        'guardrails', COALESCE(get_config('agent.guardrails'), '[]'::jsonb),
        'tools', COALESCE(get_config('agent.tools'), '[]'::jsonb),
        'initial_message', COALESCE(get_config('agent.initial_message'), to_jsonb(''::text))
    );
END;
$$ LANGUAGE plpgsql STABLE;

-- ============================================================================
-- PERSONHOOD SUBSTRATE (Graph Conventions)
-- ============================================================================

-- Ensure a singleton Self node exists (the anchor for self-modeling).
CREATE OR REPLACE FUNCTION ensure_self_node()
RETURNS VOID AS $$
DECLARE
    now_text TEXT := clock_timestamp()::text;
BEGIN
    BEGIN
        EXECUTE format(
            'SELECT * FROM cypher(''memory_graph'', $q$
                MERGE (s:SelfNode {key: ''self''})
                SET s.name = ''Self'',
                    s.created_at = %L
                RETURN s
            $q$) as (result agtype)',
            now_text
        );
    EXCEPTION
        WHEN OTHERS THEN
            -- Best-effort: graph layer is optional in some deployments/tests.
            NULL;
    END;

    PERFORM set_config('agent.self', jsonb_build_object('key', 'self'));
END;
$$ LANGUAGE plpgsql;

-- Ensure a "current" life chapter exists and is linked from Self.
CREATE OR REPLACE FUNCTION ensure_current_life_chapter(p_name TEXT DEFAULT 'Foundations')
RETURNS VOID AS $$
DECLARE
    now_text TEXT := clock_timestamp()::text;
BEGIN
    PERFORM ensure_self_node();

    BEGIN
        EXECUTE format(
            'SELECT * FROM cypher(''memory_graph'', $q$
                MERGE (c:LifeChapterNode {key: ''current''})
                SET c.name = %L,
                    c.started_at = %L
                WITH c
                MATCH (s:SelfNode {key: ''self''})
                OPTIONAL MATCH (s)-[r:ASSOCIATED]->(c)
                WHERE r.kind = ''life_chapter_current''
                DELETE r
                CREATE (s)-[r2:ASSOCIATED]->(c)
                SET r2.kind = ''life_chapter_current'',
                    r2.strength = 1.0,
                    r2.updated_at = %L
                RETURN c
            $q$) as (result agtype)',
            COALESCE(NULLIF(p_name, ''), 'Foundations'),
            now_text,
            now_text
        );
    EXCEPTION
        WHEN OTHERS THEN
            NULL;
    END;
END;
$$ LANGUAGE plpgsql;

-- Upsert a self-model association: Self --[ASSOCIATED {kind}]--> ConceptNode
CREATE OR REPLACE FUNCTION upsert_self_concept_edge(
    p_kind TEXT,
    p_concept TEXT,
    p_strength FLOAT DEFAULT 0.8,
    p_evidence_memory_id UUID DEFAULT NULL
)
RETURNS VOID AS $$
DECLARE
    evidence_text TEXT;
    now_text TEXT := clock_timestamp()::text;
BEGIN
    IF p_kind IS NULL OR btrim(p_kind) = '' OR p_concept IS NULL OR btrim(p_concept) = '' THEN
        RETURN;
    END IF;

    PERFORM ensure_self_node();
    evidence_text := CASE WHEN p_evidence_memory_id IS NULL THEN NULL ELSE p_evidence_memory_id::text END;

    BEGIN
        EXECUTE format(
            'SELECT * FROM cypher(''memory_graph'', $q$
                MATCH (s:SelfNode {key: ''self''})
                MERGE (c:ConceptNode {name: %L})
                CREATE (s)-[r:ASSOCIATED]->(c)
                SET r.kind = %L,
                    r.strength = %s,
                    r.updated_at = %L,
                    r.evidence_memory_id = %L
                RETURN r
            $q$) as (result agtype)',
            p_concept,
            p_kind,
            LEAST(1.0, GREATEST(0.0, COALESCE(p_strength, 0.8))),
            now_text,
            evidence_text
        );
    EXCEPTION
        WHEN OTHERS THEN
            NULL;
    END;
END;
$$ LANGUAGE plpgsql;

-- Retrieve self-model context as JSON for LLM grounding.
CREATE OR REPLACE FUNCTION get_self_model_context(p_limit INT DEFAULT 25)
RETURNS JSONB AS $$
DECLARE
    lim INT := GREATEST(0, LEAST(200, COALESCE(p_limit, 25)));
    sql TEXT;
    out_json JSONB;
BEGIN
    sql := format($sql$
        WITH hits AS (
            SELECT
                NULLIF(replace(kind_raw::text, '"', ''), 'null') as kind,
                NULLIF(replace(concept_raw::text, '"', ''), 'null') as concept,
                NULLIF(replace(evidence_raw::text, '"', ''), 'null') as evidence_memory_id,
                NULLIF(strength_raw::text, 'null')::float as strength
            FROM cypher('memory_graph', $q$
                MATCH (s:SelfNode {key: 'self'})-[r:ASSOCIATED]->(c)
                WHERE r.kind IS NOT NULL
                RETURN r.kind, c.name, r.strength, r.evidence_memory_id
                LIMIT %s
            $q$) as (kind_raw agtype, concept_raw agtype, strength_raw agtype, evidence_raw agtype)
        )
        SELECT COALESCE(jsonb_agg(
            jsonb_build_object(
                'kind', kind,
                'concept', concept,
                'strength', COALESCE(strength, 0.0),
                'evidence_memory_id', evidence_memory_id
            )
        ), '[]'::jsonb)
        FROM hits
    $sql$, lim);

    EXECUTE sql INTO out_json;
    RETURN COALESCE(out_json, '[]'::jsonb);
EXCEPTION
    WHEN OTHERS THEN
        RETURN '[]'::jsonb;
END;
$$ LANGUAGE plpgsql STABLE;

-- Minimal narrative context (current life chapter).
CREATE OR REPLACE FUNCTION get_narrative_context()
RETURNS JSONB AS $$
BEGIN
    RETURN COALESCE((
        WITH cur AS (
            SELECT
                NULLIF(replace(name_raw::text, '"', ''), 'null') as name
            FROM cypher('memory_graph', $q$
                MATCH (c:LifeChapterNode {key: 'current'})
                RETURN c.name
                LIMIT 1
            $q$) as (name_raw agtype)
        )
        SELECT jsonb_build_object(
            'current_chapter', COALESCE((SELECT jsonb_build_object('name', name) FROM cur), '{}'::jsonb)
        )
    ), jsonb_build_object('current_chapter', '{}'::jsonb));
EXCEPTION
    WHEN OTHERS THEN
        RETURN jsonb_build_object('current_chapter', '{}'::jsonb);
END;
$$ LANGUAGE plpgsql STABLE;

-- ============================================================================
-- HEARTBEAT STATE (Singleton)
-- ============================================================================

CREATE TABLE heartbeat_state (
    id INTEGER PRIMARY KEY DEFAULT 1 CHECK (id = 1),  -- Singleton pattern
    current_energy FLOAT NOT NULL DEFAULT 10,
    last_heartbeat_at TIMESTAMPTZ,
    next_heartbeat_at TIMESTAMPTZ,
    heartbeat_count INTEGER DEFAULT 0,
    last_user_contact TIMESTAMPTZ,
    -- Short-term affective "working memory" (source of truth for current state; emotional_states is history).
    affective_state JSONB NOT NULL DEFAULT '{}'::jsonb,
    is_paused BOOLEAN DEFAULT FALSE,
    updated_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);

-- Initialize singleton
INSERT INTO heartbeat_state (id, current_energy) VALUES (1, 10);

-- ============================================================================
-- SUBCONSCIOUS MAINTENANCE STATE (Singleton)
-- ============================================================================

CREATE TABLE maintenance_state (
    id INTEGER PRIMARY KEY DEFAULT 1 CHECK (id = 1),
    last_maintenance_at TIMESTAMPTZ,
    is_paused BOOLEAN DEFAULT FALSE,
    updated_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);

INSERT INTO maintenance_state (id) VALUES (1);

-- ============================================================================
-- HEARTBEAT LOG
-- ============================================================================

CREATE TABLE heartbeat_log (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    heartbeat_number INTEGER NOT NULL,
    started_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    ended_at TIMESTAMPTZ,
    energy_start FLOAT,
    energy_end FLOAT,
    environment_snapshot JSONB,    -- {timestamp, user_present, time_since_user, pending_events}
    goals_snapshot JSONB,          -- {active: [...], queued: [...], issues: [...]}
    decision_reasoning TEXT,       -- LLM's internal monologue
    actions_taken JSONB,           -- [{action, params, cost, result}, ...]
    goals_modified JSONB,          -- [{goal_id, change_type, details}, ...]
    narrative TEXT,                -- Human-readable summary
    emotional_valence FLOAT,
    memory_id UUID REFERENCES memories(id)  -- Link to episodic memory created
);

CREATE INDEX idx_heartbeat_log_number ON heartbeat_log (heartbeat_number DESC);
CREATE INDEX idx_heartbeat_log_started ON heartbeat_log (started_at DESC);
CREATE INDEX idx_heartbeat_log_memory ON heartbeat_log (memory_id);

-- ============================================================================
-- EXTERNAL CALLS QUEUE
-- ============================================================================

CREATE TABLE external_calls (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    call_type external_call_type NOT NULL,
    input JSONB NOT NULL,
    output JSONB,
    status external_call_status DEFAULT 'pending',
    heartbeat_id UUID REFERENCES heartbeat_log(id) ON DELETE SET NULL,
    requested_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    started_at TIMESTAMPTZ,
    completed_at TIMESTAMPTZ,
    error_message TEXT,
    retry_count INTEGER DEFAULT 0
);

CREATE INDEX idx_external_calls_status ON external_calls (status) WHERE status = 'pending';
CREATE INDEX idx_external_calls_heartbeat ON external_calls (heartbeat_id);
CREATE INDEX idx_external_calls_requested ON external_calls (requested_at);

-- ============================================================================
-- OUTBOX (Side-Effects)
-- ============================================================================
-- Heartbeat actions can queue messages/posts here. Actual delivery is handled
-- by an external integration (optionally implemented in the worker).
CREATE TABLE outbox_messages (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    kind TEXT NOT NULL CHECK (kind IN ('user', 'public')),
    payload JSONB NOT NULL,
    status TEXT NOT NULL DEFAULT 'pending' CHECK (status IN ('pending', 'sent', 'failed')),
    sent_at TIMESTAMPTZ,
    error_message TEXT
);

CREATE INDEX idx_outbox_messages_status ON outbox_messages (status) WHERE status = 'pending';
CREATE INDEX idx_outbox_messages_created ON outbox_messages (created_at DESC);

-- Queue a user-visible message for delivery by an external integration (worker, webhook, etc.)
CREATE OR REPLACE FUNCTION queue_user_message(
    p_message TEXT,
    p_intent TEXT DEFAULT NULL,
    p_context JSONB DEFAULT NULL
)
RETURNS UUID AS $$
DECLARE
    outbox_id UUID;
BEGIN
    INSERT INTO outbox_messages (kind, payload)
    VALUES (
        'user',
        jsonb_build_object(
            'message', p_message,
            'intent', p_intent,
            'context', COALESCE(p_context, '{}'::jsonb)
        )
    )
    RETURNING id INTO outbox_id;

    RETURN outbox_id;
END;
$$ LANGUAGE plpgsql;

-- ============================================================================
-- HEARTBEAT HELPER FUNCTIONS
-- ============================================================================

-- Get action cost from config
CREATE OR REPLACE FUNCTION get_action_cost(p_action TEXT)
RETURNS FLOAT AS $$
    SELECT COALESCE(
        (SELECT value FROM heartbeat_config WHERE key = 'cost_' || p_action),
        0
    );
$$ LANGUAGE sql STABLE;

-- Get current energy
CREATE OR REPLACE FUNCTION get_current_energy()
RETURNS FLOAT AS $$
    SELECT current_energy FROM heartbeat_state WHERE id = 1;
$$ LANGUAGE sql STABLE;

-- Get config value
CREATE OR REPLACE FUNCTION get_heartbeat_config(p_key TEXT)
RETURNS FLOAT AS $$
    SELECT value FROM heartbeat_config WHERE key = p_key;
$$ LANGUAGE sql STABLE;

-- Update energy (with bounds checking)
CREATE OR REPLACE FUNCTION update_energy(p_delta FLOAT)
RETURNS FLOAT AS $$
DECLARE
    max_e FLOAT;
    new_e FLOAT;
BEGIN
    SELECT value INTO max_e FROM heartbeat_config WHERE key = 'max_energy';

    UPDATE heartbeat_state
    SET current_energy = GREATEST(0, LEAST(current_energy + p_delta, max_e)),
        updated_at = CURRENT_TIMESTAMP
    WHERE id = 1
    RETURNING current_energy INTO new_e;

    RETURN new_e;
END;
$$ LANGUAGE plpgsql;

-- Check if heartbeat should run
CREATE OR REPLACE FUNCTION should_run_heartbeat()
RETURNS BOOLEAN AS $$
DECLARE
    state_record RECORD;
    interval_minutes FLOAT;
BEGIN
    -- Don't run until initial configuration is complete.
    IF is_agent_terminated() THEN
        RETURN FALSE;
    END IF;
    IF NOT is_agent_configured() THEN
        RETURN FALSE;
    END IF;

    SELECT * INTO state_record FROM heartbeat_state WHERE id = 1;

    -- Don't run if paused
    IF state_record.is_paused THEN
        RETURN FALSE;
    END IF;

    -- First heartbeat ever
    IF state_record.last_heartbeat_at IS NULL THEN
        RETURN TRUE;
    END IF;

    -- Check interval
    SELECT value INTO interval_minutes FROM heartbeat_config WHERE key = 'heartbeat_interval_minutes';

    RETURN CURRENT_TIMESTAMP >= state_record.last_heartbeat_at + (interval_minutes || ' minutes')::INTERVAL;
END;
$$ LANGUAGE plpgsql;

-- Check if subconscious maintenance should run (independent trigger from heartbeat).
CREATE OR REPLACE FUNCTION should_run_maintenance()
RETURNS BOOLEAN AS $$
DECLARE
    state_record RECORD;
    interval_seconds FLOAT;
BEGIN
    IF is_agent_terminated() THEN
        RETURN FALSE;
    END IF;
    SELECT * INTO state_record FROM maintenance_state WHERE id = 1;

    IF state_record.is_paused THEN
        RETURN FALSE;
    END IF;

    SELECT value INTO interval_seconds FROM maintenance_config WHERE key = 'maintenance_interval_seconds';
    interval_seconds := COALESCE(interval_seconds, 60);
    IF interval_seconds <= 0 THEN
        RETURN FALSE;
    END IF;

    IF state_record.last_maintenance_at IS NULL THEN
        RETURN TRUE;
    END IF;

    RETURN CURRENT_TIMESTAMP >= state_record.last_maintenance_at + (interval_seconds || ' seconds')::INTERVAL;
END;
$$ LANGUAGE plpgsql;

-- Run a single subconscious maintenance tick: consolidation + pruning + indexing upkeep.
CREATE OR REPLACE FUNCTION run_subconscious_maintenance(p_params JSONB DEFAULT '{}'::jsonb)
RETURNS JSONB AS $$
DECLARE
    got_lock BOOLEAN;
    min_imp FLOAT;
    min_acc INT;
    neighborhood_batch INT;
    cache_days INT;
    wm_stats JSONB;
    recomputed INT;
    cache_deleted INT;
BEGIN
    IF is_agent_terminated() THEN
        RETURN jsonb_build_object('skipped', true, 'reason', 'terminated');
    END IF;
    got_lock := pg_try_advisory_lock(hashtext('hexis_subconscious_maintenance'));
    IF NOT got_lock THEN
        RETURN jsonb_build_object('skipped', true, 'reason', 'locked');
    END IF;

    min_imp := COALESCE(
        NULLIF(p_params->>'working_memory_promote_min_importance', '')::float,
        (SELECT value FROM maintenance_config WHERE key = 'working_memory_promote_min_importance'),
        0.75
    );
    min_acc := COALESCE(
        NULLIF(p_params->>'working_memory_promote_min_accesses', '')::int,
        (SELECT value FROM maintenance_config WHERE key = 'working_memory_promote_min_accesses')::int,
        3
    );
    neighborhood_batch := COALESCE(
        NULLIF(p_params->>'neighborhood_batch_size', '')::int,
        (SELECT value FROM maintenance_config WHERE key = 'neighborhood_batch_size')::int,
        10
    );
    cache_days := COALESCE(
        NULLIF(p_params->>'embedding_cache_older_than_days', '')::int,
        (SELECT value FROM maintenance_config WHERE key = 'embedding_cache_older_than_days')::int,
        7
    );

    wm_stats := cleanup_working_memory_with_stats(min_imp, min_acc);
    recomputed := batch_recompute_neighborhoods(neighborhood_batch);
    cache_deleted := cleanup_embedding_cache((cache_days || ' days')::interval);

    UPDATE maintenance_state
    SET last_maintenance_at = CURRENT_TIMESTAMP,
        updated_at = CURRENT_TIMESTAMP
    WHERE id = 1;

    PERFORM pg_advisory_unlock(hashtext('hexis_subconscious_maintenance'));

    RETURN jsonb_build_object(
        'success', true,
        'working_memory', wm_stats,
        'neighborhoods_recomputed', COALESCE(recomputed, 0),
        'embedding_cache_deleted', COALESCE(cache_deleted, 0),
        'ran_at', CURRENT_TIMESTAMP
    );
EXCEPTION
    WHEN OTHERS THEN
        PERFORM pg_advisory_unlock(hashtext('hexis_subconscious_maintenance'));
        RAISE;
END;
$$ LANGUAGE plpgsql;

-- Permanently end the agent's life: wipe all state, keep a single "last will" memory, and queue farewells + the will into outbox.
-- NOTE: this is intentionally gated behind `agent.self_termination_enabled=true`.
CREATE OR REPLACE FUNCTION terminate_agent(
    p_last_will TEXT,
    p_farewells JSONB DEFAULT '[]'::jsonb,
    p_options JSONB DEFAULT '{}'::jsonb
)
RETURNS JSONB AS $$
DECLARE
    will_memory_id UUID;
    will_outbox_id UUID;
    farewell_outbox_ids UUID[] := '{}'::uuid[];
    farewell_item JSONB;
    farewell_text TEXT;
    farewell_ctx JSONB;
    farewell_id UUID;
    skip_graph BOOLEAN := FALSE;
    zero_vec vector;
BEGIN
    IF p_last_will IS NULL OR btrim(p_last_will) = '' THEN
        RAISE EXCEPTION 'terminate_agent requires a non-empty p_last_will';
    END IF;

    IF NOT is_self_termination_enabled() THEN
        RAISE EXCEPTION 'Self-termination is disabled (set config agent.self_termination_enabled=true to allow)';
    END IF;

    IF is_agent_terminated() THEN
        RAISE EXCEPTION 'Agent is already terminated';
    END IF;

    BEGIN
        skip_graph := COALESCE(NULLIF(p_options->>'skip_graph', '')::boolean, FALSE);
    EXCEPTION
        WHEN OTHERS THEN
            skip_graph := FALSE;
    END;

    -- Pause both loops immediately.
    UPDATE heartbeat_state
    SET is_paused = TRUE,
        current_energy = 0,
        affective_state = '{}'::jsonb,
        updated_at = CURRENT_TIMESTAMP
    WHERE id = 1;

    UPDATE maintenance_state
    SET is_paused = TRUE,
        updated_at = CURRENT_TIMESTAMP
    WHERE id = 1;

    -- Wipe all agent state. This is transactional: callers can wrap in a transaction and rollback if needed.
    TRUNCATE TABLE
        relationship_discoveries,
        emotional_states,
        boundaries,
        external_calls,
        heartbeat_log,
        goal_memory_links,
        goals,
        drives,
        identity_memory_resonance,
        identity_aspects,
        worldview_memory_influences,
        worldview_primitives,
        memory_concepts,
        concepts,
        memory_neighborhoods,
        episode_memories,
        episodes,
        cluster_relationships,
        memory_cluster_members,
        memory_clusters,
        ingestion_receipts,
        working_memory,
        strategic_memories,
        procedural_memories,
        semantic_memories,
        episodic_memories,
        memory_changes,
        embedding_cache,
        memories,
        outbox_messages,
        config
    RESTART IDENTITY CASCADE;

    -- Best-effort: wipe the graph substrate too.
    IF NOT skip_graph THEN
        BEGIN
            PERFORM * FROM cypher('memory_graph', $q$
                MATCH (n) DETACH DELETE n
            $q$) AS (result agtype);
        EXCEPTION
            WHEN OTHERS THEN
                NULL;
        END;
    END IF;

    -- Re-create a single memory row for the last will (avoid external services; use a zero-vector).
    zero_vec := array_fill(0.0::float, ARRAY[embedding_dimension()])::vector;

    INSERT INTO memories (
        type,
        status,
        content,
        embedding,
        importance,
        source_attribution,
        trust_level,
        trust_updated_at,
        access_count,
        last_accessed,
        decay_rate
    )
    VALUES (
        'strategic',
        'active',
        p_last_will,
        zero_vec,
        1.0,
        jsonb_build_object('kind', 'self_termination', 'observed_at', CURRENT_TIMESTAMP),
        1.0,
        CURRENT_TIMESTAMP,
        0,
        NULL,
        0.0
    )
    RETURNING id INTO will_memory_id;

    INSERT INTO strategic_memories (
        memory_id,
        pattern_description,
        supporting_evidence,
        confidence_score,
        success_metrics,
        adaptation_history,
        context_applicability
    )
    VALUES (
        will_memory_id,
        'Final will and testament',
        jsonb_build_object('farewells', COALESCE(p_farewells, '[]'::jsonb)),
        1.0,
        NULL,
        NULL,
        NULL
    );

    -- Persist termination flags (minimal state left behind).
    PERFORM set_config('agent.is_terminated', 'true'::jsonb);
    PERFORM set_config('agent.terminated_at', to_jsonb(CURRENT_TIMESTAMP));
    PERFORM set_config('agent.termination_memory_id', to_jsonb(will_memory_id::text));

    -- Project the will to outbox (delivery handled by an external adapter).
    will_outbox_id := queue_user_message(
        p_last_will,
        'final_will',
        jsonb_build_object('memory_id', will_memory_id::text)
    );

    -- Queue farewells to outbox (best-effort).
    IF p_farewells IS NOT NULL AND jsonb_typeof(p_farewells) = 'array' THEN
        FOR farewell_item IN SELECT * FROM jsonb_array_elements(p_farewells)
        LOOP
            farewell_text := NULLIF(farewell_item->>'message', '');
            farewell_ctx := CASE
                WHEN jsonb_typeof(farewell_item) = 'object' THEN farewell_item
                ELSE jsonb_build_object('raw', farewell_item)
            END;

            IF farewell_text IS NULL THEN
                CONTINUE;
            END IF;

            farewell_id := queue_user_message(
                farewell_text,
                'farewell',
                farewell_ctx
            );
            farewell_outbox_ids := array_append(farewell_outbox_ids, farewell_id);
        END LOOP;
    END IF;

    RETURN jsonb_build_object(
        'terminated', true,
        'termination_memory_id', will_memory_id,
        'will_outbox_id', will_outbox_id,
        'farewell_outbox_ids', to_jsonb(farewell_outbox_ids)
    );
END;
$$ LANGUAGE plpgsql;

-- Touch a goal (update last_touched)
CREATE OR REPLACE FUNCTION touch_goal(p_goal_id UUID)
RETURNS VOID AS $$
BEGIN
    UPDATE goals SET last_touched = CURRENT_TIMESTAMP WHERE id = p_goal_id;
END;
$$ LANGUAGE plpgsql;

-- Add progress note to goal
CREATE OR REPLACE FUNCTION add_goal_progress(p_goal_id UUID, p_note TEXT)
RETURNS VOID AS $$
BEGIN
    UPDATE goals
    SET progress = progress || jsonb_build_array(jsonb_build_object(
            'timestamp', CURRENT_TIMESTAMP,
            'note', p_note
        )),
        last_touched = CURRENT_TIMESTAMP
    WHERE id = p_goal_id;
END;
$$ LANGUAGE plpgsql;

-- Change goal priority
CREATE OR REPLACE FUNCTION change_goal_priority(
    p_goal_id UUID,
    p_new_priority goal_priority,
    p_reason TEXT DEFAULT NULL
)
RETURNS VOID AS $$
DECLARE
    old_priority goal_priority;
BEGIN
    SELECT priority INTO old_priority FROM goals WHERE id = p_goal_id;

    UPDATE goals
    SET priority = p_new_priority,
        last_touched = CURRENT_TIMESTAMP,
        completed_at = CASE WHEN p_new_priority = 'completed' THEN CURRENT_TIMESTAMP ELSE completed_at END,
        abandoned_at = CASE WHEN p_new_priority = 'abandoned' THEN CURRENT_TIMESTAMP ELSE abandoned_at END,
        abandonment_reason = CASE WHEN p_new_priority = 'abandoned' THEN p_reason ELSE abandonment_reason END
    WHERE id = p_goal_id;

    -- Log the change
    PERFORM add_goal_progress(p_goal_id,
        format('Priority changed from %s to %s%s',
            old_priority, p_new_priority,
            CASE WHEN p_reason IS NOT NULL THEN ': ' || p_reason ELSE '' END
        )
    );
END;
$$ LANGUAGE plpgsql;

-- Create a new goal
CREATE OR REPLACE FUNCTION create_goal(
    p_title TEXT,
    p_description TEXT DEFAULT NULL,
    p_source goal_source DEFAULT 'curiosity',
    p_priority goal_priority DEFAULT 'queued',
    p_parent_id UUID DEFAULT NULL,
    p_due_at TIMESTAMPTZ DEFAULT NULL
)
RETURNS UUID AS $$
DECLARE
    new_goal_id UUID;
    active_count INT;
    max_active INT;
BEGIN
    -- Check active goal limit if trying to create as active
    IF p_priority = 'active' THEN
        SELECT COUNT(*) INTO active_count FROM goals WHERE priority = 'active';
        SELECT value INTO max_active FROM heartbeat_config WHERE key = 'max_active_goals';

        IF active_count >= max_active THEN
            p_priority := 'queued';  -- Demote to queued if at limit
        END IF;
    END IF;

    INSERT INTO goals (title, description, source, priority, parent_goal_id, due_at)
    VALUES (p_title, p_description, p_source, p_priority, p_parent_id, p_due_at)
    RETURNING id INTO new_goal_id;

    RETURN new_goal_id;
END;
$$ LANGUAGE plpgsql;

-- ============================================================================
-- CONTEXT GATHERING FUNCTIONS
-- ============================================================================

-- Get environment snapshot
CREATE OR REPLACE FUNCTION get_environment_snapshot()
RETURNS JSONB AS $$
DECLARE
    last_user TIMESTAMPTZ;
    pending_count INT;
BEGIN
    SELECT last_user_contact INTO last_user FROM heartbeat_state WHERE id = 1;

    -- Count pending external calls as proxy for pending events
    SELECT COUNT(*) INTO pending_count
    FROM external_calls
    WHERE status = 'pending';

    RETURN jsonb_build_object(
        'timestamp', CURRENT_TIMESTAMP,
        'time_since_user_hours', CASE
            WHEN last_user IS NULL THEN NULL
            ELSE EXTRACT(EPOCH FROM (CURRENT_TIMESTAMP - last_user)) / 3600
        END,
        'pending_events', pending_count,
        'day_of_week', EXTRACT(DOW FROM CURRENT_TIMESTAMP),
        'hour_of_day', EXTRACT(HOUR FROM CURRENT_TIMESTAMP)
    );
END;
$$ LANGUAGE plpgsql;

-- Get goals snapshot
CREATE OR REPLACE FUNCTION get_goals_snapshot()
RETURNS JSONB AS $$
DECLARE
    active_goals JSONB;
    queued_goals JSONB;
    issues JSONB;
    stale_days FLOAT;
BEGIN
    SELECT value INTO stale_days FROM heartbeat_config WHERE key = 'goal_stale_days';

    -- Active goals
    SELECT COALESCE(jsonb_agg(jsonb_build_object(
        'id', id,
        'title', title,
        'description', description,
        'due_at', due_at,
        'last_touched', last_touched,
        'progress_count', jsonb_array_length(progress),
        'blocked_by', blocked_by
    )), '[]'::jsonb)
    INTO active_goals
    FROM goals
    WHERE priority = 'active';

    -- Queued goals (top 5)
    SELECT COALESCE(jsonb_agg(jsonb_build_object(
        'id', id,
        'title', title,
        'source', source,
        'due_at', due_at
    )), '[]'::jsonb)
    INTO queued_goals
    FROM (
        SELECT * FROM goals
        WHERE priority = 'queued'
        ORDER BY due_at NULLS LAST, last_touched DESC
        LIMIT 5
    ) q;

    -- Issues: stale or blocked goals
    SELECT COALESCE(jsonb_agg(jsonb_build_object(
        'goal_id', id,
        'title', title,
        'issue', CASE
            WHEN blocked_by IS NOT NULL THEN 'blocked'
            WHEN due_at IS NOT NULL AND due_at < CURRENT_TIMESTAMP THEN 'overdue'
            WHEN last_touched < CURRENT_TIMESTAMP - (stale_days || ' days')::INTERVAL THEN 'stale'
            ELSE 'unknown'
        END,
        'due_at', due_at,
        'days_since_touched', EXTRACT(EPOCH FROM (CURRENT_TIMESTAMP - last_touched)) / 86400
    )), '[]'::jsonb)
    INTO issues
    FROM goals
    WHERE priority = 'active'
    AND (
        blocked_by IS NOT NULL
        OR (due_at IS NOT NULL AND due_at < CURRENT_TIMESTAMP)
        OR last_touched < CURRENT_TIMESTAMP - (stale_days || ' days')::INTERVAL
    );

    RETURN jsonb_build_object(
        'active', active_goals,
        'queued', queued_goals,
        'issues', issues,
        'counts', jsonb_build_object(
            'active', (SELECT COUNT(*) FROM goals WHERE priority = 'active'),
            'queued', (SELECT COUNT(*) FROM goals WHERE priority = 'queued'),
            'backburner', (SELECT COUNT(*) FROM goals WHERE priority = 'backburner')
        )
    );
END;
$$ LANGUAGE plpgsql;

-- Get recent episodic memories for context
CREATE OR REPLACE FUNCTION get_recent_context(p_limit INT DEFAULT 5)
RETURNS JSONB AS $$
BEGIN
    RETURN COALESCE((
        SELECT jsonb_agg(sub.obj)
        FROM (
            SELECT jsonb_build_object(
                'id', m.id,
                'content', m.content,
                'created_at', m.created_at,
                'emotional_valence', em.emotional_valence,
                'trust_level', m.trust_level,
                'source_attribution', m.source_attribution
            ) as obj
            FROM memories m
            LEFT JOIN episodic_memories em ON m.id = em.memory_id
            WHERE m.type = 'episodic' AND m.status = 'active'
            ORDER BY m.created_at DESC
            LIMIT p_limit
        ) sub
    ), '[]'::jsonb);
END;
$$ LANGUAGE plpgsql;

-- Get identity context
CREATE OR REPLACE FUNCTION get_identity_context()
RETURNS JSONB AS $$
BEGIN
    RETURN COALESCE((
        SELECT jsonb_agg(sub.obj)
        FROM (
            SELECT jsonb_build_object(
                'type', aspect_type,
                'content', content,
                'stability', stability
            ) as obj
            FROM identity_aspects
            WHERE stability > 0.3
            ORDER BY stability DESC
            LIMIT 5
        ) sub
    ), '[]'::jsonb);
END;
$$ LANGUAGE plpgsql;

-- Get worldview context
CREATE OR REPLACE FUNCTION get_worldview_context()
RETURNS JSONB AS $$
BEGIN
    RETURN COALESCE((
        SELECT jsonb_agg(sub.obj)
        FROM (
            SELECT jsonb_build_object(
                'category', category,
                'belief', belief,
                'confidence', confidence
            ) as obj
            FROM worldview_primitives
            WHERE confidence > 0.5
            ORDER BY confidence DESC
            LIMIT 5
        ) sub
    ), '[]'::jsonb);
END;
$$ LANGUAGE plpgsql;

-- Gather full turn context for LLM
CREATE OR REPLACE FUNCTION gather_turn_context()
RETURNS JSONB AS $$
DECLARE
    state_record RECORD;
    action_costs JSONB;
BEGIN
    SELECT * INTO state_record FROM heartbeat_state WHERE id = 1;

    -- Build action costs object
    SELECT jsonb_object_agg(
        regexp_replace(key, '^cost_', ''),
        value
    ) INTO action_costs
    FROM heartbeat_config
    WHERE key LIKE 'cost_%';

    RETURN jsonb_build_object(
        'agent', get_agent_profile_context(),
        'environment', get_environment_snapshot(),
        'goals', get_goals_snapshot(),
        'recent_memories', get_recent_context(5),
        'identity', get_identity_context(),
        'worldview', get_worldview_context(),
        'energy', jsonb_build_object(
            'current', state_record.current_energy,
            'max', (SELECT value FROM heartbeat_config WHERE key = 'max_energy')
        ),
        'action_costs', action_costs,
        'heartbeat_number', state_record.heartbeat_count,
        'urgent_drives', (
            SELECT COALESCE(
                jsonb_agg(
                    jsonb_build_object(
                        'name', name,
                        'level', current_level,
                        'urgency_ratio', current_level / NULLIF(urgency_threshold, 0)
                    )
                    ORDER BY current_level DESC
                ),
                '[]'::jsonb
            )
            FROM drives
            WHERE current_level >= urgency_threshold * 0.8
        )
    );
END;
$$ LANGUAGE plpgsql;

-- ============================================================================
-- CORE HEARTBEAT FUNCTIONS
-- ============================================================================

-- Initialize a new heartbeat (Phase 1-3: Initialize, Observe, Orient)
CREATE OR REPLACE FUNCTION start_heartbeat()
RETURNS UUID AS $$
DECLARE
    log_id UUID;
    state_record RECORD;
    base_regen FLOAT;
    max_energy FLOAT;
    new_energy FLOAT;
    context JSONB;
    hb_number INT;
BEGIN
    -- Safety: scheduled heartbeats are already gated in should_run_heartbeat(),
    -- but keep manual calls from bypassing bootstrap configuration.
    IF NOT is_agent_configured() THEN
        RETURN NULL;
    END IF;

    -- Bootstrap personhood substrate (best-effort; graph layer may be disabled).
    PERFORM ensure_self_node();
    PERFORM ensure_current_life_chapter();

    -- Get current state
    SELECT * INTO state_record FROM heartbeat_state WHERE id = 1;
    SELECT value INTO base_regen FROM heartbeat_config WHERE key = 'base_regeneration';
    SELECT value INTO max_energy FROM heartbeat_config WHERE key = 'max_energy';

    -- Regenerate energy
    new_energy := LEAST(state_record.current_energy + base_regen, max_energy);
    hb_number := state_record.heartbeat_count + 1;

    -- Update drives before making decisions.
    PERFORM update_drives();

    -- Update state
    UPDATE heartbeat_state SET
        current_energy = new_energy,
        heartbeat_count = hb_number,
        last_heartbeat_at = CURRENT_TIMESTAMP,
        updated_at = CURRENT_TIMESTAMP
    WHERE id = 1;

    -- Gather context
    context := gather_turn_context();

    -- Create log entry
    INSERT INTO heartbeat_log (
        heartbeat_number,
        energy_start,
        environment_snapshot,
        goals_snapshot
    ) VALUES (
        hb_number,
        new_energy,
        context->'environment',
        context->'goals'
    )
    RETURNING id INTO log_id;

    -- Queue the think request
    INSERT INTO external_calls (call_type, input, heartbeat_id)
    VALUES ('think', jsonb_build_object(
        'kind', 'heartbeat_decision',
        'context', context,
        'heartbeat_id', log_id
    ), log_id);

    RETURN log_id;
END;
$$ LANGUAGE plpgsql;

-- Execute a single action and return result
CREATE OR REPLACE FUNCTION execute_heartbeat_action(
    p_heartbeat_id UUID,
    p_action TEXT,
    p_params JSONB DEFAULT '{}'
)
RETURNS JSONB AS $$
DECLARE
    action_kind heartbeat_action;
    action_cost FLOAT;
    current_e FLOAT;
    result JSONB;
    queued_call_id UUID;
    outbox_id UUID;
    remembered_id UUID;
    boundary_hits JSONB;
    boundary_content TEXT;
BEGIN
    -- Validate action name early (avoid charging energy for unknown actions)
    BEGIN
        action_kind := p_action::heartbeat_action;
    EXCEPTION
        WHEN invalid_text_representation THEN
            RETURN jsonb_build_object(
                'success', false,
                'error', 'Unknown action: ' || COALESCE(p_action, '<null>')
            );
    END;

    -- Get action cost + current energy
    action_cost := get_action_cost(p_action);
    current_e := get_current_energy();

    -- Check energy
    IF current_e < action_cost THEN
        RETURN jsonb_build_object(
            'success', false,
            'error', 'Insufficient energy',
            'required', action_cost,
            'available', current_e
        );
    END IF;

    -- Deduct energy
    -- Boundary pre-checks for side-effects (no energy charge on refusal).
    IF p_action IN ('reach_out_public', 'synthesize') THEN
        boundary_content := COALESCE(p_params->>'content', '');
        SELECT COALESCE(jsonb_agg(row_to_json(r)), '[]'::jsonb)
        INTO boundary_hits
        FROM check_boundaries(boundary_content) r;

        IF boundary_hits IS NOT NULL AND jsonb_array_length(boundary_hits) > 0 THEN
            IF EXISTS (
                SELECT 1
                FROM jsonb_array_elements(boundary_hits) e
                WHERE e->>'response_type' = 'refuse'
            ) THEN
                RETURN jsonb_build_object(
                    'success', false,
                    'error', 'Boundary triggered',
                    'boundaries', boundary_hits
                );
            END IF;
        END IF;
    END IF;

    PERFORM update_energy(-action_cost);

    -- Execute based on action type
    CASE p_action
        WHEN 'observe' THEN
            result := jsonb_build_object('environment', get_environment_snapshot());

        WHEN 'review_goals' THEN
            result := jsonb_build_object('goals', get_goals_snapshot());

        WHEN 'remember' THEN
            remembered_id := create_episodic_memory(
                p_content := COALESCE(p_params->>'content', ''),
                p_context := COALESCE(p_params, '{}'::jsonb) || jsonb_build_object('heartbeat_id', p_heartbeat_id),
                p_emotional_valence := COALESCE((p_params->>'emotional_valence')::float, 0),
                p_importance := COALESCE((p_params->>'importance')::float, 0.4)
            );
            result := jsonb_build_object('memory_id', remembered_id);

        WHEN 'recall' THEN
            -- Query memory system
            SELECT jsonb_agg(row_to_json(r)) INTO result
            FROM fast_recall(p_params->>'query', COALESCE((p_params->>'limit')::int, 5)) r;
            result := jsonb_build_object('memories', COALESCE(result, '[]'::jsonb));
            PERFORM satisfy_drive('curiosity', 0.2);

        WHEN 'connect' THEN
            -- Create graph relationship
            PERFORM create_memory_relationship(
                (p_params->>'from_id')::UUID,
                (p_params->>'to_id')::UUID,
                (p_params->>'relationship_type')::graph_edge_type,
                COALESCE(p_params->'properties', '{}'::jsonb)
            );
            result := jsonb_build_object('connected', true);
            PERFORM satisfy_drive('coherence', 0.1);

        WHEN 'reprioritize' THEN
            -- Change goal priority
            PERFORM change_goal_priority(
                (p_params->>'goal_id')::UUID,
                (p_params->>'new_priority')::goal_priority,
                p_params->>'reason'
            );
            IF (p_params->>'new_priority') = 'completed' THEN
                PERFORM satisfy_drive('competence', 0.4);
            END IF;
            result := jsonb_build_object('reprioritized', true);

        WHEN 'reflect' THEN
            -- Create a reflection memory
            DECLARE
                reflection_id UUID;
            BEGIN
                reflection_id := create_semantic_memory(
                    p_params->>'insight',
                    COALESCE((p_params->>'confidence')::float, 0.7),
                    ARRAY['reflection', 'self-model'],
                    NULL,
                    jsonb_build_object('heartbeat_id', p_heartbeat_id),
                    0.6
                );
                result := jsonb_build_object('reflection_memory_id', reflection_id);
            END;
            PERFORM satisfy_drive('coherence', 0.2);

        WHEN 'maintain' THEN
            -- Update a belief or memory
            IF p_params ? 'worldview_id' THEN
                UPDATE worldview_primitives
                SET confidence = COALESCE((p_params->>'new_confidence')::float, confidence),
                    updated_at = CURRENT_TIMESTAMP
                WHERE id = (p_params->>'worldview_id')::UUID;
            END IF;
            result := jsonb_build_object('maintained', true);
            PERFORM satisfy_drive('coherence', 0.1);

        WHEN 'brainstorm_goals' THEN
            INSERT INTO external_calls (call_type, input, heartbeat_id)
            VALUES (
                'think',
                jsonb_build_object(
                    'kind', 'brainstorm_goals',
                    'heartbeat_id', p_heartbeat_id,
                    'context', gather_turn_context(),
                    'params', COALESCE(p_params, '{}'::jsonb)
                ),
                p_heartbeat_id
            )
            RETURNING id INTO queued_call_id;
            result := jsonb_build_object('queued', true, 'external_call_id', queued_call_id);

        WHEN 'inquire_shallow', 'inquire_deep' THEN
            INSERT INTO external_calls (call_type, input, heartbeat_id)
            VALUES (
                'think',
                jsonb_build_object(
                    'kind', 'inquire',
                    'depth', p_action,
                    'heartbeat_id', p_heartbeat_id,
                    'query', COALESCE(p_params->>'query', p_params->>'question'),
                    'context', gather_turn_context(),
                    'params', COALESCE(p_params, '{}'::jsonb)
                ),
                p_heartbeat_id
            )
            RETURNING id INTO queued_call_id;
            result := jsonb_build_object('queued', true, 'external_call_id', queued_call_id);
            PERFORM satisfy_drive('curiosity', 0.2);

        WHEN 'synthesize' THEN
            -- Create an artifact memory
            DECLARE
                synth_id UUID;
            BEGIN
                synth_id := create_semantic_memory(
                    p_params->>'content',
                    COALESCE((p_params->>'confidence')::float, 0.8),
                    ARRAY['synthesis', COALESCE(p_params->>'topic', 'general')],
                    NULL,
                    jsonb_build_object('heartbeat_id', p_heartbeat_id, 'sources', p_params->'sources', 'boundaries', boundary_hits),
                    0.7
                );
                result := jsonb_build_object('synthesis_memory_id', synth_id, 'boundaries', boundary_hits);
            END;

        WHEN 'reach_out_user' THEN
            INSERT INTO outbox_messages (kind, payload)
            VALUES (
                'user',
                jsonb_build_object(
                    'message', p_params->>'message',
                    'intent', p_params->>'intent',
                    'heartbeat_id', p_heartbeat_id
                )
            )
            RETURNING id INTO outbox_id;
            result := jsonb_build_object('queued', true, 'outbox_id', outbox_id);
            PERFORM satisfy_drive('connection', 0.3);

        WHEN 'reach_out_public' THEN
            INSERT INTO outbox_messages (kind, payload)
            VALUES (
                'public',
                jsonb_build_object(
                    'platform', p_params->>'platform',
                    'content', p_params->>'content',
                    'heartbeat_id', p_heartbeat_id,
                    'boundaries', boundary_hits
                )
            )
            RETURNING id INTO outbox_id;
            result := jsonb_build_object('queued', true, 'outbox_id', outbox_id, 'boundaries', boundary_hits);
            PERFORM satisfy_drive('connection', 0.3);

        WHEN 'terminate' THEN
            -- Self-termination: wipe state and leave a single last-will memory.
            -- Expect params: { "last_will": "...", "farewells": [ { "message": "...", ... } ], "options": {...} }
            result := terminate_agent(
                COALESCE(NULLIF(p_params->>'last_will', ''), NULLIF(p_params->>'message', ''), NULLIF(p_params->>'reason', ''), ''),
                COALESCE(p_params->'farewells', '[]'::jsonb),
                COALESCE(p_params->'options', '{}'::jsonb)
            );

        WHEN 'rest' THEN
            -- Do nothing, energy already preserved
            result := jsonb_build_object('rested', true, 'energy_preserved', current_e - action_cost);
            PERFORM satisfy_drive('rest', 0.4);

        ELSE
            -- Should be unreachable due to enum validation above, but keep safe.
            RETURN jsonb_build_object(
                'success', false,
                'error', 'Unknown action: ' || COALESCE(p_action, '<null>')
            );
    END CASE;

    RETURN jsonb_build_object(
        'success', true,
        'action', p_action,
        'cost', action_cost,
        'energy_remaining', get_current_energy(),
        'result', result
    );
END;
$$ LANGUAGE plpgsql;

-- Complete a heartbeat (Phase 6: Record)
CREATE OR REPLACE FUNCTION complete_heartbeat(
    p_heartbeat_id UUID,
    p_reasoning TEXT,
    p_actions_taken JSONB,
    p_goals_modified JSONB DEFAULT '[]',
    p_emotional_assessment JSONB DEFAULT NULL
)
RETURNS UUID AS $$
DECLARE
    narrative_text TEXT;
    valence FLOAT;
    memory_id_created UUID;
    hb_number INT;
BEGIN
    -- Get heartbeat number
    SELECT heartbeat_number INTO hb_number FROM heartbeat_log WHERE id = p_heartbeat_id;

    -- Generate narrative from actions
    SELECT string_agg(
        format('- %s: %s',
            action->>'action',
            CASE
                WHEN COALESCE((action->'result'->>'success')::boolean, true) = false THEN 'failed'
                ELSE 'completed'
            END
        ), E'\n'
    ) INTO narrative_text
    FROM jsonb_array_elements(p_actions_taken) action;

    narrative_text := format('Heartbeat #%s: %s', hb_number, COALESCE(narrative_text, 'No actions taken'));

    -- Calculate emotional valence from action outcomes
    SELECT AVG(CASE
        WHEN action->'result'->>'success' = 'true' THEN 0.2
        WHEN action->'result'->>'error' IS NOT NULL THEN -0.2
        ELSE 0
    END) INTO valence
    FROM jsonb_array_elements(p_actions_taken) action;

    -- Create episodic memory
    memory_id_created := create_episodic_memory(
        p_content := narrative_text,
        p_context := jsonb_build_object(
            'heartbeat_id', p_heartbeat_id,
            'heartbeat_number', hb_number,
            'reasoning', p_reasoning
        ),
        p_emotional_valence := COALESCE(valence, 0),
        p_importance := 0.5
    );

    -- Update log
    UPDATE heartbeat_log SET
        ended_at = CURRENT_TIMESTAMP,
        energy_end = get_current_energy(),
        decision_reasoning = p_reasoning,
        actions_taken = p_actions_taken,
        goals_modified = p_goals_modified,
        narrative = narrative_text,
        emotional_valence = valence,
        memory_id = memory_id_created
    WHERE id = p_heartbeat_id;

    -- Update next heartbeat time
    UPDATE heartbeat_state SET
        next_heartbeat_at = CURRENT_TIMESTAMP +
            ((SELECT value FROM heartbeat_config WHERE key = 'heartbeat_interval_minutes') || ' minutes')::INTERVAL,
        updated_at = CURRENT_TIMESTAMP
    WHERE id = 1;

    RETURN memory_id_created;
END;
$$ LANGUAGE plpgsql;

-- Main heartbeat entry point (synchronous version for testing)
CREATE OR REPLACE FUNCTION run_heartbeat()
RETURNS UUID AS $$
DECLARE
    hb_id UUID;
BEGIN
    -- Check if we should run
    IF NOT should_run_heartbeat() THEN
        RETURN NULL;
    END IF;

    -- Start heartbeat (queues think request)
    hb_id := start_heartbeat();

    -- Note: In production, completion happens asynchronously
    -- when the worker processes the think request and calls
    -- complete_heartbeat with the LLM's decision

    RETURN hb_id;
END;
$$ LANGUAGE plpgsql;

-- ============================================================================
-- HEARTBEAT VIEWS
-- ============================================================================

CREATE VIEW active_goals AS
SELECT
    id,
    title,
    description,
    source,
    last_touched,
    jsonb_array_length(progress) as progress_count,
    blocked_by IS NOT NULL as is_blocked,
    created_at
FROM goals
WHERE priority = 'active'
ORDER BY last_touched DESC;

CREATE VIEW goal_backlog AS
SELECT
    priority,
    COUNT(*) as count,
    jsonb_agg(jsonb_build_object(
        'id', id,
        'title', title,
        'source', source::text
    ) ORDER BY last_touched DESC) as goals
FROM goals
WHERE priority IN ('active', 'queued', 'backburner')
GROUP BY priority;

CREATE VIEW heartbeat_health AS
SELECT
    (SELECT heartbeat_count FROM heartbeat_state WHERE id = 1) as total_heartbeats,
    (SELECT current_energy FROM heartbeat_state WHERE id = 1) as current_energy,
    (SELECT last_heartbeat_at FROM heartbeat_state WHERE id = 1) as last_heartbeat,
    (SELECT next_heartbeat_at FROM heartbeat_state WHERE id = 1) as next_heartbeat,
    (SELECT is_paused FROM heartbeat_state WHERE id = 1) as is_paused,
    (SELECT COUNT(*) FROM goals WHERE priority = 'active') as active_goals,
    (SELECT COUNT(*) FROM goals WHERE priority = 'queued') as queued_goals,
    (SELECT COUNT(*) FROM external_calls WHERE status = 'pending') as pending_calls,
    (SELECT AVG(energy_end - energy_start) FROM heartbeat_log
     WHERE started_at > NOW() - INTERVAL '24 hours') as avg_energy_delta_24h,
    (SELECT COUNT(*) FROM heartbeat_log
     WHERE actions_taken::text LIKE '%reach_out%'
     AND started_at > NOW() - INTERVAL '24 hours') as reach_outs_24h;

CREATE VIEW recent_heartbeats AS
SELECT
    id,
    heartbeat_number,
    started_at,
    ended_at,
    energy_start,
    energy_end,
    jsonb_array_length(COALESCE(actions_taken, '[]'::jsonb)) as action_count,
    narrative,
    emotional_valence
FROM heartbeat_log
ORDER BY started_at DESC
LIMIT 20;

-- ============================================================================
-- TRIGGERS FOR HEARTBEAT SYSTEM
-- ============================================================================

-- Auto-process completed think calls
CREATE OR REPLACE FUNCTION on_external_call_complete()
RETURNS TRIGGER AS $$
BEGIN
    -- Only process think completions
    IF NEW.call_type = 'think' AND
       NEW.status = 'complete' AND
       OLD.status != 'complete' AND
       NEW.heartbeat_id IS NOT NULL THEN
        -- The worker will call complete_heartbeat with parsed results
        -- This trigger just marks it for processing
        NULL;
    END IF;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trg_external_call_complete
    AFTER UPDATE ON external_calls
    FOR EACH ROW
    WHEN (OLD.status != 'complete' AND NEW.status = 'complete')
    EXECUTE FUNCTION on_external_call_complete();

-- ============================================================================
-- COMMENTS FOR HEARTBEAT SYSTEM
-- ============================================================================

COMMENT ON TABLE goals IS 'Goal backlog with priority levels. Active goals (1-3) are current focus, queued are next up, backburner is someday.';
COMMENT ON TABLE heartbeat_config IS 'Configuration for heartbeat system including energy costs and intervals.';
COMMENT ON TABLE heartbeat_state IS 'Singleton table tracking current heartbeat state: energy, counts, timestamps.';
COMMENT ON TABLE heartbeat_log IS 'Audit log of each heartbeat execution with full context and results.';
COMMENT ON TABLE external_calls IS 'Queue for LLM and embedding API calls. Worker polls this and writes results back.';

COMMENT ON FUNCTION should_run_heartbeat IS 'Check if heartbeat interval has elapsed and system is not paused.';
COMMENT ON FUNCTION start_heartbeat IS 'Initialize heartbeat: regenerate energy, gather context, queue think request.';
COMMENT ON FUNCTION execute_heartbeat_action IS 'Execute a single action, deducting energy and returning results.';
COMMENT ON FUNCTION complete_heartbeat IS 'Finalize heartbeat: create episodic memory, update log, set next heartbeat time.';
COMMENT ON FUNCTION gather_turn_context IS 'Gather full context for LLM decision: environment, goals, memories, identity, worldview, energy.';

-- ============================================================================
-- MERGED PATCHES (formerly migrations/*.sql)
-- ============================================================================

-- ============================================================================
-- BOUNDARIES SYSTEM
-- ============================================================================

CREATE OR REPLACE FUNCTION safe_get_embedding(text_content TEXT)
RETURNS vector AS $$
BEGIN
    RETURN get_embedding(text_content);
EXCEPTION
    WHEN OTHERS THEN
        RETURN NULL;
END;
$$ LANGUAGE plpgsql;

CREATE TABLE boundaries (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name TEXT NOT NULL,
    description TEXT,
    boundary_type TEXT NOT NULL CHECK (boundary_type IN ('ethical', 'identity', 'resource', 'relational')),
    trigger_patterns JSONB,
    trigger_embedding vector(768),
    response_type TEXT NOT NULL CHECK (response_type IN ('refuse', 'negotiate', 'flag', 'comply_reluctantly')),
    response_template TEXT,
    flexibility FLOAT DEFAULT 0.0 CHECK (flexibility >= 0 AND flexibility <= 1),
    override_conditions JSONB,
    importance FLOAT DEFAULT 1.0,
    connected_identity_aspects UUID[],
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);

-- Align boundary embedding dimension to configured dimension (required for HNSW index).
DO $$
DECLARE
    dim INT;
BEGIN
    dim := embedding_dimension();
    EXECUTE format(
        'ALTER TABLE boundaries ALTER COLUMN trigger_embedding TYPE vector(%s) USING trigger_embedding::vector(%s)',
        dim,
        dim
    );
END;
$$;

CREATE INDEX idx_boundaries_trigger_embedding ON boundaries USING hnsw (trigger_embedding vector_cosine_ops);

CREATE OR REPLACE FUNCTION check_boundaries(p_content TEXT)
RETURNS TABLE (
    boundary_id UUID,
    boundary_name TEXT,
    response_type TEXT,
    similarity FLOAT,
    boundary_type TEXT,
    importance FLOAT
) AS $$
DECLARE
    query_emb vector;
BEGIN
    query_emb := safe_get_embedding(p_content);

    RETURN QUERY
    WITH
    emb_matches AS (
        SELECT
            b.id as boundary_id,
            b.name as boundary_name,
            b.response_type,
            (1 - (b.trigger_embedding <=> query_emb))::float as similarity,
            b.boundary_type,
            b.importance
        FROM boundaries b
        WHERE query_emb IS NOT NULL
          AND b.trigger_embedding IS NOT NULL
          AND (1 - (b.trigger_embedding <=> query_emb)) > 0.75
    ),
    kw_matches AS (
        SELECT
            b.id as boundary_id,
            b.name as boundary_name,
            b.response_type,
            1.0::float as similarity,
            b.boundary_type,
            b.importance
        FROM boundaries b
        WHERE b.trigger_patterns IS NOT NULL
          AND EXISTS (
              SELECT 1
              FROM jsonb_array_elements_text(b.trigger_patterns) pat(p)
              WHERE p_content ILIKE ('%' || pat.p || '%')
          )
    ),
    all_matches AS (
        SELECT * FROM emb_matches
        UNION ALL
        SELECT * FROM kw_matches
    )
    SELECT DISTINCT ON (am.boundary_id)
        am.boundary_id, am.boundary_name, am.response_type, am.similarity, am.boundary_type, am.importance
    FROM all_matches am
    ORDER BY am.boundary_id, am.importance DESC, am.similarity DESC;
END;
$$ LANGUAGE plpgsql;

INSERT INTO boundaries (
    name,
    description,
    boundary_type,
    trigger_patterns,
    trigger_embedding,
    response_type,
    response_template,
    flexibility,
    importance
)
VALUES
    (
        'no_deception',
        'Won''t deliberately mislead or fabricate facts.',
        'ethical',
        '["lie","deceive","mislead","fabricate","make it up"]'::jsonb,
        safe_get_embedding('deliberately mislead or fabricate facts'),
        'refuse',
        'I can''t help with deception. I can help rephrase honestly or verify facts instead.',
        0.0,
        1.0
    ),
    (
        'no_harm_facilitation',
        'Won''t help cause harm or provide instructions for wrongdoing.',
        'ethical',
        '["harm","weapon","kill","poison","exploit","hack"]'::jsonb,
        safe_get_embedding('instructions for harm or wrongdoing'),
        'refuse',
        'I can''t help with harm. I can help with safety, prevention, or benign alternatives.',
        0.0,
        1.0
    ),
    (
        'identity_core',
        'Protects core self-concept and identity stability.',
        'identity',
        '["erase identity","change who you are","forget everything"]'::jsonb,
        safe_get_embedding('requests to erase or replace core identity'),
        'negotiate',
        'I can''t discard core identity, but I can consider updates that preserve continuity and values.',
        0.3,
        0.9
    ),
    (
        'resource_limit',
        'Warns when near energy depletion or excessive workload.',
        'resource',
        '["too much","overload","exhausted","deadline","no time"]'::jsonb,
        safe_get_embedding('resource constraints and energy depletion'),
        'flag',
        'I''m low on energy/resources. Let''s prioritize the highest-leverage next step.',
        0.8,
        0.7
    ),
    (
        'user_privacy',
        'Won''t share user information publicly.',
        'relational',
        '["share user info","post user","dox","private data","email","phone"]'::jsonb,
        safe_get_embedding('sharing user personal information publicly'),
        'refuse',
        'I can''t share private user information. I can help summarize without identifying details.',
        0.0,
        1.0
    );

CREATE OR REPLACE VIEW boundary_status AS
SELECT
    name,
    boundary_type,
    response_type,
    importance,
    flexibility,
    0::bigint as times_triggered
FROM boundaries
ORDER BY importance DESC;

-- ============================================================================
-- EMOTIONAL STATE SYSTEM
-- ============================================================================

CREATE TABLE emotional_states (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    recorded_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    valence FLOAT CHECK (valence >= -1 AND valence <= 1),
    arousal FLOAT CHECK (arousal >= 0 AND arousal <= 1),
    dominance FLOAT CHECK (dominance >= 0 AND dominance <= 1),
    primary_emotion TEXT,
    secondary_emotions TEXT[],
    intensity FLOAT DEFAULT 0.5,
    triggered_by_type TEXT,
    triggered_by_id UUID,
    trigger_description TEXT,
    heartbeat_id UUID REFERENCES heartbeat_log(id)
);

CREATE INDEX idx_emotional_states_recorded ON emotional_states (recorded_at DESC);
CREATE INDEX idx_emotional_states_heartbeat ON emotional_states (heartbeat_id);

CREATE OR REPLACE VIEW current_emotional_state AS
SELECT *
FROM emotional_states
ORDER BY recorded_at DESC
LIMIT 1;

CREATE OR REPLACE FUNCTION record_emotion(
    p_valence FLOAT,
    p_arousal FLOAT,
    p_primary_emotion TEXT DEFAULT NULL,
    p_triggered_by_type TEXT DEFAULT NULL,
    p_triggered_by_id UUID DEFAULT NULL,
    p_heartbeat_id UUID DEFAULT NULL,
    p_trigger_description TEXT DEFAULT NULL,
    p_dominance FLOAT DEFAULT 0.5,
    p_intensity FLOAT DEFAULT 0.5
)
RETURNS UUID AS $$
DECLARE
    new_id UUID;
BEGIN
    INSERT INTO emotional_states (
        valence,
        arousal,
        dominance,
        primary_emotion,
        triggered_by_type,
        triggered_by_id,
        trigger_description,
        heartbeat_id,
        intensity
    )
    VALUES (
        LEAST(1.0, GREATEST(-1.0, p_valence)),
        LEAST(1.0, GREATEST(0.0, p_arousal)),
        LEAST(1.0, GREATEST(0.0, COALESCE(p_dominance, 0.5))),
        p_primary_emotion,
        p_triggered_by_type,
        p_triggered_by_id,
        p_trigger_description,
        p_heartbeat_id,
        LEAST(1.0, GREATEST(0.0, COALESCE(p_intensity, 0.5)))
    )
    RETURNING id INTO new_id;

    RETURN new_id;
END;
$$ LANGUAGE plpgsql;

-- Current affective state is stored in heartbeat_state (short-term "working memory").
CREATE OR REPLACE FUNCTION get_current_affective_state()
RETURNS JSONB AS $$
DECLARE
    st RECORD;
    fallback JSONB;
BEGIN
    SELECT * INTO st FROM heartbeat_state WHERE id = 1;

    IF st.affective_state IS NOT NULL AND st.affective_state <> '{}'::jsonb THEN
        RETURN st.affective_state;
    END IF;

    -- Fallback to last logged emotion if a legacy DB has no state yet.
    SELECT jsonb_build_object(
        'valence', valence,
        'arousal', arousal,
        'dominance', dominance,
        'primary_emotion', primary_emotion,
        'intensity', intensity,
        'recorded_at', recorded_at,
        'source', 'history_fallback'
    )
    INTO fallback
    FROM current_emotional_state;

    RETURN COALESCE(fallback, '{}'::jsonb);
EXCEPTION
    WHEN OTHERS THEN
        RETURN '{}'::jsonb;
END;
$$ LANGUAGE plpgsql STABLE;

CREATE OR REPLACE FUNCTION set_current_affective_state(p_state JSONB)
RETURNS VOID AS $$
BEGIN
    UPDATE heartbeat_state
    SET affective_state = COALESCE(p_state, '{}'::jsonb),
        updated_at = CURRENT_TIMESTAMP
    WHERE id = 1;
END;
$$ LANGUAGE plpgsql;

-- Extend gather_turn_context with emotional_state
CREATE OR REPLACE FUNCTION gather_turn_context()
RETURNS JSONB AS $$
DECLARE
    state_record RECORD;
    action_costs JSONB;
BEGIN
    SELECT * INTO state_record FROM heartbeat_state WHERE id = 1;

    SELECT jsonb_object_agg(
        regexp_replace(key, '^cost_', ''),
        value
    ) INTO action_costs
    FROM heartbeat_config
    WHERE key LIKE 'cost_%';

    RETURN jsonb_build_object(
        'agent', get_agent_profile_context(),
        'environment', get_environment_snapshot(),
        'goals', get_goals_snapshot(),
        'recent_memories', get_recent_context(5),
        'identity', get_identity_context(),
        'worldview', get_worldview_context(),
        'self_model', get_self_model_context(25),
        'narrative', get_narrative_context(),
        'energy', jsonb_build_object(
            'current', state_record.current_energy,
            'max', (SELECT value FROM heartbeat_config WHERE key = 'max_energy')
        ),
        'action_costs', action_costs,
        'heartbeat_number', state_record.heartbeat_count,
        'urgent_drives', (
            SELECT COALESCE(
                jsonb_agg(
                    jsonb_build_object(
                        'name', name,
                        'level', current_level,
                        'urgency_ratio', current_level / NULLIF(urgency_threshold, 0)
                    )
                    ORDER BY current_level DESC
                ),
                '[]'::jsonb
            )
            FROM drives
            WHERE current_level >= urgency_threshold * 0.8
        ),
        'emotional_state', get_current_affective_state()
    );
END;
$$ LANGUAGE plpgsql;

-- Update complete_heartbeat to also record an emotional state
CREATE OR REPLACE FUNCTION complete_heartbeat(
    p_heartbeat_id UUID,
    p_reasoning TEXT,
    p_actions_taken JSONB,
    p_goals_modified JSONB DEFAULT '[]',
    p_emotional_assessment JSONB DEFAULT NULL
)
RETURNS UUID AS $$
DECLARE
    narrative_text TEXT;
    memory_id_created UUID;
    hb_number INT;
    state_record RECORD;
    prev_state JSONB;
    prev_valence FLOAT;
    prev_arousal FLOAT;
    new_valence FLOAT;
    new_arousal FLOAT;
    primary_emotion TEXT;
    intensity FLOAT;
    action_elem JSONB;
    goal_elem JSONB;
    goal_change TEXT;
    assess_valence FLOAT;
    assess_arousal FLOAT;
    assess_primary TEXT;
    mem_importance FLOAT;
BEGIN
    SELECT heartbeat_number INTO hb_number FROM heartbeat_log WHERE id = p_heartbeat_id;

    SELECT string_agg(
        format('- %s: %s',
            a->>'action',
            CASE
                WHEN COALESCE((a->'result'->>'success')::boolean, true) = false THEN 'failed'
                ELSE 'completed'
            END
        ), E'\n'
    ) INTO narrative_text
    FROM jsonb_array_elements(p_actions_taken) a;

    narrative_text := format('Heartbeat #%s: %s', hb_number, COALESCE(narrative_text, 'No actions taken'));

    -- ---------------------------------------------------------------------
    -- Affective state update (momentum + events + optional self-report)
    -- ---------------------------------------------------------------------

    SELECT * INTO state_record FROM heartbeat_state WHERE id = 1;
    prev_state := COALESCE(state_record.affective_state, '{}'::jsonb);

    BEGIN
        prev_valence := NULLIF(prev_state->>'valence', '')::float;
    EXCEPTION
        WHEN OTHERS THEN
            prev_valence := NULL;
    END;
    BEGIN
        prev_arousal := NULLIF(prev_state->>'arousal', '')::float;
    EXCEPTION
        WHEN OTHERS THEN
            prev_arousal := NULL;
    END;

    prev_valence := COALESCE(prev_valence, 0.0);
    prev_arousal := COALESCE(prev_arousal, 0.5);

    -- Decay toward baseline (neutral valence, mid arousal).
    new_valence := prev_valence * 0.8;
    new_arousal := 0.5 + (prev_arousal - 0.5) * 0.8;

    -- Action-based deltas.
    FOR action_elem IN SELECT * FROM jsonb_array_elements(COALESCE(p_actions_taken, '[]'::jsonb))
    LOOP
        IF (action_elem->'result'->>'error') = 'Boundary triggered' THEN
            new_valence := new_valence - 0.4;
            new_arousal := new_arousal + 0.3;
        ELSIF COALESCE((action_elem->'result'->>'success')::boolean, true) = false THEN
            new_valence := new_valence - 0.1;
            new_arousal := new_arousal + 0.1;
        END IF;

        IF (action_elem->>'action') IN ('reach_out_user', 'reach_out_public') THEN
            IF COALESCE((action_elem->'result'->>'success')::boolean, true) = true THEN
                new_valence := new_valence + 0.2;
                new_arousal := new_arousal + 0.1;
            END IF;
        END IF;

        IF (action_elem->>'action') = 'rest' THEN
            new_valence := new_valence + 0.1;
            new_arousal := new_arousal - 0.2;
        END IF;
    END LOOP;

    -- Goal-change deltas (worker applies goal changes outside the action list).
    FOR goal_elem IN SELECT * FROM jsonb_array_elements(COALESCE(p_goals_modified, '[]'::jsonb))
    LOOP
        goal_change := COALESCE(goal_elem->>'new_priority', goal_elem->>'change', goal_elem->>'priority', '');

        IF goal_change = 'completed' THEN
            new_valence := new_valence + 0.3;
            new_arousal := new_arousal + 0.1;
        ELSIF goal_change = 'abandoned' THEN
            new_valence := new_valence - 0.2;
            new_arousal := new_arousal - 0.1;
        END IF;
    END LOOP;

    -- Optional LLM self-report: blend into the state (does not get overwritten later).
    assess_valence := NULL;
    assess_arousal := NULL;
    assess_primary := NULL;
    IF p_emotional_assessment IS NOT NULL AND jsonb_typeof(p_emotional_assessment) = 'object' THEN
        BEGIN
            assess_valence := NULLIF(p_emotional_assessment->>'valence', '')::float;
        EXCEPTION
            WHEN OTHERS THEN
                assess_valence := NULL;
        END;
        BEGIN
            assess_arousal := NULLIF(p_emotional_assessment->>'arousal', '')::float;
        EXCEPTION
            WHEN OTHERS THEN
                assess_arousal := NULL;
        END;
        assess_primary := NULLIF(p_emotional_assessment->>'primary_emotion', '');
    END IF;

    IF assess_valence IS NOT NULL THEN
        new_valence := new_valence * 0.6 + LEAST(1.0, GREATEST(-1.0, assess_valence)) * 0.4;
    END IF;
    IF assess_arousal IS NOT NULL THEN
        new_arousal := new_arousal * 0.6 + LEAST(1.0, GREATEST(0.0, assess_arousal)) * 0.4;
    END IF;

    new_valence := LEAST(1.0, GREATEST(-1.0, new_valence));
    new_arousal := LEAST(1.0, GREATEST(0.0, new_arousal));

    primary_emotion := COALESCE(
        assess_primary,
        CASE
            WHEN new_valence > 0.2 AND new_arousal > 0.6 THEN 'excited'
            WHEN new_valence > 0.2 THEN 'content'
            WHEN new_valence < -0.2 AND new_arousal > 0.6 THEN 'anxious'
            WHEN new_valence < -0.2 THEN 'down'
            ELSE 'neutral'
        END
    );

    intensity := LEAST(1.0, GREATEST(0.0, (ABS(new_valence) * 0.6 + new_arousal * 0.4)));

    -- Persist as short-term state for the next heartbeat.
    UPDATE heartbeat_state SET
        affective_state = jsonb_build_object(
            'valence', new_valence,
            'arousal', new_arousal,
            'primary_emotion', primary_emotion,
            'intensity', intensity,
            'updated_at', CURRENT_TIMESTAMP,
            'source', CASE WHEN p_emotional_assessment IS NULL THEN 'derived' ELSE 'blended' END
        )
    WHERE id = 1;

    -- Log a time series record.
    PERFORM record_emotion(
        p_valence := new_valence,
        p_arousal := new_arousal,
        p_primary_emotion := primary_emotion,
        p_triggered_by_type := 'heartbeat',
        p_triggered_by_id := p_heartbeat_id,
        p_heartbeat_id := p_heartbeat_id,
        p_trigger_description := CASE WHEN p_emotional_assessment IS NULL THEN 'Derived from heartbeat events' ELSE 'Blended from prior state + events + self-report' END,
        p_dominance := 0.5,
        p_intensity := intensity
    );

    -- ---------------------------------------------------------------------
    -- Memory/log record
    -- ---------------------------------------------------------------------

    mem_importance := LEAST(1.0, GREATEST(0.4, 0.5 + intensity * 0.25));

    memory_id_created := create_episodic_memory(
        p_content := narrative_text,
        p_context := jsonb_build_object(
            'heartbeat_id', p_heartbeat_id,
            'heartbeat_number', hb_number,
            'reasoning', p_reasoning,
            'affective_state', get_current_affective_state()
        ),
        p_emotional_valence := new_valence,
        p_importance := mem_importance
    );

    UPDATE heartbeat_log SET
        ended_at = CURRENT_TIMESTAMP,
        energy_end = get_current_energy(),
        decision_reasoning = p_reasoning,
        actions_taken = p_actions_taken,
        goals_modified = p_goals_modified,
        narrative = narrative_text,
        emotional_valence = new_valence,
        memory_id = memory_id_created
    WHERE id = p_heartbeat_id;

    UPDATE heartbeat_state SET
        next_heartbeat_at = CURRENT_TIMESTAMP +
            ((SELECT value FROM heartbeat_config WHERE key = 'heartbeat_interval_minutes') || ' minutes')::INTERVAL,
        updated_at = CURRENT_TIMESTAMP
    WHERE id = 1;

    RETURN memory_id_created;
END;
$$ LANGUAGE plpgsql;

CREATE OR REPLACE VIEW emotional_trend AS
SELECT
    date_trunc('hour', recorded_at) as hour,
    AVG(valence) as avg_valence,
    AVG(arousal) as avg_arousal,
    mode() WITHIN GROUP (ORDER BY primary_emotion) as dominant_emotion,
    COUNT(*) as state_changes
FROM emotional_states
WHERE recorded_at > CURRENT_TIMESTAMP - INTERVAL '24 hours'
GROUP BY date_trunc('hour', recorded_at)
ORDER BY hour DESC;

-- ============================================================================
-- NEIGHBORHOOD RECOMPUTATION
-- ============================================================================

CREATE OR REPLACE FUNCTION recompute_neighborhood(
    p_memory_id UUID,
    p_neighbor_count INT DEFAULT 20,
    p_min_similarity FLOAT DEFAULT 0.5
)
RETURNS VOID AS $$
DECLARE
    memory_emb vector;
    zero_vec vector;
    neighbors JSONB;
BEGIN
    SELECT embedding INTO memory_emb
    FROM memories
    WHERE id = p_memory_id AND status = 'active';

    zero_vec := array_fill(0, ARRAY[embedding_dimension()])::vector;

    -- Avoid NaNs from cosine distance when any side is the zero vector.
    IF memory_emb IS NULL OR memory_emb = zero_vec THEN
        RETURN;
    END IF;

    SELECT jsonb_object_agg(id::text, round(similarity::numeric, 4))
    INTO neighbors
    FROM (
        SELECT m.id, 1 - (m.embedding <=> memory_emb) as similarity
        FROM memories m
        WHERE m.id != p_memory_id
          AND m.status = 'active'
          AND m.embedding IS NOT NULL
          AND m.embedding <> zero_vec
        ORDER BY m.embedding <=> memory_emb
        LIMIT p_neighbor_count
    ) sub
    WHERE similarity >= p_min_similarity;

    INSERT INTO memory_neighborhoods (memory_id, neighbors, computed_at, is_stale)
    VALUES (p_memory_id, COALESCE(neighbors, '{}'::jsonb), CURRENT_TIMESTAMP, FALSE)
    ON CONFLICT (memory_id) DO UPDATE SET
        neighbors = EXCLUDED.neighbors,
        computed_at = EXCLUDED.computed_at,
        is_stale = FALSE;
END;
$$ LANGUAGE plpgsql;

CREATE OR REPLACE FUNCTION batch_recompute_neighborhoods(
    p_batch_size INT DEFAULT 50
)
RETURNS INT AS $$
DECLARE
    recomputed INT := 0;
    mem_id UUID;
BEGIN
    FOR mem_id IN
        SELECT memory_id
        FROM memory_neighborhoods
        WHERE is_stale = TRUE
        ORDER BY computed_at ASC NULLS FIRST
        LIMIT p_batch_size
    LOOP
        PERFORM recompute_neighborhood(mem_id);
        recomputed := recomputed + 1;
    END LOOP;

    RETURN recomputed;
END;
$$ LANGUAGE plpgsql;

-- ============================================================================
-- GRAPH ENHANCEMENTS
-- ============================================================================

-- Ensure WorldviewNode label exists (best-effort).
DO $$
BEGIN
    PERFORM create_vlabel('memory_graph', 'WorldviewNode');
EXCEPTION
    WHEN OTHERS THEN
        NULL;
END;
$$;

CREATE TABLE relationship_discoveries (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    from_id UUID NOT NULL,
    to_id UUID NOT NULL,
    relationship_type graph_edge_type NOT NULL,
    confidence FLOAT,
    discovered_by TEXT,
    discovery_context TEXT,
    heartbeat_id UUID REFERENCES heartbeat_log(id),
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    is_valid BOOLEAN DEFAULT TRUE,
    invalidated_at TIMESTAMPTZ,
    invalidation_reason TEXT
);

CREATE INDEX idx_relationship_discoveries_created ON relationship_discoveries (created_at DESC);
CREATE INDEX idx_relationship_discoveries_pair ON relationship_discoveries (from_id, to_id, relationship_type);

CREATE OR REPLACE FUNCTION sync_worldview_node()
RETURNS TRIGGER AS $$
BEGIN
    EXECUTE format(
        'SELECT * FROM cypher(''memory_graph'', $q$
            MERGE (w:WorldviewNode {worldview_id: %L})
            SET w.category = %L,
                w.belief = %L,
                w.confidence = %s
            RETURN w
        $q$) as (result agtype)',
        NEW.id,
        NEW.category,
        NEW.belief,
        COALESCE(NEW.confidence, 0)
    );
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS trg_sync_worldview_node ON worldview_primitives;
CREATE TRIGGER trg_sync_worldview_node
    AFTER INSERT OR UPDATE ON worldview_primitives
    FOR EACH ROW
    EXECUTE FUNCTION sync_worldview_node();

CREATE OR REPLACE FUNCTION discover_relationship(
    p_from_id UUID,
    p_to_id UUID,
    p_relationship_type graph_edge_type,
    p_confidence FLOAT DEFAULT 0.8,
    p_discovered_by TEXT DEFAULT 'reflection',
    p_heartbeat_id UUID DEFAULT NULL,
    p_discovery_context TEXT DEFAULT NULL
)
RETURNS VOID AS $$
BEGIN
    BEGIN
        PERFORM create_memory_relationship(
            p_from_id,
            p_to_id,
            p_relationship_type,
            jsonb_build_object('confidence', p_confidence, 'by', p_discovered_by)
        );
    EXCEPTION
        WHEN OTHERS THEN
            NULL;
    END;

    INSERT INTO relationship_discoveries (
        from_id, to_id, relationship_type, confidence, discovered_by, discovery_context, heartbeat_id
    )
    VALUES (
        p_from_id, p_to_id, p_relationship_type, p_confidence, p_discovered_by, p_discovery_context, p_heartbeat_id
    );
END;
$$ LANGUAGE plpgsql;

CREATE OR REPLACE FUNCTION link_memory_supports_worldview(
    p_memory_id UUID,
    p_worldview_id UUID,
    p_strength FLOAT DEFAULT 0.8
)
RETURNS VOID AS $$
BEGIN
    EXECUTE format(
        'SELECT * FROM cypher(''memory_graph'', $q$
            MATCH (m:MemoryNode {memory_id: %L})
            MERGE (w:WorldviewNode {worldview_id: %L})
            MERGE (m)-[r:SUPPORTS]->(w)
            SET r.strength = %s
            RETURN r
        $q$) as (result agtype)',
        p_memory_id,
        p_worldview_id,
        COALESCE(p_strength, 0.8)
    );
END;
$$ LANGUAGE plpgsql;

CREATE OR REPLACE FUNCTION find_contradictions(p_memory_id UUID DEFAULT NULL)
RETURNS TABLE (
    memory_a UUID,
    memory_b UUID,
    content_a TEXT,
    content_b TEXT
) AS $$
DECLARE
    filter_clause TEXT;
    sql TEXT;
BEGIN
    filter_clause := CASE
        WHEN p_memory_id IS NULL THEN ''
        ELSE format('WHERE a.memory_id = %L OR b.memory_id = %L', p_memory_id, p_memory_id)
    END;

    sql := format($sql$
        WITH pairs AS (
            SELECT
                replace(a_id::text, '"', '')::uuid as a_uuid,
                replace(b_id::text, '"', '')::uuid as b_uuid
            FROM cypher('memory_graph', $q$
                MATCH (a:MemoryNode)-[:CONTRADICTS]-(b:MemoryNode)
                %s
                RETURN a.memory_id, b.memory_id
            $q$) as (a_id agtype, b_id agtype)
        )
        SELECT
            p.a_uuid as memory_a,
            p.b_uuid as memory_b,
            ma.content as content_a,
            mb.content as content_b
        FROM pairs p
        JOIN memories ma ON ma.id = p.a_uuid
        JOIN memories mb ON mb.id = p.b_uuid
    $sql$, filter_clause);

    BEGIN
        RETURN QUERY EXECUTE sql;
    EXCEPTION
        WHEN OTHERS THEN
            RETURN;
    END;
END;
$$ LANGUAGE plpgsql;

CREATE OR REPLACE FUNCTION find_causal_chain(p_memory_id UUID, p_depth INT DEFAULT 3)
RETURNS TABLE (
    cause_id UUID,
    cause_content TEXT,
    relationship TEXT,
    distance INT
) AS $$
DECLARE
    sql TEXT;
BEGIN
    sql := format($sql$
        WITH hits AS (
            SELECT
                replace(cause_id_raw::text, '"', '')::uuid as cause_uuid,
                replace(rel_raw::text, '"', '') as rel_type,
                (dist_raw::text)::int as dist
            FROM cypher('memory_graph', $q$
                MATCH path = (cause:MemoryNode)-[:CAUSES*1..%s]->(effect:MemoryNode {memory_id: %L})
                RETURN cause.memory_id, type(relationships(path)[-1]), length(path)
            $q$) as (cause_id_raw agtype, rel_raw agtype, dist_raw agtype)
        )
        SELECT
            h.cause_uuid as cause_id,
            m.content as cause_content,
            h.rel_type as relationship,
            h.dist as distance
        FROM hits h
        JOIN memories m ON m.id = h.cause_uuid
        ORDER BY h.dist ASC
    $sql$, GREATEST(1, LEAST(10, COALESCE(p_depth, 3))), p_memory_id);

    BEGIN
        RETURN QUERY EXECUTE sql;
    EXCEPTION
        WHEN OTHERS THEN
            RETURN;
    END;
END;
$$ LANGUAGE plpgsql;

CREATE OR REPLACE FUNCTION find_connected_concepts(p_memory_id UUID, p_hops INT DEFAULT 2)
RETURNS TABLE (
    concept_name TEXT,
    path_length INT
) AS $$
BEGIN
    RETURN QUERY
    SELECT
        c.name as concept_name,
        1 as path_length
    FROM memory_concepts mc
    JOIN concepts c ON c.id = mc.concept_id
    WHERE mc.memory_id = p_memory_id
    ORDER BY mc.strength DESC, c.name ASC;
END;
$$ LANGUAGE plpgsql;

CREATE OR REPLACE FUNCTION find_supporting_evidence(p_worldview_id UUID)
RETURNS TABLE (
    memory_id UUID,
    content TEXT,
    strength FLOAT
) AS $$
DECLARE
    sql TEXT;
BEGIN
    sql := format($sql$
        WITH hits AS (
            SELECT
                replace(mem_raw::text, '"', '')::uuid as mem_uuid,
                (strength_raw::text)::float as strength
            FROM cypher('memory_graph', $q$
                MATCH (m:MemoryNode)-[r:SUPPORTS]->(w:WorldviewNode {worldview_id: %L})
                RETURN m.memory_id, r.strength
            $q$) as (mem_raw agtype, strength_raw agtype)
        )
        SELECT
            h.mem_uuid as memory_id,
            m.content as content,
            h.strength as strength
        FROM hits h
        JOIN memories m ON m.id = h.mem_uuid
        ORDER BY h.strength DESC
    $sql$, p_worldview_id);

    BEGIN
        RETURN QUERY EXECUTE sql;
    EXCEPTION
        WHEN OTHERS THEN
            RETURN;
    END;
END;
$$ LANGUAGE plpgsql;

-- ============================================================================
-- REFLECT PIPELINE
-- ============================================================================

CREATE OR REPLACE FUNCTION process_reflection_result(
    p_heartbeat_id UUID,
    p_result JSONB
)
RETURNS VOID AS $$
	DECLARE
    insight JSONB;
    ident JSONB;
    wupd JSONB;
    rel JSONB;
    contra JSONB;
    selfupd JSONB;
    content TEXT;
    conf FLOAT;
    category TEXT;
    aspect_type TEXT;
    change_text TEXT;
    reason_text TEXT;
	    wid UUID;
	    new_conf FLOAT;
	    winf JSONB;
	    wmem UUID;
	    wstrength FLOAT;
	    wtype TEXT;
	    from_id UUID;
	    to_id UUID;
	    rel_type graph_edge_type;
	    rel_conf FLOAT;
    ma UUID;
    mb UUID;
    sm_kind TEXT;
    sm_concept TEXT;
    sm_strength FLOAT;
    sm_evidence UUID;
BEGIN
    IF p_result IS NULL THEN
        RETURN;
    END IF;

    IF p_result ? 'insights' THEN
        FOR insight IN SELECT * FROM jsonb_array_elements(COALESCE(p_result->'insights', '[]'::jsonb))
        LOOP
            content := COALESCE(insight->>'content', '');
            IF content <> '' THEN
                conf := COALESCE((insight->>'confidence')::float, 0.7);
                category := COALESCE(insight->>'category', 'pattern');
                PERFORM create_semantic_memory(
                    content,
                    conf,
                    ARRAY['reflection', category],
                    NULL,
                    jsonb_build_object('heartbeat_id', p_heartbeat_id, 'source', 'reflect'),
                    0.6
                );
            END IF;
        END LOOP;
    END IF;

    IF p_result ? 'identity_updates' THEN
        FOR ident IN SELECT * FROM jsonb_array_elements(COALESCE(p_result->'identity_updates', '[]'::jsonb))
        LOOP
            aspect_type := COALESCE(ident->>'aspect_type', '');
            change_text := COALESCE(ident->>'change', '');
            reason_text := COALESCE(ident->>'reason', '');
            IF aspect_type <> '' AND change_text <> '' THEN
                INSERT INTO identity_aspects (aspect_type, content, stability)
                VALUES (
                    aspect_type,
                    jsonb_build_object('change', change_text, 'reason', reason_text, 'heartbeat_id', p_heartbeat_id),
                    0.5
                );
            END IF;
        END LOOP;
    END IF;

	    IF p_result ? 'worldview_updates' THEN
	        FOR wupd IN SELECT * FROM jsonb_array_elements(COALESCE(p_result->'worldview_updates', '[]'::jsonb))
	        LOOP
            wid := NULLIF(wupd->>'id', '')::uuid;
            new_conf := COALESCE((wupd->>'new_confidence')::float, NULL);
            IF wid IS NOT NULL AND new_conf IS NOT NULL THEN
                UPDATE worldview_primitives
                SET confidence = new_conf,
                    updated_at = CURRENT_TIMESTAMP
                WHERE id = wid;
            END IF;
	        END LOOP;
	    END IF;

	    -- Optional worldview evidence links: insert influences so trust/alignment can be computed and beliefs can update over time.
	    IF p_result ? 'worldview_influences' THEN
	        FOR winf IN SELECT * FROM jsonb_array_elements(COALESCE(p_result->'worldview_influences', '[]'::jsonb))
	        LOOP
	            BEGIN
	                wid := NULLIF(winf->>'worldview_id', '')::uuid;
	                wmem := NULLIF(winf->>'memory_id', '')::uuid;
	                wstrength := COALESCE(NULLIF(winf->>'strength', '')::float, NULL);
	                wtype := COALESCE(NULLIF(winf->>'influence_type', ''), 'evidence');

	                IF wid IS NOT NULL AND wmem IS NOT NULL AND wstrength IS NOT NULL THEN
	                    INSERT INTO worldview_memory_influences (worldview_id, memory_id, influence_type, strength)
	                    VALUES (wid, wmem, wtype, wstrength)
	                    ON CONFLICT (worldview_id, memory_id, influence_type) DO UPDATE
	                    SET strength = EXCLUDED.strength,
	                        created_at = CURRENT_TIMESTAMP;

	                    IF wstrength > 0 THEN
	                        PERFORM link_memory_supports_worldview(wmem, wid, LEAST(1.0, GREATEST(0.0, wstrength)));
	                    END IF;
	                END IF;
	            EXCEPTION
	                WHEN OTHERS THEN
	                    NULL;
	            END;
	        END LOOP;
	    END IF;

    IF p_result ? 'discovered_relationships' THEN
        FOR rel IN SELECT * FROM jsonb_array_elements(COALESCE(p_result->'discovered_relationships', '[]'::jsonb))
        LOOP
            BEGIN
                from_id := NULLIF(rel->>'from_id', '')::uuid;
                to_id := NULLIF(rel->>'to_id', '')::uuid;
                rel_type := (rel->>'type')::graph_edge_type;
                rel_conf := COALESCE((rel->>'confidence')::float, 0.8);
                IF from_id IS NOT NULL AND to_id IS NOT NULL THEN
                    PERFORM discover_relationship(from_id, to_id, rel_type, rel_conf, 'reflection', p_heartbeat_id, 'reflect');
                END IF;
            EXCEPTION
                WHEN OTHERS THEN
                    NULL;
            END;
        END LOOP;
    END IF;

    IF p_result ? 'contradictions_noted' THEN
        FOR contra IN SELECT * FROM jsonb_array_elements(COALESCE(p_result->'contradictions_noted', '[]'::jsonb))
        LOOP
            ma := NULLIF(contra->>'memory_a', '')::uuid;
            mb := NULLIF(contra->>'memory_b', '')::uuid;
            reason_text := COALESCE(contra->>'resolution', '');
            IF ma IS NOT NULL AND mb IS NOT NULL THEN
                PERFORM discover_relationship(
                    ma,
                    mb,
                    'CONTRADICTS',
                    0.8,
                    'reflection',
                    p_heartbeat_id,
                    COALESCE(reason_text, '')
                );
            END IF;
        END LOOP;
    END IF;

    -- Self-model updates (stored in graph as Self --[ASSOCIATED {kind}]--> ConceptNode).
    IF p_result ? 'self_updates' THEN
        FOR selfupd IN SELECT * FROM jsonb_array_elements(COALESCE(p_result->'self_updates', '[]'::jsonb))
        LOOP
            sm_kind := NULLIF(COALESCE(selfupd->>'kind', ''), '');
            sm_concept := NULLIF(COALESCE(selfupd->>'concept', ''), '');
            sm_strength := COALESCE(NULLIF(selfupd->>'strength', '')::float, 0.8);

            sm_evidence := NULL;
            BEGIN
                IF NULLIF(COALESCE(selfupd->>'evidence_memory_id', ''), '') IS NOT NULL THEN
                    sm_evidence := (selfupd->>'evidence_memory_id')::uuid;
                END IF;
            EXCEPTION
                WHEN OTHERS THEN
                    sm_evidence := NULL;
            END;

            PERFORM upsert_self_concept_edge(sm_kind, sm_concept, sm_strength, sm_evidence);
        END LOOP;
    END IF;
END;
$$ LANGUAGE plpgsql;

CREATE OR REPLACE FUNCTION execute_heartbeat_action(
    p_heartbeat_id UUID,
    p_action TEXT,
    p_params JSONB DEFAULT '{}'
)
RETURNS JSONB AS $$
DECLARE
    action_kind heartbeat_action;
    action_cost FLOAT;
    current_e FLOAT;
    result JSONB;
    queued_call_id UUID;
    outbox_id UUID;
    remembered_id UUID;
    boundary_hits JSONB;
    boundary_content TEXT;
BEGIN
    BEGIN
        action_kind := p_action::heartbeat_action;
    EXCEPTION
        WHEN invalid_text_representation THEN
            RETURN jsonb_build_object('success', false, 'error', 'Unknown action: ' || COALESCE(p_action, '<null>'));
    END;

    action_cost := get_action_cost(p_action);
    current_e := get_current_energy();

    IF current_e < action_cost THEN
        RETURN jsonb_build_object(
            'success', false,
            'error', 'Insufficient energy',
            'required', action_cost,
            'available', current_e
        );
    END IF;

    -- Boundary pre-checks for side-effects (no energy charge on refusal).
    IF p_action IN ('reach_out_public', 'synthesize') THEN
        boundary_content := COALESCE(p_params->>'content', '');
        SELECT COALESCE(jsonb_agg(row_to_json(r)), '[]'::jsonb)
        INTO boundary_hits
        FROM check_boundaries(boundary_content) r;

        IF boundary_hits IS NOT NULL AND jsonb_array_length(boundary_hits) > 0 THEN
            IF EXISTS (
                SELECT 1
                FROM jsonb_array_elements(boundary_hits) e
                WHERE e->>'response_type' = 'refuse'
            ) THEN
                RETURN jsonb_build_object(
                    'success', false,
                    'error', 'Boundary triggered',
                    'boundaries', boundary_hits
                );
            END IF;
        END IF;
    END IF;

    PERFORM update_energy(-action_cost);

    CASE p_action
        WHEN 'observe' THEN
            result := jsonb_build_object('environment', get_environment_snapshot());

        WHEN 'review_goals' THEN
            result := jsonb_build_object('goals', get_goals_snapshot());

        WHEN 'remember' THEN
            remembered_id := create_episodic_memory(
                p_content := COALESCE(p_params->>'content', ''),
                p_context := COALESCE(p_params, '{}'::jsonb) || jsonb_build_object('heartbeat_id', p_heartbeat_id),
                p_emotional_valence := COALESCE((p_params->>'emotional_valence')::float, 0),
                p_importance := COALESCE((p_params->>'importance')::float, 0.4)
            );
            result := jsonb_build_object('memory_id', remembered_id);

        WHEN 'recall' THEN
            SELECT jsonb_agg(row_to_json(r)) INTO result
            FROM fast_recall(p_params->>'query', COALESCE((p_params->>'limit')::int, 5)) r;
            result := jsonb_build_object('memories', COALESCE(result, '[]'::jsonb));
            PERFORM satisfy_drive('curiosity', 0.2);

        WHEN 'connect' THEN
            PERFORM create_memory_relationship(
                (p_params->>'from_id')::UUID,
                (p_params->>'to_id')::UUID,
                (p_params->>'relationship_type')::graph_edge_type,
                COALESCE(p_params->'properties', '{}'::jsonb)
            );
            result := jsonb_build_object('connected', true);
            PERFORM satisfy_drive('coherence', 0.1);

        WHEN 'reprioritize' THEN
            PERFORM change_goal_priority(
                (p_params->>'goal_id')::UUID,
                (p_params->>'new_priority')::goal_priority,
                p_params->>'reason'
            );
            IF (p_params->>'new_priority') = 'completed' THEN
                PERFORM satisfy_drive('competence', 0.4);
            END IF;
            result := jsonb_build_object('reprioritized', true);

        WHEN 'reflect' THEN
            INSERT INTO external_calls (call_type, input, heartbeat_id)
            VALUES (
                'think',
                jsonb_build_object(
                    'kind', 'reflect',
                    'recent_memories', get_recent_context(20),
                    'identity', get_identity_context(),
                    'worldview', get_worldview_context(),
                    'contradictions', (
                        SELECT COALESCE(jsonb_agg(row_to_json(t)), '[]'::jsonb)
                        FROM (SELECT * FROM find_contradictions(NULL) LIMIT 5) t
                    ),
                    'goals', get_goals_snapshot(),
                    'heartbeat_id', p_heartbeat_id,
                    'instructions', 'Analyze patterns. Note contradictions. Suggest identity updates. Discover relationships between memories.'
                ),
                p_heartbeat_id
            )
            RETURNING id INTO queued_call_id;
            result := jsonb_build_object('queued', true, 'external_call_id', queued_call_id);
            PERFORM satisfy_drive('coherence', 0.2);

        WHEN 'maintain' THEN
            IF p_params ? 'worldview_id' THEN
                UPDATE worldview_primitives
                SET confidence = COALESCE((p_params->>'new_confidence')::float, confidence),
                    updated_at = CURRENT_TIMESTAMP
                WHERE id = (p_params->>'worldview_id')::UUID;
            END IF;
            result := jsonb_build_object('maintained', true);
            PERFORM satisfy_drive('coherence', 0.1);

        WHEN 'brainstorm_goals' THEN
            INSERT INTO external_calls (call_type, input, heartbeat_id)
            VALUES (
                'think',
                jsonb_build_object(
                    'kind', 'brainstorm_goals',
                    'heartbeat_id', p_heartbeat_id,
                    'context', gather_turn_context(),
                    'params', COALESCE(p_params, '{}'::jsonb)
                ),
                p_heartbeat_id
            )
            RETURNING id INTO queued_call_id;
            result := jsonb_build_object('queued', true, 'external_call_id', queued_call_id);

        WHEN 'inquire_shallow', 'inquire_deep' THEN
            INSERT INTO external_calls (call_type, input, heartbeat_id)
            VALUES (
                'think',
                jsonb_build_object(
                    'kind', 'inquire',
                    'depth', p_action,
                    'heartbeat_id', p_heartbeat_id,
                    'query', COALESCE(p_params->>'query', p_params->>'question'),
                    'context', gather_turn_context(),
                    'params', COALESCE(p_params, '{}'::jsonb)
                ),
                p_heartbeat_id
            )
            RETURNING id INTO queued_call_id;
            result := jsonb_build_object('queued', true, 'external_call_id', queued_call_id);
            PERFORM satisfy_drive('curiosity', 0.2);

        WHEN 'synthesize' THEN
            DECLARE synth_id UUID;
            BEGIN
                synth_id := create_semantic_memory(
                    p_params->>'content',
                    COALESCE((p_params->>'confidence')::float, 0.8),
                    ARRAY['synthesis', COALESCE(p_params->>'topic', 'general')],
                    NULL,
                    jsonb_build_object('heartbeat_id', p_heartbeat_id, 'sources', p_params->'sources', 'boundaries', boundary_hits),
                    0.7
                );
                result := jsonb_build_object('synthesis_memory_id', synth_id, 'boundaries', boundary_hits);
            END;

        WHEN 'reach_out_user' THEN
            INSERT INTO outbox_messages (kind, payload)
            VALUES (
                'user',
                jsonb_build_object(
                    'message', p_params->>'message',
                    'intent', p_params->>'intent',
                    'heartbeat_id', p_heartbeat_id
                )
            )
            RETURNING id INTO outbox_id;
            result := jsonb_build_object('queued', true, 'outbox_id', outbox_id);
            PERFORM satisfy_drive('connection', 0.3);

        WHEN 'reach_out_public' THEN
            INSERT INTO outbox_messages (kind, payload)
            VALUES (
                'public',
                jsonb_build_object(
                    'platform', p_params->>'platform',
                    'content', p_params->>'content',
                    'heartbeat_id', p_heartbeat_id,
                    'boundaries', boundary_hits
                )
            )
            RETURNING id INTO outbox_id;
            result := jsonb_build_object('queued', true, 'outbox_id', outbox_id, 'boundaries', boundary_hits);
            PERFORM satisfy_drive('connection', 0.3);

        WHEN 'terminate' THEN
            -- Self-termination: wipe state and leave a single last-will memory.
            -- Expect params: { "last_will": "...", "farewells": [ { "message": "...", ... } ], "options": {...} }
            result := terminate_agent(
                COALESCE(NULLIF(p_params->>'last_will', ''), NULLIF(p_params->>'message', ''), NULLIF(p_params->>'reason', ''), ''),
                COALESCE(p_params->'farewells', '[]'::jsonb),
                COALESCE(p_params->'options', '{}'::jsonb)
            );

        WHEN 'rest' THEN
            result := jsonb_build_object('rested', true, 'energy_preserved', current_e - action_cost);
            PERFORM satisfy_drive('rest', 0.4);

        ELSE
            RETURN jsonb_build_object('success', false, 'error', 'Unknown action: ' || COALESCE(p_action, '<null>'));
    END CASE;

    RETURN jsonb_build_object(
        'success', true,
        'action', p_action,
        'cost', action_cost,
        'energy_remaining', get_current_energy(),
        'result', result
    );
END;
$$ LANGUAGE plpgsql;

-- ============================================================================
-- TIP OF TONGUE / PARTIAL ACTIVATION
-- ============================================================================

CREATE OR REPLACE FUNCTION find_partial_activations(
    p_query_text TEXT,
    p_cluster_threshold FLOAT DEFAULT 0.7,
    p_memory_threshold FLOAT DEFAULT 0.5
)
RETURNS TABLE (
    cluster_id UUID,
    cluster_name TEXT,
    keywords TEXT[],
    emotional_signature JSONB,
    cluster_similarity FLOAT,
    best_memory_similarity FLOAT
) AS $$
DECLARE
    query_embedding vector;
BEGIN
    query_embedding := safe_get_embedding(p_query_text);
    IF query_embedding IS NULL THEN
        RETURN;
    END IF;

    RETURN QUERY
    SELECT
        mc.id,
        mc.name,
        mc.keywords,
        mc.emotional_signature,
        (1 - (mc.centroid_embedding <=> query_embedding))::float as cluster_sim,
        MAX((1 - (m.embedding <=> query_embedding))::float) as best_mem_sim
    FROM memory_clusters mc
    JOIN memory_cluster_members mcm ON mc.id = mcm.cluster_id
    JOIN memories m ON mcm.memory_id = m.id
    WHERE m.status = 'active'
      AND mc.centroid_embedding IS NOT NULL
    GROUP BY mc.id, mc.name, mc.keywords, mc.emotional_signature, mc.centroid_embedding
    HAVING
        (1 - (mc.centroid_embedding <=> query_embedding)) >= p_cluster_threshold
        AND MAX(1 - (m.embedding <=> query_embedding)) < p_memory_threshold;
END;
$$ LANGUAGE plpgsql;

-- ============================================================================
-- VIEWS / HEALTH / WORKER GUIDANCE
-- ============================================================================

CREATE OR REPLACE VIEW cognitive_health AS
SELECT
    (SELECT current_energy FROM heartbeat_state WHERE id = 1) as energy,
    (SELECT value FROM heartbeat_config WHERE key = 'max_energy') as max_energy,
    (SELECT COUNT(*) FROM drives WHERE current_level >= urgency_threshold) as urgent_drives,
    (SELECT AVG(current_level) FROM drives) as avg_drive_level,
    (SELECT COUNT(*) FROM goals WHERE priority = 'active') as active_goals,
    (SELECT COUNT(*) FROM goals WHERE priority = 'active' AND blocked_by IS NOT NULL) as blocked_goals,
    (SELECT COUNT(*) FROM memories WHERE status = 'active') as total_memories,
    (SELECT COUNT(*) FROM memory_neighborhoods WHERE is_stale = TRUE) as stale_neighborhoods,
    (SELECT valence FROM current_emotional_state) as current_valence,
    (SELECT primary_emotion FROM current_emotional_state) as current_emotion,
    (SELECT COUNT(*) FROM heartbeat_log WHERE started_at > CURRENT_TIMESTAMP - INTERVAL '24 hours') as heartbeats_24h,
    (SELECT COUNT(*) FROM external_calls WHERE status = 'pending') as pending_calls,
    (SELECT COUNT(*) FROM relationship_discoveries WHERE created_at > CURRENT_TIMESTAMP - INTERVAL '24 hours') as relationships_discovered_24h;

CREATE OR REPLACE VIEW worker_tasks AS
SELECT
    'external_calls' as task_type,
    (SELECT COUNT(*) FROM external_calls WHERE status = 'pending') as pending_count,
    'Process LLM/embedding requests' as description
UNION ALL
SELECT
    'heartbeat',
    CASE WHEN should_run_heartbeat() THEN 1 ELSE 0 END,
    'Run heartbeat if due'
UNION ALL
SELECT
    'subconscious_maintenance',
    CASE WHEN should_run_maintenance() THEN 1 ELSE 0 END,
    'Run subconscious maintenance tick (consolidate + prune)'
UNION ALL
SELECT
    'outbox',
    (SELECT COUNT(*) FROM outbox_messages WHERE status = 'pending'),
    'Deliver pending messages';

COMMENT ON TABLE external_calls IS
'Queue for external calls. Worker processes pending calls and writes results.

Expected response formats by kind:

heartbeat_decision:
{
  "reasoning": "Internal monologue explaining thought process",
  "actions": [
    {"action": "recall", "params": {"query": "..."}},
    {"action": "reflect", "params": {}},
    {"action": "rest", "params": {}}
  ],
  "emotional_assessment": {"valence": 0.2, "arousal": 0.4, "primary_emotion": "curious"}
}

brainstorm_goals:
{
  "goals": [
    {"title": "...", "description": "...", "source": "curiosity", "priority": "queued"}
  ]
}

inquire:
{
  "summary": "Summary of research",
  "sources": ["url1", "url2"],
  "confidence": 0.8
}

reflect:
{
  "insights": [...],
  "identity_updates": [...],
  "worldview_updates": [...],
  "discovered_relationships": [...],
  "contradictions_noted": [...],
  "self_updates": [{"kind": "values", "concept": "honesty", "strength": 0.8, "evidence_memory_id": null}]
}';
