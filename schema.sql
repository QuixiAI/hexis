-- ============================================================================
-- AGI MEMORY SYSTEM - FINAL SCHEMA
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

LOAD 'age';
SET search_path = ag_catalog, "$user", public;

-- ============================================================================
-- GRAPH INITIALIZATION
-- ============================================================================

SELECT create_graph('memory_graph');
SELECT create_vlabel('memory_graph', 'MemoryNode');
SELECT create_vlabel('memory_graph', 'ConceptNode');

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
    expiry TIMESTAMPTZ
);

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
    influence_type TEXT,
    strength FLOAT,
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);

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
RETURNS vector(768) AS $$
DECLARE
    service_url TEXT;
    response http_response;
    request_body TEXT;
    embedding_array FLOAT[];
    embedding_json JSONB;
    v_content_hash TEXT;
    cached_embedding vector(768);
BEGIN
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

    -- Make HTTP request
    SELECT * INTO response FROM http_post(
        service_url,
        request_body,
        'application/json'
    );

    -- Check response status
    IF response.status != 200 THEN
        RAISE EXCEPTION 'Embedding service error: % - %', response.status, response.content;
    END IF;

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
    IF array_length(embedding_array, 1) != 768 THEN
        RAISE EXCEPTION 'Invalid embedding dimension: expected 768, got %', array_length(embedding_array, 1);
    END IF;

    -- Cache the result
    INSERT INTO embedding_cache (content_hash, embedding)
    VALUES (v_content_hash, embedding_array::vector(768))
    ON CONFLICT DO NOTHING;

    RETURN embedding_array::vector(768);
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
    query_embedding vector(768);
BEGIN
    query_embedding := get_embedding(p_query_text);
    
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
        ORDER BY m.embedding <=> query_embedding
        LIMIT 5
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
            calculate_relevance(m.importance, m.decay_rate, m.created_at, m.last_accessed) * 0.05,
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
    WHERE m.status = 'active'
    ORDER BY final_score DESC
    LIMIT p_limit;
END;
$$ LANGUAGE plpgsql;

-- Create memory (base function) - generates embedding automatically
CREATE OR REPLACE FUNCTION create_memory(
    p_type memory_type,
    p_content TEXT,
    p_importance FLOAT DEFAULT 0.5
) RETURNS UUID AS $$
DECLARE
    new_memory_id UUID;
    embedding_vec vector(768);
BEGIN
    -- Generate embedding
    embedding_vec := get_embedding(p_content);
    
    INSERT INTO memories (type, content, embedding, importance)
    VALUES (p_type, p_content, embedding_vec, p_importance)
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
    p_importance FLOAT DEFAULT 0.5
) RETURNS UUID AS $$
DECLARE
    new_memory_id UUID;
BEGIN
    new_memory_id := create_memory('episodic', p_content, p_importance);
    
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
    p_importance FLOAT DEFAULT 0.5
) RETURNS UUID AS $$
DECLARE
    new_memory_id UUID;
BEGIN
    new_memory_id := create_memory('semantic', p_content, p_importance);
    
    INSERT INTO semantic_memories (
        memory_id, confidence, category, related_concepts,
        source_references, last_validated
    ) VALUES (
        new_memory_id, p_confidence, p_category, p_related_concepts,
        p_source_references, CURRENT_TIMESTAMP
    );
    
    RETURN new_memory_id;
END;
$$ LANGUAGE plpgsql;

-- Create procedural memory
CREATE OR REPLACE FUNCTION create_procedural_memory(
    p_content TEXT,
    p_steps JSONB,
    p_prerequisites JSONB DEFAULT NULL,
    p_importance FLOAT DEFAULT 0.5
) RETURNS UUID AS $$
DECLARE
    new_memory_id UUID;
BEGIN
    new_memory_id := create_memory('procedural', p_content, p_importance);
    
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
    p_importance FLOAT DEFAULT 0.5
) RETURNS UUID AS $$
DECLARE
    new_memory_id UUID;
BEGIN
    new_memory_id := create_memory('strategic', p_content, p_importance);
    
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
    query_embedding vector(768);
BEGIN
    query_embedding := get_embedding(p_query_text);
    
    RETURN QUERY
    SELECT 
        m.id,
        m.content,
        m.type,
        1 - (m.embedding <=> query_embedding) as similarity,
        m.importance
    FROM memories m
    WHERE m.status = 'active'
    AND (p_memory_types IS NULL OR m.type = ANY(p_memory_types))
    AND m.importance >= p_min_importance
    ORDER BY m.embedding <=> query_embedding
    LIMIT p_limit;
END;
$$ LANGUAGE plpgsql;

-- Assign memory to clusters based on similarity
CREATE OR REPLACE FUNCTION assign_memory_to_clusters(
    p_memory_id UUID, 
    p_max_clusters INT DEFAULT 3
) RETURNS VOID AS $$
DECLARE
    memory_embedding vector(768);
    cluster_record RECORD;
    similarity_threshold FLOAT := 0.7;
    assigned_count INT := 0;
BEGIN
    SELECT embedding INTO memory_embedding
    FROM memories WHERE id = p_memory_id;
    
    FOR cluster_record IN 
        SELECT id, 1 - (centroid_embedding <=> memory_embedding) as similarity
        FROM memory_clusters
        WHERE centroid_embedding IS NOT NULL
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
    new_centroid vector(768);
BEGIN
    SELECT AVG(m.embedding)::vector(768)
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

-- Clean expired working memory
CREATE OR REPLACE FUNCTION cleanup_working_memory()
RETURNS INT AS $$
DECLARE
    deleted_count INT;
BEGIN
    WITH deleted AS (
        DELETE FROM working_memory
        WHERE expiry < CURRENT_TIMESTAMP
        RETURNING 1
    )
    SELECT COUNT(*) INTO deleted_count FROM deleted;
    
    RETURN deleted_count;
END;
$$ LANGUAGE plpgsql;

-- Add to working memory with auto-embedding
CREATE OR REPLACE FUNCTION add_to_working_memory(
    p_content TEXT,
    p_expiry INTERVAL DEFAULT INTERVAL '1 hour'
) RETURNS UUID AS $$
DECLARE
    new_id UUID;
    embedding_vec vector(768);
BEGIN
    embedding_vec := get_embedding(p_content);
    
    INSERT INTO working_memory (content, embedding, expiry)
    VALUES (p_content, embedding_vec, CURRENT_TIMESTAMP + p_expiry)
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
    query_embedding vector(768);
BEGIN
    query_embedding := get_embedding(p_query_text);
    
    -- Clean expired first
    PERFORM cleanup_working_memory();
    
    RETURN QUERY
    SELECT 
        wm.id,
        wm.content,
        1 - (wm.embedding <=> query_embedding) as similarity,
        wm.created_at
    FROM working_memory wm
    ORDER BY wm.embedding <=> query_embedding
    LIMIT p_limit;
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