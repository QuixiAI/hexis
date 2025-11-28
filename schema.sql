-- Required extensions
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS age;
CREATE EXTENSION IF NOT EXISTS btree_gist;
CREATE EXTENSION IF NOT EXISTS pg_trgm;
CREATE EXTENSION IF NOT EXISTS cube;
CREATE EXTENSION IF NOT EXISTS http;

-- Load AGE extension explicitly
LOAD 'age';
SET search_path = ag_catalog, "$user", public;

-- Create the graph
SELECT create_graph('memory_graph');
SELECT create_vlabel('memory_graph', 'MemoryNode');

-- Switch to public schema for our tables
SET search_path = public, ag_catalog, "$user";

-- Enums for memory types and status
CREATE TYPE memory_type AS ENUM ('episodic', 'semantic', 'procedural', 'strategic');
CREATE TYPE memory_status AS ENUM ('active', 'archived', 'invalidated');
CREATE TYPE cluster_type AS ENUM ('theme', 'emotion', 'temporal', 'person', 'pattern', 'mixed');

-- Working Memory (temporary table or in-memory structure)
CREATE TABLE working_memory (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    content TEXT NOT NULL,
    embedding vector(768) NOT NULL,
    expiry TIMESTAMPTZ
);

CREATE OR REPLACE FUNCTION age_in_days(created_at TIMESTAMPTZ) 
RETURNS FLOAT
IMMUTABLE
AS $$
BEGIN
    RETURN extract(epoch from (now() - created_at))/86400.0;
END;
$$ LANGUAGE plpgsql;

-- Base memory table with vector embeddings
CREATE TABLE memories (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    type memory_type NOT NULL,
    status memory_status DEFAULT 'active',
    content TEXT NOT NULL,
    embedding vector(768) NOT NULL,
    importance FLOAT DEFAULT 0.0,
    access_count INTEGER DEFAULT 0,
    last_accessed TIMESTAMPTZ,
    decay_rate FLOAT DEFAULT 0.01,
    relevance_score FLOAT GENERATED ALWAYS AS (
        importance * exp(-decay_rate * age_in_days(created_at))
    ) STORED
);

-- Memory clusters for thematic grouping
CREATE TABLE memory_clusters (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    cluster_type cluster_type NOT NULL,
    name TEXT NOT NULL,
    description TEXT,
    centroid_embedding vector(768), -- Average embedding of all memories in cluster
    emotional_signature JSONB, -- Common emotional patterns
    keywords TEXT[], -- Key terms associated with this cluster
    importance_score FLOAT DEFAULT 0.0,
    coherence_score FLOAT, -- How tightly related are the memories
    last_activated TIMESTAMPTZ,
    activation_count INTEGER DEFAULT 0,
    worldview_alignment FLOAT -- How much this cluster aligns with current worldview
);

-- Mapping between memories and clusters (many-to-many)
CREATE TABLE memory_cluster_members (
    cluster_id UUID REFERENCES memory_clusters(id) ON DELETE CASCADE,
    memory_id UUID REFERENCES memories(id) ON DELETE CASCADE,
    membership_strength FLOAT DEFAULT 1.0, -- How strongly this memory belongs to cluster
    added_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    contribution_to_centroid FLOAT, -- How much this memory shapes the cluster
    PRIMARY KEY (cluster_id, memory_id)
);

-- Relationships between clusters
CREATE TABLE cluster_relationships (
    from_cluster_id UUID REFERENCES memory_clusters(id) ON DELETE CASCADE,
    to_cluster_id UUID REFERENCES memory_clusters(id) ON DELETE CASCADE,
    relationship_type TEXT NOT NULL, -- 'causes', 'contradicts', 'supports', 'evolves_into'
    strength FLOAT DEFAULT 0.5,
    discovered_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    evidence_memories UUID[], -- Memory IDs that support this relationship
    PRIMARY KEY (from_cluster_id, to_cluster_id, relationship_type)
);

-- Cluster activation patterns
CREATE TABLE cluster_activation_history (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    cluster_id UUID REFERENCES memory_clusters(id),
    activated_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    activation_context TEXT, -- What triggered this activation
    activation_strength FLOAT,
    co_activated_clusters UUID[], -- Other clusters activated at same time
    resulting_insights JSONB -- Any new connections discovered
);

-- Episodic memories
CREATE TABLE episodic_memories (
    memory_id UUID PRIMARY KEY REFERENCES memories(id),
    action_taken JSONB,
    context JSONB,
    result JSONB,
    emotional_valence FLOAT,
    verification_status BOOLEAN,
    event_time TIMESTAMPTZ,
    CONSTRAINT valid_emotion CHECK (emotional_valence >= -1 AND emotional_valence <= 1)
);

-- Semantic memories
CREATE TABLE semantic_memories (
    memory_id UUID PRIMARY KEY REFERENCES memories(id),
    confidence FLOAT NOT NULL,
    last_validated TIMESTAMPTZ,
    source_references JSONB,
    contradictions JSONB,
    category TEXT[],
    related_concepts TEXT[],
    CONSTRAINT valid_confidence CHECK (confidence >= 0 AND confidence <= 1)
);

-- Procedural memories
CREATE TABLE procedural_memories (
    memory_id UUID PRIMARY KEY REFERENCES memories(id),
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

-- Strategic memories
CREATE TABLE strategic_memories (
    memory_id UUID PRIMARY KEY REFERENCES memories(id),
    pattern_description TEXT NOT NULL,
    supporting_evidence JSONB,
    confidence_score FLOAT,
    success_metrics JSONB,
    adaptation_history JSONB,
    context_applicability JSONB,
    CONSTRAINT valid_confidence CHECK (confidence_score >= 0 AND confidence_score <= 1)
);

-- Worldview primitives
CREATE TABLE worldview_primitives (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    category TEXT NOT NULL,
    belief TEXT NOT NULL,
    confidence FLOAT,
    emotional_valence FLOAT,
    stability_score FLOAT,
    connected_beliefs UUID[],
    activation_patterns JSONB,
    memory_filter_rules JSONB,
    influence_patterns JSONB,
    preferred_clusters UUID[] -- Clusters that align with this worldview
);

-- Track how worldview affects memory interpretation
CREATE TABLE worldview_memory_influences (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    worldview_id UUID REFERENCES worldview_primitives(id),
    memory_id UUID REFERENCES memories(id),
    influence_type TEXT,
    strength FLOAT,
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);

-- Identity model
CREATE TABLE identity_model (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    self_concept JSONB,
    agency_beliefs JSONB,
    purpose_framework JSONB,
    group_identifications JSONB,
    boundary_definitions JSONB,
    emotional_baseline JSONB,
    threat_sensitivity FLOAT,
    change_resistance FLOAT,
    core_memory_clusters UUID[] -- Clusters central to identity
);

-- Bridge between memories and identity
CREATE TABLE identity_memory_resonance (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    memory_id UUID REFERENCES memories(id),
    identity_aspect UUID REFERENCES identity_model(id),
    resonance_strength FLOAT,
    integration_status TEXT
);

-- Temporal tracking
CREATE TABLE memory_changes (
    change_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    memory_id UUID REFERENCES memories(id),
    changed_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    change_type TEXT NOT NULL,
    old_value JSONB,
    new_value JSONB
);

-- Indexes for performance
CREATE INDEX ON memories USING hnsw (embedding vector_cosine_ops);
CREATE INDEX ON memories (status);
CREATE INDEX ON memories USING GIN (content gin_trgm_ops);
CREATE INDEX ON memories (relevance_score DESC) WHERE status = 'active';
CREATE INDEX ON memory_clusters USING hnsw (centroid_embedding vector_cosine_ops);
CREATE INDEX ON memory_clusters (cluster_type, importance_score DESC);
CREATE INDEX ON memory_clusters (last_activated DESC);
CREATE INDEX ON memory_cluster_members (memory_id);
CREATE INDEX ON memory_cluster_members (cluster_id, membership_strength DESC);
CREATE INDEX ON cluster_relationships (from_cluster_id);
CREATE INDEX ON cluster_relationships (to_cluster_id);
CREATE INDEX ON worldview_memory_influences (memory_id, strength DESC);
CREATE INDEX ON identity_memory_resonance (memory_id, resonance_strength DESC);

-- Functions for memory management

-- Update memory timestamp
CREATE OR REPLACE FUNCTION update_memory_timestamp()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Update memory importance based on access
CREATE OR REPLACE FUNCTION update_memory_importance()
RETURNS TRIGGER AS $$
BEGIN
    NEW.importance = NEW.importance * (1.0 + (ln(NEW.access_count + 1) * 0.1));
    NEW.last_accessed = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Update cluster when accessed
CREATE OR REPLACE FUNCTION update_cluster_activation()
RETURNS TRIGGER AS $$
BEGIN
    NEW.last_activated = CURRENT_TIMESTAMP;
    NEW.activation_count = NEW.activation_count + 1;
    NEW.importance_score = NEW.importance_score * (1.0 + (ln(NEW.activation_count + 1) * 0.05));
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Function to recalculate cluster centroid
CREATE OR REPLACE FUNCTION recalculate_cluster_centroid(cluster_uuid UUID)
RETURNS VOID AS $$
DECLARE
    new_centroid vector(768);
BEGIN
    -- Calculate average embedding of all active memories in cluster
    SELECT AVG(m.embedding)::vector(768)
    INTO new_centroid
    FROM memories m
    JOIN memory_cluster_members mcm ON m.id = mcm.memory_id
    WHERE mcm.cluster_id = cluster_uuid
    AND m.status = 'active'
    AND mcm.membership_strength > 0.3;
    
    UPDATE memory_clusters
    SET centroid_embedding = new_centroid,
        updated_at = CURRENT_TIMESTAMP
    WHERE id = cluster_uuid;
END;
$$ LANGUAGE plpgsql;

-- Function to find or create cluster for a memory
CREATE OR REPLACE FUNCTION assign_memory_to_clusters(memory_uuid UUID, max_clusters INT DEFAULT 3)
RETURNS VOID AS $$
DECLARE
    memory_embedding vector(768);
    memory_content TEXT;
    cluster_record RECORD;
    similarity_threshold FLOAT := 0.7;
    assigned_count INT := 0;
    zero_vector vector(768) := array_fill(0::float, ARRAY[768])::vector;
BEGIN
    -- Get memory details
    SELECT embedding, content INTO memory_embedding, memory_content
    FROM memories WHERE id = memory_uuid;
    
    -- Find similar clusters
    FOR cluster_record IN 
        SELECT id, 1 - dist as similarity
        FROM (
            SELECT id, centroid_embedding <=> memory_embedding as dist
            FROM memory_clusters
            WHERE centroid_embedding IS NOT NULL
              AND centroid_embedding <> zero_vector
        ) distances
        WHERE dist::text <> 'NaN'
        ORDER BY dist
        LIMIT 10
    LOOP
        IF cluster_record.similarity >= similarity_threshold AND assigned_count < max_clusters THEN
            -- Add to cluster
            INSERT INTO memory_cluster_members (cluster_id, memory_id, membership_strength)
            VALUES (cluster_record.id, memory_uuid, cluster_record.similarity)
            ON CONFLICT DO NOTHING;
            
            assigned_count := assigned_count + 1;
        END IF;
    END LOOP;
    
    -- If no suitable clusters found, consider creating a new one
    -- (This would be triggered by application logic based on themes)
END;
$$ LANGUAGE plpgsql;

-- Create memory relationship in graph
CREATE OR REPLACE FUNCTION create_memory_relationship(
    from_id UUID,
    to_id UUID,
    relationship_type TEXT,
    properties JSONB DEFAULT '{}'
) RETURNS VOID AS $$
BEGIN
    EXECUTE format(
        'SELECT * FROM cypher(''memory_graph'', $q$
            MATCH (a:MemoryNode), (b:MemoryNode)
            WHERE a.memory_id = %L AND b.memory_id = %L
            CREATE (a)-[r:%s %s]->(b)
            RETURN r
        $q$) as (result agtype)',
        from_id,
        to_id,
        relationship_type,
        case when properties = '{}'::jsonb 
             then '' 
             else format('{%s}', 
                  (SELECT string_agg(format('%I: %s', key, value), ', ')
                   FROM jsonb_each(properties)))
        end
    );
END;
$$ LANGUAGE plpgsql;

-- Triggers
CREATE TRIGGER update_memory_timestamp
    BEFORE UPDATE ON memories
    FOR EACH ROW
    EXECUTE FUNCTION update_memory_timestamp();

CREATE TRIGGER update_importance_on_access
    BEFORE UPDATE ON memories
    FOR EACH ROW
    WHEN (NEW.access_count != OLD.access_count)
    EXECUTE FUNCTION update_memory_importance();

CREATE TRIGGER update_cluster_on_access
    BEFORE UPDATE ON memory_clusters
    FOR EACH ROW
    WHEN (NEW.activation_count != OLD.activation_count)
    EXECUTE FUNCTION update_cluster_activation();

-- Views for memory analysis

CREATE VIEW memory_health AS
SELECT 
    type,
    count(*) as total_memories,
    avg(importance) as avg_importance,
    avg(access_count) as avg_access_count,
    count(*) FILTER (WHERE last_accessed > CURRENT_TIMESTAMP - INTERVAL '1 day') as accessed_last_day,
    avg(relevance_score) as avg_relevance
FROM memories
GROUP BY type;

CREATE VIEW cluster_insights AS
SELECT 
    mc.id,
    mc.name,
    mc.cluster_type,
    mc.importance_score,
    mc.coherence_score,
    count(mcm.memory_id) as memory_count,
    mc.last_activated,
    array_agg(DISTINCT cr.to_cluster_id) as related_clusters
FROM memory_clusters mc
LEFT JOIN memory_cluster_members mcm ON mc.id = mcm.cluster_id
LEFT JOIN cluster_relationships cr ON mc.id = cr.from_cluster_id
GROUP BY mc.id, mc.name, mc.cluster_type, mc.importance_score, mc.coherence_score, mc.last_activated
ORDER BY mc.importance_score DESC;

CREATE VIEW active_themes AS
SELECT 
    mc.name as theme,
    mc.emotional_signature,
    mc.keywords,
    count(DISTINCT mch.id) as recent_activations,
    array_agg(DISTINCT mch.co_activated_clusters) FILTER (WHERE mch.co_activated_clusters IS NOT NULL) as associated_themes
FROM memory_clusters mc
JOIN cluster_activation_history mch ON mc.id = mch.cluster_id
WHERE mch.activated_at > CURRENT_TIMESTAMP - INTERVAL '7 days'
GROUP BY mc.id, mc.name, mc.emotional_signature, mc.keywords
ORDER BY count(DISTINCT mch.id) DESC;

-- Configuration table for embeddings service
CREATE TABLE IF NOT EXISTS embedding_config (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

-- Set default embeddings service URL (can be updated as needed)
INSERT INTO embedding_config (key, value) 
VALUES ('service_url', 'http://embeddings:80/embed')
ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value;

-- Embedding cache table for performance
CREATE TABLE IF NOT EXISTS embedding_cache (
    content_hash TEXT PRIMARY KEY,
    embedding vector(768) NOT NULL,
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);

-- Index for cache cleanup
CREATE INDEX ON embedding_cache (created_at);

-- Core function to get embeddings from the service
CREATE OR REPLACE FUNCTION get_embedding(text_content TEXT) 
RETURNS vector(768) AS $$
DECLARE
    service_url TEXT;
    response http_response;
    request_body TEXT;
    embedding_array FLOAT[];
    embedding_json JSONB;
    content_hash TEXT;
    cached_embedding vector(768);
BEGIN
    -- Generate hash for caching
    content_hash := encode(sha256(text_content::bytea), 'hex');
    
    -- Check cache first
    SELECT ec.embedding INTO cached_embedding
    FROM embedding_cache ec
    WHERE ec.content_hash = get_embedding.content_hash;
    
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
        -- Format: {"embeddings": [[...numbers...]]}
        embedding_array := ARRAY(
            SELECT jsonb_array_elements_text((embedding_json->'embeddings')->0)::FLOAT
        );
    ELSIF embedding_json ? 'embedding' THEN
        -- Format: {"embedding": [...numbers...]}
        embedding_array := ARRAY(
            SELECT jsonb_array_elements_text(embedding_json->'embedding')::FLOAT
        );
    ELSIF embedding_json ? 'data' THEN
        -- Format: {"data": [{"embedding": [...numbers...]}]}
        embedding_array := ARRAY(
            SELECT jsonb_array_elements_text((embedding_json->'data')->0->'embedding')::FLOAT
        );
    ELSE
        -- Direct array format: [...numbers...]
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
    VALUES (content_hash, embedding_array::vector(768))
    ON CONFLICT DO NOTHING;
    
    RETURN embedding_array::vector(768);
EXCEPTION
    WHEN OTHERS THEN
        RAISE EXCEPTION 'Failed to get embedding: %', SQLERRM;
END;
$$ LANGUAGE plpgsql;

-- Main function to create a memory with automatic embedding
CREATE OR REPLACE FUNCTION create_memory(
    p_type memory_type,
    p_content TEXT,
    p_importance FLOAT DEFAULT 0.5
) RETURNS UUID AS $$
DECLARE
    memory_id UUID;
    embedding_vec vector(768);
BEGIN
    -- Generate embedding
    embedding_vec := get_embedding(p_content);
    
    -- Insert memory
    INSERT INTO memories (type, content, embedding, importance)
    VALUES (p_type, p_content, embedding_vec, p_importance)
    RETURNING id INTO memory_id;
    
    -- Assign to clusters
    PERFORM assign_memory_to_clusters(memory_id);
    
    -- Create graph node
    EXECUTE format(
        'SELECT * FROM cypher(''memory_graph'', $q$
            CREATE (n:MemoryNode {memory_id: %L, type: %L, created_at: %L})
            RETURN n
        $q$) as (result agtype)',
        memory_id,
        p_type,
        CURRENT_TIMESTAMP
    );
    
    RETURN memory_id;
END;
$$ LANGUAGE plpgsql;

-- Create episodic memory with automatic embedding
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
    memory_id UUID;
BEGIN
    -- Create base memory
    memory_id := create_memory('episodic', p_content, p_importance);
    
    -- Insert episodic details
    INSERT INTO episodic_memories (
        memory_id, action_taken, context, result, 
        emotional_valence, event_time
    ) VALUES (
        memory_id, p_action_taken, p_context, p_result,
        p_emotional_valence, p_event_time
    );
    
    RETURN memory_id;
END;
$$ LANGUAGE plpgsql;

-- Create semantic memory with automatic embedding
CREATE OR REPLACE FUNCTION create_semantic_memory(
    p_content TEXT,
    p_confidence FLOAT,
    p_category TEXT[] DEFAULT NULL,
    p_related_concepts TEXT[] DEFAULT NULL,
    p_source_references JSONB DEFAULT NULL,
    p_importance FLOAT DEFAULT 0.5
) RETURNS UUID AS $$
DECLARE
    memory_id UUID;
BEGIN
    -- Create base memory
    memory_id := create_memory('semantic', p_content, p_importance);
    
    -- Insert semantic details
    INSERT INTO semantic_memories (
        memory_id, confidence, category, related_concepts,
        source_references, last_validated
    ) VALUES (
        memory_id, p_confidence, p_category, p_related_concepts,
        p_source_references, CURRENT_TIMESTAMP
    );
    
    RETURN memory_id;
END;
$$ LANGUAGE plpgsql;

-- Create procedural memory with automatic embedding
CREATE OR REPLACE FUNCTION create_procedural_memory(
    p_content TEXT,
    p_steps JSONB,
    p_prerequisites JSONB DEFAULT NULL,
    p_importance FLOAT DEFAULT 0.5
) RETURNS UUID AS $$
DECLARE
    memory_id UUID;
BEGIN
    -- Create base memory
    memory_id := create_memory('procedural', p_content, p_importance);
    
    -- Insert procedural details
    INSERT INTO procedural_memories (
        memory_id, steps, prerequisites
    ) VALUES (
        memory_id, p_steps, p_prerequisites
    );
    
    RETURN memory_id;
END;
$$ LANGUAGE plpgsql;

-- Create strategic memory with automatic embedding
CREATE OR REPLACE FUNCTION create_strategic_memory(
    p_content TEXT,
    p_pattern_description TEXT,
    p_confidence_score FLOAT,
    p_supporting_evidence JSONB DEFAULT NULL,
    p_context_applicability JSONB DEFAULT NULL,
    p_importance FLOAT DEFAULT 0.5
) RETURNS UUID AS $$
DECLARE
    memory_id UUID;
BEGIN
    -- Create base memory
    memory_id := create_memory('strategic', p_content, p_importance);
    
    -- Insert strategic details
    INSERT INTO strategic_memories (
        memory_id, pattern_description, confidence_score,
        supporting_evidence, context_applicability
    ) VALUES (
        memory_id, p_pattern_description, p_confidence_score,
        p_supporting_evidence, p_context_applicability
    );
    
    RETURN memory_id;
END;
$$ LANGUAGE plpgsql;

-- Function to add memory to working memory with automatic embedding
CREATE OR REPLACE FUNCTION add_to_working_memory(
    p_content TEXT,
    p_expiry INTERVAL DEFAULT INTERVAL '1 hour'
) RETURNS UUID AS $$
DECLARE
    memory_id UUID;
    embedding_vec vector(768);
BEGIN
    -- Generate embedding
    embedding_vec := get_embedding(p_content);
    
    -- Insert into working memory
    INSERT INTO working_memory (content, embedding, expiry)
    VALUES (p_content, embedding_vec, CURRENT_TIMESTAMP + p_expiry)
    RETURNING id INTO memory_id;
    
    RETURN memory_id;
END;
$$ LANGUAGE plpgsql;

-- Search memories by semantic similarity (with automatic embedding)
CREATE OR REPLACE FUNCTION search_similar_memories(
    p_query_text TEXT,
    p_limit INT DEFAULT 10,
    p_memory_types memory_type[] DEFAULT NULL,
    p_min_relevance FLOAT DEFAULT 0.0
) RETURNS TABLE (
    memory_id UUID,
    content TEXT,
    type memory_type,
    similarity FLOAT,
    relevance_score FLOAT,
    importance FLOAT
) AS $$
DECLARE
    query_embedding vector(768);
BEGIN
    -- Generate embedding for query
    query_embedding := get_embedding(p_query_text);
    
    -- Search memories
    RETURN QUERY
    SELECT 
        m.id,
        m.content,
        m.type,
        1 - (m.embedding <=> query_embedding) as similarity,
        m.relevance_score,
        m.importance
    FROM memories m
    WHERE m.status = 'active'
    AND (p_memory_types IS NULL OR m.type = ANY(p_memory_types))
    AND m.relevance_score >= p_min_relevance
    ORDER BY m.embedding <=> query_embedding
    LIMIT p_limit;
END;
$$ LANGUAGE plpgsql;

-- Find memories in working memory by similarity
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
    -- Generate embedding for query
    query_embedding := get_embedding(p_query_text);
    
    -- Clean expired memories first
    DELETE FROM working_memory WHERE expiry < CURRENT_TIMESTAMP;
    
    -- Search working memory
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

-- Create new cluster with automatic centroid embedding
CREATE OR REPLACE FUNCTION create_memory_cluster(
    p_name TEXT,
    p_cluster_type cluster_type,
    p_description TEXT DEFAULT NULL,
    p_initial_memories UUID[] DEFAULT NULL
) RETURNS UUID AS $$
DECLARE
    cluster_id UUID;
    centroid_vec vector(768);
    memory_embeddings vector(768)[];
    i INT;
BEGIN
    -- If initial memories provided, calculate centroid
    IF p_initial_memories IS NOT NULL AND array_length(p_initial_memories, 1) > 0 THEN
        -- Calculate average (simplified - in production, use proper vector averaging)
        SELECT AVG(embedding)::vector(768) INTO centroid_vec
        FROM memories
        WHERE id = ANY(p_initial_memories)
        AND status = 'active';
    END IF;
    
    -- Create cluster
    INSERT INTO memory_clusters (name, cluster_type, description, centroid_embedding)
    VALUES (p_name, p_cluster_type, p_description, centroid_vec)
    RETURNING id INTO cluster_id;
    
    -- Add initial memories to cluster
    IF p_initial_memories IS NOT NULL THEN
        FOR i IN 1..array_length(p_initial_memories, 1) LOOP
            INSERT INTO memory_cluster_members (cluster_id, memory_id)
            VALUES (cluster_id, p_initial_memories[i]);
        END LOOP;
    END IF;
    
    RETURN cluster_id;
END;
$$ LANGUAGE plpgsql;

-- Batch create memories (more efficient for multiple memories)
CREATE OR REPLACE FUNCTION batch_create_memories(
    p_memories JSONB -- Array of {type, content, importance}
) RETURNS UUID[] AS $$
DECLARE
    memory_ids UUID[];
    memory_record JSONB;
    new_memory_id UUID;
BEGIN
    -- Process each memory
    FOR memory_record IN SELECT * FROM jsonb_array_elements(p_memories)
    LOOP
        new_memory_id := create_memory(
            (memory_record->>'type')::memory_type,
            memory_record->>'content',
            COALESCE((memory_record->>'importance')::FLOAT, 0.5)
        );
        memory_ids := array_append(memory_ids, new_memory_id);
    END LOOP;
    
    RETURN memory_ids;
END;
$$ LANGUAGE plpgsql;

-- Monitor embedding service health
CREATE OR REPLACE FUNCTION check_embedding_service_health()
RETURNS BOOLEAN AS $$
DECLARE
    service_url TEXT;
    response http_response;
BEGIN
    SELECT value INTO service_url FROM embedding_config WHERE key = 'service_url';
    
    SELECT * INTO response FROM http_get(replace(service_url, '/embed', '/health'));
    
    RETURN response.status = 200;
EXCEPTION
    WHEN OTHERS THEN
        RETURN FALSE;
END;
$$ LANGUAGE plpgsql;

-- Clean old embedding cache entries
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

-- Function comments for documentation
COMMENT ON FUNCTION create_memory IS 'Creates a memory with automatic embedding generation. Usage: SELECT create_memory(''semantic'', ''The user prefers dark mode interfaces'', 0.7);';

COMMENT ON FUNCTION search_similar_memories IS 'Search memories by semantic similarity. Usage: SELECT * FROM search_similar_memories(''user interface preferences'', 10);';

COMMENT ON FUNCTION create_episodic_memory IS 'Creates an episodic memory with automatic embedding. Usage: SELECT create_episodic_memory(''User clicked help button'', ''{"action": "click"}'', ''{"page": "settings"}'', ''{"displayed": "help_modal"}'', 0.0);';

COMMENT ON FUNCTION create_semantic_memory IS 'Creates a semantic memory with automatic embedding. Usage: SELECT create_semantic_memory(''User prefers dark themes'', 0.9, ARRAY[''preference''], ARRAY[''UI'', ''theme'']);';

COMMENT ON FUNCTION create_procedural_memory IS 'Creates a procedural memory with automatic embedding. Usage: SELECT create_procedural_memory(''How to reset password'', ''{"steps": ["click forgot", "enter email", "check inbox"]}'');';

COMMENT ON FUNCTION create_strategic_memory IS 'Creates a strategic memory with automatic embedding. Usage: SELECT create_strategic_memory(''User engagement pattern'', ''Users engage more with visual content'', 0.8);';

COMMENT ON FUNCTION batch_create_memories IS 'Batch create multiple memories efficiently. Usage: SELECT batch_create_memories(''[{"type": "semantic", "content": "fact1"}, {"type": "episodic", "content": "event1"}]'');';

COMMENT ON FUNCTION check_embedding_service_health IS 'Check if the embedding service is healthy. Usage: SELECT check_embedding_service_health();';
