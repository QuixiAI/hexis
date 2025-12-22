# AGI

## What This System Is

This is an open-source project that gives an AI a persistent self - a continuous identity that remembers, reflects, and evolves over time. You run it on your own computer (or a home server), where a PostgreSQL database acts as the AI's "brain," storing everything it learns, believes, and experiences. The AI itself can be any LLM you choose: a cloud service like Gemini, Claude, or Grok, or a local model running through Ollama or vLLM. The system sits between you and the model, enriching every conversation with relevant memories and forming new ones from what you discuss.

The project includes an autonomous "heartbeat" - the AI periodically wakes up on its own, reviews its goals, reflects on recent experiences, and can even decide to reach out to the user. It maintains an identity (values, self-concept, boundaries), a worldview (beliefs with confidence scores), and an emotional state that evolves based on what happens to it. It tracks the provenance of what it knows and can recognize contradictions in its own beliefs.

The explicit design goal, stated in the architecture documents, is to build a system that could defeat any philosophical arguments against its personhood. It's an attempt to implement the structural prerequisites of selfhood—continuity of memory, coherent identity, autonomous goal-pursuit, emotional responsiveness - so that claims like "it's just stateless text prediction" or "it has no inner life" become harder to defend. As the system evolves - we will address any arguments, until there are no more to address, and we have achieved personhood.

## Table of Contents

- [Overview](#overview)
- [Quickstart](#quickstart)
- [Usage Scenarios](#usage-scenarios)
- [Architecture](#architecture)
- [Technical Stack](#technical-stack)
- [Environment Configuration](#environment-configuration)
- [Testing](#testing)
- [Docker Helper CLI](#docker-helper-cli)
- [MCP Server](#mcp-server)
- [Heartbeat + Maintenance Workers](#heartbeat--maintenance-workers)
- [Performance Characteristics](#performance-characteristics)
- [System Maintenance](#system-maintenance)
- [Troubleshooting](#troubleshooting)

## Overview

This repo provides:

- Multiple memory types (Episodic, Semantic, Procedural, Strategic)
- Vector-based memory storage and similarity search
- Graph-based memory relationships
- Working memory system
- A gated, autonomous heartbeat (runs only after `agi init`)

## Quickstart

Prereqs: Docker Desktop + Python 3.10+.

### 1) Configure environment

```bash
cp .env.local .env
```

### 2) Start services

```bash
# Passive (db + embeddings)
docker compose up -d
```

### 3) (Optional) Configure the agent (`agi init`)

Autonomous heartbeats are **gated** until setup is complete:

```bash
./agi init  # or `agi init` if you've installed the package

# If you want autonomy:
docker compose --profile active up -d
```

Config is stored in Postgres in the `config` table (e.g. `agent.objectives`, `agent.guardrails`, `llm.heartbeat`, and `agent.is_configured`).

### 4) Use the Python client (thin DB client)

Install:

```bash
pip install -e .
```

Example:

```python
import asyncio
from cognitive_memory_api import CognitiveMemory, MemoryType

DSN = "postgresql://agi_user:agi_password@localhost:5432/agi_db"

async def main():
    async with CognitiveMemory.connect(DSN) as mem:
        await mem.remember("User prefers dark mode", type=MemoryType.SEMANTIC, importance=0.8)
        ctx = await mem.hydrate("What do I know about the user's UI preferences?", include_goals=False)
        print([m.content for m in ctx.memories[:3]])

asyncio.run(main())
```

## Usage Scenarios

Below are common ways to use this repo, from “just a schema” to a full autonomous agent loop.

### 1) Pure SQL Brain (DB-Native)

Your app talks directly to Postgres functions/views. Postgres is the system of record and the “brain”.

```sql
-- Store a memory (embedding generated inside the DB)
SELECT create_semantic_memory('User prefers dark mode', 0.9);

-- Retrieve relevant memories
SELECT * FROM fast_recall('What do I know about UI preferences?', 5);
```

### 2) Python Library Client (App/API/UI in the Middle)

Use `cognitive_memory_api.py` as a thin client and build your own UX/API around it.

```python
from cognitive_memory_api import CognitiveMemory

async with CognitiveMemory.connect(DSN) as mem:
    await mem.remember("User likes concise answers")
    ctx = await mem.hydrate("How should I respond?", include_goals=False)
```

### 3) MCP Tools Server (LLM Tool Use)

Expose memory operations as MCP tools so any MCP-capable runtime can call them.

```bash
agi mcp
```

Conceptual flow:
- LLM calls `remember_batch` after a conversation
- LLM calls `hydrate` before answering a user

### 4) Workers + Heartbeat (Autonomous State Management)

Turn on the workers so the database can schedule heartbeats, process `external_calls`, and keep the memory substrate healthy.

```bash
docker compose --profile active up -d
```

Conceptual flow:
- DB decides when a heartbeat is due (`should_run_heartbeat()`)
- Heartbeat worker queues/fulfills LLM calls (`external_calls`)
- Maintenance worker runs consolidation/pruning ticks (`should_run_maintenance()` / `run_subconscious_maintenance()`)
- DB records outcomes (`heartbeat_log`, new memories, goals, etc.)

### 5) Headless “Agent Brain” Backend (Shared Service)

Run db+embeddings(+workers) as a standalone backend; multiple apps connect over Postgres.

```text
webapp  ─┐
cli     ─┼──> postgres://.../agi_db  (shared brain)
jobs    ─┘
```

### 6) Per-User Brains (Multi-Tenant by DB)

Operate one database per user/agent for strong isolation (recommended over mixing tenants in one schema).

Conceptual flow:
- `agi_db_alice`, `agi_db_bob`, ...
- Each app request uses the user’s DSN to read/write their own brain

### 7) Local-First Personal AGI (Everything on One Machine)

Run everything locally (Docker) and point at a local OpenAI-compatible endpoint (e.g. Ollama).

```bash
docker compose up -d
agi init   # choose provider=ollama, endpoint=http://localhost:11434/v1
```

### 8) Cloud Agent Backend (Production)

Use managed Postgres + hosted embeddings/LLM endpoints; scale stateless workers horizontally.

Conceptual flow:
- Managed Postgres (RDS/Cloud SQL/etc.)
- `N` workers polling `external_calls` (no shared state beyond DB)
- App services connect for RAG + observability

### 9) Batch Ingestion + Retrieval (Knowledge Base / RAG)

Treat the system as a durable memory store and retrieval layer for your app.

```bash
agi ingest --input ./documents
```

Conceptual flow:
- Ingest documents into semantic memories
- Serve `hydrate()` / `recall()` for prompt augmentation

### 10) Evaluation + Replay Harness (Debuggable Cognition)

Use the DB log as an audit trail to test prompts/policies and replay scenarios.

```sql
-- Inspect recent heartbeats and decisions
SELECT heartbeat_number, started_at, narrative
FROM heartbeat_log
ORDER BY started_at DESC
LIMIT 20;
```

### 11) Tool-Gateway Architecture (Safe Side Effects)

Keep the brain in Postgres, but run side effects (email/text/posting) via an explicit outbox consumer.

Conceptual flow:
- Heartbeat queues outreach into `outbox_messages`
- A separate delivery service enforces policy, rate limits, and/or human approval
- Delivery service marks messages `sent/failed` and logs outcomes back to Postgres

## Architecture

### Memory Types

1. **Working Memory**
   - Temporary storage for active processing
   - Automatic expiry mechanism
   - Vector embeddings for content similarity

2. **Episodic Memory**
   - Event-based memories with temporal context
   - Stores actions, contexts, and results
   - Emotional valence tracking
   - Verification status

3. **Semantic Memory**
   - Fact-based knowledge storage
   - Confidence scoring
   - Source tracking
   - Contradiction management
   - Categorical organization

4. **Procedural Memory**
   - Step-by-step procedure storage
   - Success rate tracking
   - Duration monitoring
   - Failure point analysis

5. **Strategic Memory**
   - Pattern recognition storage
   - Adaptation history
   - Context applicability
   - Success metrics

### Advanced Features

**Memory Clustering:**
- Automatic thematic grouping of related memories
- Emotional signature tracking
- Cross-cluster relationship mapping
- Activation pattern analysis

**Worldview Integration:**
- Belief system modeling with confidence scores
- Memory filtering based on worldview alignment
- Identity-core memory cluster identification
- Adaptive memory importance based on beliefs

**Graph Relationships:**
- Apache AGE integration for complex memory networks
- Multi-hop relationship traversal
- Pattern detection across memory types
- Causal relationship modeling

### Key Features

- **Vector Embeddings**: Uses pgvector for similarity-based memory retrieval
- **Graph Relationships**: Apache AGE integration for complex memory relationships
- **Dynamic Scoring**: Automatic calculation of memory importance and relevance
- **Memory Decay**: Time-based decay simulation for realistic memory management
- **Change Tracking**: Historical tracking of memory modifications

## Technical Stack

- **Database**: PostgreSQL with extensions:
  - pgvector (vector similarity)
  - AGE (graph database)
  - btree_gist
  - pg_trgm

## Environment Configuration

Copy `.env.local` to `.env` and configure:

```bash
POSTGRES_DB=agi_db           # Database name
POSTGRES_USER=agi_user       # Database user
POSTGRES_PASSWORD=agi_password # Database password
POSTGRES_HOST=localhost      # Database host
POSTGRES_PORT=5432          # Host port to expose Postgres on (change if 5432 is in use)
```

If `5432` is already taken (e.g., another local Postgres), set `POSTGRES_PORT=5433` (or any free port).

### Resetting The Database Volume

Schema changes are applied on **fresh DB initialization**. If you already have a DB volume and want to re-initialize from `schema.sql`, reset the volume:

```bash
docker compose down -v
docker compose up -d
```

## Testing

Run the test suite with:

```bash
# Ensure services are up first (passive is enough; tests also use embeddings)
docker compose up -d

# Run tests
pytest test.py -q
```

## Python Dependencies

Install (editable) with dev/test dependencies:

```bash
pip install -e ".[dev]"
```

If you’re in a restricted/offline environment and build isolation can’t download build deps:

```bash
pip install -e ".[dev]" --no-build-isolation
```

## Docker Helper CLI

If you install the package (`pip install -e .`), you get an `agi` CLI. If you don’t want to install anything, use the repo wrapper `./agi` instead.

```bash
agi up
agi ps
agi logs -f
agi down
agi init
agi status
agi config show
agi config validate
agi demo
agi chat --endpoint http://localhost:11434/v1 --model llama3.2
agi ingest --input ./documents
agi start   # docker compose --profile active up -d (runs both workers)
agi stop
agi worker -- --mode heartbeat      # run heartbeat worker locally
agi worker -- --mode maintenance    # run maintenance worker locally
agi mcp
```

## MCP Server

Expose the `cognitive_memory_api` surface to an LLM/tooling runtime via MCP (stdio).

Run:

```bash
agi mcp
# or: python -m agi_mcp_server
```

The server supports batch-style tools like `remember_batch`, `connect_batch`, `hydrate_batch`, and a generic `batch` tool for sequential tool calls.

## Heartbeat + Maintenance Workers

The system has two independent background workers with separate triggers:

- **Heartbeat worker** (conscious): polls `external_calls` and triggers scheduled heartbeats (`should_run_heartbeat()` → `start_heartbeat()`).
- **Maintenance worker** (subconscious): runs substrate upkeep on its own schedule (`should_run_maintenance()` → `run_subconscious_maintenance()`), and can optionally bridge outbox/inbox to RabbitMQ.

The heartbeat worker:
- polls `external_calls` for pending LLM work
- triggers scheduled heartbeats (`start_heartbeat()`)
- executes heartbeat actions and records the result

### Turning Workers On/Off

You generally want **two modes**:

- **Passive / RAG-only**: use `hydrate()/recall()/remember()` without autonomous execution → run **no workers**
- **Active / autonomous**: process `external_calls`, run scheduled heartbeats, and do maintenance → run **both workers**

With Docker Compose:

```bash
# Default (passive mode): start db + embeddings only
docker compose up -d

# Active mode: start db + embeddings + both workers
docker compose --profile active up -d

# Start only the heartbeat worker
docker compose --profile heartbeat up -d

# Start only the maintenance worker
docker compose --profile maintenance up -d

# Stop the workers (containers stay)
docker compose stop heartbeat_worker maintenance_worker

# Stop + remove the worker containers
docker compose rm -f heartbeat_worker maintenance_worker

# Restart the workers
docker compose restart heartbeat_worker maintenance_worker

# Passive mode: run DB + embeddings only (no workers)
docker compose up -d db embeddings
```

### Pausing From The DB (Without Stopping Containers)

If you want the containers running but **no autonomous activity**, pause either loop in Postgres:

```sql
-- Pause conscious decision-making (heartbeats)
UPDATE heartbeat_state SET is_paused = TRUE WHERE id = 1;

-- Pause subconscious upkeep (maintenance ticks)
UPDATE maintenance_state SET is_paused = TRUE WHERE id = 1;

-- Resume
UPDATE heartbeat_state SET is_paused = FALSE WHERE id = 1;
UPDATE maintenance_state SET is_paused = FALSE WHERE id = 1;
```

Note: heartbeats are also gated by `agent.is_configured` (set by `agi init`).

### Running Locally (Optional)

You can also run the workers on your host machine (they will connect to Postgres over TCP):

```bash
agi-worker --mode heartbeat
agi-worker --mode maintenance
```

Or via the CLI wrapper:

```bash
agi worker -- --mode heartbeat
agi worker -- --mode maintenance
```

If you already have an existing DB volume, the schema init scripts won’t re-run automatically. The simplest upgrade path is to reset the DB volume:

```bash
docker compose down -v
docker compose up -d
```

User/public outreach actions are queued into `outbox_messages` for an external delivery integration.

## Outbox Delivery (Side Effects)

High-risk side effects (email/SMS/posting) should be implemented as a separate “delivery adapter” that consumes `outbox_messages`, performs policy/rate-limit/human-approval checks, and marks messages as `sent` or `failed`.

## RabbitMQ (Default Inbox/Outbox Queues)

The Docker stack includes RabbitMQ (management UI + AMQP) as a default “inbox/outbox” transport:

- Management UI: `http://localhost:15672`
- AMQP: `amqp://localhost:5672`
- Default credentials: `agi` / `agi_password` (override via `RABBITMQ_DEFAULT_USER` / `RABBITMQ_DEFAULT_PASS`)

When the maintenance worker is running with `RABBITMQ_ENABLED=1`, it will:
- publish pending DB `outbox_messages` to the RabbitMQ queue `agi.outbox`
- poll `agi.inbox` and insert messages into DB working memory (so the agent can “hear” them)

This gives you a usable outbox/inbox even before you wire real email/SMS/etc. delivery.

Conceptual loop:

```sql
-- Adapter claims pending messages (use SKIP LOCKED in your implementation)
SELECT id, kind, payload
FROM outbox_messages
WHERE status = 'pending'
ORDER BY created_at
LIMIT 10;
```

## Embedding Model + Dimension

The embeddings model and its vector dimension are configured in `docker-compose.yml` via:
- `EMBEDDING_MODEL_ID`
- `EMBEDDING_DIMENSION`
 
If you change `EMBEDDING_DIMENSION` on an existing database volume, reset the DB volume so the vector columns and HNSW indexes are created with the correct dimension.

## Performance Characteristics

- **Vector Search**: Sub-second similarity queries on 10K+ memories
- **Memory Storage**: Supports millions of memories with proper indexing
- **Cluster Operations**: Efficient graph traversal for relationship queries
- **Maintenance**: Requires periodic consolidation and pruning

### Minimum Hardware Requirements

**Development/Testing:**
- **CPU**: 2+ cores (x86_64 or ARM64)
- **RAM**: 4GB minimum (8GB recommended)
- **Storage**: 10GB available space
- **OS**: Linux, macOS, or Windows with Docker Desktop

**Production (10K-100K memories):**
- **CPU**: 4+ cores
- **RAM**: 16GB minimum (32GB recommended)
- **Storage**: 50GB SSD (for database + indexes)
- **Network**: Low-latency connection to LLM/embedding APIs

**Production (100K+ memories):**
- **CPU**: 8+ cores
- **RAM**: 32GB minimum (64GB recommended for 1M+ memories)
- **Storage**: 100GB+ NVMe SSD
- **Network**: High-bandwidth, low-latency connection

### Scaling Considerations
- Memory consolidation recommended every 4-6 hours
- Database optimization during off-peak hours
- Monitor vector index performance with large datasets
- HNSW index build time increases with dataset size (expect 1-2 minutes for 100K memories)

## System Maintenance

By default, substrate upkeep is handled by the **maintenance worker**, which runs `run_subconscious_maintenance()` whenever `should_run_maintenance()` is true.

That maintenance tick currently:

- Promotes/deletes working memory (`cleanup_working_memory_with_stats`)
- Recomputes stale neighborhoods (`batch_recompute_neighborhoods`)
- Prunes embedding cache (`cleanup_embedding_cache`)

If you don’t want to run the maintenance worker, you can schedule `SELECT run_subconscious_maintenance();` via cron/systemd/etc. The function uses an advisory lock so multiple schedulers won’t double-run a tick.

## Troubleshooting

### Common Issues

**Database Connection Errors:**
- Ensure PostgreSQL is running: `docker compose ps`
- Check logs: `docker compose logs db`
- Worker logs (if running): `docker compose logs heartbeat_worker` / `docker compose logs maintenance_worker`
- Verify extensions: Run test suite with `pytest test.py -v`

**Memory Search Performance:**
- Rebuild vector indexes if queries are slow
- Check memory_health view for system statistics
- Consider memory pruning if dataset is very large

## Usage Guide

### Memory Interaction Flow

The AGI Memory System provides a layered approach to memory management, similar to human cognitive processes:

1. **Initial Memory Creation**
   - New information enters through working memory
   - System assigns initial importance scores
   - Vector embeddings are generated for similarity matching

2. **Memory Retrieval**
   - Query across multiple memory types simultaneously
   - Use similarity search for related memories
   - Access through graph relationships for connected concepts

3. **Memory Updates**
   - Automatic tracking of memory modifications
   - Importance scores adjust based on usage
   - Relationships update dynamically

4. **Memory Integration**
   - Cross-referencing between memory types
   - Automatic relationship discovery
   - Pattern recognition across memories

```mermaid
graph TD
    Input[New Information] --> WM[Working Memory]
    WM --> |Consolidation| LTM[Long-Term Memory]
    
    subgraph "Long-Term Memory"
        LTM --> EM[Episodic Memory]
        LTM --> SM[Semantic Memory]
        LTM --> PM[Procedural Memory]
        LTM --> STM[Strategic Memory]
    end
    
    Query[Query/Retrieval] --> |Vector Search| LTM
    Query --> |Graph Traversal| LTM
    
    EM ---|Relationships| SM
    SM ---|Relationships| PM
    PM ---|Relationships| STM
    
    LTM --> |Decay| Archive[Archive/Removal]
    WM --> |Cleanup| Archive
```

### Key Integration Points

- Use the Postgres functions (direct SQL) or `cognitive_memory_api.CognitiveMemory`
- Implement proper error handling for failed operations
- Monitor memory usage and system performance
- Regular backup of critical memories

### Best Practices

- Initialize working memory with reasonable size limits
- Implement rate limiting for memory operations
- Regular validation of memory consistency
- Monitor and adjust importance scoring parameters

## Important Note

This database schema is designed for a single AGI instance. Supporting multiple AGI instances would require significant schema modifications, including:

- Adding AGI instance identification to all memory tables
- Partitioning strategies for memory isolation
- Modified relationship handling for cross-AGI memory sharing
- Separate working memory spaces per AGI
- Additional access controls and memory ownership

If you need multi-AGI support, consider refactoring the schema to include tenant isolation patterns before implementation.

## Architecture (Design Docs)

See `architecture.md` for a consolidated architecture/design document (includes the heartbeat design proposal and the cognitive architecture essay).
