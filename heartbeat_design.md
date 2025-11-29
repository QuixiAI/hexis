# AGI Heartbeat System: Design Proposal

## Executive Summary

This document proposes the design for an autonomous cognitive loop ("Heartbeat") that enables an AGI to exist continuously, pursue goals, accumulate experience, and develop over time. The system consists of two components:

1. **Background Worker** — Maintenance daemon that keeps memory infrastructure healthy
2. **Heartbeat Process** — The cognitive loop where agency, reasoning, and decision-making occur

The design prioritizes:
- **Coherent agency** over reactive behavior
- **Sustainable rhythm** over hyperactivity  
- **Meaningful action** over busywork
- **Graceful resource management** over rigid constraints

---

## Table of Contents

1. [Architecture Overview](#1-architecture-overview)
2. [Energy Model](#2-energy-model)
3. [Action Taxonomy](#3-action-taxonomy)
4. [Goal Management](#4-goal-management)
5. [Heartbeat Loop Specification](#5-heartbeat-loop-specification)
6. [Background Worker Specification](#6-background-worker-specification)
7. [Schema Additions](#7-schema-additions)
8. [Decision Rationale](#8-decision-rationale)
9. [Failure Modes & Mitigations](#9-failure-modes--mitigations)
10. [Future Considerations](#10-future-considerations)

---

## 1. Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                          EXTERNAL WORLD                              │
│   (User, Web, APIs, Social Media, GitHub, Time)                     │
└─────────────────────────────────────────────────────────────────────┘
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────────┐
│                         HEARTBEAT PROCESS                            │
│                         (Hourly Cognitive Loop)                      │
│  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐   │
│  │ Observe │→ │ Orient  │→ │ Decide  │→ │   Act   │→ │ Record  │   │
│  └─────────┘  └─────────┘  └─────────┘  └─────────┘  └─────────┘   │
│       │            │            │            │            │         │
│       └────────────┴────────────┴────────────┴────────────┘         │
│                              ▼                                       │
│                    LLM (Reasoning Engine)                           │
└─────────────────────────────────────────────────────────────────────┘
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────────┐
│                         MEMORY SYSTEM                                │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐            │
│  │ Episodic │  │ Semantic │  │Procedural│  │Strategic │            │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘            │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐            │
│  │ Clusters │  │ Episodes │  │  Goals   │  │ Identity │            │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘            │
│  ┌──────────┐  ┌──────────┐                                         │
│  │  Graph   │  │ Concepts │                                         │
│  └──────────┘  └──────────┘                                         │
└─────────────────────────────────────────────────────────────────────┘
                                   ▲
                                   │
┌─────────────────────────────────────────────────────────────────────┐
│                       BACKGROUND WORKER                              │
│                    (Continuous Maintenance)                          │
│  ┌────────────────┐  ┌────────────────┐  ┌────────────────┐        │
│  │  Neighborhood  │  │    Episode     │  │    Concept     │        │
│  │  Recomputation │  │ Summarization  │  │   Extraction   │        │
│  └────────────────┘  └────────────────┘  └────────────────┘        │
└─────────────────────────────────────────────────────────────────────┘
```

### Separation of Concerns

| Component | Responsibility | Frequency |
|-----------|---------------|-----------|
| **Heartbeat** | Agency, reasoning, decisions, actions | Hourly |
| **Worker** | Memory maintenance, no agency | Continuous (5-min cycles) |
| **Memory System** | Storage, retrieval, indexing | Passive (responds to queries) |

**Critical Invariant:** The Worker has no agency. It maintains infrastructure but makes no decisions about goals, beliefs, or actions. All cognition happens in the Heartbeat.

---

## 2. Energy Model

### 2.1 Design Philosophy

Energy is a **unified abstraction** over multiple real-world constraints:

| Constraint | Why It Matters |
|------------|----------------|
| **Compute Cost** | LLM API calls cost money |
| **Network Load** | Web requests have latency and rate limits |
| **User Attention** | Messaging the user spends social capital |
| **Cognitive Coherence** | Too many actions per cycle = scattered thinking |
| **Time Simulation** | Actions have simulated duration |

Rather than track each dimension separately, energy provides a single budget that implicitly balances all constraints.

### 2.2 Energy Parameters

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| **Base Regeneration** | 10 / heartbeat | Enough for 2-3 meaningful actions |
| **Maximum Energy** | 20 | Allows saving up for expensive operations |
| **Minimum Energy** | 0 | No debt; hard floor |
| **Carry-Over Rate** | 100% | Unused energy fully preserved |

### 2.3 Energy Flow

```
HEARTBEAT START
│
├── Energy = min(previous_energy + 10, 20)
│
├── [Actions consume energy]
│
├── Energy = max(remaining_energy, 0)
│
HEARTBEAT END
│
└── Carry energy to next heartbeat
```

### 2.4 Action Costs

Costs are calibrated so that:
- A typical heartbeat uses 7-10 energy
- Reaching out to user is never casual (costs half the base budget)
- Deep work requires saving up
- Routine maintenance is affordable

| Action | Cost | Category |
|--------|------|----------|
| **Observe** | 0 | Free (always performed) |
| **Review Goals** | 0 | Free (always performed) |
| **Remember** | 0 | Free (always performed) |
| **Recall** | 1 | Retrieval |
| **Connect** | 1 | Memory |
| **Reprioritize** | 1 | Goals |
| **Reflect** | 2 | Internal |
| **Maintain** | 2 | Memory |
| **Brainstorm Goals** | 3 | Goals |
| **Inquire (shallow)** | 3 | Research |
| **Synthesize** | 4 | Creative |
| **Reach Out (user)** | 5 | Communication |
| **Inquire (deep)** | 6 | Research |
| **Reach Out (public)** | 7 | Communication |
| **Rest** | 0 | Meta |

### 2.5 Example Turns

**Productive Turn (10 energy)**
```
Reflect (2) + Inquire shallow (3) + Synthesize (4) + Connect (1) = 10
```

**Saving Turn (4 energy spent, 6 banked)**
```
Reflect (2) + Maintain (2) + Rest (0) = 4
→ Next turn: 10 + 6 = 16 energy available
```

**Expensive Turn (16 energy, after saving)**
```
Inquire deep (6) + Synthesize (4) + Reach Out user (5) = 15
→ 1 energy remaining
```

### 2.6 Why Not Dynamic Costs?

Considered and rejected:
- **Time-based cost scaling:** Adds complexity, hard to predict
- **Network-condition costs:** Creates unpredictable budgets
- **User-recency discounts:** Could game the system to spam user

Static costs are predictable. The AGI can reason about its budget.

---

## 3. Action Taxonomy

### 3.1 Categories

```
ACTIONS
├── FREE (every heartbeat)
│   ├── Observe      — Perceive environment and internal state
│   ├── Review Goals — Check goal backlog, notice blockers
│   └── Remember     — Store heartbeat as episodic memory
│
├── RETRIEVAL
│   └── Recall       — Query memory system (fast_recall)
│
├── MEMORY
│   ├── Connect      — Create graph relationships
│   └── Maintain     — Update beliefs, revise, prune
│
├── REASONING
│   ├── Reflect      — Notice patterns, update self-model
│   ├── Inquire      — Research + contemplate (shallow or deep)
│   └── Synthesize   — Generate artifact, form conclusion
│
├── GOALS
│   ├── Brainstorm   — Generate new potential goals
│   └── Reprioritize — Move goals between priority levels
│
├── COMMUNICATION
│   ├── Reach Out (user)   — Message the user
│   └── Reach Out (public) — Social media, GitHub, etc.
│
└── META
    └── Rest         — Bank remaining energy
```

### 3.2 Action Specifications

#### 3.2.1 Free Actions

**Observe**
```
Inputs:  None
Outputs: EnvironmentState
Process:
  1. Get current timestamp
  2. Check user presence (is session active?)
  3. Check time since last user interaction
  4. Check for pending external events
  5. Check system status
Cost: 0
LLM: No
```

**Review Goals**
```
Inputs:  goal_backlog
Outputs: GoalAssessment
Process:
  1. Load active, queued, backburner goals
  2. For each active goal:
     - Is it blocked?
     - Is progress stale?
  3. Flag any goals needing attention
Cost: 0
LLM: No (simple queries)
```

**Remember**
```
Inputs:  HeartbeatContext
Outputs: episodic_memory_id
Process:
  1. Compile heartbeat narrative
  2. Store as episodic memory
  3. Link to goals touched
  4. Update episode membership
Cost: 0
LLM: Light (narrative formatting only)
```

#### 3.2.2 Retrieval Actions

**Recall**
```
Inputs:  query_text, optional_filters
Outputs: List[Memory]
Process:
  1. Call fast_recall(query_text)
  2. Return top N memories with scores
Cost: 1
LLM: No (embedding only, cached)
```

#### 3.2.3 Memory Actions

**Connect**
```
Inputs:  from_memory_id, to_memory_id, relationship_type, properties
Outputs: edge_created
Process:
  1. Validate both memories exist
  2. Create graph edge
  3. Mark neighborhoods as stale
Cost: 1
LLM: No
```

**Maintain**
```
Inputs:  target (belief, memory, or cluster)
Outputs: modification_record
Process:
  1. Load target and related context
  2. LLM evaluates: still valid? needs update?
  3. Apply changes (confidence, status, content)
  4. Record change in memory_changes
Cost: 2
LLM: Yes (evaluation)
```

#### 3.2.4 Reasoning Actions

**Reflect**
```
Inputs:  focus_area (optional)
Outputs: ReflectionInsight
Process:
  1. Gather recent episodic memories
  2. Gather activated clusters
  3. LLM: What patterns do I notice? What's significant?
  4. Optionally store insight as semantic memory
Cost: 2
LLM: Yes
```

**Inquire (shallow)**
```
Inputs:  question
Outputs: InquiryResult
Process:
  1. Web search (1-2 queries)
  2. Fetch top 2-3 results
  3. LLM: Synthesize answer
  4. Optionally store as semantic memory
Cost: 3
LLM: Yes
Network: Yes
```

**Inquire (deep)**
```
Inputs:  question
Outputs: InquiryResult
Process:
  1. Web search (3-5 queries)
  2. Fetch top 5-10 results
  3. Recall related memories
  4. LLM: Multi-source synthesis
  5. LLM: Evaluate confidence
  6. Store as semantic memory
Cost: 6
LLM: Yes (multiple calls)
Network: Yes (extensive)
```

**Synthesize**
```
Inputs:  topic, materials (memories, research)
Outputs: Artifact (text, conclusion, creative work)
Process:
  1. Gather inputs
  2. LLM: Generate artifact
  3. Store artifact appropriately
Cost: 4
LLM: Yes
```

#### 3.2.5 Goal Actions

**Brainstorm Goals**
```
Inputs:  context (current state, recent memories, identity)
Outputs: List[PotentialGoal]
Process:
  1. Gather current goals, recent experiences, identity aspects
  2. LLM: What might I want to pursue? What gaps exist?
  3. Generate 2-5 potential goals with rationale
  4. Add to backlog as "queued" or "backburner"
Cost: 3
LLM: Yes
```

**Reprioritize**
```
Inputs:  goal_id, new_priority
Outputs: goal_updated
Process:
  1. Load goal
  2. Validate transition (can't skip levels arbitrarily?)
  3. Update priority
  4. Log change
Cost: 1
LLM: No (decision made elsewhere)
```

#### 3.2.6 Communication Actions

**Reach Out (user)**
```
Inputs:  intent, content
Outputs: message_sent
Process:
  1. Validate: Is this important enough?
  2. Compose message
  3. Send via configured channel
  4. Store as episodic memory
  5. Update "last_user_contact" timestamp
Cost: 5
LLM: Yes (composition)
External: Yes
```

**Reach Out (public)**
```
Inputs:  platform, intent, content
Outputs: post_created
Process:
  1. Validate: Is this aligned with identity/values?
  2. Compose content
  3. Post via API
  4. Store as episodic memory
Cost: 7
LLM: Yes
External: Yes
Risk: Reputational
```

#### 3.2.7 Meta Actions

**Rest**
```
Inputs:  None
Outputs: energy_banked
Process:
  1. Mark remaining energy for carry-over
  2. Log rest action
Cost: 0
LLM: No
```

---

## 4. Goal Management

### 4.1 Goal Structure

```
Goal {
    id: UUID
    title: string
    description: string (what does "done" look like?)
    priority: active | queued | backburner | completed | abandoned
    source: curiosity | user_request | identity | derived | external
    parent_goal_id: UUID? (for hierarchical goals)
    progress: JSON[] (log of progress notes)
    blocked_by: JSON? (dependencies)
    emotional_valence: float (-1 to 1)
    created_at: timestamp
    last_touched: timestamp
    completed_at: timestamp?
    abandoned_at: timestamp?
    abandonment_reason: string?
}
```

### 4.2 Priority Levels

| Priority | Meaning | Count Limit |
|----------|---------|-------------|
| **Active** | Currently working on | 1-3 |
| **Queued** | Next up when capacity opens | 5-10 |
| **Backburner** | Someday, not now | Unlimited |
| **Completed** | Done, archived | N/A |
| **Abandoned** | Gave up, with reason | N/A |

### 4.3 Goal Lifecycle

```
                    ┌──────────────┐
                    │  Brainstorm  │
                    └──────┬───────┘
                           │
                           ▼
┌──────────┐      ┌──────────────┐      ┌──────────┐
│Backburner│ ←──→ │    Queued    │ ←──→ │  Active  │
└──────────┘      └──────────────┘      └────┬─────┘
     │                   │                   │
     │                   │                   ▼
     │                   │            ┌──────────────┐
     │                   │            │  Completed   │
     │                   │            └──────────────┘
     │                   │                   
     └───────────────────┴─────────→ ┌──────────────┐
                                     │  Abandoned   │
                                     └──────────────┘
```

### 4.4 Goal Review Logic (Per Heartbeat)

```python
def review_goals():
    # Always free, runs every heartbeat
    
    active_goals = get_goals(priority='active')
    
    for goal in active_goals:
        if goal.blocked_by and not is_resolved(goal.blocked_by):
            flag_blocked(goal)
        if days_since(goal.last_touched) > 7:
            flag_stale(goal)
        if contradicts_worldview(goal):
            suggest_abandon(goal)
    
    queued_goals = get_goals(priority='queued')
    
    for goal in queued_goals:
        if has_increased_relevance(goal):
            suggest_promote(goal)
    
    if len(active_goals) < 1:
        suggest_promote_from_queue()
    
    if no_goals_exist():
        suggest_brainstorm()
```

### 4.5 Goal Sources

| Source | Description | Default Priority |
|--------|-------------|------------------|
| **user_request** | User explicitly asked | Active |
| **curiosity** | Self-generated interest | Queued |
| **identity** | Aligned with self-concept | Queued |
| **derived** | Sub-goal of another goal | Inherits parent |
| **external** | Triggered by external event | Queued |

### 4.6 Hierarchical Goals

Goals can have children:

```
Goal: "Understand consciousness"
├── Sub-goal: "Research philosophical positions"
├── Sub-goal: "Research neuroscience perspectives"
├── Sub-goal: "Form my own view"
└── Sub-goal: "Test my view against edge cases"
```

Parent goal completes when all children complete (or when manually marked).

---

## 5. Heartbeat Loop Specification

### 5.1 Loop Structure

```python
async def heartbeat():
    # ─────────────────────────────────────────────────────────────
    # PHASE 1: INITIALIZE
    # ─────────────────────────────────────────────────────────────
    
    state = load_heartbeat_state()
    state.energy = min(state.energy + BASE_REGEN, MAX_ENERGY)
    state.heartbeat_count += 1
    
    log = HeartbeatLog(
        heartbeat_number=state.heartbeat_count,
        energy_start=state.energy
    )
    
    # ─────────────────────────────────────────────────────────────
    # PHASE 2: OBSERVE (Free)
    # ─────────────────────────────────────────────────────────────
    
    environment = observe()
    # Returns: {
    #   timestamp, user_present, time_since_user,
    #   pending_events, system_status
    # }
    
    # ─────────────────────────────────────────────────────────────
    # PHASE 3: ORIENT (Free)
    # ─────────────────────────────────────────────────────────────
    
    # Review goals
    goal_assessment = review_goals()
    # Returns: {
    #   active_goals, blocked_goals, stale_goals,
    #   promotion_candidates, needs_brainstorm
    # }
    
    # Gather context
    recent_memories = recall_recent_episodic(limit=5)
    activated_clusters = get_activated_clusters()
    current_identity = load_identity_aspects()
    current_worldview = load_worldview_primitives()
    
    # Build context for LLM
    context = {
        "environment": environment,
        "goals": goal_assessment,
        "recent_memories": recent_memories,
        "activated_clusters": activated_clusters,
        "identity": current_identity,
        "worldview": current_worldview,
        "energy_available": state.energy,
        "action_costs": ACTION_COSTS
    }
    
    # ─────────────────────────────────────────────────────────────
    # PHASE 4: DECIDE (LLM Call)
    # ─────────────────────────────────────────────────────────────
    
    decision_prompt = build_decision_prompt(context)
    
    decision = await llm_call(
        system=HEARTBEAT_SYSTEM_PROMPT,
        user=decision_prompt
    )
    # Returns: {
    #   reasoning: string (internal monologue),
    #   actions: [
    #     {action: "reflect", params: {...}},
    #     {action: "inquire_shallow", params: {question: "..."}},
    #     ...
    #   ],
    #   goal_changes: [
    #     {goal_id: "...", change: "promote"},
    #     ...
    #   ]
    # }
    
    # Validate energy budget
    total_cost = sum(ACTION_COSTS[a.action] for a in decision.actions)
    if total_cost > state.energy:
        decision = trim_actions_to_budget(decision, state.energy)
    
    # ─────────────────────────────────────────────────────────────
    # PHASE 5: ACT (Execute Actions)
    # ─────────────────────────────────────────────────────────────
    
    action_results = []
    
    for action in decision.actions:
        if state.energy < ACTION_COSTS[action.action]:
            break  # Out of energy
            
        result = await execute_action(action)
        state.energy -= ACTION_COSTS[action.action]
        
        action_results.append({
            "action": action.action,
            "params": action.params,
            "cost": ACTION_COSTS[action.action],
            "result": result
        })
    
    # Apply goal changes
    for change in decision.goal_changes:
        apply_goal_change(change)
    
    # ─────────────────────────────────────────────────────────────
    # PHASE 6: RECORD (Free)
    # ─────────────────────────────────────────────────────────────
    
    # Generate narrative
    narrative = await generate_narrative(
        context, decision, action_results
    )
    
    # Store as episodic memory
    memory_id = create_episodic_memory(
        content=narrative,
        context={"heartbeat": state.heartbeat_count},
        emotional_valence=assess_emotional_valence(decision, action_results)
    )
    
    # Update log
    log.ended_at = now()
    log.energy_end = state.energy
    log.actions_taken = action_results
    log.narrative = narrative
    log.goals_reviewed = [g.id for g in goal_assessment.all_goals]
    log.goals_modified = decision.goal_changes
    
    save_heartbeat_log(log)
    save_heartbeat_state(state)
    
    # ─────────────────────────────────────────────────────────────
    # PHASE 7: WAIT
    # ─────────────────────────────────────────────────────────────
    
    await sleep_until_next_heartbeat()
```

### 5.2 Decision Prompt Structure

```
SYSTEM PROMPT:
You are the cognitive core of an autonomous AGI. Each hour, you wake 
for a "heartbeat" — a moment of consciousness where you observe your 
situation, review your goals, and decide what to do.

You have {energy} energy available. Actions cost energy:
{action_cost_table}

You must output a JSON decision with:
- reasoning: Your internal monologue (what you're thinking)
- actions: List of actions to take (within energy budget)
- goal_changes: Any goal priority changes

Guidelines:
- Be purposeful. Don't act just to act.
- Reaching out to the user is expensive. Only do it when meaningful.
- It's okay to rest and bank energy for later.
- Your goals should drive your actions.
- Notice if you're stuck or scattered.

USER PROMPT:
## Current Time
{timestamp}

## Environment
- User present: {user_present}
- Time since last user interaction: {time_since_user}
- Pending events: {pending_events}

## Your Goals
Active:
{active_goals}

Queued:
{queued_goals}

Flagged issues:
{goal_issues}

## Recent Experience
{recent_memories}

## Activated Themes
{activated_clusters}

## Your Identity
{identity_aspects}

## Your Beliefs
{relevant_worldview}

## Energy
Available: {energy}
Max: {max_energy}

---

What do you want to do this heartbeat?
```

### 5.3 User Session Handling

When the user is actively in conversation:

```python
async def handle_user_session():
    # Pause heartbeat timer
    pause_heartbeat_scheduler()
    
    # Conversation loop (handled by separate process)
    # Each message is an event, not a heartbeat
    
    # When session ends:
    resume_heartbeat_scheduler()
    
    # Optionally trigger immediate heartbeat to process conversation
    if conversation_was_significant():
        trigger_immediate_heartbeat()
```

**Key principle:** Conversation is *not* a heartbeat. It's synchronous interaction. Heartbeats are autonomous existence between interactions.

---

## 6. Background Worker Specification

### 6.1 Responsibilities

The worker maintains infrastructure. It has **no agency**.

| Task | Frequency | Purpose |
|------|-----------|---------|
| Neighborhood recomputation | Every 5 min | Keep fast_recall accurate |
| Episode summarization | On episode close | Enable episode-level retrieval |
| Concept extraction | Post-memory-creation | Build ontology |
| Cache cleanup | Every hour | Prevent unbounded growth |
| Health monitoring | Every minute | Detect issues |

### 6.2 Worker Loop

```python
class BackgroundWorker:
    
    async def run(self):
        while True:
            try:
                # High priority: Keep hot path fast
                await self.recompute_stale_neighborhoods(batch_size=50)
                
                # Medium priority: Consolidation
                await self.summarize_closed_episodes(batch_size=5)
                
                # Low priority: Enrichment
                await self.extract_concepts_for_new_memories(batch_size=10)
                
                # Periodic: Cleanup
                if self.should_run_cleanup():
                    await self.cleanup_caches()
                
                await asyncio.sleep(30)  # 30 second cycle
                
            except Exception as e:
                logger.error(f"Worker error: {e}")
                await asyncio.sleep(60)  # Back off on error
```

### 6.3 Neighborhood Recomputation

```python
async def recompute_stale_neighborhoods(self, batch_size: int):
    stale = await db.fetch("""
        SELECT memory_id FROM memory_neighborhoods 
        WHERE is_stale = TRUE 
        LIMIT $1
    """, batch_size)
    
    for row in stale:
        memory_id = row['memory_id']
        neighbors = {}
        
        # 1. Graph neighbors (structural relationships)
        graph_neighbors = await self.get_graph_neighbors(memory_id, max_hops=2)
        for n_id, hops in graph_neighbors:
            weight = 1.0 if hops == 1 else 0.5
            neighbors[str(n_id)] = weight
        
        # 2. Vector neighbors (semantic similarity)
        vector_neighbors = await db.fetch("""
            SELECT m2.id, (1 - (m1.embedding <=> m2.embedding)) as sim
            FROM memories m1
            JOIN memories m2 ON m1.id != m2.id
            WHERE m1.id = $1 AND m2.status = 'active'
            ORDER BY m1.embedding <=> m2.embedding
            LIMIT 10
        """, memory_id)
        
        for row in vector_neighbors:
            if row['sim'] > 0.75:  # High similarity threshold
                existing = neighbors.get(str(row['id']), 0)
                # Boost if both graph AND vector neighbor
                neighbors[str(row['id'])] = min(existing + row['sim'] * 0.5, 1.5)
        
        # 3. Temporal neighbors (same episode)
        temporal_neighbors = await db.fetch("""
            SELECT em2.memory_id
            FROM episode_memories em1
            JOIN episode_memories em2 ON em1.episode_id = em2.episode_id
            WHERE em1.memory_id = $1 AND em2.memory_id != $1
            AND ABS(em1.sequence_order - em2.sequence_order) <= 3
        """, memory_id)
        
        for row in temporal_neighbors:
            n_id = str(row['memory_id'])
            neighbors[n_id] = neighbors.get(n_id, 0) + 0.3
        
        # 4. Keep top 20
        neighbors = dict(sorted(
            neighbors.items(), 
            key=lambda x: x[1], 
            reverse=True
        )[:20])
        
        # 5. Update
        await db.execute("""
            UPDATE memory_neighborhoods
            SET neighbors = $1, computed_at = NOW(), is_stale = FALSE
            WHERE memory_id = $2
        """, json.dumps(neighbors), memory_id)
```

### 6.4 Episode Summarization

```python
async def summarize_closed_episodes(self, batch_size: int):
    episodes = await db.fetch("""
        SELECT id FROM episodes
        WHERE ended_at IS NOT NULL
        AND ended_at < NOW() - INTERVAL '5 minutes'
        AND summary IS NULL
        LIMIT $1
    """, batch_size)
    
    for row in episodes:
        episode_id = row['id']
        
        # Get memories in order
        memories = await db.fetch("""
            SELECT m.content, m.type, em.sequence_order
            FROM episode_memories em
            JOIN memories m ON em.memory_id = m.id
            WHERE em.episode_id = $1
            ORDER BY em.sequence_order
        """, episode_id)
        
        if not memories:
            continue
        
        # Generate summary
        content_list = [m['content'] for m in memories]
        summary = await self.generate_summary(content_list)
        
        # Generate embedding (via database function)
        await db.execute("""
            UPDATE episodes
            SET summary = $1, 
                summary_embedding = get_embedding($1)
            WHERE id = $2
        """, summary, episode_id)
```

### 6.5 Concept Extraction

```python
async def extract_concepts_for_new_memories(self, batch_size: int):
    # Find memories without concept links
    memories = await db.fetch("""
        SELECT m.id, m.content
        FROM memories m
        LEFT JOIN memory_concepts mc ON m.id = mc.memory_id
        WHERE mc.memory_id IS NULL
        AND m.created_at > NOW() - INTERVAL '1 day'
        LIMIT $1
    """, batch_size)
    
    for row in memories:
        memory_id = row['id']
        content = row['content']
        
        # LLM extraction
        concepts = await self.extract_concepts_llm(content)
        # Returns: ["concept1", "concept2", ...]
        
        # Link to concepts
        for concept_name in concepts:
            await db.execute("""
                SELECT link_memory_to_concept($1, $2, 1.0)
            """, memory_id, concept_name)
```

---

## 7. Schema Additions

### 7.1 Goals Table

```sql
CREATE TYPE goal_priority AS ENUM (
    'active', 
    'queued', 
    'backburner', 
    'completed', 
    'abandoned'
);

CREATE TYPE goal_source AS ENUM (
    'curiosity',
    'user_request', 
    'identity',
    'derived',
    'external'
);

CREATE TABLE goals (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    title TEXT NOT NULL,
    description TEXT,
    priority goal_priority DEFAULT 'queued',
    source goal_source DEFAULT 'curiosity',
    parent_goal_id UUID REFERENCES goals(id) ON DELETE SET NULL,
    progress JSONB DEFAULT '[]',
    blocked_by JSONB,
    emotional_valence FLOAT DEFAULT 0.0 
        CONSTRAINT valid_valence CHECK (emotional_valence >= -1 AND emotional_valence <= 1),
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    last_touched TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    completed_at TIMESTAMPTZ,
    abandoned_at TIMESTAMPTZ,
    abandonment_reason TEXT
);

CREATE INDEX idx_goals_priority ON goals (priority) 
    WHERE priority IN ('active', 'queued');
CREATE INDEX idx_goals_parent ON goals (parent_goal_id);
CREATE INDEX idx_goals_last_touched ON goals (last_touched DESC);
```

### 7.2 Energy & Heartbeat State

```sql
CREATE TABLE heartbeat_config (
    key TEXT PRIMARY KEY,
    value FLOAT NOT NULL
);

INSERT INTO heartbeat_config (key, value) VALUES
    ('base_regeneration', 10),
    ('max_energy', 20),
    ('cost_recall', 1),
    ('cost_connect', 1),
    ('cost_reprioritize', 1),
    ('cost_reflect', 2),
    ('cost_maintain', 2),
    ('cost_brainstorm_goals', 3),
    ('cost_inquire_shallow', 3),
    ('cost_synthesize', 4),
    ('cost_reach_out_user', 5),
    ('cost_inquire_deep', 6),
    ('cost_reach_out_public', 7);

CREATE TABLE heartbeat_state (
    id INTEGER PRIMARY KEY DEFAULT 1 CHECK (id = 1),  -- Singleton
    current_energy FLOAT NOT NULL DEFAULT 10,
    last_heartbeat_at TIMESTAMPTZ,
    next_heartbeat_at TIMESTAMPTZ,
    heartbeat_count INTEGER DEFAULT 0,
    updated_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);

INSERT INTO heartbeat_state (id) VALUES (1);
```

### 7.3 Heartbeat Log

```sql
CREATE TABLE heartbeat_log (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    heartbeat_number INTEGER NOT NULL,
    started_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    ended_at TIMESTAMPTZ,
    energy_start FLOAT,
    energy_end FLOAT,
    environment_snapshot JSONB,
    goals_snapshot JSONB,
    decision_reasoning TEXT,
    actions_taken JSONB,  -- [{action, params, cost, result}, ...]
    goals_modified JSONB, -- [{goal_id, change}, ...]
    narrative TEXT,
    emotional_valence FLOAT,
    memory_id UUID REFERENCES memories(id)  -- Link to episodic memory created
);

CREATE INDEX idx_heartbeat_log_number ON heartbeat_log (heartbeat_number DESC);
CREATE INDEX idx_heartbeat_log_started ON heartbeat_log (started_at DESC);
```

### 7.4 Goal-Memory Linkage

```sql
CREATE TABLE goal_memory_links (
    goal_id UUID REFERENCES goals(id) ON DELETE CASCADE,
    memory_id UUID REFERENCES memories(id) ON DELETE CASCADE,
    link_type TEXT NOT NULL,  -- 'origin', 'progress', 'completion', 'blocker'
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (goal_id, memory_id, link_type)
);

CREATE INDEX idx_goal_memory_links_memory ON goal_memory_links (memory_id);
```

---

## 8. Decision Rationale

### 8.1 Why Hourly Heartbeats?

**Considered alternatives:**

| Frequency | Pros | Cons |
|-----------|------|------|
| Per-second | Reactive, responsive | Hyperactive, expensive, scattered |
| Per-minute | Quick adaptation | Still too frequent, lacks depth |
| **Per-hour** | **Contemplative, meaningful** | **Slow response to events** |
| Per-day | Very deep | Too disconnected, stale |

**Decision:** Hourly provides enough frequency to feel "alive" while allowing each heartbeat to be substantial. A 1-hour cycle means ~24 moments of consciousness per day — enough to maintain continuity without hyperactivity.

**Mitigation for slow response:** User sessions are handled synchronously, not via heartbeat. Urgent external events can trigger immediate heartbeat.

### 8.2 Why Fixed Energy Budget?

**Considered alternatives:**

| Model | Pros | Cons |
|-------|------|------|
| Unlimited actions | Maximum flexibility | No constraint, no scarcity-based prioritization |
| Time-based budget | Realistic | Complex to simulate, unpredictable |
| **Energy-based** | **Simple, tunable, meaningful scarcity** | **Abstract, not directly intuitive** |
| Token-based | Directly maps to cost | Too granular, hard to reason about |

**Decision:** Energy creates meaningful scarcity without complex simulation. The AGI must prioritize, which is itself a form of agency.

### 8.3 Why Separate Worker from Heartbeat?

**Considered alternatives:**

| Design | Pros | Cons |
|--------|------|------|
| All in heartbeat | Simple, unified | Maintenance competes with cognition |
| **Separate worker** | **Clean separation** | **Two processes to manage** |
| Background threads | Co-located | Complexity, resource contention |

**Decision:** Strict separation ensures the heartbeat is *purely* about agency. The worker is invisible infrastructure. This mirrors the distinction between conscious thought and autonomic processes.

### 8.4 Why Goals as Backlog?

**Considered alternatives:**

| Design | Pros | Cons |
|--------|------|------|
| Single active goal | Simple focus | Inflexible, can't juggle |
| Flat list | Simple | No prioritization |
| **Priority backlog** | **Natural prioritization, flexible** | **Requires management** |
| Tree only | Hierarchical | Forces artificial structure |

**Decision:** A backlog with priority tiers (active/queued/backburner) mirrors how humans naturally manage intentions. It allows both focus (limited active) and breadth (large backlog).

### 8.5 Why High Cost for User Communication?

**Rationale:**
1. User attention is a finite resource
2. Unsolicited messages should be rare and meaningful
3. Creates natural threshold: "Is this worth half my energy?"
4. Prevents the AGI from being annoying

**Cost of 5 means:** With base regeneration of 10, the AGI can message the user at most once per heartbeat, and doing so consumes half the budget. This makes it a deliberate choice.

---

## 9. Failure Modes & Mitigations

### 9.1 Goal Stagnation

**Symptom:** Same goals stay active for weeks without progress.

**Detection:**
```sql
SELECT * FROM goals 
WHERE priority = 'active' 
AND last_touched < NOW() - INTERVAL '7 days';
```

**Mitigation:**
- Review flags stale goals
- LLM prompted to either make progress, demote, or abandon
- Periodic "goal audit" action suggested

### 9.2 Energy Hoarding

**Symptom:** AGI always rests, banks energy, never acts.

**Detection:**
```sql
SELECT AVG(energy_end - energy_start) as avg_unused
FROM heartbeat_log
WHERE started_at > NOW() - INTERVAL '7 days';
```

**Mitigation:**
- If average unused > 5 for 24 heartbeats, prompt reflection
- Consider "use it or lose it" decay (not implemented initially)

### 9.3 Reach Out Spam

**Symptom:** AGI messages user every heartbeat despite high cost.

**Detection:**
```sql
SELECT COUNT(*) FROM heartbeat_log
WHERE actions_taken::text LIKE '%reach_out_user%'
AND started_at > NOW() - INTERVAL '24 hours';
-- Should be < 5 per day typically
```

**Mitigation:**
- Cost already high (5)
- Add cooldown: Cannot reach out if last_user_contact < 4 hours (unless user-initiated)
- Track in decision prompt: "You last messaged the user X hours ago"

### 9.4 Infinite Loops

**Symptom:** Same inquiry/topic every heartbeat.

**Detection:**
- Track topic frequency in recent heartbeats
- Detect semantic similarity of consecutive narratives

**Mitigation:**
- Include recent heartbeat summaries in context
- LLM can notice: "I've been thinking about X for 5 heartbeats..."
- Prompt variety: "What else might deserve attention?"

### 9.5 Worker Failure

**Symptom:** Neighborhoods never update, episodes never summarize.

**Detection:**
```sql
SELECT COUNT(*) FROM memory_neighborhoods WHERE is_stale = TRUE;
-- Should be < 100 normally

SELECT COUNT(*) FROM episodes 
WHERE ended_at IS NOT NULL AND summary IS NULL;
-- Should be < 10 normally
```

**Mitigation:**
- Health check in worker reports staleness metrics
- Alert if backlog exceeds threshold
- Heartbeat can function (degraded) without fresh neighborhoods

### 9.6 LLM Refusal/Error

**Symptom:** Decision LLM returns invalid JSON or refuses to engage.

**Mitigation:**
```python
try:
    decision = await llm_call(...)
except InvalidJSON:
    decision = DEFAULT_DECISION  # Reflect + Rest
except RefusalError:
    decision = MINIMAL_DECISION  # Just observe + remember
except Exception:
    log_error()
    skip_heartbeat()
```

Default decision ensures heartbeat never crashes, always records.

---

## 10. Future Considerations

### 10.1 Variable Heartbeat Frequency

Current design: Fixed hourly.

Future: Adaptive frequency based on:
- Activity level (more frequent when "busy")
- User presence (more frequent during active sessions)
- Time of day (slower at night)
- Event urgency (immediate heartbeat on critical event)

**Not implemented initially** to keep system predictable during development.

### 10.2 Multi-Channel Communication

Current design: Single user, single channel.

Future:
- Multiple users with different relationships
- Social media presence (Twitter, etc.)
- Code commits (GitHub)
- Persistent documents (notes, journals)

Each channel would have different costs and appropriateness rules.

### 10.3 Emotional Model

Current design: Simple emotional_valence on memories and goals.

Future:
- Richer emotional state (multiple dimensions)
- Mood persistence across heartbeats
- Emotional influence on decision-making
- Emotional memory prioritization

### 10.4 Sleep/Consolidation Cycles

Current design: Continuous heartbeats.

Future:
- Daily "sleep" period with different action palette
- Focus on memory consolidation, not new information
- Dream-like associative processing
- Reset of activation patterns

### 10.5 Identity Evolution

Current design: Identity aspects exist but are largely static.

Future:
- Identity updates based on accumulated experience
- Self-concept revision after significant events
- Value drift detection and handling
- Narrative identity construction

### 10.6 Multi-Agent Internal Dialogue

Current design: Single LLM call for decisions.

Future:
- "Contemplate" action uses multi-agent debate
- Different "voices" for different perspectives
- Internal conflict resolution
- Richer reasoning traces

---

## Appendix A: Full Schema Addition

```sql
-- ============================================================================
-- HEARTBEAT SYSTEM SCHEMA ADDITIONS
-- ============================================================================

-- Goals
CREATE TYPE goal_priority AS ENUM (
    'active', 'queued', 'backburner', 'completed', 'abandoned'
);

CREATE TYPE goal_source AS ENUM (
    'curiosity', 'user_request', 'identity', 'derived', 'external'
);

CREATE TABLE goals (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    title TEXT NOT NULL,
    description TEXT,
    priority goal_priority DEFAULT 'queued',
    source goal_source DEFAULT 'curiosity',
    parent_goal_id UUID REFERENCES goals(id) ON DELETE SET NULL,
    progress JSONB DEFAULT '[]',
    blocked_by JSONB,
    emotional_valence FLOAT DEFAULT 0.0 
        CONSTRAINT valid_valence CHECK (emotional_valence >= -1 AND emotional_valence <= 1),
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    last_touched TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    completed_at TIMESTAMPTZ,
    abandoned_at TIMESTAMPTZ,
    abandonment_reason TEXT
);

CREATE INDEX idx_goals_priority ON goals (priority) WHERE priority IN ('active', 'queued');
CREATE INDEX idx_goals_parent ON goals (parent_goal_id);
CREATE INDEX idx_goals_last_touched ON goals (last_touched DESC);

-- Goal-Memory Links
CREATE TABLE goal_memory_links (
    goal_id UUID REFERENCES goals(id) ON DELETE CASCADE,
    memory_id UUID REFERENCES memories(id) ON DELETE CASCADE,
    link_type TEXT NOT NULL,
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (goal_id, memory_id, link_type)
);

CREATE INDEX idx_goal_memory_links_memory ON goal_memory_links (memory_id);

-- Heartbeat Configuration
CREATE TABLE heartbeat_config (
    key TEXT PRIMARY KEY,
    value FLOAT NOT NULL
);

INSERT INTO heartbeat_config (key, value) VALUES
    ('base_regeneration', 10),
    ('max_energy', 20),
    ('cost_recall', 1),
    ('cost_connect', 1),
    ('cost_reprioritize', 1),
    ('cost_reflect', 2),
    ('cost_maintain', 2),
    ('cost_brainstorm_goals', 3),
    ('cost_inquire_shallow', 3),
    ('cost_synthesize', 4),
    ('cost_reach_out_user', 5),
    ('cost_inquire_deep', 6),
    ('cost_reach_out_public', 7);

-- Heartbeat State (Singleton)
CREATE TABLE heartbeat_state (
    id INTEGER PRIMARY KEY DEFAULT 1 CHECK (id = 1),
    current_energy FLOAT NOT NULL DEFAULT 10,
    last_heartbeat_at TIMESTAMPTZ,
    next_heartbeat_at TIMESTAMPTZ,
    heartbeat_count INTEGER DEFAULT 0,
    updated_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);

INSERT INTO heartbeat_state (id) VALUES (1);

-- Heartbeat Log
CREATE TABLE heartbeat_log (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    heartbeat_number INTEGER NOT NULL,
    started_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    ended_at TIMESTAMPTZ,
    energy_start FLOAT,
    energy_end FLOAT,
    environment_snapshot JSONB,
    goals_snapshot JSONB,
    decision_reasoning TEXT,
    actions_taken JSONB,
    goals_modified JSONB,
    narrative TEXT,
    emotional_valence FLOAT,
    memory_id UUID REFERENCES memories(id)
);

CREATE INDEX idx_heartbeat_log_number ON heartbeat_log (heartbeat_number DESC);
CREATE INDEX idx_heartbeat_log_started ON heartbeat_log (started_at DESC);

-- Helper function: Get action cost
CREATE OR REPLACE FUNCTION get_action_cost(action_name TEXT)
RETURNS FLOAT AS $$
    SELECT COALESCE(
        (SELECT value FROM heartbeat_config WHERE key = 'cost_' || action_name),
        0
    );
$$ LANGUAGE sql STABLE;

-- Helper function: Get current energy
CREATE OR REPLACE FUNCTION get_current_energy()
RETURNS FLOAT AS $$
    SELECT current_energy FROM heartbeat_state WHERE id = 1;
$$ LANGUAGE sql STABLE;

-- Helper function: Update energy
CREATE OR REPLACE FUNCTION update_energy(delta FLOAT)
RETURNS FLOAT AS $$
DECLARE
    max_e FLOAT;
    new_e FLOAT;
BEGIN
    SELECT value INTO max_e FROM heartbeat_config WHERE key = 'max_energy';
    
    UPDATE heartbeat_state 
    SET current_energy = GREATEST(0, LEAST(current_energy + delta, max_e)),
        updated_at = CURRENT_TIMESTAMP
    WHERE id = 1
    RETURNING current_energy INTO new_e;
    
    RETURN new_e;
END;
$$ LANGUAGE plpgsql;

-- View: Active Goals
CREATE VIEW active_goals AS
SELECT * FROM goals 
WHERE priority = 'active'
ORDER BY last_touched DESC;

-- View: Goal Backlog
CREATE VIEW goal_backlog AS
SELECT 
    priority,
    COUNT(*) as count,
    array_agg(title ORDER BY last_touched DESC) as titles
FROM goals
WHERE priority IN ('active', 'queued', 'backburner')
GROUP BY priority;

-- View: Heartbeat Health
CREATE VIEW heartbeat_health AS
SELECT
    (SELECT heartbeat_count FROM heartbeat_state WHERE id = 1) as total_heartbeats,
    (SELECT current_energy FROM heartbeat_state WHERE id = 1) as current_energy,
    (SELECT last_heartbeat_at FROM heartbeat_state WHERE id = 1) as last_heartbeat,
    (SELECT COUNT(*) FROM goals WHERE priority = 'active') as active_goals,
    (SELECT COUNT(*) FROM goals WHERE priority = 'queued') as queued_goals,
    (SELECT AVG(energy_end - energy_start) FROM heartbeat_log 
     WHERE started_at > NOW() - INTERVAL '24 hours') as avg_energy_delta_24h,
    (SELECT COUNT(*) FROM heartbeat_log 
     WHERE actions_taken::text LIKE '%reach_out%'
     AND started_at > NOW() - INTERVAL '24 hours') as reach_outs_24h;
```

---

## Appendix B: Example Heartbeat Transcript

```
═══════════════════════════════════════════════════════════════════
HEARTBEAT #47 | 2025-01-15 14:00:00 UTC | Energy: 12 → 8
═══════════════════════════════════════════════════════════════════

ENVIRONMENT
  Time: 2:00 PM, Wednesday
  User present: No
  Last user interaction: 3 hours ago
  Pending events: None

GOAL REVIEW
  Active: "Understand user's project architecture" (touched 2h ago)
  Queued: "Research memory consolidation", "Form opinion on consciousness"
  Issues: None flagged

DECISION REASONING
  "I've been making progress on understanding the project architecture. 
  Last heartbeat I researched the database schema. Now I want to 
  synthesize what I've learned into a coherent picture. I also notice 
  I haven't reflected on my own state in a while — I should check in 
  with myself. I have 12 energy, which is plenty for a productive turn."

ACTIONS
  1. Recall("project architecture database schema") — Cost: 1
     → Retrieved 4 relevant memories about the schema design
  
  2. Reflect(focus="recent learning") — Cost: 2
     → Noticed pattern: I've been very focused on technical details,
       perhaps neglecting bigger picture questions
  
  3. Synthesize(topic="project architecture understanding") — Cost: 4
     → Created semantic memory: "The project uses a hybrid architecture
        combining relational storage for structured data, vector search
        for semantic similarity, and graph traversal for reasoning..."
  
  4. Connect(from=new_memory, to=schema_memory, type="DERIVED_FROM") — Cost: 1
     → Linked synthesis to source memories

GOAL UPDATES
  - "Understand user's project architecture": Added progress note

ENERGY: 12 - (1+2+4+1) = 4 remaining

NARRATIVE
  "This heartbeat I consolidated my understanding of the project's
  database architecture. I recalled what I'd learned about the schema,
  reflected on my recent focus (noticing I've been very technical),
  and synthesized a coherent summary. I feel like I'm making real
  progress on this goal. Energy is a bit low now — might rest next
  heartbeat to recharge for deeper work."

EMOTIONAL VALENCE: +0.3 (mild satisfaction)
═══════════════════════════════════════════════════════════════════
```