# Week 3 -> Day 3 ->  Knowledge Graph RAG
---

## Table of Contents

1. [Knowledge Graph RAG - What & Why](#1-knowledge-graph-rag---what--why)
    *   [1.1 What is a Knowledge Graph?](#11-what-is-a-knowledge-graph)
    *   [1.2 What is Knowledge Graph RAG?](#12-what-is-knowledge-graph-rag)
    *   [1.3 Why Knowledge Graph RAG?](#13-why-knowledge-graph-rag)
2. [Traditional RAG vs Knowledge Graph RAG](#2-traditional-rag-vs-knowledge-graph-rag)
    *   [2.1 Architecture Comparison](#21-architecture-comparison)
    *   [2.2 Detailed Feature Comparison](#22-detailed-feature-comparison)
    *   [2.3 When to Choose Which](#23-when-to-choose-which)
3. [LLMGraphTransformer vs Graphiti](LLMGraphTransformer_vs_Graphiti.md)
4. [Real-World Example — Customer Support Knowledge Graph](#4-real-world-example--customer-support-knowledge-graph)


# Knowledge Graph RAG

---

## 1. Knowledge Graph RAG - What & Why

### 1.1 What is a Knowledge Graph?

A **Knowledge Graph (KG)** is a structured representation of real-world entities and the relationships between them. It stores facts as **triplets**:

```
(Subject)  ──[Relationship]──▶  (Object)

Examples:
  (Alice)        ──[works_at]──▶       (TechCorp)
  (TechCorp)     ──[located_in]──▶     (San Francisco)
  (Alice)        ──[manages]──▶        (Bob)
  (Bob)          ──[works_on]──▶       (ProjectX)
  (ProjectX)     ──[uses_technology]──▶(Python)
```

These triplets form a **graph** where:
- **Nodes** = Entities (people, places, things, concepts)
- **Edges** = Relationships between entities
- **Properties** = Attributes on nodes and edges (name, date, confidence score, etc.)

---

### 1.2 What is Knowledge Graph RAG?

**Knowledge Graph RAG (Graph RAG)** is an advanced RAG architecture that replaces (or augments) a flat vector store with a **structured knowledge graph** as its retrieval backbone.

Instead of retrieving isolated text chunks by semantic similarity, Graph RAG **traverses graph relationships** to find contextually connected information — enabling multi-hop reasoning across linked entities.

```
Traditional RAG:
  Query ──▶ [Vector Search] ──▶ Top-K text chunks ──▶ LLM ──▶ Answer

Knowledge Graph RAG:
  Query ──▶ [Entity Extraction] ──▶ [Graph Traversal] ──▶
            Subgraph of connected facts ──▶ LLM ──▶ Answer
```

---

### 1.3 Why Knowledge Graph RAG?

#### The Core Problem with Traditional RAG

Traditional RAG stores text as disconnected chunks. It answers "what" questions well, but struggles with "how are these things related?" and "what changed over time?" questions.

**Scenario:** You have a corporate knowledge base. A user asks:

> *"Which team members who previously worked on the cancelled Project Apollo are now working on Project Mercury, and what Python libraries are they using?"*

**Traditional RAG fails** because:
1. This requires linking people → projects → technologies across multiple documents
2. It needs to know about the past state (Project Apollo was cancelled) and current state (now on Mercury)
3. Keyword/semantic search returns isolated chunks about Apollo, Mercury, and people separately — with no way to connect them

**Knowledge Graph RAG succeeds** because:
1. It walks the graph: `Person → worked_on → ProjectApollo` → `Person → now_works_on → ProjectMercury` → `ProjectMercury → uses → Python libraries`
2. It has temporal awareness — it knows what was true then vs. now
3. It returns a connected subgraph that directly answers all three parts of the question

---

#### Key Reasons to Use Graph RAG

| Pain Point | Traditional RAG | Knowledge Graph RAG |
|---|---|---|
| **Multi-hop questions** | ❌ Needs all facts in one chunk | ✅ Traverses edges across hops |
| **Relationship queries** | ❌ Misses entity links | ✅ Relationships are first-class |
| **Temporal reasoning** | ❌ No concept of "was" vs "is" | ✅ Bi-temporal edge validity |
| **Global context** | ❌ Retrieves local chunks only | ✅ Community summaries, hierarchies |
| **Contradiction handling** | ❌ Returns conflicting chunks | ✅ Invalidates outdated edges |
| **Source traceability** | ⚠️ Approximate (by chunk) | ✅ Precise (entity + edge level) |
| **Dynamic data** | ❌ Requires full re-indexing | ✅ Incremental graph updates |
| **Data integration** | ❌ One embedding space | ✅ Any data type as nodes/edges |

---

## 2. Traditional RAG vs Knowledge Graph RAG

### 2.1 Architecture Comparison

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

TRADITIONAL RAG
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  ┌─────────────────────────────────────────────────┐
  │               INDEXING (Offline)                │
  │                                                 │
  │  Documents ──▶ Chunks ──▶ Embeddings ──▶ FAISS │
  │                                                 │
  │  "Alice manages Bob. Bob works on ProjectX      │
  │   which uses Python 3.11 and FastAPI."          │
  │            ↓                                    │
  │  [chunk_1: "Alice manages Bob..."]              │
  │  [chunk_2: "Bob works on ProjectX..."]          │
  │  [chunk_3: "ProjectX uses Python..."]           │
  └─────────────────────────────────────────────────┘

  ┌─────────────────────────────────────────────────┐
  │                 QUERY (Online)                  │
  │                                                 │
  │  Query: "What does Alice's team use?"           │
  │    ↓                                            │
  │  Embed query ──▶ Cosine similarity              │
  │    ↓                                            │
  │  Returns: chunk_1 ("Alice manages Bob")         │
  │           chunk_3 ("ProjectX uses Python")      │
  │                                                 │
  │  ⚠️  PROBLEM: chunk_2 ("Bob works on ProjectX") │
  │     might NOT be in top-K → broken link         │
  │     The connection Alice→Bob→ProjectX is LOST   │
  └─────────────────────────────────────────────────┘

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

KNOWLEDGE GRAPH RAG
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  ┌─────────────────────────────────────────────────┐
  │               INDEXING (Offline)                │
  │                                                 │
  │  Documents ──▶ Entity Extraction ──▶ Graph DB  │
  │                                                 │
  │  Stored as a connected graph:                   │
  │                                                 │
  │  [Alice] ──manages──▶ [Bob]                    │
  │                         │                       │
  │                    works_on                     │
  │                         │                       │
  │                         ▼                       │
  │                   [ProjectX]                    │
  │                         │                       │
  │                     uses                        │
  │                         │                       │
  │                    ┌────┴────┐                  │
  │                    ▼         ▼                  │
  │               [Python]   [FastAPI]              │
  └─────────────────────────────────────────────────┘

  ┌─────────────────────────────────────────────────┐
  │                 QUERY (Online)                  │
  │                                                 │
  │  Query: "What does Alice's team use?"           │
  │    ↓                                            │
  │  Extract entity: Alice                          │
  │    ↓                                            │
  │  Graph traversal (2 hops):                      │
  │    Alice ──manages──▶ Bob                       │
  │    Bob ──works_on──▶ ProjectX                   │
  │    ProjectX ──uses──▶ Python, FastAPI           │
  │    ↓                                            │
  │  Returns: full connected subgraph ✅            │
  │  Answer: "Alice's team (Bob) uses Python        │
  │            and FastAPI on ProjectX."            │
  └─────────────────────────────────────────────────┘
```

---

### 2.2 Detailed Feature Comparison

| Feature | Traditional RAG | Knowledge Graph RAG |
|---|---|---|
| **Data storage** | Flat vector embeddings | Graph nodes + edges + embeddings |
| **Retrieval method** | Cosine/dot-product similarity | Graph traversal + semantic search |
| **Relationship awareness** | ❌ Implicit in text only | ✅ Explicit as typed edges |
| **Multi-hop reasoning** | ❌ Limited by chunk boundaries | ✅ Native via graph traversal |
| **Temporal tracking** | ❌ No built-in concept | ✅ Edge validity time ranges |
| **Update mechanism** | Reindex affected chunks | Add/invalidate edges incrementally |
| **Contradiction handling** | Both versions retrieved | Old edge invalidated, new one added |
| **Global summarization** | Not possible | ✅ Community detection algorithms |
| **Query types excelled** | Semantic similarity, keyword | Relational, multi-hop, temporal |
| **Setup complexity** | Low — FAISS/Chroma + embeddings | High — Graph DB + entity extraction |
| **Best for** | Document Q&A, search | Agents, enterprise knowledge, CRM |

---

### 2.3 When to Choose Which

```
Use Traditional RAG when:
  ✅ Questions are self-contained ("What is the return policy?")
  ✅ Data is static documents (PDFs, manuals, wikis)
  ✅ No need for relationship traversal
  ✅ Quick setup is a priority
  ✅ Dataset is < 100K documents

Use Knowledge Graph RAG when:
  ✅ Questions need multi-hop reasoning
     ("Which engineers who left team A are now on team B?")
  ✅ Data has rich entity relationships
     (CRM, org charts, medical records, supply chains)
  ✅ Data is dynamic and frequently updated
  ✅ Temporal queries matter ("What was true in January?")
  ✅ You need precise source attribution at entity level
  ✅ You're building persistent agent memory
```

---

## 4. Real-World Example — Customer Support Knowledge Graph

### Scenario

We are building an AI support assistant for a **SaaS company** called **CloudSync**. The assistant must:

1. Know the current status of each customer (plan, usage, open tickets)
2. Answer relational questions ("Which enterprise customers have open billing issues?")
3. Track how customer situations change over time
4. Support multi-hop queries ("Who is the account manager for customers with > 3 unresolved tickets?")

---

### 4.1 Project Setup with `uv`

```bash
# Step 1: Create project folder
mkdir cloudsync-graphrag
cd cloudsync-graphrag

# Step 2: Initialise uv project
uv init
uv python pin 3.11

# Step 3: Add all dependencies
uv add \
  graphiti-core \
  langchain \
  langchain-core \
  langchain-openai \
  langchain-community \
  langchain-neo4j \
  neo4j \
  python-dotenv \
  asyncio

# Step 4: Start Neo4j via Docker (required graph database)
docker run \
  --name neo4j-graphrag \
  -p 7474:7474 \
  -p 7687:7687 \
  -d \
  -e NEO4J_AUTH=neo4j/password123 \
  neo4j:5.26

# Step 5: Create .env file
cat > .env << 'EOF'
OPENAI_API_KEY=sk-...
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=password123
DEFAULT_DATABASE=neo4j
EOF

# Step 6: Create project files
touch 01_ingest_episodes.py
touch 02_query_graph.py
touch 03_langchain_rag_chain.py
touch 04_advanced_queries.py
```

Final project structure:

```
cloudsync-graphrag/
├── .venv/
├── .python-version
├── .env
├── pyproject.toml
├── uv.lock
├── 01_ingest_episodes.py     ← Build the knowledge graph
├── 02_query_graph.py         ← Direct graph queries
├── 03_langchain_rag_chain.py ← Full RAG pipeline
└── 04_advanced_queries.py    ← Multi-hop + temporal queries
```

---

### 4.2 Step 1 — Ingest Episodes (Build the Knowledge Base)

```python
# 01_ingest_episodes.py
# ================================================================
# PURPOSE: Feed real-world business events into the knowledge graph
# Each episode represents a customer interaction, support ticket,
# or account change — automatically parsed into entities + edges
# ================================================================

import asyncio
import os
from datetime import datetime, timezone, timedelta
from dotenv import load_dotenv
from graphiti_core import Graphiti
from graphiti_core.nodes import EpisodeType

load_dotenv()

# ── Initialize Graphiti client ───────────────────────────────────
graphiti = Graphiti(
    uri=os.getenv("NEO4J_URI", "bolt://localhost:7687"),
    user=os.getenv("NEO4J_USER", "neo4j"),
    password=os.getenv("NEO4J_PASSWORD", "password123")
)

# ── Helper: build a timestamp N days ago ─────────────────────────
def days_ago(n: int) -> datetime:
    return datetime.now(timezone.utc) - timedelta(days=n)

async def ingest_all_episodes():
    """
    Feed a stream of business events into the knowledge graph.
    Graphiti automatically extracts entities and relationships from
    each episode and stores them with temporal validity.
    """

    print("🏗️  Building indices and constraints...")
    await graphiti.build_indices_and_constraints()

    # ── EPISODE GROUP 1: Customer Account Setup ───────────────────
    # These represent historical onboarding events

    print("\n📥 Ingesting customer account episodes...")

    await graphiti.add_episode(
        name="customer_acme_onboarding",
        episode_body=(
            "ACME Corp signed up for the CloudSync Enterprise Plan on January 10, 2024. "
            "Their account manager is Sarah Johnson. The primary contact is David Lee "
            "(david.lee@acme.com). ACME has 250 seats licensed."
        ),
        source=EpisodeType.text,
        source_description="CRM onboarding record",
        reference_time=days_ago(180),       # happened 6 months ago
        group_id="cloudsync_support"        # namespace for multi-tenancy
    )
    # Graphiti automatically extracts:
    # Entities: ACME Corp, Enterprise Plan, Sarah Johnson, David Lee
    # Edges:    ACME Corp --[subscribed_to]--> Enterprise Plan
    #           Sarah Johnson --[manages_account]--> ACME Corp
    #           David Lee --[primary_contact_of]--> ACME Corp
    # Temporal: all edges valid from 180 days ago

    await graphiti.add_episode(
        name="customer_beta_onboarding",
        episode_body=(
            "BetaStart Inc joined CloudSync on the Pro Plan on March 1, 2024. "
            "Account manager is Mike Chen. The contact is Emily Torres "
            "(emily.torres@betastart.com). They have 50 seats."
        ),
        source=EpisodeType.text,
        source_description="CRM onboarding record",
        reference_time=days_ago(120),
        group_id="cloudsync_support"
    )

    await graphiti.add_episode(
        name="customer_gamma_onboarding",
        episode_body=(
            "GammaTech Ltd signed up for the Starter Plan on April 15, 2024. "
            "Account manager is Sarah Johnson. Contact is Raj Patel "
            "(raj.patel@gammatech.io). They have 10 seats."
        ),
        source=EpisodeType.text,
        source_description="CRM onboarding record",
        reference_time=days_ago(90),
        group_id="cloudsync_support"
    )

    print("   ✅ Account episodes ingested")

    # ── EPISODE GROUP 2: Plan Changes ─────────────────────────────
    # These episodes will UPDATE the graph — Graphiti invalidates
    # the old plan edges and creates new ones automatically

    print("\n📥 Ingesting plan change episodes...")

    await graphiti.add_episode(
        name="acme_plan_upgrade",
        episode_body=(
            "ACME Corp upgraded from Enterprise Plan to Enterprise Plus Plan "
            "on June 1, 2024, adding 100 more seats (now 350 total). "
            "The upgrade was approved by David Lee after a successful Q1 review."
        ),
        source=EpisodeType.text,
        source_description="Billing system event",
        reference_time=days_ago(60),
        group_id="cloudsync_support"
    )
    # Graphiti detects conflict:
    # Old: ACME Corp --[subscribed_to]--> Enterprise Plan  (t_valid=180d ago)
    # New: ACME Corp --[subscribed_to]--> Enterprise Plus  (t_valid=60d ago)
    # Action: Invalidates old edge, creates new one → history preserved!

    await graphiti.add_episode(
        name="betastart_plan_downgrade",
        episode_body=(
            "BetaStart Inc downgraded from Pro Plan to Starter Plan on July 15, 2024 "
            "due to budget constraints. Their seat count reduced from 50 to 15. "
            "Emily Torres confirmed the change."
        ),
        source=EpisodeType.text,
        source_description="Billing system event",
        reference_time=days_ago(30),
        group_id="cloudsync_support"
    )

    print("   ✅ Plan change episodes ingested")

    # ── EPISODE GROUP 3: Support Tickets ──────────────────────────

    print("\n📥 Ingesting support ticket episodes...")

    await graphiti.add_episode(
        name="ticket_001_acme_sync_issue",
        episode_body=(
            "Support Ticket #001: ACME Corp reported a data synchronization failure "
            "on June 10, 2024. Assigned to support engineer Tom Wilson. "
            "The issue affects their Salesforce integration. Status: Open. Priority: High."
        ),
        source=EpisodeType.text,
        source_description="Support ticket system",
        reference_time=days_ago(50),
        group_id="cloudsync_support"
    )

    await graphiti.add_episode(
        name="ticket_002_acme_billing",
        episode_body=(
            "Support Ticket #002: ACME Corp raised a billing discrepancy on July 1, 2024. "
            "They were charged incorrectly for the Enterprise Plus plan. "
            "Assigned to billing specialist Lisa Park. Status: Open. Priority: Medium."
        ),
        source=EpisodeType.text,
        source_description="Support ticket system",
        reference_time=days_ago(35),
        group_id="cloudsync_support"
    )

    await graphiti.add_episode(
        name="ticket_003_gamma_login",
        episode_body=(
            "Support Ticket #003: GammaTech Ltd reported users unable to login "
            "on July 20, 2024. Root cause: SSO misconfiguration. "
            "Assigned to Tom Wilson. Status: Resolved on July 22, 2024. Priority: Critical."
        ),
        source=EpisodeType.text,
        source_description="Support ticket system",
        reference_time=days_ago(25),
        group_id="cloudsync_support"
    )

    await graphiti.add_episode(
        name="ticket_004_betastart_perf",
        episode_body=(
            "Support Ticket #004: BetaStart Inc reported slow performance after "
            "downgrading to Starter Plan on July 28, 2024. "
            "Emily Torres suspects it's related to the reduced seat allocation. "
            "Assigned to Tom Wilson. Status: Open. Priority: Low."
        ),
        source=EpisodeType.text,
        source_description="Support ticket system",
        reference_time=days_ago(15),
        group_id="cloudsync_support"
    )

    print("   ✅ Support ticket episodes ingested")

    # ── EPISODE GROUP 4: Structured JSON Data ─────────────────────
    # Graphiti can also ingest structured JSON directly

    print("\n📥 Ingesting structured JSON data...")

    usage_data = {
        "event": "monthly_usage_report",
        "date": "2024-07-31",
        "customers": [
            {
                "name": "ACME Corp",
                "api_calls_this_month": 1250000,
                "storage_gb": 450,
                "active_users": 320,
                "plan": "Enterprise Plus"
            },
            {
                "name": "BetaStart Inc",
                "api_calls_this_month": 45000,
                "storage_gb": 12,
                "active_users": 14,
                "plan": "Starter"
            },
            {
                "name": "GammaTech Ltd",
                "api_calls_this_month": 28000,
                "storage_gb": 8,
                "active_users": 9,
                "plan": "Starter"
            }
        ]
    }

    import json
    await graphiti.add_episode(
        name="monthly_usage_report_july",
        episode_body=json.dumps(usage_data),
        source=EpisodeType.json,
        source_description="Monthly usage analytics report",
        reference_time=days_ago(7),
        group_id="cloudsync_support"
    )

    print("   ✅ JSON data ingested")
    print("\n🎉 All episodes ingested! Graph is ready for querying.")

# ── Run ──────────────────────────────────────────────────────────
asyncio.run(ingest_all_episodes())
```

Run it:
```bash
uv run 01_ingest_episodes.py
```

**What the graph looks like after ingestion:**

```
Knowledge Graph State (simplified):

[ACME Corp] ──subscribed_to (CURRENT)──▶ [Enterprise Plus Plan]
            ──subscribed_to (EXPIRED)───▶ [Enterprise Plan]      ← kept in history
            ──has_contact──────────────▶ [David Lee]
            ──managed_by───────────────▶ [Sarah Johnson]
            ──has_ticket───────────────▶ [Ticket #001: Sync Issue]  (Open)
            ──has_ticket───────────────▶ [Ticket #002: Billing]     (Open)
            ──api_calls_july───────────▶ 1,250,000

[BetaStart Inc] ──subscribed_to (CURRENT)──▶ [Starter Plan]
                ──subscribed_to (EXPIRED)───▶ [Pro Plan]
                ──has_contact──────────────▶ [Emily Torres]
                ──managed_by───────────────▶ [Mike Chen]
                ──has_ticket───────────────▶ [Ticket #004: Perf]    (Open)

[GammaTech Ltd] ──subscribed_to──▶ [Starter Plan]
                ──has_contact────▶ [Raj Patel]
                ──managed_by─────▶ [Sarah Johnson]
                ──has_ticket─────▶ [Ticket #003: Login]  (Resolved)

[Tom Wilson] ──assigned_to──▶ [Ticket #001]
             ──assigned_to──▶ [Ticket #003]
             ──assigned_to──▶ [Ticket #004]

[Sarah Johnson] ──manages_account──▶ [ACME Corp]
                ──manages_account──▶ [GammaTech Ltd]
```

---

### 4.3 Step 2 — Direct Graph Queries

```python
# 02_query_graph.py
# ================================================================
# PURPOSE: Demonstrate direct Graphiti search — returning facts
# from the knowledge graph with temporal context
# ================================================================

import asyncio
import os
from dotenv import load_dotenv
from graphiti_core import Graphiti

load_dotenv()

graphiti = Graphiti(
    uri=os.getenv("NEO4J_URI"),
    user=os.getenv("NEO4J_USER"),
    password=os.getenv("NEO4J_PASSWORD")
)

async def demo_queries():

    # ── Query 1: Current state of a customer ─────────────────────
    print("=" * 60)
    print("Query 1: What is ACME Corp's current status?")
    print("=" * 60)

    results = await graphiti.search(
        query="What is ACME Corp's current plan and open support tickets?",
        num_results=5,
        group_ids=["cloudsync_support"]     # filter by namespace
    )

    for edge in results:
        print(f"\n  Fact:       {edge.fact}")
        print(f"  Valid From: {edge.valid_at}")
        print(f"  Valid To:   {edge.invalid_at or 'NOW (current)'}")

    # ── Query 2: Multi-hop — account managers and their issues ───
    print("\n" + "=" * 60)
    print("Query 2: Which customers managed by Sarah Johnson have open tickets?")
    print("=" * 60)

    results = await graphiti.search(
        query="customers managed by Sarah Johnson with open support tickets",
        num_results=5,
        group_ids=["cloudsync_support"]
    )

    for edge in results:
        print(f"\n  Fact: {edge.fact}")
        print(f"  From: {edge.valid_at}")

    # ── Query 3: Temporal query — plan history ────────────────────
    print("\n" + "=" * 60)
    print("Query 3: What plan was ACME Corp on before the upgrade?")
    print("=" * 60)

    results = await graphiti.search(
        query="ACME Corp subscription plan history changes",
        num_results=5,
        group_ids=["cloudsync_support"]
    )

    # Graphiti returns both current AND expired edges
    for edge in results:
        status = "CURRENT" if edge.invalid_at is None else f"EXPIRED on {edge.invalid_at.date()}"
        print(f"\n  Fact:   {edge.fact}")
        print(f"  Status: {status}")

    # ── Query 4: Find overloaded support engineer ─────────────────
    print("\n" + "=" * 60)
    print("Query 4: How many open tickets does Tom Wilson have?")
    print("=" * 60)

    results = await graphiti.search(
        query="tickets assigned to Tom Wilson status open",
        num_results=10,
        group_ids=["cloudsync_support"]
    )

    for edge in results:
        print(f"\n  Fact: {edge.fact}")

asyncio.run(demo_queries())
```

Run it:
```bash
uv run 02_query_graph.py
```

**Example Output:**
```
============================================================
Query 1: What is ACME Corp's current status?
============================================================

  Fact:       ACME Corp subscribed to Enterprise Plus Plan
  Valid From: 2024-06-01
  Valid To:   NOW (current)

  Fact:       ACME Corp has open Ticket #001 for sync failure
  Valid From: 2024-06-10
  Valid To:   NOW (current)

  Fact:       ACME Corp has open Ticket #002 for billing discrepancy
  Valid From: 2024-07-01
  Valid To:   NOW (current)

============================================================
Query 3: ACME Corp plan history
============================================================

  Fact:   ACME Corp subscribed to Enterprise Plus Plan
  Status: CURRENT

  Fact:   ACME Corp subscribed to Enterprise Plan
  Status: EXPIRED on 2024-06-01     ← history preserved!
```

---

### 4.4 Step 3 — Full LangChain RAG Chain

```python
# 03_langchain_rag_chain.py
# ================================================================
# PURPOSE: Wire Graphiti retrieval into a full LangChain RAG chain
# using LCEL for clean, composable pipeline
# ================================================================

import asyncio
import os
from dotenv import load_dotenv
from graphiti_core import Graphiti
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnablePassthrough, RunnableParallel

load_dotenv()

# ── Initialize Graphiti ──────────────────────────────────────────
graphiti = Graphiti(
    uri=os.getenv("NEO4J_URI"),
    user=os.getenv("NEO4J_USER"),
    password=os.getenv("NEO4J_PASSWORD")
)

# ── Initialize LLM ───────────────────────────────────────────────
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# ── Step 1: Graphiti Retriever ────────────────────────────────────
# Wraps graphiti.search() as an async-compatible retriever function

async def graphiti_retriever(query: str, num_results: int = 8) -> str:
    """
    Retrieve relevant facts from the knowledge graph.
    Returns a formatted string of facts with temporal context.
    """
    results = await graphiti.search(
        query=query,
        num_results=num_results,
        group_ids=["cloudsync_support"]
    )

    if not results:
        return "No relevant facts found in the knowledge graph."

    formatted_facts = []
    for i, edge in enumerate(results, 1):
        validity = (
            f"Valid: {edge.valid_at.strftime('%Y-%m-%d') if edge.valid_at else 'unknown'}"
            + (f" → {edge.invalid_at.strftime('%Y-%m-%d')}" if edge.invalid_at else " → PRESENT")
        )
        formatted_facts.append(
            f"[Fact {i}] {edge.fact}\n"
            f"           {validity}"
        )

    return "\n\n".join(formatted_facts)

# ── Step 2: RAG Prompt ────────────────────────────────────────────
support_prompt = ChatPromptTemplate.from_template("""
You are an intelligent customer support AI for CloudSync, a SaaS platform.
You have access to a real-time knowledge graph containing customer information,
support tickets, account history, and usage data.

Your retrieved knowledge graph context is shown below. Each fact includes
TEMPORAL validity — use this to distinguish current facts from historical ones.
ALWAYS prefer facts marked "→ PRESENT" (currently valid) over expired ones.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
KNOWLEDGE GRAPH CONTEXT:
{context}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Support Question: {question}

Instructions:
- Answer based on CURRENT facts (→ PRESENT) for the current status
- Reference historical facts (with end dates) when explaining what changed
- If information is missing, say so clearly
- Cite the source fact numbers when relevant

Answer:
""")

# ── Step 3: Build the LCEL RAG Chain ─────────────────────────────
#
# Chain flow:
#   question ──▶ graphiti_retriever (async) ──▶ format context
#             ──▶ pass question through
#             ──▶ fill prompt
#             ──▶ llm
#             ──▶ parse output

async def run_graphrag_chain(question: str) -> str:
    """Full Knowledge Graph RAG pipeline."""

    # Step A: Retrieve from knowledge graph
    print(f"\n🔍 Retrieving from graph for: '{question}'")
    context = await graphiti_retriever(question)

    # Step B: Build prompt
    filled_prompt = support_prompt.format_messages(
        context=context,
        question=question
    )

    # Step C: Generate answer
    response = await llm.ainvoke(filled_prompt)
    answer   = response.content

    return answer

# ── Step 4: Run Example Questions ────────────────────────────────

async def main():

    questions = [
        # ── Standard relational query ──
        "What is ACME Corp's current subscription plan and how many seats do they have?",

        # ── Multi-hop query ──
        "Who is the account manager for the customer with the most open support tickets?",

        # ── Temporal query ──
        "What plan was BetaStart Inc on before their most recent plan change, and why did they change?",

        # ── Aggregation + relationship query ──
        "Which support engineer has the most open tickets right now? List the tickets.",

        # ── Business intelligence query ──
        "Which of Sarah Johnson's customers might need attention based on their ticket status?",
    ]

    for question in questions:
        print("\n" + "═" * 65)
        print(f"❓ QUESTION: {question}")
        print("═" * 65)

        answer = await run_graphrag_chain(question)
        print(f"\n💬 ANSWER:\n{answer}")
        print()

asyncio.run(main())
```

Run it:
```bash
uv run 03_langchain_rag_chain.py
```

**Expected Answers:**

```
═════════════════════════════════════════════════════════════════
❓ QUESTION: What is ACME Corp's current plan and how many seats?
═════════════════════════════════════════════════════════════════
💬 ANSWER:
ACME Corp is currently on the Enterprise Plus Plan (Fact 1, → PRESENT)
with 350 seats. They upgraded from the Enterprise Plan on June 1, 2024,
adding 100 more seats to their original 250 (Fact 3, expired 2024-06-01).

═════════════════════════════════════════════════════════════════
❓ QUESTION: Which support engineer has the most open tickets?
═════════════════════════════════════════════════════════════════
💬 ANSWER:
Tom Wilson has 3 open tickets (→ PRESENT):
- Ticket #001: ACME Corp data synchronization failure (High priority)
- Ticket #003: GammaTech Ltd SSO login issue (this was Resolved, July 22)
- Ticket #004: BetaStart Inc slow performance after downgrade (Low priority)

Note: Tom Wilson's workload is notably high. Lisa Park has 1 open ticket
(Ticket #002: ACME Corp billing discrepancy).
```

---

### 4.5 Step 4 — Advanced Multi-Hop & Temporal Queries

```python
# 04_advanced_queries.py
# ================================================================
# PURPOSE: Demonstrate advanced Graph RAG capabilities that would
# be IMPOSSIBLE with traditional vector-only RAG
# ================================================================

import asyncio
import os
from datetime import datetime, timezone, timedelta
from dotenv import load_dotenv
from graphiti_core import Graphiti
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

graphiti = Graphiti(
    uri=os.getenv("NEO4J_URI"),
    user=os.getenv("NEO4J_USER"),
    password=os.getenv("NEO4J_PASSWORD")
)
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# ── Advanced Query 1: Point-in-Time Historical Lookup ─────────────
async def historical_state_query():
    """
    Ask: "What plan was ACME Corp on in May 2024?"
    This is IMPOSSIBLE with traditional RAG — it requires temporal
    graph traversal to find which edge was valid at that date.
    """
    print("\n🕐 ADVANCED QUERY 1: Point-in-Time Historical State")
    print("     'What was ACME Corp's plan in May 2024?'")
    print("-" * 60)

    # Graphiti stores edge validity intervals — search at a point in time
    results = await graphiti.search(
        query="ACME Corp subscription plan May 2024",
        num_results=5,
        group_ids=["cloudsync_support"]
    )

    for edge in results:
        # Check if this edge was valid in May 2024
        may_2024 = datetime(2024, 5, 15, tzinfo=timezone.utc)
        if edge.valid_at and edge.valid_at <= may_2024:
            expired_after = edge.invalid_at is None or edge.invalid_at >= may_2024
            if expired_after:
                print(f"  ✅ VALID IN MAY 2024: {edge.fact}")
            else:
                print(f"  ❌ Already expired by May: {edge.fact}")

# ── Advanced Query 2: Churn Risk Detection ────────────────────────
async def churn_risk_analysis():
    """
    Detect customers at risk of churning based on:
    - Recent plan downgrade
    - Open support tickets
    - Low usage
    This requires connecting MULTIPLE entity types across hops.
    """
    print("\n⚠️  ADVANCED QUERY 2: Churn Risk Analysis")
    print("     'Which customers show signs of churn risk?'")
    print("-" * 60)

    churn_signals = await graphiti.search(
        query="customers who downgraded plan and have open support tickets low usage",
        num_results=10,
        group_ids=["cloudsync_support"]
    )

    # Compile into a report
    risk_prompt = ChatPromptTemplate.from_template("""
    You are a customer success analyst. Based on the following knowledge graph facts
    about CloudSync customers, identify which customers are at risk of churning.

    Look for signals like:
    - Recent plan downgrades
    - Multiple open support tickets
    - Low API usage relative to their plan
    - Billing complaints

    Knowledge Graph Facts:
    {facts}

    Produce a churn risk report ranking customers from HIGH to LOW risk.
    For each customer, list the risk signals found and a recommended action.
    """)

    facts_text = "\n".join(f"- {e.fact}" for e in churn_signals)

    response = await llm.ainvoke(
        churn_risk_prompt := churn_risk_analysis_prompt := churn_prompt := risk_prompt.format_messages(
            facts=facts_text
        )
    )
    print(response.content)

# ── Advanced Query 3: Real-Time Update + Query ────────────────────
async def realtime_update_demo():
    """
    Demonstrate real-time graph update + immediate query.
    Traditional RAG: would need to re-embed and re-index.
    Graphiti: update is available for querying within milliseconds.
    """
    print("\n⚡ ADVANCED QUERY 3: Real-Time Update")
    print("     Adding a new ticket NOW and querying immediately...")
    print("-" * 60)

    # Add a new critical event RIGHT NOW
    await graphiti.add_episode(
        name="ticket_005_acme_critical_outage",
        episode_body=(
            "CRITICAL: Support Ticket #005 opened right now. "
            "ACME Corp is experiencing a complete service outage. "
            "All 320 active users cannot access CloudSync. "
            "CEO Jane Smith is personally escalating to our VP of Engineering, "
            "Robert Brown. Assigned to Tom Wilson and Lisa Park jointly. "
            "Status: Open. Priority: P0 - Critical."
        ),
        source=EpisodeType.text,
        source_description="Emergency escalation — real-time",
        reference_time=datetime.now(timezone.utc),
        group_id="cloudsync_support"
    )

    print("   ✅ New P0 ticket added to graph")
    print("   🔍 Querying immediately (no re-indexing needed)...\n")

    # Query the graph IMMEDIATELY — no batch recompute needed
    results = await graphiti.search(
        query="ACME Corp critical open tickets current escalations",
        num_results=5,
        group_ids=["cloudsync_support"]
    )

    for edge in results:
        print(f"   [{edge.valid_at.strftime('%H:%M:%S') if edge.valid_at else 'now'}] {edge.fact}")

# ── Advanced Query 4: Cross-Entity Aggregation ────────────────────
async def cross_entity_intelligence():
    """
    'Who is the best account manager to handle enterprise escalations?'
    Requires traversing: Customers → Plans → Tickets → Account Managers
    """
    print("\n🧠 ADVANCED QUERY 4: Cross-Entity Intelligence")
    print("     'Who should handle the ACME Corp P0 escalation?'")
    print("-" * 60)

    # Retrieve all relevant relationship context
    team_context = await graphiti.search(
        query="account managers customers enterprise plan open critical tickets assignments",
        num_results=10,
        group_ids=["cloudsync_support"]
    )

    escalation_prompt = ChatPromptTemplate.from_template("""
    You are a support operations manager. A critical P0 outage has been reported
    at ACME Corp. Using the knowledge graph context below, answer:

    1. Who is ACME Corp's current account manager?
    2. Which support engineers are already assigned to ACME tickets?
    3. Given current workloads, who should lead the P0 response?
    4. What is the escalation path (who to notify up the chain)?

    Knowledge Graph Context:
    {context}

    Provide a clear escalation plan with specific names and roles.
    """)

    facts = "\n".join(f"• {e.fact}" for e in team_context)
    response = await llm.ainvoke(
        escalation_prompt.format_messages(context=facts)
    )
    print(response.content)

# ── Run all advanced queries ──────────────────────────────────────
async def main():
    from graphiti_core.nodes import EpisodeType  # needed for realtime demo
    await historical_state_query()
    await churn_risk_analysis()
    await realtime_update_demo()
    await cross_entity_intelligence()

asyncio.run(main())
```

Run it:
```bash
uv run 04_advanced_queries.py
```

---

### 4.6 What Makes This Impossible with Traditional RAG

```
QUESTION: "Which customers managed by Sarah Johnson recently downgraded
           their plan AND have more than 1 open ticket?"

TRADITIONAL RAG ATTEMPT:
  Step 1: Embed the query
  Step 2: Find top-K semantically similar chunks
  
  Result: Returns chunks about:
    - "Sarah Johnson manages ACME Corp" (chunk A)
    - "BetaStart downgraded plan" (chunk B)
    - "Ticket #001 open" (chunk C)
    - "Ticket #002 open" (chunk D)
    
  ❌ PROBLEM: Chunks A, B, C, D are returned as isolated facts.
     The system has NO WAY to:
     - Know chunk B (BetaStart downgrade) is NOT managed by Sarah
     - Connect the two tickets in C and D to the same customer
     - Filter by "Sarah's customers" AND "downgraded" AND "open tickets"
     
  Result: LLM hallucinates a connection or says "I don't know"

KNOWLEDGE GRAPH RAG:
  Step 1: Extract entities: Sarah Johnson, plan changes, open tickets
  Step 2: Traverse from Sarah Johnson node:
    Sarah Johnson ──manages──▶ ACME Corp ──has_ticket──▶ Ticket #001 (Open)
                                          ──has_ticket──▶ Ticket #002 (Open)
                                          ──plan_change──▶ Upgraded (NOT downgraded)
    Sarah Johnson ──manages──▶ GammaTech ──has_ticket──▶ Ticket #003 (RESOLVED)
                                          ──subscribed_to──▶ Starter (no downgrade)
  Step 3: Filter: downgraded? No. More than 1 open ticket? No.
  
  ✅ RESULT: "None of Sarah Johnson's current customers match all criteria."
              (Correct! BetaStart downgraded but is managed by Mike Chen, not Sarah)
```

---

## 5. Complete Architecture Summary

```
┌─────────────────────────────────────────────────────────────────────┐
│              KNOWLEDGE GRAPH RAG — FULL PIPELINE                    │
│                                                                     │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │              KNOWLEDGE INGESTION (Offline + Real-Time)      │   │
│  │                                                             │   │
│  │  Text / JSON / Chat ──▶ Graphiti.add_episode()             │   │
│  │                              ↓                              │   │
│  │              LLM Entity + Relation Extraction               │   │
│  │                              ↓                              │   │
│  │         Conflict Detection + Temporal Invalidation          │   │
│  │                              ↓                              │   │
│  │            Neo4j: Nodes + Edges + Episodes                  │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                              ↕ (real-time sync)                     │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │              HYBRID RETRIEVAL (Online, per query)           │   │
│  │                                                             │   │
│  │  User Query ──▶ Graphiti.search()                          │   │
│  │                     ├── Semantic (embedding similarity)     │   │
│  │                     ├── Keyword (BM25)                      │   │
│  │                     ├── Graph traversal (hops)              │   │
│  │                     └── Temporal filter (valid_at range)    │   │
│  │                              ↓                              │   │
│  │                  Ranked, dated facts (edges)                │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                              ↓                                      │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │                 GENERATION (LangChain LCEL)                 │   │
│  │                                                             │   │
│  │  Retrieved Facts + Temporal Context                         │   │
│  │          ↓                                                  │   │
│  │  ChatPromptTemplate (with temporal instructions)            │   │
│  │          ↓                                                  │   │
│  │  LLM (gpt-4o-mini / Claude / Gemini)                       │   │
│  │          ↓                                                  │   │
│  │  Grounded, temporally-accurate Answer                       │   │
│  └─────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 6. Key Takeaways

| Concept | Summary |
|---|---|
| **Knowledge Graph** | Entities + Relationships stored as graph triplets |
| **Why Graph RAG** | Multi-hop reasoning, temporal tracking, relationship queries |
| **vs Traditional RAG** | Structured edges vs. flat chunks; traversal vs. similarity |
| **Graphiti** | Open-source, real-time, temporally-aware KG framework by Zep AI |
| **Bi-temporal model** | Every edge has `t_valid` and `t_invalid` — history never deleted |
| **Episodes** | Unit of ingestion — text, JSON, or chat messages |
| **Hybrid retrieval** | Semantic + BM25 + graph traversal + reranking |
| **Real-time updates** | New facts available instantly — no batch recompute |
| **LangChain integration** | `graphiti.search()` wraps as a retriever in LCEL chains |

---

# Assignment: Knowledge Graph RAG



# Knowledge Graph RAG — End-to-End Knowledge Graph RAG Application

> **What we're building:** A real KG-RAG system using the existing `nexacore_knowledge_report.pdf`.
> We ingest it into a Neo4j knowledge graph, query it with Graphiti's hybrid retrieval,
> and compare answers against Traditional RAG on the same hard questions.

---

## The PDF — NexaCore Technologies Knowledge Report

The `nexacore_knowledge_report.pdf` is a **7-page fictional corporate intelligence document**.
It is deliberately designed with deep cross-entity relationships that span multiple pages —
exactly the kind of content that breaks Traditional RAG but where Knowledge Graph RAG thrives.

| Pages | Content | What makes it hard |
|---|---|---|
| 1–2 | Org structure, reporting chains, Projects Helios & Aurora | 9 named executives whose roles interlink across later pages |
| 3–4 | Products, 3 flagship clients, technology stack | Clients link to people, modules, and cloud providers |
| 5–6 | P0 outage, MIT partnership, DataFlow acquisition | 6-person incident chain; conflict of interest requires prior-history join |
| 7 | ARR financials, roadmap, risk register | 6 risks each owned by a named person from earlier pages |

**Example multi-hop chains hidden in the PDF:**

```
Chain 1 (3 hops, pages 1 + 5):
  GlobalBank outage ──caused by── Elena Vasquez's Kafka change
    ──part of── Project Helios
    ──approved by── Dr. Priya Mehta

Chain 2 (5 hops, pages 1 + 5 + 6):
  DataFlow delay ──blocks── Tomasz Kowalski's integration
    ──delays── Project Helios Q1 milestone
    ──delays── NIP v4.0
    ──loses── GlobalBank as pilot client
    ──risks── $8.2M ARR renewal

Chain 3 (prior history + current role, page 5):
  Nathan Brooks (DataFlow CEO) + Elena Vasquez
    ──both worked at── StreamCore Inc. (2019)
    ──Elena must evaluate── DataFlow technology
    ──conflict flagged by── James Okafor → Risk R-003
```

Traditional RAG retrieves each of these facts as isolated chunks with no ability to JOIN them.
Knowledge Graph RAG traverses the edges and returns the full connected chain.

---

## Project Setup with `uv`

```bash
# Step 1 — Create project folder
mkdir nexacore-kgrag
cd nexacore-kgrag

# Step 2 — Initialise uv project
uv init
uv python pin 3.11

# Step 3 — Add all dependencies
uv add \
  graphiti-core \
  langchain \
  langchain-core \
  langchain-openai \
  langchain-community \
  langchain-huggingface \
  pypdf \
  faiss-cpu \
  sentence-transformers \
  python-dotenv \
  neo4j \
  rich \
  streamlit

# Step 4 — Start Neo4j via Docker
docker run \
  --name neo4j-nexacore \
  -p 7474:7474 -p 7687:7687 \
  -d \
  -e NEO4J_AUTH=neo4j/nexacore123 \
  neo4j:5.26

# Step 5 — Configure environment
cat > .env << 'EOF'
OPENAI_API_KEY=sk-...
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=nexacore123
EOF

# Step 6 — Copy the existing PDF into the project root
cp /path/to/nexacore_knowledge_report.pdf .

# Step 7 — Create application files
touch 01_ingest_pdf.py
touch 02_kg_rag_chain.py
touch 03_compare_rag.py
touch 04_interactive_qa.py
touch 05_streamlit_app.py
```

Final project structure:

```
nexacore-kgrag/
├── .venv/
├── .python-version
├── .env
├── pyproject.toml
├── uv.lock
├── nexacore_knowledge_report.pdf   ← existing PDF — just copy it here
├── 01_ingest_pdf.py                ← parse PDF → chunk → build Knowledge Graph in Neo4j
├── 02_kg_rag_chain.py              ← KG-RAG chain (Graphiti retrieval + LCEL generation)
├── 03_compare_rag.py               ← KG-RAG vs Traditional RAG side-by-side
├── 04_interactive_qa.py            ← interactive rich terminal Q&A app
└── 05_streamlit_app.py             ← Streamlit web UI (KG-RAG + Traditional RAG comparison)
```

---

## File 1 — Ingest the PDF into the Knowledge Graph

```python
# 01_ingest_pdf.py
# ================================================================
# PURPOSE: Read nexacore_knowledge_report.pdf, split it into
# overlapping text chunks, and feed each chunk into Graphiti
# as an "episode". Graphiti's LLM pipeline automatically extracts
# entities and typed relationships from every chunk and stores
# them as a structured graph in Neo4j.
# ================================================================
#
# HOW IT WORKS — STEP BY STEP:
#
#  Step 1 — Build Neo4j indices
#    graphiti.build_indices_and_constraints() creates:
#      • A vector index on edge fact embeddings (for semantic search)
#      • A fulltext BM25 index on edge fact strings (for keyword search)
#      • Uniqueness constraints on entity node UUIDs
#    Only needs to run once per fresh Neo4j instance.
#
#  Step 2 — Extract text page-by-page using pypdf
#    We read each page individually rather than dumping the whole
#    PDF as one string. This preserves logical document structure
#    and lets us tag every episode with its source page number
#    for provenance tracking later.
#
#  Step 3 — Chunk each page into 800-char overlapping segments
#    Large pages are split at sentence boundaries ('. ') with
#    150-char overlap between consecutive chunks.
#
#    Why sentence-boundary splitting?
#      A fact like "Elena Vasquez, who leads Platform Core, caused
#      the GlobalBank outage" must NOT be cut between her name and
#      "caused the outage" — that would break entity-relationship
#      extraction in the LLM step.
#
#    Why 150-char overlap?
#      If chunk N ends with "...approved by Dr. Mehta" and chunk N+1
#      starts with "Project Helios has a $4.2M budget", the overlap
#      carries the tail of chunk N into chunk N+1 so the LLM sees
#      the connecting context and correctly extracts the edge:
#        Project Helios ──[approved_by]──▶ Dr. Priya Mehta
#
#  Step 4 — Add each chunk to Graphiti via add_episode()
#    Graphiti internally:
#      a) Sends the chunk to an LLM → extracts named entities
#         (people, projects, companies, technologies, dates)
#      b) Sends the chunk to an LLM → extracts typed relationships
#         (manages, works_on, caused, approved_by, integrates_with…)
#      c) Checks for conflicts with edges already in the graph and
#         invalidates any outdated facts (bi-temporal model)
#      d) Generates a vector embedding per edge fact string
#      e) Stores Entity nodes, Episode nodes, typed edges + all
#         metadata (valid_at, group_id, source_description) in Neo4j
#
#  Step 5 — Print summary + Neo4j inspection Cypher query
# ================================================================

import asyncio
import os
import re
from datetime import datetime, timezone
from dotenv import load_dotenv
from pypdf import PdfReader
from graphiti_core import Graphiti
from graphiti_core.nodes import EpisodeType

load_dotenv()

# ── Graphiti client — wraps all Neo4j communication ─────────────
graphiti = Graphiti(
    uri=os.getenv("NEO4J_URI",       "bolt://localhost:7687"),
    user=os.getenv("NEO4J_USER",     "neo4j"),
    password=os.getenv("NEO4J_PASSWORD", "nexacore123"),
)

def extract_pages(pdf_path: str) -> list[dict]:
    """
    Open the PDF and return the text of each non-empty page.

    Returns:
        List of { page_num: int, text: str, char_count: int }

    Pages that render as blank (e.g. table-only pages that pypdf
    cannot parse as text) are silently skipped.
    """
    reader = PdfReader(pdf_path)
    pages  = []
    for i, page in enumerate(reader.pages, 1):
        text = (page.extract_text() or "").strip()
        if text:
            pages.append({"page_num": i, "text": text, "char_count": len(text)})
    return pages

def chunk_text(text: str, chunk_size: int = 800, overlap: int = 150) -> list[str]:
    """
    Split page text into overlapping chunks at sentence boundaries.

    Algorithm:
      1. Split on sentence-ending punctuation (. ! ?)
      2. Accumulate sentences until chunk_size is reached
      3. Save the chunk, then start the next chunk with the last
         `overlap` characters of the previous chunk as a seed
         so both chunks share that context

    Args:
        text       : raw page text
        chunk_size : max chars per chunk (800 ≈ ~150 tokens)
        overlap    : chars from end of previous chunk prepended
                     to next chunk (150 chars ≈ 1–2 sentences)
    """
    sentences            = re.split(r'(?<=[.!?])\s+', text)
    chunks, current      = [], ""

    for sentence in sentences:
        if len(current) + len(sentence) <= chunk_size:
            current = (current + " " + sentence).strip()
        else:
            if current:
                chunks.append(current)
            overlap_seed = current[-overlap:] if len(current) > overlap else current
            current      = (overlap_seed + " " + sentence).strip()

    if current:
        chunks.append(current)
    return chunks

async def ingest_pdf(pdf_path: str):
    print("=" * 62)
    print("  NexaCore KG-RAG — PDF Ingestion Pipeline")
    print("=" * 62)

    # ── Step 1: Initialise Neo4j indices ─────────────────────────
    print("\n🏗️  Step 1: Building Neo4j indices and constraints...")
    await graphiti.build_indices_and_constraints()
    print("   ✅ Vector index, BM25 index, and constraints ready")

    # ── Step 2: Extract text page by page ────────────────────────
    print(f"\n📖 Step 2: Reading '{pdf_path}'...")
    pages = extract_pages(pdf_path)
    print(f"   Non-empty pages: {len(pages)}")
    for p in pages:
        print(f"   Page {p['page_num']:2d}  →  {p['char_count']} chars")

    # ── Step 3: Build the chunk list ─────────────────────────────
    print("\n✂️  Step 3: Chunking pages (800 chars, 150 overlap)...")
    all_chunks = []
    for page in pages:
        for j, chunk in enumerate(chunk_text(page["text"]), 1):
            all_chunks.append({
                "name":  f"nexacore_p{page['page_num']}_c{j}",
                "page":  page["page_num"],
                "chunk": j,
                "text":  chunk,
            })
    print(f"   Total chunks to ingest: {len(all_chunks)}")
    for c in all_chunks:
        preview = c["text"][:75].replace("\n", " ")
        print(f"   {c['name']:22s}  ({len(c['text'])} chars)  \"{preview}...\"")

    # ── Step 4: Feed each chunk into Graphiti ─────────────────────
    print("\n🧠 Step 4: Ingesting episodes → Knowledge Graph...")
    print("   Each episode triggers LLM entity + relation extraction.\n")

    for i, chunk in enumerate(all_chunks, 1):
        print(f"   [{i:02d}/{len(all_chunks)}] Ingesting: {chunk['name']}")
        preview = chunk["text"][:100].replace("\n", " ")
        print(f"            Preview : \"{preview}...\"")

        await graphiti.add_episode(
            name=chunk["name"],
            episode_body=chunk["text"],
            source=EpisodeType.text,
            # source_description is stored on every edge in Neo4j —
            # lets you trace which PDF page a relationship came from
            source_description=(
                f"NexaCore Technologies Internal Report — "
                f"Page {chunk['page']}, Chunk {chunk['chunk']}"
            ),
            # reference_time stamps when this fact became known
            reference_time=datetime.now(timezone.utc),
            # group_id namespaces all nodes + edges from this PDF
            # so searches can be scoped to nexacore_report only
            group_id="nexacore_report",
        )
        print(f"            ✅ Entities + edges stored in Neo4j\n")

    # ── Step 5: Summary ───────────────────────────────────────────
    print("=" * 62)
    print("  ✅ INGESTION COMPLETE")
    print("=" * 62)
    print(f"   Pages processed : {len(pages)}")
    print(f"   Chunks ingested : {len(all_chunks)}")
    print(f"   Neo4j group_id  : nexacore_report")
    print()
    print("   What is now in Neo4j:")
    print("   ├─ Entity nodes   — people, projects, companies, tools")
    print("   ├─ Typed edges    — manages, caused, approved_by, works_on…")
    print("   ├─ Episode nodes  — one per chunk (provenance records)")
    print("   ├─ Edge embeddings — for semantic vector search")
    print("   └─ BM25 fulltext  — for keyword search")
    print()
    print("   Inspect the graph → http://localhost:7474")
    print("   Cypher: MATCH (n)-[r]->(m) RETURN n, r, m LIMIT 100")

asyncio.run(ingest_pdf("nexacore_knowledge_report.pdf"))
```

Run it:
```bash
uv run 01_ingest_pdf.py
```

**What Graphiti extracts from a single chunk — example:**

```
Input chunk (from page 1):
  "Project Helios is jointly owned by Elena Vasquez and Dr. Arjun Nair.
   Budget: $4.2M approved by Dr. Priya Mehta. Target: Q1 2025."

Graphiti LLM extraction output:

  Entity nodes created / merged in Neo4j:
    ● Project Helios   (type: Project)
    ● Elena Vasquez    (type: Person)
    ● Dr. Arjun Nair   (type: Person)
    ● Dr. Priya Mehta  (type: Person)

  Typed edges stored with temporal metadata:
    Elena Vasquez  ──[co_owns]──────▶  Project Helios
    Dr. Arjun Nair ──[co_owns]──────▶  Project Helios
    Project Helios ──[approved_by]──▶  Dr. Priya Mehta
    Project Helios ──[has_budget]───▶  $4.2M
    Project Helios ──[target_date]──▶  Q1 2025

  Each edge also carries:
    valid_at   = datetime.now()   ← when this fact became known
    invalid_at = None             ← null means "still true → PRESENT"
    group_id   = "nexacore_report"
    embedding  = vector(edge.fact_string)
```

---

## File 2 — Knowledge Graph RAG Chain

```python
# 02_kg_rag_chain.py
# ================================================================
# PURPOSE: The core KG-RAG pipeline. Accepts a user question,
# retrieves relevant facts from the Neo4j knowledge graph via
# Graphiti's hybrid search, formats them with temporal context,
# and passes them through a LangChain LCEL chain to generate
# a grounded, multi-hop-aware answer.
# ================================================================
#
# HOW IT WORKS — STEP BY STEP:
#
#  Step 1 — retrieve_from_graph(question)
#    Calls graphiti.search() which runs FOUR strategies in parallel:
#      a) Semantic search   — embeds the question as a dense vector,
#         scores all stored edge facts by cosine similarity
#      b) BM25 keyword      — tokenises the question, scores edge
#         facts by term frequency / inverse document frequency
#      c) Graph traversal   — from the entity nodes matched in (a)
#         and (b), walks the Neo4j graph N hops outward to collect
#         connected facts that may not be semantically similar but
#         are relationally relevant
#      d) Cross-encoder     — reranks all candidates from (a)–(c)
#         with a bi-encoder model for final precision ordering
#    Returns the top-10 edges, each with valid_at / invalid_at.
#
#  Step 2 — format_context(edges)
#    Converts raw edge objects into a numbered, human-readable
#    fact list. Each line includes the edge's time window so the
#    LLM can distinguish "→ PRESENT" (current) from expired facts.
#
#  Step 3 — KG_RAG_PROMPT
#    A ChatPromptTemplate that instructs the LLM to:
#      • Prefer "→ PRESENT" facts for current-state questions
#      • Trace multi-hop chains explicitly with each hop named
#      • Cite [Fact NN] numbers for every claim
#      • Acknowledge when the graph lacks enough information
#
#  Step 4 — gpt-4o generation
#    filled prompt → ChatOpenAI → StrOutputParser → string answer
# ================================================================

import asyncio
import os
from dotenv import load_dotenv
from graphiti_core import Graphiti
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

graphiti = Graphiti(
    uri=os.getenv("NEO4J_URI"),
    user=os.getenv("NEO4J_USER"),
    password=os.getenv("NEO4J_PASSWORD"),
)
llm = ChatOpenAI(model="gpt-4o", temperature=0)

# ── Step 1: Graphiti hybrid retriever ────────────────────────────
async def retrieve_from_graph(question: str, num_results: int = 10) -> tuple[str, list]:
    """
    Run Graphiti hybrid retrieval scoped to the NexaCore graph.

    group_ids=["nexacore_report"] ensures the search only touches
    edges extracted from this PDF — not other documents in Neo4j.

    Returns:
        formatted_context : numbered fact string ready for the prompt
        edges             : raw edge objects (for verbose display)
    """
    edges = await graphiti.search(
        query=question,
        num_results=num_results,
        group_ids=["nexacore_report"],
    )

    if not edges:
        return "No relevant facts found in the knowledge graph.", []

    lines = []
    for i, edge in enumerate(edges, 1):
        valid_from = edge.valid_at.strftime("%Y-%m-%d")  if edge.valid_at  else "unknown"
        valid_to   = edge.invalid_at.strftime("%Y-%m-%d") if edge.invalid_at else "PRESENT"
        lines.append(
            f"[Fact {i:02d}] {edge.fact}\n"
            f"             Time window: {valid_from} → {valid_to}"
        )

    return "\n\n".join(lines), edges

# ── Step 2 & 3: Prompt ───────────────────────────────────────────
KG_RAG_PROMPT = ChatPromptTemplate.from_template("""
You are an expert enterprise analyst for NexaCore Technologies.
You have been given facts retrieved from a structured Knowledge Graph
built from NexaCore's internal intelligence report.

Each fact includes a TIME WINDOW:
  "YYYY-MM-DD → PRESENT"    = this fact is currently true
  "YYYY-MM-DD → YYYY-MM-DD" = this fact is historical (expired)

Always prefer "→ PRESENT" facts for current-state questions.
Use expired facts only when the question is explicitly about history.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
KNOWLEDGE GRAPH FACTS:
{context}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

QUESTION: {question}

INSTRUCTIONS:
- Answer strictly from the facts above.
- For multi-hop questions, trace each link in the chain explicitly.
- Cite the [Fact NN] number(s) supporting each claim.
- If the facts do not fully answer the question, say so clearly.

ANSWER:
""")

# ── Step 4: Full KG-RAG pipeline ─────────────────────────────────
async def kg_rag(question: str, verbose: bool = True) -> dict:
    """
    End-to-end KG-RAG.

    LCEL flow:
      question
        ──▶ retrieve_from_graph()          (Graphiti hybrid search)
        ──▶ format facts with timestamps   (context string)
        ──▶ KG_RAG_PROMPT.format_messages  (fill template)
        ──▶ llm.ainvoke()                  (gpt-4o generation)
        ──▶ .content                       (extract string answer)
    """
    if verbose:
        print(f"\n{'═'*65}")
        print(f"  🔍 KG-RAG  |  {question}")
        print(f"{'═'*65}")

    context, edges = await retrieve_from_graph(question)

    if verbose:
        print(f"\n  📊 {len(edges)} facts retrieved:")
        for i, e in enumerate(edges[:6], 1):
            print(f"     [{i:02d}] {e.fact[:85]}...")

    messages = KG_RAG_PROMPT.format_messages(context=context, question=question)
    response = await llm.ainvoke(messages)
    answer   = response.content

    if verbose:
        print(f"\n  💬 ANSWER:\n{answer}\n")

    return {
        "question":            question,
        "context":             context,
        "answer":              answer,
        "num_facts_retrieved": len(edges),
    }

# ── Example queries ───────────────────────────────────────────────
async def main():
    questions = [
        "Who caused the GlobalBank outage and what project were they working on?",
        "What is the link between the DataFlow acquisition and NIP v4.0?",
        "Which person appears in the most open risk items?",
    ]
    for q in questions:
        await kg_rag(q)

asyncio.run(main())
```

Run it:
```bash
uv run 02_kg_rag_chain.py
```

---

## File 3 — Traditional RAG vs Knowledge Graph RAG Comparison

```python
# 03_compare_rag.py
# ================================================================
# PURPOSE: Run the same four hard questions through both
# Traditional RAG and KG-RAG using the same PDF and the same
# LLM (gpt-4o) so that any quality difference is entirely due
# to the retrieval method — not the generator.
# ================================================================
#
# TRADITIONAL RAG PIPELINE:
#   pypdf → all pages as one string
#   → RecursiveCharacterTextSplitter (500 chars, 100 overlap)
#   → HuggingFaceEmbeddings (all-MiniLM-L6-v2, CPU)
#   → FAISS in-memory vector store
#   → Retriever: top-4 chunks by cosine similarity
#   → ChatPromptTemplate → gpt-4o → answer
#
#   Limitation: the retriever returns FLAT chunks with no entity
#   awareness. It cannot JOIN facts across chunks or walk
#   relationships. Whatever doesn't fit in 4 chunks is silently
#   dropped — even if it's the key connecting fact.
#
# KG-RAG PIPELINE (built in 01_ingest_pdf.py + 02_kg_rag_chain.py):
#   Neo4j knowledge graph
#   → graphiti.search() (semantic + BM25 + graph traversal + rerank)
#   → 10 temporally-tagged edge facts
#   → ChatPromptTemplate → gpt-4o → answer
# ================================================================

import asyncio
import os
from dotenv import load_dotenv

from pypdf import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from graphiti_core import Graphiti

load_dotenv()

# ── Shared LLM — identical for both pipelines ────────────────────
llm = ChatOpenAI(model="gpt-4o", temperature=0)

# ════════════════════════════════════════════════════════════════
# TRADITIONAL RAG
# ════════════════════════════════════════════════════════════════
def build_traditional_rag(pdf_path: str):
    """
    Build a FAISS retriever from the PDF.

    The entire PDF is dumped into one string, split into 500-char
    chunks, embedded, and stored in FAISS. The retriever returns
    the top-4 chunks closest to the query embedding.

    Key structural limitation:
      Chunks are FLAT text with no entity or relationship metadata.
      To answer a 3-hop question the retriever would need to return
      the 3 specific chunks that each contain one hop — but cosine
      similarity only guarantees relevance to the query string, not
      completeness of the reasoning chain.
    """
    print("⚙️  Building Traditional RAG (FAISS + HuggingFace)...")
    reader   = PdfReader(pdf_path)
    raw_text = "\n\n".join(p.extract_text() for p in reader.pages if p.extract_text())

    splitter    = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    chunks      = splitter.create_documents([raw_text])
    print(f"   Chunks: {len(chunks)}")

    embeddings  = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
    vectorstore = FAISS.from_documents(chunks, embeddings)
    retriever   = vectorstore.as_retriever(search_kwargs={"k": 4})
    print("   ✅ FAISS index ready\n")
    return retriever

TRAD_PROMPT = ChatPromptTemplate.from_template("""
Answer the question using ONLY the context below.
If you cannot find enough information say:
"Information not found in retrieved context."

Context:
{context}

Question: {question}
Answer:
""")

def run_traditional_rag(retriever, question: str) -> str:
    docs    = retriever.invoke(question)
    context = "\n\n---\n\n".join(d.page_content for d in docs)
    return (TRAD_PROMPT | llm | StrOutputParser()).invoke(
        {"context": context, "question": question}
    )

# ════════════════════════════════════════════════════════════════
# KG-RAG
# ════════════════════════════════════════════════════════════════
graphiti = Graphiti(
    uri=os.getenv("NEO4J_URI"),
    user=os.getenv("NEO4J_USER"),
    password=os.getenv("NEO4J_PASSWORD"),
)

KG_PROMPT = ChatPromptTemplate.from_template("""
You are an enterprise analyst. Answer using the knowledge graph facts below.
Trace multi-hop relationship chains explicitly. Cite [Fact NN] numbers.

KNOWLEDGE GRAPH FACTS:
{context}

Question: {question}
Answer:
""")

async def run_kg_rag(question: str) -> str:
    edges   = await graphiti.search(
        query=question, num_results=10, group_ids=["nexacore_report"]
    )
    context = "\n\n".join(
        f"[{i:02d}] {e.fact}  "
        f"(valid: {e.valid_at.strftime('%Y-%m-%d') if e.valid_at else '?'} "
        f"→ {e.invalid_at.strftime('%Y-%m-%d') if e.invalid_at else 'PRESENT'})"
        for i, e in enumerate(edges, 1)
    )
    return (KG_PROMPT | llm | StrOutputParser()).invoke(
        {"context": context, "question": question}
    )

# ════════════════════════════════════════════════════════════════
# BENCHMARK QUESTIONS
# ════════════════════════════════════════════════════════════════
QUESTIONS = [
    {
        "id": "Q1",
        "label": "3-hop reasoning  |  pages 1 & 5",
        "question": (
            "Who is responsible for the work that caused the GlobalBank outage "
            "and who originally approved that project?"
        ),
        "why_hard": (
            "Hop 1: Outage → Elena Vasquez's Kafka change (page 5)\n"
            "Hop 2: Kafka change → part of Project Helios (page 1)\n"
            "Hop 3: Project Helios → approved by Dr. Priya Mehta (page 1)\n"
            "The three chunks have low cosine similarity to each other."
        ),
    },
    {
        "id": "Q2",
        "label": "5-hop dependency chain  |  pages 1, 5 & 6",
        "question": (
            "Why does a delay in the DataFlow acquisition directly threaten "
            "the NIP v4.0 release date?"
        ),
        "why_hard": (
            "DataFlow replaces Kafka (p5) → Kafka caused outage (p5)\n"
            "→ GlobalBank is NIP v4.0 pilot (p6)\n"
            "→ Pilot needs outage post-mortem closed (p6)\n"
            "→ Helios needs DataFlow before Q1 2025 (p6)\n"
            "5 hops across 3 pages — impossible for cosine retrieval to JOIN."
        ),
    },
    {
        "id": "Q3",
        "label": "Prior history + current role  |  pages 1, 5 & 7",
        "question": (
            "What conflict of interest exists in the DataFlow acquisition "
            "and who flagged it?"
        ),
        "why_hard": (
            "Requires joining: Nathan Brooks (DataFlow CEO, p5)\n"
            "+ Elena Vasquez (Platform Core lead, p1)\n"
            "+ StreamCore Inc. shared history (p5)\n"
            "+ Vasquez evaluates DataFlow technology (p5)\n"
            "+ James Okafor flagged it → Risk R-003 (p7)\n"
            "All 5 sub-facts arrive as separate isolated chunks."
        ),
    },
    {
        "id": "Q4",
        "label": "Aggregation + risk cross-reference  |  pages 3–4 & 7",
        "question": (
            "Which account manager is responsible for the most total ARR "
            "and what active risks are threatening those accounts right now?"
        ),
        "why_hard": (
            "Must SUM: Linda Zhao → GlobalBank $8.2M + MediTech $5.1M\n"
            "+ RetailGiant $3.4M = $16.7M (pages 3–4)\n"
            "Then JOIN with Risk Register: R-001 owner Linda Zhao (page 7)\n"
            "Traditional RAG cannot aggregate across chunks or join tables."
        ),
    },
]

# ════════════════════════════════════════════════════════════════
# RUN COMPARISON
# ════════════════════════════════════════════════════════════════
async def run_comparison(pdf_path: str):
    retriever = build_traditional_rag(pdf_path)

    for q in QUESTIONS:
        print("\n" + "█" * 65)
        print(f"  {q['id']}  [{q['label']}]")
        print("█" * 65)
        print(f"\n❓ QUESTION:\n   {q['question']}")
        print(f"\n💡 WHY TRADITIONAL RAG FAILS:")
        for line in q["why_hard"].splitlines():
            print(f"   {line}")

        print("\n" + "─" * 65)
        print("  📦 TRADITIONAL RAG  (FAISS + cosine similarity)")
        print("─" * 65)
        print(run_traditional_rag(retriever, q["question"]))

        print("\n" + "─" * 65)
        print("  🧠 KNOWLEDGE GRAPH RAG  (Graphiti + Neo4j)")
        print("─" * 65)
        print(await run_kg_rag(q["question"]))

asyncio.run(run_comparison("nexacore_knowledge_report.pdf"))
```

Run it:
```bash
uv run 03_compare_rag.py
```

---

## File 4 — Interactive Q&A Terminal App

```python
# 04_interactive_qa.py
# ================================================================
# PURPOSE: A real-time interactive terminal app. Ask any question
# about the NexaCore PDF and see the retrieved Knowledge Graph
# facts (with time windows) and the LLM-generated answer.
# ================================================================
#
# HOW IT WORKS:
#   1. Welcome panel displays on startup
#   2. User types a question → Enter
#   3. Graphiti hybrid search runs (rich spinner shown)
#   4. Retrieved facts displayed in a formatted rich Table
#   5. LLM generates the answer (rich spinner shown)
#   6. Answer displayed in a coloured Panel
#   7. Loop back to step 2 until user types 'exit'
#
# SPECIAL COMMANDS:
#   help  → show table of 8 curated sample questions
#   exit  → quit
# ================================================================

import asyncio
import os
from dotenv import load_dotenv
from graphiti_core import Graphiti
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

load_dotenv()

console  = Console()
llm      = ChatOpenAI(model="gpt-4o", temperature=0)
graphiti = Graphiti(
    uri=os.getenv("NEO4J_URI"),
    user=os.getenv("NEO4J_USER"),
    password=os.getenv("NEO4J_PASSWORD"),
)

KG_PROMPT = ChatPromptTemplate.from_template("""
You are an expert enterprise analyst for NexaCore Technologies.
Answer the question using only the Knowledge Graph facts below.
Trace multi-hop chains explicitly. Cite [Fact NN] numbers.

KNOWLEDGE GRAPH FACTS:
{context}

QUESTION: {question}
ANSWER:
""")

SAMPLE_QUESTIONS = [
    "Who caused the GlobalBank outage and who approved their project?",
    "Why does a DataFlow acquisition delay threaten NIP v4.0?",
    "What conflict of interest exists in the DataFlow acquisition?",
    "Which account manager owns the most ARR and what risks threaten those accounts?",
    "What projects and risks is Elena Vasquez currently involved in?",
    "Which technology caused the GlobalBank outage and what is replacing it?",
    "How are Dr. Arjun Nair, MIT Lincoln Laboratory, and Project Aurora connected?",
    "What remediation steps were taken after the August 2024 GlobalBank outage?",
]

async def answer_question(question: str) -> None:
    # ── Retrieve facts ────────────────────────────────────────────
    with console.status("[bold cyan]🔍 Searching Knowledge Graph...[/]"):
        edges = await graphiti.search(
            query=question, num_results=10, group_ids=["nexacore_report"]
        )

    if not edges:
        console.print("[red]No relevant facts found in the knowledge graph.[/]")
        return

    # ── Display facts table ───────────────────────────────────────
    tbl = Table(
        title=f"📊 Retrieved {len(edges)} Knowledge Graph Facts",
        header_style="bold blue", border_style="blue", min_width=82,
    )
    tbl.add_column("#",     style="cyan",  width=4)
    tbl.add_column("Fact",  style="white", min_width=57)
    tbl.add_column("Valid", style="green", width=18)

    context_lines = []
    for i, edge in enumerate(edges, 1):
        vf = edge.valid_at.strftime("%Y-%m-%d")   if edge.valid_at   else "?"
        vt = edge.invalid_at.strftime("%Y-%m-%d") if edge.invalid_at else "PRESENT"
        tbl.add_row(str(i), edge.fact, f"{vf} → {vt}")
        context_lines.append(f"[{i:02d}] {edge.fact}  (valid: {vf} → {vt})")

    console.print(tbl)

    # ── Generate answer ───────────────────────────────────────────
    context = "\n\n".join(context_lines)
    with console.status("[bold cyan]🧠 Generating answer...[/]"):
        messages = KG_PROMPT.format_messages(context=context, question=question)
        response = await llm.ainvoke(messages)

    console.print(Panel(
        response.content,
        title="[bold green]💬 Knowledge Graph RAG — Answer[/]",
        border_style="green", padding=(1, 2),
    ))

async def interactive_loop():
    console.print(Panel(
        "[bold cyan]NexaCore Technologies — Knowledge Graph RAG[/]\n\n"
        "[white]Ask any question about the NexaCore report.[/]\n"
        "[dim]  help  → sample questions   |   exit  → quit[/]",
        border_style="cyan",
        title="[bold]🧠 KG-RAG Interactive Terminal[/]",
        padding=(1, 2),
    ))

    while True:
        console.print()
        try:
            question = console.input("[bold yellow]❓ Question: [/]").strip()
        except (EOFError, KeyboardInterrupt):
            console.print("\n[dim]Goodbye![/]")
            break

        if not question:
            continue
        if question.lower() in ("exit", "quit", "q"):
            console.print("[dim]Goodbye![/]")
            break
        if question.lower() == "help":
            t = Table(title="Sample Questions", border_style="yellow", min_width=72)
            t.add_column("#",        style="yellow", width=4)
            t.add_column("Question", style="white")
            for i, q in enumerate(SAMPLE_QUESTIONS, 1):
                t.add_row(str(i), q)
            console.print(t)
            continue

        await answer_question(question)

asyncio.run(interactive_loop())
```

Run it:
```bash
uv run 04_interactive_qa.py
```

---

## Sample Questions & Expected Outputs

### Q1 — 3-Hop Multi-Hop (Pages 1 & 5)

**Question:**
> *"Who is responsible for the work that caused the GlobalBank outage and who approved their project?"*

**Why Traditional RAG fails:**

```
The three required facts sit in different sections with very different vocabulary.
Cosine similarity cannot connect them:

  Chunk A  (page 5 — incident section)
    "Root cause: Kafka memory leak introduced by Elena Vasquez's
     Platform Core team during the Project Helios migration."
    Vocab: outage, Kafka, memory leak, migration

  Chunk B  (page 1 — projects section)
    "Project Helios is jointly owned by Elena Vasquez and Dr. Arjun Nair."
    Vocab: project, owned, microservices

  Chunk C  (page 1 — projects section)
    "Budget: $4.2M approved by Dr. Priya Mehta and the Board."
    Vocab: budget, approved, board

  FAISS retrieves A (closest to "outage") and maybe B,
  but C has near-zero cosine similarity to "GlobalBank outage".
  The LLM cannot answer "who approved the project".
```

**Traditional RAG Output:**
```
Elena Vasquez's Platform Core team introduced the Kafka configuration
change during the Project Helios migration that caused the GlobalBank outage.

Information about who originally approved Project Helios is not found
in the retrieved context.                                               ❌
```

**KG-RAG Output:**
```
Tracing the 3-hop chain from the Knowledge Graph:

[Fact 01] Elena Vasquez's Platform Core team introduced a Kafka config
          change during Project Helios that caused the GlobalBank outage
          (valid: 2024-08-14 → PRESENT)

[Fact 03] Elena Vasquez co-owns Project Helios with Dr. Arjun Nair
          (valid: 2024-07-01 → PRESENT)

[Fact 05] Project Helios was approved by Dr. Priya Mehta (CEO) and
          the Board of Directors with a $4.2M budget
          (valid: 2024-07-01 → PRESENT)

ANSWER:
Elena Vasquez (Team Lead, Platform Core) is responsible — her team
introduced the Kafka configuration change as part of Project Helios.
Project Helios was jointly owned by Elena Vasquez and Dr. Arjun Nair,
and was originally approved by CEO Dr. Priya Mehta and the Board.

3-hop chain:
  GlobalBank Outage
    ← Kafka change by Elena Vasquez's team     [Fact 01]
    ← Part of Project Helios (Vasquez co-owns) [Fact 03]
    ← Helios approved by Dr. Priya Mehta       [Fact 05]       ✅
```

---

### Q2 — 5-Hop Dependency Chain (Pages 1, 5 & 6)

**Question:**
> *"Why does a delay in the DataFlow acquisition directly threaten the NIP v4.0 release date?"*

**KG-RAG Output:**
```
5-step dependency chain retrieved from the graph:

[Fact 02] DataFlow's streaming middleware will replace the Kafka layer,
          integrated by Tomasz Kowalski's team (→ PRESENT)
[Fact 04] The Kafka layer caused the GlobalBank P0 outage on Aug 14 2024
          (→ PRESENT)
[Fact 07] GlobalBank conditionally committed as NIP v4.0 pilot client
          (→ PRESENT)
[Fact 08] GlobalBank's pilot commitment requires outage post-mortem
          sign-off by James Okafor and Raymond Chan (→ PRESENT)
[Fact 10] DataFlow must close by Nov 30, 2024 for Kowalski's team to
          meet the Q1 2025 Project Helios milestone (→ PRESENT)

DEPENDENCY CHAIN:
  DataFlow delay past Nov 30
    → Kowalski misses Helios Q1 2025 milestone        [Fact 10]
    → Helios incomplete → NIP v4.0 delayed            [Fact 02]
    → GlobalBank pilot launch blocked                 [Fact 07]
    → $8.2M ARR renewal at risk (R-001 churn risk)    [Fact 08]

Secondary risk: DataFlow replaces the exact Kafka config that caused
the outage — delay means the root cause stays unresolved while the
GlobalBank relationship is already under strain.                     ✅
```

---

### Q3 — Conflict of Interest (Prior History + Current Role)

**Question:**
> *"What conflict of interest exists in the DataFlow acquisition and who flagged it?"*

**KG-RAG Output:**
```
[Fact 01] Nathan Brooks is DataFlow Systems' CEO; joins NexaCore as VP
          Data Infrastructure post-acquisition (→ PRESENT)
[Fact 03] Nathan Brooks and Elena Vasquez both previously worked at
          StreamCore Inc., acquired by Oracle in 2019 (→ PRESENT)
[Fact 05] Elena Vasquez's Platform Core team is responsible for
          technically evaluating DataFlow's technology (→ PRESENT)
[Fact 06] CTO James Okafor flagged their prior relationship as a
          potential conflict of interest (→ PRESENT)
[Fact 07] Risk R-003: "Vasquez/Brooks conflict of interest" —
          owner: James Okafor, mitigation: appoint neutral evaluator
          (→ PRESENT)

ANSWER:
Elena Vasquez must objectively evaluate DataFlow's technology for NIP
adoption — but she has a pre-existing working relationship with DataFlow's
CEO Nathan Brooks from their shared time at StreamCore Inc. (Oracle, 2019).
This creates a perceived bias risk in the technical evaluation.

CTO James Okafor identified and formally flagged the conflict. It is
tracked as Risk R-003 with the approved mitigation of appointing a
neutral technical evaluator independent of Vasquez's team.          ✅
```

---

### Q4 — Financial Aggregation + Risk Cross-Reference

**Question:**
> *"Which account manager is responsible for the most total ARR and what active risks are threatening those accounts right now?"*

**KG-RAG Output:**
```
Graph traversal: Account Managers → Clients → ARR values → Risk Register

[Fact 01] Linda Zhao manages GlobalBank Financial Group: $8.2M (→ PRESENT)
[Fact 02] Linda Zhao manages MediTech Solutions: $5.1M (→ PRESENT)
[Fact 03] Linda Zhao manages RetailGiant Corp: $3.4M (→ PRESENT)
[Fact 05] Risk R-001: GlobalBank churn risk post-outage — owner: Linda Zhao
          — status: Mitigating (→ PRESENT)
[Fact 07] Linda Zhao negotiated a $1.23M SLA credit with GlobalBank
          after the August 2024 outage (→ PRESENT)
[Fact 08] RetailGiant renewal expected at $4.1M in Q4 2024 (→ PRESENT)

ANSWER:
Linda Zhao manages the highest ARR portfolio:
  GlobalBank    $8.2M
  MediTech      $5.1M
  RetailGiant   $3.4M
  ─────────────────────
  Total        $16.7M  →  43.5% of NexaCore's $38.4M total ARR

Active risks:

  R-001 — GlobalBank churn (HIGH)
    Post-outage relationship strain. Linda negotiated a $1.23M SLA
    credit and made an in-person Singapore visit. The Q4 renewal is
    contingent on the Helios post-mortem being closed by James Okafor
    and Raymond Chan (GlobalBank CTO). The credit also reduces Q4
    net-new ARR by ~8%.

  RetailGiant renewal risk (MEDIUM)
    Renewal not yet signed — expected $4.1M. DataFlow delay (R-002)
    could affect their NIP-Connect integration (built by Tomasz
    Kowalski) if Kafka replacement is not completed in time.         ✅
```

---

## File 5 — Streamlit Web UI

```python
# 05_streamlit_app.py
# ================================================================
# PURPOSE: A fully interactive browser-based UI for the KG-RAG
# system. Provides three tabs:
#
#   Tab 1 — KG-RAG Q&A
#     User types a question → Graphiti hybrid search → retrieved
#     facts displayed as a formatted table → gpt-4o answer rendered
#     with Markdown. Full fact metadata (time windows) shown inline.
#
#   Tab 2 — Side-by-Side Comparison
#     Runs the same question through both Traditional RAG (FAISS)
#     and KG-RAG simultaneously and renders both answers in two
#     columns so differences are immediately visible.
#
#   Tab 3 — Graph Explorer
#     Displays the full entity/relationship summary from Neo4j
#     and lists all stored facts with their temporal metadata.
#     Useful for inspecting what the ingestion pipeline extracted.
#
# HOW TO RUN:
#   uv run streamlit run 05_streamlit_app.py
#   → opens http://localhost:8501 in your browser
#
# REQUIREMENTS:
#   • Neo4j must be running (docker container neo4j-nexacore)
#   • 01_ingest_pdf.py must have been run first to populate the graph
#   • .env must contain OPENAI_API_KEY, NEO4J_URI, NEO4J_USER,
#     NEO4J_PASSWORD
# ================================================================

import asyncio
import os
import streamlit as st
from dotenv import load_dotenv

from graphiti_core import Graphiti
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from pypdf import PdfReader

load_dotenv()

# ── Page config ──────────────────────────────────────────────────
st.set_page_config(
    page_title="NexaCore KG-RAG",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Async helper: Streamlit runs synchronously, so we need a
#    dedicated event loop for Graphiti's async API calls ──────────
def run_async(coro):
    """Run an async coroutine from synchronous Streamlit code."""
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()

# ── Cached resources — initialised once per Streamlit session ────
@st.cache_resource
def get_graphiti():
    """Graphiti client — one connection pool for the whole session."""
    return Graphiti(
        uri=os.getenv("NEO4J_URI", "bolt://localhost:7687"),
        user=os.getenv("NEO4J_USER", "neo4j"),
        password=os.getenv("NEO4J_PASSWORD", "nexacore123"),
    )

@st.cache_resource
def get_llm():
    """Shared gpt-4o LLM — cached so the model is not re-instantiated
    on every Streamlit rerun triggered by user interaction."""
    return ChatOpenAI(model="gpt-4o", temperature=0)

@st.cache_resource
def get_trad_retriever():
    """
    Build and cache the Traditional RAG retriever once per session.

    Reads nexacore_knowledge_report.pdf → splits into 500-char chunks
    → embeds with all-MiniLM-L6-v2 → stores in FAISS in memory.
    Cached so the ~10s embedding step only runs once.
    """
    pdf_path = "nexacore_knowledge_report.pdf"
    if not os.path.exists(pdf_path):
        return None  # Caller checks for None and shows a warning

    reader   = PdfReader(pdf_path)
    raw_text = "\n\n".join(p.extract_text() for p in reader.pages if p.extract_text())
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    chunks   = splitter.create_documents([raw_text])

    embeddings  = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
    vectorstore = FAISS.from_documents(chunks, embeddings)
    return vectorstore.as_retriever(search_kwargs={"k": 4})

# ── KG-RAG prompt (same as 02_kg_rag_chain.py) ───────────────────
KG_RAG_PROMPT = ChatPromptTemplate.from_template("""
You are an expert enterprise analyst for NexaCore Technologies.
Answer the question using only the Knowledge Graph facts below.
Trace multi-hop chains explicitly. Cite [Fact NN] numbers.

KNOWLEDGE GRAPH FACTS:
{context}

QUESTION: {question}
ANSWER:
""")

# ── Traditional RAG prompt ────────────────────────────────────────
TRAD_PROMPT = ChatPromptTemplate.from_template("""
Answer the question using ONLY the context below.
If you cannot find enough information say:
"Information not found in retrieved context."

Context:
{context}

Question: {question}
Answer:
""")

# ── Core async functions ──────────────────────────────────────────
async def kg_rag_query(question: str, graphiti, llm):
    """Run a KG-RAG query and return facts + answer."""
    edges = await graphiti.search(
        query=question,
        num_results=10,
        group_ids=["nexacore_report"],
    )
    if not edges:
        return [], "No relevant facts found in the knowledge graph."

    context_lines = []
    for i, edge in enumerate(edges, 1):
        vf = edge.valid_at.strftime("%Y-%m-%d")   if edge.valid_at   else "unknown"
        vt = edge.invalid_at.strftime("%Y-%m-%d") if edge.invalid_at else "PRESENT"
        context_lines.append(
            f"[Fact {i:02d}] {edge.fact}  (valid: {vf} → {vt})"
        )

    context  = "\n\n".join(context_lines)
    messages = KG_RAG_PROMPT.format_messages(context=context, question=question)
    response = await llm.ainvoke(messages)
    return edges, response.content

async def trad_rag_query(question: str, retriever, llm):
    """Run a Traditional RAG query and return answer string."""
    docs    = retriever.invoke(question)
    context = "\n\n---\n\n".join(d.page_content for d in docs)
    chain   = TRAD_PROMPT | llm | StrOutputParser()
    return chain.invoke({"context": context, "question": question})

async def fetch_graph_summary(graphiti):
    """
    Fetch all edges from Neo4j scoped to nexacore_report and return
    a flat list of fact strings with their time windows.

    Uses graphiti.search() with a broad entity query to pull a
    representative sample of the graph — not every edge, but enough
    to give the Graph Explorer tab useful content.
    """
    edges = await graphiti.search(
        query="NexaCore Technologies organization projects clients risks",
        num_results=50,
        group_ids=["nexacore_report"],
    )
    return edges

# ── Sidebar ───────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/4/4e/Graph-based_knowledge_representation.svg/240px-Graph-based_knowledge_representation.svg.png", width=120)
    st.title("🧠 NexaCore KG-RAG")
    st.caption("Knowledge Graph RAG — Powered by Graphiti + Neo4j + gpt-4o")
    st.divider()

    st.subheader("💡 Sample Questions")
    SAMPLE_QUESTIONS = [
        "Who caused the GlobalBank outage and who approved their project?",
        "Why does a DataFlow acquisition delay threaten NIP v4.0?",
        "What conflict of interest exists in the DataFlow acquisition?",
        "Which account manager owns the most ARR and what risks threaten those accounts?",
        "What projects and risks is Elena Vasquez currently involved in?",
        "How are Dr. Arjun Nair, MIT Lincoln Laboratory, and Project Aurora connected?",
        "What remediation steps were taken after the August 2024 GlobalBank outage?",
        "Which technology caused the GlobalBank outage and what is replacing it?",
    ]
    for q in SAMPLE_QUESTIONS:
        if st.button(q, key=f"sample_{q[:30]}", use_container_width=True):
            st.session_state["prefill_question"] = q

    st.divider()
    st.subheader("⚙️ Configuration")
    st.code(
        f"NEO4J_URI: {os.getenv('NEO4J_URI', 'bolt://localhost:7687')}\n"
        f"Model: gpt-4o\n"
        f"Retrieval: Graphiti hybrid (top 10)\n"
        f"Group ID: nexacore_report",
        language="text",
    )

# ── Tabs ──────────────────────────────────────────────────────────
tab_kgrag, tab_compare, tab_explorer = st.tabs([
    "🧠 KG-RAG Q&A",
    "⚖️ Side-by-Side Comparison",
    "🔍 Graph Explorer",
])

# ═══════════════════════════════════════════════════════════════════
# TAB 1 — KG-RAG Q&A
# ═══════════════════════════════════════════════════════════════════
with tab_kgrag:
    st.header("Knowledge Graph RAG — Q&A")
    st.caption(
        "Ask any question about the NexaCore internal report. "
        "Graphiti runs semantic + BM25 + graph traversal retrieval "
        "and gpt-4o generates a grounded, multi-hop-aware answer."
    )

    # Pre-fill from sidebar sample question buttons
    default_q = st.session_state.pop("prefill_question", "")
    question  = st.text_input(
        "Your question",
        value=default_q,
        placeholder="e.g. Who caused the GlobalBank outage?",
        key="kgrag_question",
    )

    if st.button("🔍 Ask the Knowledge Graph", type="primary", key="kgrag_run"):
        if not question.strip():
            st.warning("Please enter a question.")
        else:
            graphiti = get_graphiti()
            llm      = get_llm()

            with st.spinner("🔍 Searching Knowledge Graph…"):
                edges, answer = run_async(kg_rag_query(question, graphiti, llm))

            # ── Display retrieved facts ───────────────────────────
            if edges:
                st.subheader(f"📊 Retrieved {len(edges)} Knowledge Graph Facts")
                rows = []
                for i, edge in enumerate(edges, 1):
                    vf = edge.valid_at.strftime("%Y-%m-%d")   if edge.valid_at   else "?"
                    vt = edge.invalid_at.strftime("%Y-%m-%d") if edge.invalid_at else "PRESENT"
                    rows.append({
                        "#":           i,
                        "Fact":        edge.fact,
                        "Valid From":  vf,
                        "Valid To":    vt,
                    })
                st.dataframe(
                    rows,
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        "#":          st.column_config.NumberColumn(width="small"),
                        "Fact":       st.column_config.TextColumn(width="large"),
                        "Valid From": st.column_config.TextColumn(width="medium"),
                        "Valid To":   st.column_config.TextColumn(width="medium"),
                    },
                )

            # ── Display answer ────────────────────────────────────
            st.subheader("💬 Answer")
            st.markdown(answer)

# ═══════════════════════════════════════════════════════════════════
# TAB 2 — SIDE-BY-SIDE COMPARISON
# ═══════════════════════════════════════════════════════════════════
with tab_compare:
    st.header("KG-RAG vs Traditional RAG — Side-by-Side")
    st.caption(
        "Both pipelines use the same question and the same gpt-4o LLM. "
        "The only difference is retrieval: flat cosine similarity (Traditional) "
        "vs graph-traversal-aware hybrid retrieval (KG-RAG)."
    )

    compare_q = st.text_input(
        "Question for comparison",
        placeholder="e.g. Why does a DataFlow acquisition delay threaten NIP v4.0?",
        key="compare_question",
    )

    # Quick-select benchmark questions
    st.caption("Or pick a benchmark question:")
    benchmark_cols = st.columns(2)
    BENCHMARK_QS = [
        "Who is responsible for the work that caused the GlobalBank outage and who originally approved that project?",
        "Why does a delay in the DataFlow acquisition directly threaten the NIP v4.0 release date?",
        "What conflict of interest exists in the DataFlow acquisition and who flagged it?",
        "Which account manager is responsible for the most total ARR and what active risks are threatening those accounts right now?",
    ]
    for idx, bq in enumerate(BENCHMARK_QS):
        col = benchmark_cols[idx % 2]
        with col:
            if st.button(f"Q{idx+1}: {bq[:65]}…", key=f"bq_{idx}", use_container_width=True):
                st.session_state["compare_prefill"] = bq

    if "compare_prefill" in st.session_state:
        compare_q = st.session_state.pop("compare_prefill")

    if st.button("⚖️ Run Comparison", type="primary", key="compare_run"):
        if not compare_q.strip():
            st.warning("Please enter a question.")
        else:
            graphiti  = get_graphiti()
            llm       = get_llm()
            retriever = get_trad_retriever()

            if retriever is None:
                st.error(
                    "nexacore_knowledge_report.pdf not found in the project root. "
                    "Copy the PDF here and restart Streamlit."
                )
            else:
                col_trad, col_kg = st.columns(2)

                with col_trad:
                    st.subheader("📦 Traditional RAG")
                    st.caption("FAISS cosine similarity · top-4 chunks · no graph traversal")
                    with st.spinner("Retrieving from FAISS…"):
                        trad_answer = run_async(trad_rag_query(compare_q, retriever, llm))
                    st.markdown(trad_answer)

                with col_kg:
                    st.subheader("🧠 Knowledge Graph RAG")
                    st.caption("Graphiti hybrid (semantic + BM25 + traversal) · top-10 facts")
                    with st.spinner("Searching Knowledge Graph…"):
                        edges, kg_answer = run_async(kg_rag_query(compare_q, graphiti, llm))
                    st.markdown(kg_answer)
                    if edges:
                        with st.expander(f"View {len(edges)} retrieved facts"):
                            for i, edge in enumerate(edges, 1):
                                vt = edge.invalid_at.strftime("%Y-%m-%d") if edge.invalid_at else "PRESENT"
                                vf = edge.valid_at.strftime("%Y-%m-%d") if edge.valid_at else "?"
                                st.markdown(f"**[{i:02d}]** {edge.fact}  \n`{vf} → {vt}`")

# ═══════════════════════════════════════════════════════════════════
# TAB 3 — GRAPH EXPLORER
# ═══════════════════════════════════════════════════════════════════
with tab_explorer:
    st.header("🔍 Knowledge Graph Explorer")
    st.caption(
        "Browse the facts extracted from the NexaCore PDF and stored "
        "in Neo4j. Click 'Load Graph Summary' to fetch a representative "
        "sample of edges with their temporal metadata."
    )

    st.info(
        "**Tip:** Open the full interactive Neo4j Browser at "
        "[http://localhost:7474](http://localhost:7474) and run:\n\n"
        "```cypher\nMATCH (n)-[r]->(m) RETURN n, r, m LIMIT 100\n```",
        icon="💡",
    )

    if st.button("📡 Load Graph Summary", key="load_graph"):
        graphiti = get_graphiti()
        with st.spinner("Fetching edges from Neo4j…"):
            edges = run_async(fetch_graph_summary(graphiti))

        st.success(f"✅ Loaded {len(edges)} representative edges from the knowledge graph.")
        st.subheader("Stored Knowledge Graph Facts")

        rows = []
        for i, edge in enumerate(edges, 1):
            vf = edge.valid_at.strftime("%Y-%m-%d")   if edge.valid_at   else "?"
            vt = edge.invalid_at.strftime("%Y-%m-%d") if edge.invalid_at else "PRESENT"
            rows.append({
                "#":          i,
                "Fact":       edge.fact,
                "Valid From": vf,
                "Valid To":   vt,
                "Current":    vt == "PRESENT",
            })

        st.dataframe(
            rows,
            use_container_width=True,
            hide_index=True,
            column_config={
                "#":          st.column_config.NumberColumn(width="small"),
                "Fact":       st.column_config.TextColumn(width="large"),
                "Valid From": st.column_config.TextColumn(width="medium"),
                "Valid To":   st.column_config.TextColumn(width="medium"),
                "Current":    st.column_config.CheckboxColumn(
                    "Current?", help="True if this fact is still valid (no expiry date)"
                ),
            },
        )

        # ── Summary metrics ───────────────────────────────────────
        st.divider()
        current_count  = sum(1 for r in rows if r["Current"])
        expired_count  = len(rows) - current_count

        m1, m2, m3 = st.columns(3)
        m1.metric("Total Facts Shown",  len(rows))
        m2.metric("Current (→ PRESENT)", current_count)
        m3.metric("Historical (expired)", expired_count)
```

Run it:
```bash
uv run streamlit run 05_streamlit_app.py
```

Then open **http://localhost:8501** in your browser.

### What Each Tab Does

| Tab | Purpose | What you'll see |
|---|---|---|
| **🧠 KG-RAG Q&A** | Ask free-form questions against the Knowledge Graph | Fact table with time windows + gpt-4o answer |
| **⚖️ Side-by-Side Comparison** | Same question through both pipelines simultaneously | Two-column view showing where Traditional RAG fails vs KG-RAG succeeds |
| **🔍 Graph Explorer** | Browse all extracted facts stored in Neo4j | Full fact list with temporal metadata + link to Neo4j Browser |

### Sidebar Features

- **Sample question buttons** — click any pre-written question to auto-fill the input box in Tab 1
- **Configuration panel** — shows current Neo4j URI, model, and group ID at a glance

---

```
nexacore_knowledge_report.pdf   ← existing PDF — copy into project root
              │
              ▼
┌─────────────────────────────────────────────────────────────────┐
│  01_ingest_pdf.py                                               │
│                                                                 │
│  pypdf.PdfReader → page text (page-by-page)                    │
│    → chunk_text() — 800 chars, 150 overlap, sentence boundary  │
│    → graphiti.add_episode() per chunk                          │
│          ├─ LLM: extract Entity nodes                          │
│          ├─ LLM: extract typed Relationship edges              │
│          ├─ Conflict resolution: invalidate stale edges        │
│          └─ Embed + store in Neo4j                             │
│                                                                 │
│  Output: Entity nodes + typed edges + temporal validity +      │
│          vector index + BM25 index — all in Neo4j              │
└─────────────────────────────────────────────────────────────────┘
              │
              ▼
┌─────────────────────────────────────────────────────────────────┐
│  02_kg_rag_chain.py  /  04_interactive_qa.py                    │
│  05_streamlit_app.py (Tab 1 & Tab 2)                            │
│                                                                 │
│  User Question                                                  │
│    → graphiti.search(num_results=10, group_ids=[...])           │
│          ├─ Semantic   (embedding cosine similarity)            │
│          ├─ BM25       (keyword scoring on edge facts)          │
│          ├─ Traversal  (walk graph N hops from entities)        │
│          └─ Reranker   (cross-encoder final sort)               │
│    → format facts with temporal windows                        │
│    → KG_RAG_PROMPT + gpt-4o + StrOutputParser                  │
│    → grounded, multi-hop, temporally-accurate answer            │
└─────────────────────────────────────────────────────────────────┘
              │
              ▼  (05_streamlit_app.py Tab 3)
┌─────────────────────────────────────────────────────────────────┐
│  Graph Explorer                                                 │
│                                                                 │
│  graphiti.search(broad query, num_results=50)                  │
│    → all stored edge facts with temporal metadata              │
│    → Streamlit dataframe with Current? checkbox column         │
│    → metrics: total / current / expired fact counts            │
│    → link to Neo4j Browser (http://localhost:7474)             │
└─────────────────────────────────────────────────────────────────┘
```

---

## Quick Start Recap

```bash
# 1. Create project and install dependencies
mkdir nexacore-kgrag && cd nexacore-kgrag
uv init && uv python pin 3.11
uv add graphiti-core langchain langchain-openai langchain-community \
       langchain-huggingface pypdf faiss-cpu sentence-transformers \
       python-dotenv neo4j rich streamlit

# 2. Start Neo4j
docker run --name neo4j-nexacore -p 7474:7474 -p 7687:7687 -d \
           -e NEO4J_AUTH=neo4j/nexacore123 neo4j:5.26

# 3. Configure .env
echo 'OPENAI_API_KEY=sk-...'            >  .env
echo 'NEO4J_URI=bolt://localhost:7687'  >> .env
echo 'NEO4J_USER=neo4j'                >> .env
echo 'NEO4J_PASSWORD=nexacore123'       >> .env

# 4. Copy the existing PDF into the project (no generation needed)
cp /path/to/nexacore_knowledge_report.pdf .

# 5. Build the Knowledge Graph  (~5 min — LLM calls per chunk)
uv run 01_ingest_pdf.py

# 6. Run KG-RAG queries
uv run 02_kg_rag_chain.py

# 7. Compare KG-RAG vs Traditional RAG
uv run 03_compare_rag.py

# 8. Interactive terminal Q&A
uv run 04_interactive_qa.py

# 9. Launch the Streamlit web UI  → opens http://localhost:8501
uv run streamlit run 05_streamlit_app.py

# Inspect the graph visually
# Open  → http://localhost:7474
# Query → MATCH (n)-[r]->(m) RETURN n, r, m LIMIT 100
```

---

*Next in your LangChain journey → **LangChain Agents & Tools** — building autonomous AI agents that can use Graph RAG as one of many tools in a multi-step reasoning workflow 🦜🔗⚡*
