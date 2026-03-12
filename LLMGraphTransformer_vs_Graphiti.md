# LLMGraphTransformer vs Graphiti

---

## Table of Contents

1. [Overview](#overview)
2. [What is LLMGraphTransformer?](#what-is-llmgraphtransformer)
3. [What is Graphiti?](#what-is-graphiti)
4. [Architecture Comparison](#architecture-comparison)
5. [LLMGraphTransformer: Deep Dive](#llmgraphtransformer-deep-dive)
6. [Graphiti: Deep Dive](#graphiti-deep-dive)
7. [Side-by-Side Feature Comparison](#side-by-side-feature-comparison)
8. [Use Cases](#use-cases)
9. [When to Choose Which](#when-to-choose-which)
10. [Summary](#summary)

---

## Overview

Both **LLMGraphTransformer** and **Graphiti** are frameworks that use Large Language Models (LLMs) to extract structured knowledge as **graphs** from unstructured text. However, they differ significantly in their design philosophy, use cases, and capabilities.

| Dimension | LLMGraphTransformer | Graphiti |
|---|---|---|
| Origin | LangChain ecosystem | Zep AI (open source) |
| Primary Goal | Batch text → Knowledge Graph | Real-time temporal memory graph |
| Storage | Neo4j, Memgraph, etc. | Neo4j |
| Time-awareness | No | Yes (bi-temporal) |
| Conflict resolution | No | Yes (automatic) |
| Best For | Static document indexing | AI agent long-term memory |

---

## What is LLMGraphTransformer?

**LLMGraphTransformer** is a LangChain component that converts unstructured text documents into a structured **property graph** using an LLM. It extracts entities (nodes) and relationships (edges) from documents and stores them in a graph database.

### Core Idea

```
Raw Text → LLM (extraction) → Nodes + Relationships → Graph DB
```

It is primarily a **one-shot, batch extraction** tool. You feed it a document (or many), and it produces a static knowledge graph.

### Key Components

- **`LLMGraphTransformer`** — The main class; wraps an LLM to do extraction
- **`GraphDocument`** — Output object containing nodes and relationships
- **`Neo4jGraph`** (or other backends) — Stores the resulting graph
- **`Node` / `Relationship`** — The atomic units of the extracted graph

---

## What is Graphiti?

**Graphiti** is a framework by Zep AI designed specifically for building **temporal, episodic knowledge graphs** for AI agents. Rather than batch-processing documents, it continuously ingests "episodes" (messages, events, facts) and maintains a live, evolving memory graph.

### Core Idea

```
Episode (event/message) → LLM (extraction + deduplication) → Temporal Graph → Agent Memory
```

It is designed for **streaming, real-time, agent-facing** workloads where facts change over time and conflicts must be resolved automatically.

### Key Components

- **`Graphiti`** — The main client class
- **`EpisodicEdge`** — Represents a source episode (raw input)
- **`EntityNode`** — Represents a real-world entity
- **`EntityEdge`** — Represents a relationship between entities, with validity timestamps
- **`CommunityNode`** — Clusters of related entities
- **`search()`** — Hybrid semantic + BM25 + graph traversal search

---

## Architecture Comparison

### LLMGraphTransformer Architecture

```
┌─────────────────────────────────────────────────────┐
│                   Input Layer                       │
│   Document 1   Document 2   Document 3  ...         │
└──────────────────────┬──────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────┐
│              LLMGraphTransformer                    │
│  ┌────────────────────────────────────────────┐     │
│  │  LLM (GPT-4 / Claude / Llama)              │     │
│  │  Prompt: "Extract nodes & relationships"   │     │
│  │  Output: JSON → GraphDocument              │     │
│  └────────────────────────────────────────────┘     │
└──────────────────────┬──────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────┐
│              Graph Database (Neo4j)                 │
│                                                     │
│   (Person)-[:WORKS_AT]->(Company)                   │
│   (Company)-[:LOCATED_IN]->(City)                   │
│   (Person)-[:KNOWS]->(Person)                       │
└─────────────────────────────────────────────────────┘
```

### Graphiti Architecture

```
┌─────────────────────────────────────────────────────┐
│                  Episode Stream                     │
│   msg_1 (t=0)  msg_2 (t=1)  msg_3 (t=2)  ...      │
└──────────────────────┬──────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────┐
│                  Graphiti Core                      │
│  ┌──────────────┐  ┌──────────────┐                │
│  │   Extractor  │  │  Resolver    │                │
│  │  (LLM-based) │  │ (dedup/merge)│                │
│  └──────┬───────┘  └──────┬───────┘                │
│         │                 │                         │
│         ▼                 ▼                         │
│  ┌──────────────────────────────┐                   │
│  │   Contradiction Detector     │                   │
│  │   (invalidates old facts)    │                   │
│  └──────────────┬───────────────┘                   │
└─────────────────┼───────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────────┐
│              Neo4j (Temporal Graph)                 │
│                                                     │
│   EntityNode ──[valid_from: t0, valid_to: t2]──▶   │
│   EntityNode ──[valid_from: t2, valid_to: NULL]──▶  │
│   EpisodicEdge (source of truth)                    │
└─────────────────────────────────────────────────────┘
```

---

## LLMGraphTransformer: Deep Dive

### Installation

```bash
pip install langchain langchain-openai langchain-community neo4j
```

### Step-by-Step Example: Building a Knowledge Graph from Wikipedia Text

#### Step 1 — Set Up the LLM and Graph Connection

```python
from langchain_openai import ChatOpenAI
from langchain_community.graphs import Neo4jGraph

# Connect to Neo4j
graph = Neo4jGraph(
    url="bolt://localhost:7687",
    username="neo4j",
    password="your_password"
)

# Initialize the LLM
llm = ChatOpenAI(
    model="gpt-4o",
    temperature=0,
    openai_api_key="YOUR_OPENAI_KEY"
)
```

#### Step 2 — Initialize LLMGraphTransformer

```python
from langchain_experimental.graph_transformers import LLMGraphTransformer

# Basic initialization — lets LLM decide all entity/relationship types
transformer = LLMGraphTransformer(llm=llm)

# OR: Constrained initialization (recommended for consistency)
transformer = LLMGraphTransformer(
    llm=llm,
    allowed_nodes=["Person", "Organization", "Location", "Product"],
    allowed_relationships=["WORKS_AT", "LOCATED_IN", "FOUNDED_BY", "KNOWS"],
    node_properties=["description", "birth_year"],
    relationship_properties=["since", "role"]
)
```

**Why constrain?**
Without constraints, the LLM may use inconsistent labels (`Company` vs `Organization` vs `Corp`). Constraining ensures a clean, queryable schema.

#### Step 3 — Prepare and Transform Documents

```python
from langchain_core.documents import Document

# Your source text
text = """
Elon Musk founded SpaceX in 2002. SpaceX is headquartered in Hawthorne, California.
Musk also co-founded Tesla in 2003 alongside Martin Eberhard and Marc Tarpenning.
Tesla is an American electric vehicle company located in Austin, Texas.
Elon Musk serves as CEO of both Tesla and SpaceX.
"""

# Wrap text in a Document object
documents = [Document(page_content=text)]

# Transform: LLM reads text and extracts graph structure
graph_documents = transformer.convert_to_graph_documents(documents)
```

**What happens internally:**
1. The transformer sends the text + extraction prompt to the LLM
2. The LLM returns a structured JSON with nodes and relationships
3. The result is parsed into `GraphDocument` objects

#### Step 4 — Inspect the Extracted Graph

```python
# See extracted nodes
for node in graph_documents[0].nodes:
    print(f"NODE: {node.id} [{node.type}] — properties: {node.properties}")

# OUTPUT:
# NODE: Elon Musk [Person] — properties: {}
# NODE: SpaceX [Organization] — properties: {}
# NODE: Tesla [Organization] — properties: {}
# NODE: Hawthorne [Location] — properties: {}
# NODE: Austin [Location] — properties: {}
# NODE: Martin Eberhard [Person] — properties: {}

# See extracted relationships
for rel in graph_documents[0].relationships:
    print(f"REL: ({rel.source.id}) -[{rel.type}]-> ({rel.target.id})")

# OUTPUT:
# REL: (Elon Musk) -[FOUNDED]-> (SpaceX)
# REL: (SpaceX) -[LOCATED_IN]-> (Hawthorne)
# REL: (Elon Musk) -[CO_FOUNDED]-> (Tesla)
# REL: (Tesla) -[LOCATED_IN]-> (Austin)
# REL: (Elon Musk) -[WORKS_AT]-> (Tesla)
# REL: (Elon Musk) -[WORKS_AT]-> (SpaceX)
```

#### Step 5 — Store in Neo4j

```python
# Add to graph database
graph.add_graph_documents(
    graph_documents,
    baseEntityLabel=True,   # Adds __Entity__ label to all nodes for easy querying
    include_source=True      # Links graph nodes back to source Document
)

print("Graph stored successfully!")
```

#### Step 6 — Query the Graph

```python
# Run a Cypher query
result = graph.query("""
    MATCH (p:Person)-[:FOUNDED]->(o:Organization)-[:LOCATED_IN]->(l:Location)
    RETURN p.id AS founder, o.id AS company, l.id AS city
""")

for row in result:
    print(f"{row['founder']} founded {row['company']}, based in {row['city']}")

# OUTPUT:
# Elon Musk founded SpaceX, based in Hawthorne
```

#### Step 7 — Use with RAG (GraphRAG)

```python
from langchain.chains import GraphCypherQAChain

chain = GraphCypherQAChain.from_llm(
    llm=llm,
    graph=graph,
    verbose=True
)

response = chain.run("Who founded Tesla and where is it located?")
print(response)
# OUTPUT: Tesla was co-founded by Elon Musk, Martin Eberhard, and Marc Tarpenning.
#         It is located in Austin, Texas.
```

---

### LLMGraphTransformer: Advanced Features

#### Batch Processing Multiple Documents

```python
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Split a large document into chunks
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
docs = splitter.create_documents([long_text])

# Transform all chunks in batch
all_graph_docs = transformer.convert_to_graph_documents(docs)

# Add all at once
graph.add_graph_documents(all_graph_docs, include_source=True)
```

#### Adding Custom Properties

```python
transformer = LLMGraphTransformer(
    llm=llm,
    allowed_nodes=["Person", "Company"],
    node_properties=["founded_year", "industry", "headquarters"],
    relationship_properties=["start_date", "end_date", "role"]
)
```

---

## Graphiti: Deep Dive

### Installation

```bash
pip install graphiti-core
```

You also need a running **Neo4j** instance (v5.x+ with APOC plugin).

### Step-by-Step Example: AI Agent with Evolving Memory

#### Step 1 — Set Up Graphiti Client

```python
import asyncio
from graphiti_core import Graphiti
from graphiti_core.llm_client import OpenAIClient
from graphiti_core.embedder import OpenAIEmbedder

# Initialize Graphiti
graphiti = Graphiti(
    uri="bolt://localhost:7687",
    user="neo4j",
    password="your_password",
    llm_client=OpenAIClient(api_key="YOUR_KEY", model="gpt-4o"),
    embedder=OpenAIEmbedder(api_key="YOUR_KEY")
)

# IMPORTANT: Build indices on first run
async def setup():
    await graphiti.build_indices_and_constraints()

asyncio.run(setup())
```

#### Step 2 — Add Episodes (Ingest Data)

An **episode** is any atomic piece of information — a message, document chunk, or event.

```python
from graphiti_core.nodes import EpisodeType
from datetime import datetime, timezone

async def add_episodes():
    # Episode 1: Initial fact
    await graphiti.add_episode(
        name="user_profile_1",
        episode_body="Alice is a software engineer at Google. She lives in San Francisco.",
        source=EpisodeType.text,
        source_description="user profile update",
        reference_time=datetime(2024, 1, 1, tzinfo=timezone.utc)
    )

    # Episode 2: New fact (no conflict)
    await graphiti.add_episode(
        name="user_profile_2",
        episode_body="Alice enjoys hiking and photography in her free time.",
        source=EpisodeType.text,
        source_description="user interests",
        reference_time=datetime(2024, 3, 1, tzinfo=timezone.utc)
    )

    # Episode 3: CONFLICTING fact (Alice changed jobs)
    await graphiti.add_episode(
        name="user_profile_3",
        episode_body="Alice recently joined OpenAI as a research engineer.",
        source=EpisodeType.text,
        source_description="job update",
        reference_time=datetime(2024, 6, 1, tzinfo=timezone.utc)
    )

asyncio.run(add_episodes())
```

**What happens internally when Episode 3 is added:**
1. LLM extracts: `(Alice)-[WORKS_AT]->(OpenAI)`
2. Graphiti detects this conflicts with `(Alice)-[WORKS_AT]->(Google)`
3. The old edge is **invalidated** with `valid_to = 2024-06-01`
4. The new edge is created with `valid_from = 2024-06-01, valid_to = None`
5. Both edges are preserved — full temporal history is maintained

#### Step 3 — Query the Graph (Current State)

```python
from graphiti_core.search.search_config import SearchConfig

async def search_memory():
    results = await graphiti.search(
        query="Where does Alice work?",
        config=SearchConfig(num_results=5)
    )

    for edge in results:
        print(f"Fact: {edge.fact}")
        print(f"Valid from: {edge.valid_at}")
        print(f"Valid to: {edge.invalid_at}")  # None = still current
        print("---")

asyncio.run(search_memory())

# OUTPUT:
# Fact: Alice works at OpenAI as a research engineer
# Valid from: 2024-06-01
# Valid to: None
# ---
# Fact: Alice previously worked at Google as a software engineer
# Valid from: 2024-01-01
# Valid to: 2024-06-01
```

#### Step 4 — Temporal Point-in-Time Query

```python
async def historical_query():
    # What was true on Feb 1, 2024? (before the job change)
    results = await graphiti.search(
        query="Where does Alice work?",
        config=SearchConfig(
            num_results=5,
            reference_time=datetime(2024, 2, 1, tzinfo=timezone.utc)  # Point in time
        )
    )

    for edge in results:
        print(f"Fact (as of Feb 2024): {edge.fact}")

# OUTPUT:
# Fact (as of Feb 2024): Alice works at Google as a software engineer
```

#### Step 5 — Use in an AI Agent (LangChain Integration)

```python
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain.tools import Tool

llm = ChatOpenAI(model="gpt-4o")

# Wrap Graphiti search as an agent tool
async def memory_search(query: str) -> str:
    results = await graphiti.search(query, config=SearchConfig(num_results=3))
    if not results:
        return "No relevant memories found."
    return "\n".join([r.fact for r in results])

memory_tool = Tool(
    name="long_term_memory",
    func=lambda q: asyncio.run(memory_search(q)),
    description="Search the agent's long-term memory for facts about users or events."
)

# Build agent with memory
agent = create_openai_functions_agent(llm=llm, tools=[memory_tool], prompt=your_prompt)
agent_executor = AgentExecutor(agent=agent, tools=[memory_tool])

# Now the agent can recall: "Alice works at OpenAI" automatically
response = agent_executor.invoke({"input": "What company does Alice currently work at?"})
```

#### Step 6 — Ingest Message History (Chat Use Case)

```python
async def ingest_conversation():
    messages = [
        {"role": "user", "content": "My name is Bob and I'm allergic to peanuts."},
        {"role": "assistant", "content": "Noted! I'll remember your peanut allergy."},
        {"role": "user", "content": "I also recently moved to Seattle from Boston."},
    ]

    for i, msg in enumerate(messages):
        await graphiti.add_episode(
            name=f"chat_msg_{i}",
            episode_body=f"{msg['role']}: {msg['content']}",
            source=EpisodeType.message,
            source_description="chat session",
            reference_time=datetime.now(timezone.utc)
        )

asyncio.run(ingest_conversation())
```

The graph will now contain:
- `(Bob)-[HAS_ALLERGY]->(Peanuts)`
- `(Bob)-[LIVES_IN]->(Seattle)` with `valid_from = now`
- `(Bob)-[PREVIOUSLY_LIVED_IN]->(Boston)` (auto-invalidated)

---

### Graphiti: Advanced Features

#### Custom Entity Types (Ontology)

```python
from pydantic import BaseModel
from graphiti_core.nodes import EntityNode

class MedicalRecord(BaseModel):
    condition: str
    diagnosed_date: str
    severity: str

# Register custom entity type
await graphiti.add_episode(
    name="medical_note",
    episode_body="Patient diagnosed with Type 2 Diabetes on 2024-01-15. Severity: moderate.",
    source=EpisodeType.text,
    entity_types={"MedicalRecord": MedicalRecord}
)
```

#### Community Detection

Graphiti automatically clusters related entities into **communities** (like topic clusters or user groups):

```python
# Trigger community building (run periodically)
await graphiti.build_communities()

# Search within communities
results = await graphiti.search(
    query="tech companies",
    config=SearchConfig(include_community=True)
)
```

---

## Side-by-Side Feature Comparison

| Feature | LLMGraphTransformer | Graphiti |
|---|---|---|
| **Ingestion style** | Batch (documents) | Streaming (episodes) |
| **Time-awareness** | ❌ None | ✅ Bi-temporal (event time + ingestion time) |
| **Conflict resolution** | ❌ Creates duplicate edges | ✅ Auto-detects & invalidates contradictions |
| **Deduplication** | ❌ Manual / none | ✅ Entity resolution built-in |
| **Schema control** | ✅ allowed_nodes/relationships | ✅ Custom Pydantic entity types |
| **Search** | Cypher queries | Hybrid (semantic + BM25 + graph) |
| **Community detection** | ❌ | ✅ |
| **Graph backend** | Neo4j, Memgraph, Amazon Neptune | Neo4j only |
| **LangChain integration** | ✅ Native | ✅ Via tool wrapping |
| **Async support** | Partial | ✅ Fully async |
| **Memory for agents** | ❌ Needs extra wiring | ✅ Purpose-built |
| **Open source** | ✅ | ✅ |
| **Ease of setup** | ⭐⭐⭐⭐ (simple) | ⭐⭐⭐ (requires Neo4j tuning) |
| **Production maturity** | ✅ Mature (LangChain) | 🔶 Growing (Zep AI backed) |

---

## Use Cases

### LLMGraphTransformer Use Cases

#### 1. Enterprise Knowledge Base from Documents

**Scenario:** A law firm wants to extract all entities (cases, judges, parties, laws) from thousands of legal documents and build a searchable knowledge graph.

```python
# Process a legal corpus
from langchain_community.document_loaders import DirectoryLoader

loader = DirectoryLoader("./legal_docs/", glob="*.txt")
docs = loader.load()

transformer = LLMGraphTransformer(
    llm=llm,
    allowed_nodes=["Person", "LegalCase", "Law", "Court", "Organization"],
    allowed_relationships=["RULED_ON", "CITED", "REPRESENTED_BY", "FILED_IN"]
)

graph_docs = transformer.convert_to_graph_documents(docs)
graph.add_graph_documents(graph_docs, include_source=True)

# Now query: "Which judges have ruled on IP cases?"
```

#### 2. Research Paper Graph

**Scenario:** A biotech company indexes scientific papers to find connections between researchers, drugs, and diseases.

```python
transformer = LLMGraphTransformer(
    llm=llm,
    allowed_nodes=["Researcher", "Drug", "Disease", "Institution"],
    allowed_relationships=["RESEARCHES", "TREATS", "AFFILIATED_WITH", "CO_AUTHORED"],
    node_properties=["year", "journal"]
)
```

#### 3. GraphRAG Application

Build a Retrieval-Augmented Generation system where answers come from traversing a knowledge graph rather than vector similarity alone — better for multi-hop questions like "Who is the CEO of the company that acquired the startup founded by person X?"

---

### Graphiti Use Cases

#### 1. AI Customer Support Agent with Memory

**Scenario:** A customer support chatbot needs to remember a customer's past issues, preferences, and resolutions across sessions.

```python
# Each support ticket becomes an episode
await graphiti.add_episode(
    name=f"ticket_{ticket_id}",
    episode_body=f"Customer {customer_id} reported: {issue_description}. Resolution: {resolution}",
    source=EpisodeType.text,
    reference_time=ticket_created_at
)

# Agent can now recall: "This customer had a billing issue in March that was resolved by..."
```

#### 2. Personal AI Assistant (Life Memory)

**Scenario:** A personal assistant app tracks your evolving life facts — job changes, preferences, relationships — and never forgets or gets confused when things change.

```python
# User tells assistant they moved
await graphiti.add_episode(
    name="life_update_42",
    episode_body="I moved from New York to Toronto last month for work.",
    source=EpisodeType.message,
    reference_time=datetime.now(timezone.utc)
)
# Old "lives in New York" edge is auto-invalidated. "Lives in Toronto" is the new truth.
```

#### 3. Medical Patient History

**Scenario:** A clinical AI assistant tracks patient health records over time — medications, diagnoses, allergies — where history matters and outdated facts must not override current ones.

```python
await graphiti.add_episode(
    name="patient_update_2024",
    episode_body="Patient discontinued metformin. Now prescribed semaglutide for diabetes.",
    source=EpisodeType.text,
    reference_time=datetime(2024, 9, 1, tzinfo=timezone.utc)
)
# Old "takes metformin" edge: valid_to = 2024-09-01
# New "takes semaglutide" edge: valid_from = 2024-09-01
```

#### 4. Multi-Agent Collaborative Memory

**Scenario:** Multiple AI agents share a common Graphiti graph as shared memory. Agent A learns new info, Agent B immediately benefits.

```python
# Agent A ingests a new fact
await graphiti_shared.add_episode(name="agent_a_obs_1", ...)

# Agent B queries the shared graph moments later
results = await graphiti_shared.search("latest product pricing")
```

---

## When to Choose Which

### Choose **LLMGraphTransformer** when:

- You have a **static corpus** of documents (PDFs, articles, wikis) to index
- You want to build a **GraphRAG** pipeline for question answering
- Your data **doesn't change frequently** (or you can rebuild the graph)
- You need **multiple graph database backends** (not just Neo4j)
- You're already deep in the **LangChain ecosystem**
- You need a **quick, simple setup** with minimal infrastructure

### Choose **Graphiti** when:

- You're building a **long-running AI agent** that needs persistent memory
- Your facts **change over time** and you need automatic conflict resolution
- You need **temporal queries** ("what was true in January?")
- You're building **personalized AI** (assistants, copilots, customer agents)
- You need **real-time ingestion** (streaming events, messages, IoT)
- You need **entity deduplication** and resolution out of the box
- You need to serve **multiple users** with isolated or shared memory graphs

### Use Both Together when:

- You want to **bootstrap** a Graphiti graph with domain knowledge from LLMGraphTransformer (batch ingest articles → Graphiti handles live updates)
- You need a **hybrid system**: static knowledge (LLMGraphTransformer) + dynamic agent memory (Graphiti)

```python
# Phase 1: Batch ingest domain knowledge via LLMGraphTransformer
graph_docs = transformer.convert_to_graph_documents(domain_docs)
graph.add_graph_documents(graph_docs)

# Phase 2: Live agent interactions via Graphiti
await graphiti.add_episode(
    name="live_interaction_001",
    episode_body=user_message,
    source=EpisodeType.message,
    reference_time=datetime.now(timezone.utc)
)
```

---

## Summary

```
LLMGraphTransformer
├── Philosophy: "Extract once, query forever"
├── Strength: Simplicity, LangChain ecosystem, multiple backends
├── Weakness: No time-awareness, no conflict resolution
└── Sweet spot: Document knowledge bases, GraphRAG, static corpora

Graphiti
├── Philosophy: "Living memory that grows and self-corrects"
├── Strength: Temporal reasoning, deduplication, agent memory
├── Weakness: Neo4j-only, more complex setup, newer ecosystem
└── Sweet spot: AI agents, personalization, evolving knowledge
```

Both tools represent the cutting edge of **LLM-powered knowledge graphs**. LLMGraphTransformer excels at turning documents into queryable structure, while Graphiti excels at giving AI agents a brain that remembers, updates, and reasons across time.

---

*Generated with Claude — Last updated: 2025*
