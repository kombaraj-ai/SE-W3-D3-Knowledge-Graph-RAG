# Assignment: Knowledge Graph RAG
---

## Table of Contents
1. [Use Case: The Bangalore Nexus](#use-case-the-bangalore-nexus)
2. [Step-by-Step Implementation](#step-by-step-implementation)
   - [Step 1 — Install Dependencies](#step-1--install-dependencies)
   - [Step 2 — Set Up Neo4j](#step-2--set-up-neo4j)
   - [Step 3 — Run the Streamlit App](#step-3--run-the-streamlit-app)
   - [Step 4 — Architecture Visual](#step-4--architecture-visual)
   - [Step 5 — Expected Output](#step-5--expected-output)
3. [Pro Tips & Cypher Queries](#pro-tips--cypher-queries)
4. [When to Use KG RAG vs Normal RAG](#when-to-use-kg-rag-vs-normal-rag)

---

## Use Case: The Bangalore Nexus

The test document **"The Bangalore Nexus"** is a 4-page fictional Indian corporate saga containing:

- **16 people** (founders, CEOs, lawyers, regulators, journalists)
- **12 organisations** (conglomerates, startups, banks, regulators)
- **3 regulatory bodies** (RBI, SEBI, Delhi High Court)
- **6+ hidden conflict-of-interest chains**

### Key Characters

| Person | Role / Title | Organisation |
|---|---|---|
| Arvind Krishnamurthy | Chairman & Founder | KVPL / PayNest |
| Rohan Krishnamurthy | CEO | KV Pharma Solutions |
| Priya Krishnamurthy | Founder & CEO | NexaLearn |
| Suresh Malhotra | Chairman & Founder | MalhotraCorp |
| Deepika Rao | Co-founder & CTO | PayNest Technologies |
| Karthik Subramaniam | CFO | Pharmedge Inc. |
| Dr. Sunita Pillai | Head of R&D | KV Pharma Solutions |
| Shankar Iyer | Head of Real Estate | KV Realty |
| Kavitha Iyer | Investigative Journalist | Deccan Chronicle |
| Ashok Menon | Deputy Governor | Reserve Bank of India |
| Manisha Kulkarni | Investigation Officer | SEBI |
| Arjun Mehta | Investment Banker | Goldman Sachs India |
| Vikram Oberoi | Managing Director | Goldman Sachs India |
| Adv. Nitin Sharma | Senior Partner | Sharma & Associates |
| Dr. Ritu Sharma | CEO | Pharmedge Inc. |

---

## Step-by-Step Implementation

### Step 1 — Install Dependencies

```bash
pip install langchain langchain-community langchain-openai langchain-experimental
pip install neo4j pypdf python-dotenv
```

**What each package does:**

| Package | Purpose |
|---|---|
| `langchain` | The core framework — chains, prompts, document loaders |
| `langchain-openai` | Connects LangChain to OpenAI's GPT models |
| `langchain-experimental` | Contains `LLMGraphTransformer` (PDF → Graph) |
| `neo4j` | Python driver to talk to the Neo4j graph database |
| `pypdf` | Reads and extracts text from PDF files |
| `python-dotenv` | Loads API keys from a `.env` file |

---

### Step 2 — Set Up Neo4j

Go to **[sandbox.neo4j.com](https://sandbox.neo4j.com)** → Create a blank sandbox → note down:
- Bolt URL (e.g., `bolt://54.x.x.x:7687`)
- Username: `neo4j`
- Password: (shown in dashboard)

Create a `.env` file in your project root:

```env
OPENAI_API_KEY=sk-xxxxxxxxxxxx
NEO4J_URI=bolt://your-sandbox-url:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=your-sandbox-password
```

---

### Step 3 — Run the Streamlit App

The full implementation is a single-file Streamlit application: **`kg_rag_app.py`**

```bash
streamlit run kg_rag_app.py
```

The app opens at `http://localhost:8501` in your browser.

---

### App Structure — 3 Tabs

#### Tab 1: Build Graph

This tab handles the entire ingestion pipeline through the UI:

```
[Upload PDF]  →  [Choose Node/Rel Types]  →  [Click "Build Graph"]
                                                      │
                          ┌───────────────────────────┴──────────────────────────┐
                          │                                                       │
               MODULE: PyPDFLoader                                 MODULE: TokenTextSplitter
               Reads each PDF page as a                            Splits pages into 512-token
               Document object with                                chunks with 24-token overlap
               .page_content + .metadata                           to prevent split relationships
                          │                                                       │
                          └───────────────────────────┬──────────────────────────┘
                                                      │
                                         MODULE: LLMGraphTransformer
                                         Sends each chunk to GPT-4o.
                                         Extracts nodes + relationships.
                                         Returns GraphDocument objects.
                                                      │
                                         MODULE: Neo4jGraph.add_graph_documents()
                                         Writes nodes and edges into Neo4j.
                                         baseEntityLabel=True → generic _Entity label
                                         include_source=True  → links back to source chunk
                                                      │
                                         MODULE: GraphCypherQAChain (built and cached)
                                         Ready for querying in Tab 2.
```

**Sidebar controls available during build:**
- Chunk size slider (256–1024 tokens)
- Chunk overlap slider (0–128 tokens)
- LLM model selector (GPT-4o / GPT-4-turbo / GPT-3.5-turbo)
- Node type multiselect (Person, Organisation, Role, Location, etc.)
- Relationship type multiselect (WORKS_AT, FOUNDED, MARRIED_TO, etc.)

---

#### Tab 2: Query

```
User types plain English question
              │
              ▼
GraphCypherQAChain.invoke({"query": question})
              │
    ┌─────────┴──────────┐
    │                    │
Step 1                Step 2
LLM generates         Neo4j runs
Cypher query          the Cypher
    │                    │
    └─────────┬──────────┘
              │
         Step 3: LLM formats
         raw graph result
         into natural language
              │
              ▼
    Displayed in answer box
    + optional Cypher view
    + optional raw JSON results
```

**Preset questions included:**
- Who is the CEO of KV Pharma Solutions?
- Who is investigating PayNest Technologies?
- Who is Priya Krishnamurthy's boyfriend?
- Which lawyer has a conflict of interest?
- What companies does Goldman Sachs connect to?
- Who is leaking KVPL's information to MalhotraCorp?
- How is Dr. Ritu Sharma connected to PayNest?
- What is the connection between Arvind and the RBI investigation?

---

#### Tab 3: History

Stores every question + answer + Cypher query in session state. Displays as a scrollable log with full Q&A expandable detail per entry. Includes a **Clear History** button.

---

### Code Modules Explained

| Module | What it does |
|---|---|
| `PyPDFLoader` | Reads PDF page by page into LangChain `Document` objects |
| `TokenTextSplitter` | Splits documents by token count (not characters) for LLM-safe chunking |
| `ChatOpenAI` | Initialises the GPT-4o LLM with `temperature=0` for deterministic output |
| `Neo4jGraph` | LangChain wrapper for Neo4j — handles connection, schema refresh, and writes |
| `LLMGraphTransformer` | Prompts the LLM to extract structured entities + relationships from text chunks |
| `GraphCypherQAChain` | Converts plain English → Cypher → Neo4j result → natural language answer |
| `st.session_state` | Persists graph, QA chain, stats, and query history across Streamlit reruns |

---

### Step 4 — Architecture Visual

```
┌──────────────────────────────────────────────────────┐
│              STREAMLIT UI (kg_rag_app.py)            │
│                                                      │
│  Sidebar          Tab 1             Tab 2            │
│  ─────────        ─────────         ─────────        │
│  API Keys    →   Upload PDF    →   Ask Question      │
│  Neo4j URI       Build Graph       View Answer       │
│  Chunk size      See Stats         See Cypher        │
│  Model pick      Copy Cypher       Tab 3             │
│                  snippets          ─────────         │
│                                    History log       │
└──────────────────┬───────────────────────────────────┘
                   │
         ┌─────────▼─────────┐
         │   PyPDFLoader     │  Reads PDF → Document objects
         │   TokenTextSplitter│  Splits into 512-token chunks
         └─────────┬─────────┘
                   │
         ┌─────────▼──────────────┐
         │  LLMGraphTransformer   │  GPT-4o extracts:
         │  (GPT-4o, temp=0)      │  Node: [Person] Rohan
         │                        │  Node: [Org] KV Pharma
         │  "Rohan is CEO of      │  Edge: Rohan-[WORKS_AT]->
         │   KV Pharma Solutions" │        KV Pharma
         └─────────┬──────────────┘
                   │
         ┌─────────▼─────────┐
         │      Neo4j        │  Stores graph permanently
         │  (Graph Database) │  Nodes = entities
         │                   │  Edges = relationships
         └─────────┬─────────┘
                   │
         ┌─────────▼──────────────┐
         │  GraphCypherQAChain    │  English → Cypher → Result
         │                        │  → Natural language answer
         │  "Who founded KVPL?"   │
         │        ↓               │
         │  MATCH (p:Person)      │
         │  -[:FOUNDED]->         │
         │  (o {id:'KVPL'})       │
         │  RETURN p.id           │
         │        ↓               │
         │  "Arvind Krishnamurthy"│
         └────────────────────────┘
```

---

### Step 5 — Expected Output

```
❓ QUESTION: Which lawyer represents both PayNest and Suresh Malhotra?

🔍 Generated Cypher:
   MATCH (l:Person)-[:REPRESENTS]->(p:Organisation {id:'PayNest'})
   MATCH (l)-[:REPRESENTS]->(s:Person {id:'Suresh Malhotra'})
   RETURN l.id

📊 Raw Graph Results: [{'l.id': 'Adv. Nitin Sharma'}]

💬 ANSWER: Adv. Nitin Sharma of Sharma & Associates represents
           both PayNest Technologies and Suresh Malhotra, which
           became controversial during the QuickRupee acquisition.
```

---

## Pro Tips & Cypher Queries

### Visualise the Entire Graph in Neo4j Browser

```cypher
MATCH (n)-[r]->(m) RETURN n, r, m LIMIT 100
```

### Find All Hidden Conflicts of Interest

```cypher
MATCH (p:Person)-[r1]->(o1:Organisation)
MATCH (p)-[r2]->(o2:Organisation)
WHERE o1 <> o2
RETURN p.id, type(r1), o1.id, type(r2), o2.id
```

### Find Family Members Inside Rival Companies

```cypher
MATCH (p1:Person)-[:MARRIED_TO|PARENT_OF|RELATED_TO]-(p2:Person)
MATCH (p1)-[:WORKS_AT]->(o1)
MATCH (p2)-[:WORKS_AT]->(o2)
WHERE o1 <> o2
RETURN p1.id, o1.id, p2.id, o2.id
```

### Find All Regulatory Investigations

```cypher
MATCH (o)-[:INVESTIGATED_BY]->(r)
RETURN o.id AS Entity, r.id AS RegulatingBody
```

### Trace a Chain from a Person to a Competitor

```cypher
MATCH path = (p:Person {id:'Karthik Subramaniam'})-[*1..4]-(end)
RETURN path LIMIT 25
```

---

## When to Use KG RAG vs Normal RAG

| Use KG RAG when... | Use Normal RAG when... |
|---|---|
| You need **relationship** answers | You need **summary** answers |
| Questions involve **multiple hops** | Questions are **simple lookups** |
| Data has clear **entities** (people, drugs, products) | Data is **unstructured prose** |
| You need **precise facts** | You need **general context** |
| Detecting **conflicts of interest** | Generating **creative content** |
| **Fraud / compliance** analysis | **Customer FAQ** bots |

---

> **The Bangalore Nexus** has at least **6 conflict-of-interest chains** built in — the Knowledge Graph should surface all of them automatically when queried correctly.

---

*Document generated for KG RAG testing | LangChain + Neo4j + GPT-4o*
