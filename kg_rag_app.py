"""
╔══════════════════════════════════════════════════════════════════╗
║        KNOWLEDGE GRAPH RAG — STREAMLIT APPLICATION              ║
║        Built with LangChain + Neo4j + OpenAI + Streamlit        ║
╚══════════════════════════════════════════════════════════════════╝

Run with:
    streamlit run kg_rag_app.py

Requirements:
    pip install streamlit langchain langchain-community langchain-openai
                langchain-experimental neo4j pypdf python-dotenv
"""

import os
import time
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

# ─── Page Config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="KG RAG — Bangalore Nexus",
    page_icon="🕸️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ──────────────────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Mono:ital,wght@0,300;0,400;1,300&family=DM+Sans:wght@300;400;500&display=swap');

  /* ── Global ── */
  html, body, [class*="css"] {
      font-family: 'DM Sans', sans-serif;
      background-color: #0d0f1a;
      color: #e8e6f0;
  }
  .stApp { background-color: #0d0f1a; }

  /* ── Hide default Streamlit chrome ── */
  #MainMenu, footer, header { visibility: hidden; }
  .block-container { padding-top: 1.8rem; padding-bottom: 2rem; }

  /* ── Top banner ── */
  .top-banner {
      background: linear-gradient(135deg, #12142a 0%, #1a1d3a 50%, #0f1120 100%);
      border: 1px solid #2a2d4a;
      border-radius: 14px;
      padding: 2rem 2.5rem 1.6rem;
      margin-bottom: 1.8rem;
      position: relative;
      overflow: hidden;
  }
  .top-banner::before {
      content: '';
      position: absolute;
      top: -60px; right: -60px;
      width: 220px; height: 220px;
      background: radial-gradient(circle, rgba(255,100,80,0.12) 0%, transparent 70%);
      border-radius: 50%;
  }
  .top-banner::after {
      content: '';
      position: absolute;
      bottom: -40px; left: 30%;
      width: 300px; height: 150px;
      background: radial-gradient(ellipse, rgba(100,120,255,0.08) 0%, transparent 70%);
  }
  .banner-title {
      font-family: 'Syne', sans-serif;
      font-weight: 800;
      font-size: 2rem;
      color: #ffffff;
      letter-spacing: -0.03em;
      margin: 0 0 0.3rem;
      line-height: 1.1;
  }
  .banner-title span { color: #ff6450; }
  .banner-subtitle {
      font-family: 'DM Sans', sans-serif;
      font-weight: 300;
      font-size: 0.92rem;
      color: #8885aa;
      margin: 0;
      letter-spacing: 0.02em;
  }
  .banner-badges {
      display: flex; gap: 0.5rem; flex-wrap: wrap;
      margin-top: 1rem;
  }
  .badge {
      font-family: 'DM Mono', monospace;
      font-size: 0.7rem;
      padding: 0.25rem 0.65rem;
      border-radius: 20px;
      border: 1px solid;
      letter-spacing: 0.05em;
  }
  .badge-red   { color: #ff6450; border-color: rgba(255,100,80,0.35); background: rgba(255,100,80,0.08); }
  .badge-blue  { color: #7b9cff; border-color: rgba(123,156,255,0.35); background: rgba(123,156,255,0.08); }
  .badge-green { color: #5ddb9a; border-color: rgba(93,219,154,0.35); background: rgba(93,219,154,0.08); }
  .badge-amber { color: #ffbb55; border-color: rgba(255,187,85,0.35); background: rgba(255,187,85,0.08); }

  /* ── Sidebar ── */
  [data-testid="stSidebar"] {
      background: #101220 !important;
      border-right: 1px solid #1e2040;
  }
  [data-testid="stSidebar"] .stMarkdown h3 {
      font-family: 'Syne', sans-serif;
      font-size: 0.78rem;
      font-weight: 700;
      color: #ff6450;
      letter-spacing: 0.12em;
      text-transform: uppercase;
      margin-bottom: 0.6rem;
  }

  /* ── Cards ── */
  .card {
      background: #13152a;
      border: 1px solid #1e2145;
      border-radius: 12px;
      padding: 1.4rem 1.6rem;
      margin-bottom: 1rem;
  }
  .card-title {
      font-family: 'Syne', sans-serif;
      font-weight: 700;
      font-size: 0.78rem;
      color: #ff6450;
      letter-spacing: 0.1em;
      text-transform: uppercase;
      margin-bottom: 0.8rem;
  }

  /* ── Step indicators ── */
  .step-row {
      display: flex; align-items: flex-start; gap: 1rem;
      margin-bottom: 1.1rem;
  }
  .step-num {
      font-family: 'DM Mono', monospace;
      font-size: 0.72rem;
      font-weight: 400;
      color: #0d0f1a;
      background: #ff6450;
      border-radius: 50%;
      width: 22px; height: 22px;
      display: flex; align-items: center; justify-content: center;
      flex-shrink: 0; margin-top: 1px;
  }
  .step-content { flex: 1; }
  .step-label {
      font-family: 'Syne', sans-serif;
      font-size: 0.85rem;
      font-weight: 600;
      color: #e8e6f0;
      margin-bottom: 0.15rem;
  }
  .step-desc {
      font-size: 0.8rem;
      color: #6e6b8a;
      line-height: 1.5;
  }

  /* ── Inputs ── */
  .stTextInput > div > div > input,
  .stTextArea > div > div > textarea {
      background: #0d0f1a !important;
      border: 1px solid #2a2d4a !important;
      border-radius: 8px !important;
      color: #e8e6f0 !important;
      font-family: 'DM Mono', monospace !important;
      font-size: 0.82rem !important;
  }
  .stTextInput > div > div > input:focus,
  .stTextArea > div > div > textarea:focus {
      border-color: #ff6450 !important;
      box-shadow: 0 0 0 2px rgba(255,100,80,0.15) !important;
  }
  .stSelectbox > div > div {
      background: #0d0f1a !important;
      border: 1px solid #2a2d4a !important;
      border-radius: 8px !important;
      color: #e8e6f0 !important;
  }

  /* ── Buttons ── */
  .stButton > button {
      font-family: 'Syne', sans-serif !important;
      font-weight: 700 !important;
      font-size: 0.82rem !important;
      letter-spacing: 0.05em !important;
      border-radius: 8px !important;
      border: none !important;
      padding: 0.55rem 1.4rem !important;
      transition: all 0.2s ease !important;
  }
  .stButton > button[kind="primary"] {
      background: linear-gradient(135deg, #ff6450, #ff4433) !important;
      color: white !important;
  }
  .stButton > button[kind="primary"]:hover {
      transform: translateY(-1px) !important;
      box-shadow: 0 6px 20px rgba(255,100,80,0.35) !important;
  }
  .stButton > button[kind="secondary"] {
      background: #1a1d3a !important;
      color: #e8e6f0 !important;
      border: 1px solid #2a2d4a !important;
  }

  /* ── Answer box ── */
  .answer-box {
      background: linear-gradient(135deg, #0f1428 0%, #131830 100%);
      border: 1px solid #2a3060;
      border-left: 3px solid #ff6450;
      border-radius: 10px;
      padding: 1.2rem 1.4rem;
      margin-top: 1rem;
  }
  .answer-label {
      font-family: 'DM Mono', monospace;
      font-size: 0.68rem;
      color: #ff6450;
      letter-spacing: 0.1em;
      text-transform: uppercase;
      margin-bottom: 0.5rem;
  }
  .answer-text {
      font-size: 0.92rem;
      color: #d8d5ee;
      line-height: 1.7;
  }

  /* ── Cypher box ── */
  .cypher-box {
      background: #080a14;
      border: 1px solid #1e2040;
      border-radius: 8px;
      padding: 0.9rem 1.1rem;
      margin-top: 0.8rem;
      font-family: 'DM Mono', monospace;
      font-size: 0.78rem;
      color: #7b9cff;
      line-height: 1.6;
      white-space: pre-wrap;
      overflow-x: auto;
  }
  .cypher-label {
      font-family: 'DM Mono', monospace;
      font-size: 0.68rem;
      color: #7b9cff;
      letter-spacing: 0.1em;
      text-transform: uppercase;
      margin-bottom: 0.4rem;
  }

  /* ── Stats row ── */
  .stats-row {
      display: flex; gap: 0.8rem; flex-wrap: wrap;
      margin: 1rem 0;
  }
  .stat-chip {
      background: #13152a;
      border: 1px solid #1e2145;
      border-radius: 8px;
      padding: 0.6rem 1rem;
      text-align: center;
      flex: 1; min-width: 90px;
  }
  .stat-num {
      font-family: 'Syne', sans-serif;
      font-size: 1.5rem;
      font-weight: 800;
      color: #ff6450;
      line-height: 1;
  }
  .stat-lbl {
      font-size: 0.72rem;
      color: #6e6b8a;
      margin-top: 0.2rem;
  }

  /* ── History items ── */
  .history-item {
      background: #0f1120;
      border: 1px solid #1a1d38;
      border-radius: 8px;
      padding: 0.8rem 1rem;
      margin-bottom: 0.6rem;
      cursor: pointer;
      transition: border-color 0.2s;
  }
  .history-item:hover { border-color: #ff6450; }
  .history-q {
      font-size: 0.8rem;
      color: #c8c5e0;
      font-weight: 500;
      margin-bottom: 0.25rem;
      white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
  }
  .history-time {
      font-family: 'DM Mono', monospace;
      font-size: 0.68rem;
      color: #4e4b6a;
  }

  /* ── Tag pills ── */
  .tag {
      display: inline-block;
      font-family: 'DM Mono', monospace;
      font-size: 0.68rem;
      color: #5ddb9a;
      background: rgba(93,219,154,0.08);
      border: 1px solid rgba(93,219,154,0.25);
      border-radius: 4px;
      padding: 0.15rem 0.45rem;
      margin: 0.15rem;
  }

  /* ── Divider ── */
  hr { border-color: #1a1d38 !important; margin: 1.2rem 0; }

  /* ── Expander ── */
  .streamlit-expanderHeader {
      font-family: 'Syne', sans-serif !important;
      font-size: 0.82rem !important;
      color: #8885aa !important;
      background: #0f1120 !important;
      border-radius: 6px !important;
  }

  /* ── File uploader ── */
  [data-testid="stFileUploader"] {
      background: #0d0f1a;
      border: 1px dashed #2a2d4a;
      border-radius: 10px;
  }

  /* ── Progress / spinner tint ── */
  .stProgress > div > div { background: #ff6450 !important; }

  /* ── Scrollbar ── */
  ::-webkit-scrollbar { width: 5px; height: 5px; }
  ::-webkit-scrollbar-track { background: #0d0f1a; }
  ::-webkit-scrollbar-thumb { background: #2a2d4a; border-radius: 4px; }
  ::-webkit-scrollbar-thumb:hover { background: #ff6450; }
</style>
""", unsafe_allow_html=True)


# ─── Session State Init ───────────────────────────────────────────────────────
for key, default in {
    "graph_built": False,
    "graph_stats": {"nodes": 0, "rels": 0, "chunks": 0},
    "query_history": [],
    "neo4j_connected": False,
    "openai_ready": False,
    "graph": None,
    "qa_chain": None,
    "llm": None,
}.items():
    if key not in st.session_state:
        st.session_state[key] = default


# ─── Helper: lazy imports (avoid crashing if not installed) ──────────────────
@st.cache_resource
def load_langchain_modules():
    try:
        from langchain_community.document_loaders import PyPDFLoader
        from langchain_text_splitters import TokenTextSplitter
        from langchain_openai import ChatOpenAI
        from langchain_community.graphs import Neo4jGraph
        from langchain_experimental.graph_transformers import LLMGraphTransformer
        from langchain.chains import GraphCypherQAChain
        return {
            "PyPDFLoader": PyPDFLoader,
            "TokenTextSplitter": TokenTextSplitter,
            "ChatOpenAI": ChatOpenAI,
            "Neo4jGraph": Neo4jGraph,
            "LLMGraphTransformer": LLMGraphTransformer,
            "GraphCypherQAChain": GraphCypherQAChain,
        }
    except ImportError as e:
        return {"error": str(e)}


# ─── TOP BANNER ──────────────────────────────────────────────────────────────
st.markdown("""
<div class="top-banner">
  <div class="banner-title">Knowledge Graph <span>RAG</span></div>
  <p class="banner-subtitle">PDF → Entity Extraction → Neo4j Graph → Natural Language Q&A</p>
  <div class="banner-badges">
    <span class="badge badge-red">LangChain</span>
    <span class="badge badge-blue">Neo4j</span>
    <span class="badge badge-green">GPT-4o</span>
    <span class="badge badge-amber">Streamlit</span>
  </div>
</div>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("### ⚙️ Configuration")

    # OpenAI
    openai_key = st.text_input(
        "OpenAI API Key",
        value=os.getenv("OPENAI_API_KEY", ""),
        type="password",
        placeholder="sk-...",
        help="Your OpenAI API key for GPT-4o"
    )
    if openai_key:
        os.environ["OPENAI_API_KEY"] = openai_key
        st.session_state.openai_ready = True
        st.markdown('<span class="badge badge-green">✓ OpenAI Key Set</span>', unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 🗄️ Neo4j Connection")

    neo4j_uri = st.text_input(
        "Bolt URI",
        value=os.getenv("NEO4J_URI", "bolt://localhost:7687"),
        placeholder="bolt://localhost:7687"
    )
    neo4j_user = st.text_input(
        "Username",
        value=os.getenv("NEO4J_USERNAME", "neo4j")
    )
    neo4j_pass = st.text_input(
        "Password",
        value=os.getenv("NEO4J_PASSWORD", ""),
        type="password",
        placeholder="neo4j password"
    )

    if st.button("Test Connection", use_container_width=True):
        try:
            mods = load_langchain_modules()
            if "error" in mods:
                st.error(f"Import error: {mods['error']}")
            else:
                g = mods["Neo4jGraph"](url=neo4j_uri, username=neo4j_user, password=neo4j_pass)
                g.refresh_schema()
                st.session_state.neo4j_connected = True
                st.session_state.graph = g
                st.success("✅ Connected to Neo4j!")
        except Exception as e:
            st.error(f"Connection failed: {e}")

    if st.session_state.neo4j_connected:
        st.markdown('<span class="badge badge-green">✓ Neo4j Connected</span>', unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 🎛️ Graph Settings")

    chunk_size = st.slider("Chunk Size (tokens)", 256, 1024, 512, 64,
                           help="Larger chunks = more context per extraction call")
    chunk_overlap = st.slider("Chunk Overlap (tokens)", 0, 128, 24, 8,
                              help="Overlap prevents relationships being split across chunks")
    model_choice = st.selectbox("LLM Model", ["gpt-4o", "gpt-4-turbo", "gpt-3.5-turbo"],
                                help="GPT-4o gives best extraction quality")

    st.markdown("---")
    st.markdown("### 📖 Quick Help")
    with st.expander("How it works"):
        st.markdown("""
**Step 1 — Upload PDF**
Your document is loaded page by page.

**Step 2 — Build Graph**
GPT-4o reads each chunk and extracts entities (people, orgs, places) and their relationships. These are stored in Neo4j.

**Step 3 — Ask Questions**
Your plain-English question is converted to a Cypher graph query, run against Neo4j, and answered in natural language.
        """)

    with st.expander("Sample questions to try"):
        st.markdown("""
- Who founded KVPL and when?
- Who is investigating PayNest?
- Which lawyer has a conflict of interest?
- How is Ritu Sharma connected to PayNest?
- Who is leaking information to MalhotraCorp?
- What companies does Goldman Sachs connect to?
        """)


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN CONTENT — 3 TABS
# ═══════════════════════════════════════════════════════════════════════════════
tab1, tab2, tab3 = st.tabs(["  📄  Build Graph  ", "  🔍  Query  ", "  📜  History  "])


# ─────────────────────────────────────────────────────────────────────────────
# TAB 1 — BUILD GRAPH
# ─────────────────────────────────────────────────────────────────────────────
with tab1:
    col_left, col_right = st.columns([1.1, 0.9], gap="large")

    with col_left:
        st.markdown('<div class="card-title">📂 Upload PDF Document</div>', unsafe_allow_html=True)

        uploaded_file = st.file_uploader(
            "Drop your PDF here",
            type=["pdf"],
            help="Upload the PDF you want to build a Knowledge Graph from"
        )

        if uploaded_file:
            st.markdown(f"""
            <div class="card">
              <div class="card-title">File Loaded</div>
              <div style="font-family:'DM Mono',monospace;font-size:0.82rem;color:#5ddb9a;">
                📄 {uploaded_file.name}
              </div>
              <div style="font-size:0.78rem;color:#6e6b8a;margin-top:0.3rem;">
                {round(uploaded_file.size/1024, 1)} KB
              </div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("---")

        # Node/relationship type configuration
        st.markdown('<div class="card-title">🔵 Node Types to Extract</div>', unsafe_allow_html=True)
        node_types = st.multiselect(
            "Select entity types",
            ["Person", "Organisation", "Role", "Location", "Product", "LegalCase", "Event", "Concept"],
            default=["Person", "Organisation", "Role", "Location", "Product", "LegalCase", "Event"],
            label_visibility="collapsed"
        )

        st.markdown('<div class="card-title" style="margin-top:0.8rem;">🔗 Relationship Types</div>',
                    unsafe_allow_html=True)
        rel_types = st.multiselect(
            "Select relationship types",
            ["WORKS_AT", "FOUNDED", "MARRIED_TO", "PARENT_OF", "FRIEND_OF",
             "REPORTS_TO", "COMPETES_WITH", "FUNDED_BY", "INVESTIGATED_BY",
             "REPRESENTS", "OWNS", "RELATED_TO", "LOCATED_IN", "INVOLVED_IN"],
            default=["WORKS_AT", "FOUNDED", "MARRIED_TO", "PARENT_OF", "FRIEND_OF",
                     "REPORTS_TO", "COMPETES_WITH", "FUNDED_BY", "INVESTIGATED_BY",
                     "REPRESENTS", "OWNS", "RELATED_TO", "LOCATED_IN", "INVOLVED_IN"],
            label_visibility="collapsed"
        )

        st.markdown("---")

        build_disabled = not (uploaded_file and st.session_state.neo4j_connected
                              and st.session_state.openai_ready)
        if build_disabled:
            missing = []
            if not uploaded_file: missing.append("PDF")
            if not st.session_state.openai_ready: missing.append("OpenAI Key")
            if not st.session_state.neo4j_connected: missing.append("Neo4j Connection")
            st.warning(f"⚠️ Missing: {', '.join(missing)}")

        if st.button("🚀 Build Knowledge Graph", type="primary",
                     use_container_width=True, disabled=build_disabled):

            mods = load_langchain_modules()
            if "error" in mods:
                st.error(f"Missing packages: {mods['error']}\n\nRun: pip install langchain langchain-community langchain-openai langchain-experimental neo4j pypdf")
            else:
                # Save PDF temp
                import tempfile
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                    tmp.write(uploaded_file.read())
                    tmp_path = tmp.name

                progress = st.progress(0, text="Loading PDF...")

                try:
                    # Step 1: Load
                    loader = mods["PyPDFLoader"](tmp_path)
                    pages = loader.load()
                    progress.progress(20, text=f"✅ Loaded {len(pages)} pages. Splitting into chunks...")

                    # Step 2: Split
                    splitter = mods["TokenTextSplitter"](
                        chunk_size=chunk_size, chunk_overlap=chunk_overlap
                    )
                    docs = splitter.split_documents(pages)
                    progress.progress(35, text=f"✅ {len(docs)} chunks ready. Initialising LLM...")

                    # Step 3: LLM
                    llm = mods["ChatOpenAI"](model=model_choice, temperature=0)
                    st.session_state.llm = llm
                    progress.progress(45, text="✅ LLM ready. Extracting entities & relationships...")

                    # Step 4: Transform
                    transformer = mods["LLMGraphTransformer"](
                        llm=llm,
                        allowed_nodes=node_types,
                        allowed_relationships=rel_types
                    )
                    graph_docs = transformer.convert_to_graph_documents(docs)
                    progress.progress(75, text="✅ Extraction complete. Storing in Neo4j...")

                    # Step 5: Store
                    graph = st.session_state.graph
                    graph.add_graph_documents(
                        graph_docs,
                        baseEntityLabel=True,
                        include_source=True
                    )
                    graph.refresh_schema()
                    progress.progress(90, text="✅ Graph stored. Building QA chain...")

                    # Step 6: QA Chain
                    qa_chain = mods["GraphCypherQAChain"].from_llm(
                        llm=llm,
                        graph=graph,
                        verbose=True,
                        allow_dangerous_requests=True,
                        return_intermediate_steps=True
                    )
                    st.session_state.qa_chain = qa_chain
                    progress.progress(100, text="✅ Knowledge Graph ready!")

                    # Stats
                    total_nodes = sum(len(gd.nodes) for gd in graph_docs)
                    total_rels  = sum(len(gd.relationships) for gd in graph_docs)
                    st.session_state.graph_stats = {
                        "nodes": total_nodes,
                        "rels": total_rels,
                        "chunks": len(docs)
                    }
                    st.session_state.graph_built = True

                    os.unlink(tmp_path)
                    time.sleep(0.5)
                    st.rerun()

                except Exception as e:
                    progress.empty()
                    st.error(f"❌ Error building graph: {e}")
                    os.unlink(tmp_path)

    with col_right:
        st.markdown('<div class="card-title">📊 Graph Status</div>', unsafe_allow_html=True)

        if st.session_state.graph_built:
            stats = st.session_state.graph_stats
            st.markdown(f"""
            <div class="stats-row">
              <div class="stat-chip">
                <div class="stat-num">{stats['nodes']}</div>
                <div class="stat-lbl">Nodes</div>
              </div>
              <div class="stat-chip">
                <div class="stat-num">{stats['rels']}</div>
                <div class="stat-lbl">Relationships</div>
              </div>
              <div class="stat-chip">
                <div class="stat-num">{stats['chunks']}</div>
                <div class="stat-lbl">Chunks</div>
              </div>
            </div>
            """, unsafe_allow_html=True)

            st.success("✅ Knowledge Graph is live and ready to query!")

            # Schema preview
            if st.session_state.graph:
                with st.expander("🗂️ View Graph Schema"):
                    schema = st.session_state.graph.schema
                    st.code(schema, language="text")

            # Useful Cypher queries
            st.markdown('<div class="card-title" style="margin-top:1rem;">🔎 Explore in Neo4j Browser</div>',
                        unsafe_allow_html=True)
            cypher_snippets = {
                "All nodes & relationships (limit 100)":
                    "MATCH (n)-[r]->(m) RETURN n, r, m LIMIT 100",
                "All Person nodes":
                    "MATCH (p:Person) RETURN p.id ORDER BY p.id",
                "All Organisation nodes":
                    "MATCH (o:Organisation) RETURN o.id ORDER BY o.id",
                "Conflict of interest (person in 2 orgs)":
                    "MATCH (p:Person)-[r1]->(o1)\nMATCH (p)-[r2]->(o2)\nWHERE o1 <> o2\nRETURN p.id, type(r1), o1.id, type(r2), o2.id",
                "Family in rival companies":
                    "MATCH (p1:Person)-[:MARRIED_TO|PARENT_OF]-(p2:Person)\nMATCH (p1)-[:WORKS_AT]->(o1)\nMATCH (p2)-[:WORKS_AT]->(o2)\nWHERE o1 <> o2\nRETURN p1.id, o1.id, p2.id, o2.id",
            }
            selected_cypher = st.selectbox("Copy a Cypher snippet →", list(cypher_snippets.keys()))
            st.markdown(f'<div class="cypher-label">Cypher</div><div class="cypher-box">{cypher_snippets[selected_cypher]}</div>',
                        unsafe_allow_html=True)
        else:
            # Pipeline walkthrough
            st.markdown('<div class="card-title">Pipeline Steps</div>', unsafe_allow_html=True)
            steps = [
                ("1", "Upload PDF", "Drop your document using the uploader on the left"),
                ("2", "Configure credentials", "Add OpenAI key + Neo4j connection in the sidebar"),
                ("3", "Choose entity types", "Select which node and relationship types to extract"),
                ("4", "Build Graph", "Click the button — GPT-4o extracts entities and stores them in Neo4j"),
                ("5", "Query", "Switch to the Query tab and ask questions in plain English"),
            ]
            for num, label, desc in steps:
                st.markdown(f"""
                <div class="step-row">
                  <div class="step-num">{num}</div>
                  <div class="step-content">
                    <div class="step-label">{label}</div>
                    <div class="step-desc">{desc}</div>
                  </div>
                </div>
                """, unsafe_allow_html=True)

            st.markdown("---")
            st.markdown("""
            <div style="font-size:0.78rem;color:#4e4b6a;line-height:1.7;">
            💡 <strong style="color:#6e6b8a;">No Neo4j yet?</strong><br>
            Create a free sandbox at<br>
            <span style="color:#7b9cff;font-family:'DM Mono',monospace;">sandbox.neo4j.com</span>
            </div>
            """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# TAB 2 — QUERY
# ─────────────────────────────────────────────────────────────────────────────
with tab2:
    if not st.session_state.graph_built:
        st.markdown("""
        <div style="text-align:center;padding:3rem 1rem;color:#4e4b6a;">
          <div style="font-size:2.5rem;margin-bottom:0.8rem;">🕸️</div>
          <div style="font-family:'Syne',sans-serif;font-size:1rem;font-weight:600;color:#6e6b8a;">
            No graph built yet
          </div>
          <div style="font-size:0.82rem;margin-top:0.4rem;">
            Go to the <strong>Build Graph</strong> tab to get started.
          </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        col_q, col_ans = st.columns([1, 1.1], gap="large")

        with col_q:
            st.markdown('<div class="card-title">❓ Ask a Question</div>', unsafe_allow_html=True)

            # Preset questions
            preset_questions = [
                "— choose a preset —",
                "Who is the CEO of KV Pharma Solutions?",
                "Who is investigating PayNest Technologies?",
                "Who is Priya Krishnamurthy's boyfriend?",
                "Which lawyer has a conflict of interest?",
                "What companies does Goldman Sachs connect to?",
                "Who is leaking KVPL's information to MalhotraCorp?",
                "How is Dr. Ritu Sharma connected to PayNest?",
                "What is the connection between Arvind and the RBI investigation?",
            ]
            preset = st.selectbox("Quick presets", preset_questions, label_visibility="visible")

            user_question = st.text_area(
                "Or type your own question",
                value="" if preset == "— choose a preset —" else preset,
                height=100,
                placeholder="e.g. Who founded KVPL and what other companies are they connected to?",
                label_visibility="visible"
            )

            show_cypher = st.checkbox("Show generated Cypher query", value=True)
            show_raw    = st.checkbox("Show raw graph results", value=False)

            ask_disabled = not user_question.strip()
            if st.button("🔍 Run Query", type="primary", use_container_width=True,
                         disabled=ask_disabled):
                with col_ans:
                    with st.spinner("Querying the Knowledge Graph..."):
                        try:
                            response = st.session_state.qa_chain.invoke(
                                {"query": user_question}
                            )
                            answer = response.get("result", "No answer returned.")
                            steps  = response.get("intermediate_steps", [])

                            cypher_query = ""
                            raw_results  = []
                            if steps:
                                cypher_query = steps[0].get("query", "")
                                raw_results  = steps[1].get("context", []) if len(steps) > 1 else []

                            # Store in history
                            st.session_state.query_history.append({
                                "question": user_question,
                                "answer": answer,
                                "cypher": cypher_query,
                                "raw": raw_results,
                                "time": time.strftime("%H:%M:%S"),
                            })

                        except Exception as e:
                            answer       = f"Error: {e}"
                            cypher_query = ""
                            raw_results  = []

                    # Answer
                    st.markdown(f"""
                    <div class="answer-box">
                      <div class="answer-label">Answer</div>
                      <div class="answer-text">{answer}</div>
                    </div>
                    """, unsafe_allow_html=True)

                    # Cypher
                    if show_cypher and cypher_query:
                        st.markdown(f"""
                        <div style="margin-top:0.8rem;">
                          <div class="cypher-label">Generated Cypher Query</div>
                          <div class="cypher-box">{cypher_query}</div>
                        </div>
                        """, unsafe_allow_html=True)

                    # Raw
                    if show_raw and raw_results:
                        with st.expander("📊 Raw graph results"):
                            st.json(raw_results[:5])

        with col_ans:
            if not st.session_state.query_history:
                st.markdown("""
                <div style="padding:2rem 0;color:#4e4b6a;text-align:center;">
                  <div style="font-size:0.85rem;">
                    Ask a question on the left to see the answer here.
                  </div>
                </div>
                """, unsafe_allow_html=True)
            else:
                # Show latest answer (already rendered in button block)
                latest = st.session_state.query_history[-1]
                st.markdown(f"""
                <div class="answer-box">
                  <div class="answer-label">Latest Answer</div>
                  <div class="answer-text">{latest['answer']}</div>
                </div>
                """, unsafe_allow_html=True)
                if show_cypher and latest.get("cypher"):
                    st.markdown(f"""
                    <div style="margin-top:0.8rem;">
                      <div class="cypher-label">Generated Cypher Query</div>
                      <div class="cypher-box">{latest['cypher']}</div>
                    </div>
                    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# TAB 3 — HISTORY
# ─────────────────────────────────────────────────────────────────────────────
with tab3:
    if not st.session_state.query_history:
        st.markdown("""
        <div style="text-align:center;padding:3rem 1rem;color:#4e4b6a;">
          <div style="font-size:2.5rem;margin-bottom:0.8rem;">📜</div>
          <div style="font-family:'Syne',sans-serif;font-size:1rem;font-weight:600;color:#6e6b8a;">
            No queries yet
          </div>
          <div style="font-size:0.82rem;margin-top:0.4rem;">
            Ask questions in the Query tab — they'll appear here.
          </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        col_hist, col_detail = st.columns([0.9, 1.1], gap="large")

        with col_hist:
            st.markdown(f'<div class="card-title">🕐 Query History ({len(st.session_state.query_history)})</div>',
                        unsafe_allow_html=True)

            if st.button("🗑️ Clear History", type="secondary"):
                st.session_state.query_history = []
                st.rerun()

            for i, item in enumerate(reversed(st.session_state.query_history)):
                idx = len(st.session_state.query_history) - 1 - i
                st.markdown(f"""
                <div class="history-item">
                  <div class="history-q">#{idx+1} — {item['question']}</div>
                  <div class="history-time">⏱ {item['time']}</div>
                </div>
                """, unsafe_allow_html=True)

        with col_detail:
            st.markdown('<div class="card-title">📋 Full Q&A Log</div>', unsafe_allow_html=True)

            for i, item in enumerate(reversed(st.session_state.query_history)):
                with st.expander(f"Q{len(st.session_state.query_history)-i}: {item['question'][:60]}..."):
                    st.markdown(f"""
                    <div class="answer-box">
                      <div class="answer-label">Answer</div>
                      <div class="answer-text">{item['answer']}</div>
                    </div>
                    """, unsafe_allow_html=True)

                    if item.get("cypher"):
                        st.markdown(f"""
                        <div style="margin-top:0.6rem;">
                          <div class="cypher-label">Cypher</div>
                          <div class="cypher-box">{item['cypher']}</div>
                        </div>
                        """, unsafe_allow_html=True)

                    if item.get("raw"):
                        st.caption("Raw graph data:")
                        st.json(item["raw"][:3])
