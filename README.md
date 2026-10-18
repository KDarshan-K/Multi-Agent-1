# Research Agent — Multi-Agent Stock/News Research Pipeline

<<<<<<< HEAD
A LangGraph-style multi-agent orchestrator that gathers market data, news,
and web search results, retrieves context from a knowledge base, and
produces a cited research report — with a real critic/revision feedback
loop.



-| 1 | Project skeleton, state schema, stdlib graph engine, Orchestrator, Market Agent, plain-template report |
-| 2 | News Agent + Search Agent, fan-out/converge execution, shared evidence store |
-| 3 | Unified MCP-style tool layer (mock MCP server + client); agents stop importing tool modules directly; added company_tools.py |
-| 4 | RAG/knowledge base — real TF-IDF + cosine similarity retrieval, as its own shared node (not nested under one agent) |
-| 5 | Real Analyst node — genuine HTTPS calls to the live Anthropic API, numbered [N] citations, graceful template fallback |
-| 6 | **Critic node + real conditional revise-loop** back to the Orchestrator, bounded by MAX_REVISIONS |
-| 7 | Checkpointing/persistence, `.gitignore`, final polish, this README |
=======
A production-grade, **LangGraph-driven** multi-agent orchestrator that gathers market data, company overviews, SEC filings, news, and live web search results, retrieves context from a hybrid semantic vector RAG engine, and produces a cited financial research report — powered by a bounded critic/revision feedback loop.

---

## 🌟 Key Upgrades & Architecture Enhancements
>>>>>>> review-multiagent-system-16512030288312706240

- **Official LangGraph Orchestration (`langgraph`):** Converted workflow orchestration to official `langgraph.graph.StateGraph` with state channel reducers (`Annotated[list, operator.add]`) and partial update dicts.
- **Smart Tool API Caching (`tools/cache.py`):** Added a lightweight TTL caching layer for external API calls (Alpha Vantage, Finnhub, SEC EDGAR, Tavily) to prevent rate-limit quota exhaustion across revision loops.
- **Enhanced Ticker Resolution:** Combines explicit ticker matching, possessive query cleaning, expanded offline company mapping (Apple, Tesla, Costco, Nvidia, AMD, Intel, Disney, etc.), and automated Alpha Vantage symbol search fallback.
- **Hybrid Semantic Vector RAG (`tools/rag_tools.py`):** Upgraded RAG retrieval to scikit-learn sublinear TF-IDF vector embeddings with 1-2 n-grams and cosine similarity for context-aware semantic search.
- **Pytest Test Suite (`tests/`):** Added unit and integration tests covering tool caching, ticker resolution, LangGraph node execution, critic routing, and RAG retrieval.

---

## 🔄 System Workflow & Graph Topology

```
                         ┌─────────────────┐
                         │   User Query    │
                         └────────┬────────┘
                                  │
                                  ▼
                        ┌──────────────────┐
                        │   Orchestrator   │ (Extract Tickers / Plan)
                        └────────┬─────────┘
                                  │
         ┌────────────────────────┼────────────────────────┐
         │ (Fan-out)              │                        │
         ▼                        ▼                        ▼
┌──────────────────┐    ┌──────────────────┐    ┌──────────────────┐
│   Market Agent   │    │    News Agent    │    │   Search Agent   │
└────────┬─────────┘    └────────┬─────────┘    └────────┬─────────┘
         │                        │                        │
         └────────────────────────┼────────────────────────┘
                                  │ (Fan-in / Converge)
                                  ▼
                        ┌──────────────────┐
                        │  RAG Retrieval   │ (Semantic Vector Context)
                        └────────┬─────────┘
                                  │
                                  ▼
                        ┌──────────────────┐
                        │  Analyst Node    │ (LLM Synthesis + Citations [N])
                        └────────┬─────────┘
                                  │
                                  ▼
                        ┌──────────────────┐
                        │   Critic Node    │ (Sufficiency Check)
                        └────────┬─────────┘
                                 │
                   ┌─────────────┴─────────────┐
        "revise"   │                           │ "done"
     ┌─────────────┴────────┐                  ▼
     │  (Revision Loop)     │        ┌──────────────────┐
     └──────────►───────────┘        │  Report Writer   │
                                     └────────┬─────────┘
                                              │
                                              ▼
                                     ┌──────────────────┐
                                     │   Final Report   │
                                     └──────────────────┘
```

### The Revision Feedback Loop
1. **Sufficiency Check:** The Critic evaluates the draft report against gathered evidence using an LLM JSON verdict (or a coverage heuristic when no API key is present).
2. **On "revise":** The Critic generates a specific follow-up query (`next_search_query`). The Orchestrator builds a minimal revision plan so Market/News agents become no-ops (avoiding duplicate API calls) while Search and RAG fetch targeted follow-up information.
3. **Termination:** Bounded by `MAX_REVISIONS` (default: 2). If the limit is reached with open concerns, the report finalizes and explicitly notes unresolved critiques.

---

## 📁 Project Structure

```
research_agent/
├── main.py                  # CLI entrypoint: run query, --list-runs, --resume <id>
├── state.py                 # ResearchState dataclass with LangGraph channel reducers
├── graph.py                 # LangGraph workflow definition (StateGraph)
├── checkpoint.py            # State persistence & JSON checkpointing
├── config.py                # Environment configurations (.env auto-loader)
├── llm.py                   # Multi-provider LLM client (Anthropic, Gemini, OpenAI)
├── requirements.txt         # Project dependencies
├── agents/
│   ├── orchestrator.py      # Ticker extraction, symbol search, plan building
│   ├── market_agent.py      # Quotes, company profiles, SEC filings
│   ├── news_agent.py        # Finnhub company news
│   ├── search_agent.py      # Tavily web search agent
│   ├── rag_agent.py          # Shared semantic vector RAG node
│   ├── analyst.py           # LLM report synthesis with [N] citations
│   ├── critic.py             # Sufficiency assessment & revision router
│   └── report_writer.py     # Report formatting, citations appendix, notes
├── mcp_server/
│   └── server.py            # In-process mock MCP server & tool registry
├── tools/
│   ├── mcp_client.py        # Unified agent tool client
│   ├── cache.py             # Lightweight TTL API response caching
│   ├── market_tools.py      # Alpha Vantage market quote & symbol search
│   ├── company_tools.py     # Alpha Vantage company profile overview
│   ├── news_tools.py        # Finnhub company news
│   ├── search_tools.py      # Tavily web search tool
│   ├── sec_tools.py         # SEC EDGAR filings links
│   ├── knowledge_base.py    # Sample financial document corpus
│   └── rag_tools.py         # Hybrid semantic vector TF-IDF + cosine similarity search
└── tests/
    ├── test_cache.py        # Unit tests for tool TTL caching
    ├── test_critic.py       # Unit tests for critic routing & revision cap
    ├── test_graph.py        # End-to-end LangGraph execution tests
    ├── test_rag.py          # Semantic vector search tests
    └── test_ticker_resolution.py # Ticker parsing & resolution tests
```

---

## ⚡ Quick Start

### 1. Install Dependencies
```bash
cd research_agent
pip install -r requirements.txt
```

### 2. Configure Environment Variables
Copy `.env.example` to `.env` and configure your API keys:
```bash
cp .env.example .env
```
Edit `.env` to configure:
- `LLM_PROVIDER`: `anthropic`, `gemini`, or `openai`
- `ANTHROPIC_API_KEY` / `GEMINI_API_KEY` / `OPENAI_API_KEY`
- `TAVILY_API_KEY` (Web Search)
- `ALPHA_VANTAGE_API_KEY` (Market Data)
- `FINNHUB_API_KEY` (Company News)
- `SEC_USER_AGENT` (SEC EDGAR Contact Header)

### 3. Run a Research Query
```bash
python3 main.py "Give me a report on Tesla and Costco earnings and margins"
```

### 4. Saved Runs & Checkpoints
```bash
# List all saved run IDs
python3 main.py --list-runs

# Resume/view a saved report without re-running API tools
python3 main.py --resume <RUN_ID>
```

---

## 🧪 Running Tests

Run the `pytest` test suite:
```bash
python3 -m pytest
```

---

## 🛡️ Fallback & Resilience Features

- **No API Key Fallback:** If no LLM key is configured, the pipeline runs in deterministic fallback mode (template synthesis + coverage heuristic) so execution never crashes.
- **Tool Level Error Handling:** If any live data provider (Alpha Vantage, Finnhub, SEC EDGAR, Tavily) fails or hits rate limits, the failure is recorded in `state.data_gaps` and the pipeline proceeds gracefully with remaining evidence.
