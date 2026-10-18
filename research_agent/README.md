# Research Agent — Multi-Agent Stock/News Research Pipeline

A LangGraph-style multi-agent orchestrator that gathers market data, news,
and web search results, retrieves context from a knowledge base, and
produces a cited research report — with a real critic/revision feedback
loop. Built incrementally, 15% at a time, as a direct response to a
review of an initial architecture diagram.

## Status: 100% — all 7 phases complete

| Phase | Adds |
|---|---|
| 1 (15%) | Project skeleton, state schema, stdlib graph engine, Orchestrator, Market Agent, plain-template report |
| 2 (30%) | News Agent + Search Agent, fan-out/converge execution, shared evidence store |
| 3 (45%) | Unified MCP-style tool layer (mock MCP server + client); agents stop importing tool modules directly; added company_tools.py |
| 4 (60%) | RAG/knowledge base — real TF-IDF + cosine similarity retrieval, as its own shared node (not nested under one agent) |
| 5 (75%) | Real Analyst node — genuine HTTPS calls to the live Anthropic API, numbered [N] citations, graceful template fallback |
| 6 (90%) | **Critic node + real conditional revise-loop** back to the Orchestrator, bounded by MAX_REVISIONS |
| 7 (100%) | Checkpointing/persistence, `.gitignore`, final polish, this README |

## ⚠️ Sandbox constraints (read this first)

This project was built in a sandbox **with no general internet access**:

- `langgraph`/`pydantic` etc. can't be `pip install`-ed. `graph_engine.py`
  is a from-scratch, stdlib-only re-implementation of LangGraph's core
  API (`StateGraph`, `add_node`, `add_edge`, `add_conditional_edges`,
  `.compile().invoke()`). Swapping to the real package is a near-zero
  import change — see that file's docstring.
- Market data and company profiles use Alpha Vantage; company news uses
  Finnhub; SEC filing links come from EDGAR; and general Web Search uses
  Tavily when their corresponding keys are configured. The RAG retriever
  still runs against this project's local sample corpus.
- **`llm.py` supports 3 providers** (Anthropic, Gemini, OpenAI — pick via
  `LLM_PROVIDER` in `.env`) since Anthropic has no persistent free API
  tier while Gemini's free tier is more usable for ongoing experimentation.
  **Verification is uneven across providers**: this sandbox's network
  allowlist only permits `api.anthropic.com` — confirmed reachable via
  `curl` (got a real HTTP 401, not a connection failure), and both the
  "no key" and "invalid key" fallback paths were tested end-to-end for
  real. `generativelanguage.googleapis.com` (Gemini) and `api.openai.com`
  are both blocked in this sandbox, so those two paths are written
  correctly against each provider's documented REST shape but were
  **never actually round-tripped** — test them yourself before relying on
  them. **No valid key for any provider was available**, so no real LLM
  synthesis or LLM-backed critique was ever observed — only the failure/
  fallback paths were verified.

Everything else — state management, evidence tracking, graph wiring,
control flow, **the critic feedback loop actually looping and
terminating**, the TF-IDF retriever, checkpointing — is real, tested
logic, not pseudocode. Every phase in this build was run and its output
inspected before moving to the next; nothing was written on faith.

## Project structure

```
research_agent/
├── main.py                  # CLI: run a query, --list-runs, --resume <id>
├── state.py                 # ResearchState + Evidence dataclasses
├── graph.py                 # graph wiring: fan-out → rag → analyst → critic ⟲
├── graph_engine.py           # stdlib stand-in for LangGraph's StateGraph
├── checkpoint.py              # save/load full run state as JSON
├── config.py                  # env-based settings (API key, model, revision cap)
├── llm.py                     # REAL Anthropic API wrapper via requests
├── .env.example                # template for ANTHROPIC_API_KEY
├── requirements.txt
├── agents/
│   ├── orchestrator.py       # builds initial plan; minimal plan on revision passes
│   ├── market_agent.py       # MCP: get_stock_quote, get_company_profile
│   ├── news_agent.py         # MCP: get_news
│   ├── search_agent.py       # MCP: web_search; tracks distinct queries run
│   ├── rag_agent.py           # MCP: knowledge_base_search (shared context)
│   ├── analyst.py             # REAL LLM synthesis, [N] citations, template fallback
│   ├── critic.py              # REAL feedback loop: LLM or heuristic critique + router
│   └── report_writer.py      # finalize step: references appendix + notes
├── mcp_server/
│   └── server.py               # tool registry, call_tool/list_tools (mock MCP server)
└── tools/
    ├── mcp_client.py           # unified interface every agent actually calls
    ├── market_tools.py         # live Alpha Vantage market data
    ├── company_tools.py        # live Alpha Vantage company profile
    ├── news_tools.py           # live Finnhub company news (latest year)
    ├── search_tools.py         # live Tavily-backed general web search
    ├── sec_tools.py            # live primary-source SEC EDGAR filings
    ├── knowledge_base.py        # mock document corpus
    └── rag_tools.py             # REAL TF-IDF + cosine similarity retriever
```

## Running it

```bash
cd research_agent
pip install -r requirements.txt   # just `requests`
cp .env.example .env              # then edit .env: set LLM_PROVIDER, its key, and TAVILY_API_KEY

python3 main.py "Give me a report on TSLA earnings and margins"
python3 main.py --list-runs
python3 main.py --resume <run_id>
```

**`.env` is loaded automatically** — `config.py` reads it on import (see
`_load_dotenv()`), so just editing the file is enough. You do NOT need to
`export` anything or set a shell environment variable. (An earlier version
of this README said to `export ANTHROPIC_API_KEY=...` — that's bash-only
syntax that silently does nothing on Windows PowerShell, which caused a
real "why isn't it picking up my key" report. Auto-loading fixes that on
every OS.) If you do set a real environment variable yourself on top of
`.env` (e.g. `$env:GEMINI_API_KEY="..."` in PowerShell, or `export` in
bash), that takes priority over the `.env` file.

Set `ALPHA_VANTAGE_API_KEY`, `FINNHUB_API_KEY`, `SEC_USER_AGENT`, and
`TAVILY_API_KEY` as well to enable all live evidence tools. LLM fallback
still works without an LLM key, but a missing live-data key now stops the
relevant tool rather than silently substituting invented evidence.

## Live web search with Tavily

The general `web_search` tool uses Tavily for every research topic; it has
no Tesla- or ticker-specific behavior. Create a Tavily API key at
https://app.tavily.com and add `TAVILY_API_KEY=...` to `.env`. A missing or
failed Tavily request stops the run instead of inserting invented search
results, so report citations always point to real result URLs. Market,
company, news, and SEC filing tools are connected to their live providers
as described above. The local RAG corpus remains sample content.

## The critic feedback loop — how it actually works

```
              ┌─ market_agent ─┐
    orchestrator ─ news_agent  ─┼─→ rag_retrieval → analyst → critic
              └─ search_agent ─┘                              │  │
                     ▲                                        │  │
                     └──────────── "revise" ────────────────┘  │
                                                                 "done"
                                                                  ↓
                                                            report_writer → END
```

- **Critic's check:** LLM-backed sufficiency verdict when an API key is
  configured; otherwise a deterministic heuristic (requires ≥2 distinct
  search queries before accepting the report — always resolvable in one
  round).
- **On revise:** the orchestrator builds a *minimal* plan containing only
  the critic's follow-up query. `market_agent`/`news_agent` see nothing
  matching their prefix in that plan and become no-ops — no duplicate
  data is re-fetched.
- **Bounded by `MAX_REVISIONS`** (default 2, in `config.py`). If the cap
  is hit with a gap still open, the report ships anyway with the
  unresolved concern stated plainly in the Notes section — never hidden.
- **Verified, not assumed:** ran with no API key (heuristic path — looped
  exactly once, then terminated), with `MAX_REVISIONS=0` (cap-reached path
  — terminated immediately with the gap surfaced), and with an invalid
  API key (LLM critique correctly fails over to the heuristic).

## Checkpointing

`checkpoint.py` saves the complete final state (including all evidence,
the draft, revision count, everything) to `checkpoints/<run_id>.json`
after every run. `--resume <run_id>` reloads and reprints that saved
report without rerunning the pipeline.

**This is a stub, not a full LangGraph checkpointer**: it saves once at
the end of a run, not after every node — so it can't resume mid-graph
after a crash between, say, the fan-out and the analyst. See
`checkpoint.py`'s docstring for exactly what a full implementation would
add.

## Known simplifications (honest list)

- `graph_engine.py`'s fan-out executes sequentially, not via real
  threads/async — matters most now since analyst/critic each make a real
  (slower) network call, and a revision round doubles that cost.
- `tools/mcp_client.py` talks to `mcp_server/server.py` in-process, not
  over a real stdio/SSE transport.
- RAG retrieval is TF-IDF (keyword-based), not embeddings — see
  `tools/rag_tools.py`'s docstring for the swap to a real vector store.
- `llm.py` uses raw `requests` against the REST API instead of the
  official `anthropic` SDK (PyPI wasn't reachable to install it here) —
  functionally equivalent; see that file's docstring for the SDK swap.
- Ticker extraction (`orchestrator.py`) is a regex + a small hardcoded
  company-name-to-ticker map, not real NER — covers the ~11 companies
  listed in `_COMPANY_NAME_TO_TICKER` (Apple, Amazon, Tesla, etc.) by
  matching plain-English names in the query, in addition to explicit
  tickers. A query naming a company outside that list still falls back
  to AAPL — a real fix would use an NER model or a ticker-lookup API.
  (This was caught via a real bug report: an early version only matched
  tickers and silently defaulted every unrecognized query to AAPL,
  producing a wrong-but-confident report for "earnings of amazon.")
- Checkpointing saves only the final state, not per-node (see above).
- `agents/critic.py`'s revision-round follow-up query prefers a RAG doc
  tagged with the ticker actually being researched, using the max score
  across all retrieved docs — not just the first one inserted into
  `state.rag_context`. This was also caught via a real bug report: a
  genuine TF-IDF score TIE (0.1195 both) between a generic doc and a
  company-specific one was broken by list order rather than relevance,
  picking the generic doc and producing a low-value follow-up search
  with no company connection at all. Fixed by checking each doc's
  `tags` against the round's current symbols, falling back to
  highest-score-overall only when no tagged doc exists.

## Architecture (as built — matches the original diagram's intent, fixed)

```
User
 → Orchestrator (shared state; receives the critic's revise-loop)
     ├─ Market Agent   ─┐
     ├─ News Agent      ├─→ MCP tool layer → live APIs
     └─ Search Agent    ┘
 → Evidence Store (facts + sources, shared across every agent)
 → Knowledge Base / RAG (its own shared node, not nested under one branch)
 → Analyst (LLM synthesis with [N] citations, template fallback)
 → Critic (checks sufficiency; loops back to Orchestrator, bounded)
 → Report Writer (references appendix + notes)
 → Final Report (+ checkpoint saved to disk)
```

This resolves every issue raised in the original architecture review:
duplicate/parallel tool layers → one MCP layer; RAG nested under one
branch → its own shared node; no feedback loop → a real, bounded,
tested one; no shared state → `ResearchState` threaded through every
node; no source attribution → every claim is an `Evidence` object with
`source`/`url`, cited by number end to end.
