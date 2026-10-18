"""
Mock MCP server.

This is the unified tool layer that fixes the "two parallel tool layers"
problem from the original architecture: market data, news, search, and
company/fundamentals data all sit behind ONE interface here, instead of
each agent importing a different tool module directly.

WHY "MOCK": a real MCP server runs as a separate process and talks to
clients over stdio or SSE using the Model Context Protocol's JSON-RPC-based
wire format (see https://modelcontextprotocol.io). Standing up a real
server + transport needs the `mcp` Python SDK and, for network-backed
tools, actual internet access — neither is available in this sandbox.

This module instead implements the two calls that matter
(`list_tools`, `call_tool`) in-process, with the exact same *shape* a real
MCP server would expose: each tool has a name, a description, an input
schema, and `call_tool` dispatches by name and returns structured content.

TO GO LIVE with a real MCP server:
    1. `pip install mcp`
    2. Wrap each function below in an `@server.tool()` decorator (or the
       SDK's equivalent registration call) inside a `Server` instance from
       `mcp.server`.
    3. Run it as its own process (stdio or SSE transport).
    4. Point `tools/mcp_client.py` at that process instead of importing
       this module directly — see that file's docstring for specifics.

Nothing in agents/ needs to change when you do this migration; they only
ever talk to `tools/mcp_client.py`.
"""

from __future__ import annotations
from typing import Any, Callable

from tools import (company_tools, market_tools, news_tools, rag_tools,
                   search_tools, sec_tools)


class ToolSpec:
    def __init__(self, name: str, description: str,
                 input_schema: dict, fn: Callable[..., Any]):
        self.name = name
        self.description = description
        self.input_schema = input_schema
        self.fn = fn


_TOOLS: dict[str, ToolSpec] = {}


def _register(name: str, description: str, input_schema: dict,
              fn: Callable[..., Any]) -> None:
    _TOOLS[name] = ToolSpec(name, description, input_schema, fn)


_register(
    name="get_stock_quote",
    description="Get the current price, % change, and volume for a stock ticker.",
    input_schema={"symbol": "string"},
    fn=lambda args: market_tools.get_quote(args["symbol"]),
)

_register(
    name="search_symbol",
    description="Resolve a free-text company name to candidate ticker "
                 "symbols, ranked by relevance.",
    input_schema={"keywords": "string"},
    fn=lambda args: market_tools.search_symbol(args["keywords"]),
)

_register(
    name="get_company_profile",
    description="Get sector, market cap, and a description for a company by ticker.",
    input_schema={"symbol": "string"},
    fn=lambda args: company_tools.get_company_profile(args["symbol"]),
)

_register(
    name="get_sec_filings",
    description="Get recent primary-source SEC filings for a U.S. ticker.",
    input_schema={"symbol": "string", "limit": "integer (optional)"},
    fn=lambda args: sec_tools.get_recent_filings(
        args["symbol"], limit=args.get("limit", 8)
    ),
)

_register(
    name="get_news",
    description="Get recent news headlines for a stock ticker.",
    input_schema={"symbol": "string", "limit": "integer (optional)"},
    fn=lambda args: news_tools.get_news(
        args["symbol"], limit=args.get("limit", 3)
    ),
)

_register(
    name="web_search",
    description="Run a general web search for a free-text query.",
    input_schema={"query": "string", "max_results": "integer (optional)"},
    fn=lambda args: search_tools.search(
        args["query"], max_results=args.get("max_results", 3)
    ),
)

_register(
    name="knowledge_base_search",
    description="Search the internal knowledge base of past reports, "
                 "filings notes, and analyst commentary for context "
                 "relevant to a query. Uses TF-IDF relevance ranking.",
    input_schema={"query": "string", "top_k": "integer (optional)"},
    fn=lambda args: rag_tools.search(
        args["query"], top_k=args.get("top_k", 3)
    ),
)


def list_tools() -> list[dict]:
    """Mirrors MCP's tools/list response shape."""
    return [
        {"name": t.name, "description": t.description,
         "input_schema": t.input_schema}
        for t in _TOOLS.values()
    ]


def call_tool(name: str, arguments: dict) -> Any:
    """Mirrors MCP's tools/call response shape (dispatch by name)."""
    tool = _TOOLS.get(name)
    if tool is None:
        raise ValueError(
            f"Unknown tool '{name}'. Available: {list(_TOOLS.keys())}"
        )
    return tool.fn(arguments)
