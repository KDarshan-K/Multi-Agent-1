"""
MCP client.

Every agent calls tools through THIS module — never by importing
market_tools/news_tools/search_tools/company_tools directly. That's the
whole point of this phase: one consistent tool-calling interface for every
agent, matching the "unified MCP tool layer" from the architecture review.

CURRENT IMPLEMENTATION: talks directly (in-process, no network) to
`mcp_server/server.py`'s `call_tool` function. This sandbox can't run a
real MCP server as a separate process (no network for stdio/SSE transport
setup in this environment), so this is a same-process stand-in.

TO GO LIVE with a real MCP server + the real `mcp` SDK:
    from mcp import ClientSession
    from mcp.client.stdio import stdio_client

    async def call_tool(name: str, **kwargs):
        async with stdio_client(server_params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                result = await session.call_tool(name, arguments=kwargs)
                return result

The function signature `call_tool(name, **kwargs)` below is written so
that swap is a drop-in replacement — nothing in agents/ needs to change,
only this file's internals (and it'll need to become async, which the
graph engine would need to support — noted as a phase 7+ consideration).
"""

from __future__ import annotations
from typing import Any
from mcp_server import server as _mock_server


def call_tool(name: str, **kwargs: Any) -> Any:
    return _mock_server.call_tool(name, arguments=kwargs)


def list_tools() -> list[dict]:
    return _mock_server.list_tools()
