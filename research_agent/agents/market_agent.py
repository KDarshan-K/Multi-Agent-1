"""
Market agent node.

Reads the plan for `fetch_market_data:<SYMBOL>` entries, calls the stock
quote, company profile, and SEC filings tools via the MCP client for
each, stores raw results in state.market_data, and logs each fact as
Evidence so the final report can cite its source.

Calls tools through tools/mcp_client.py (the unified MCP-style layer) —
never imports market_tools/company_tools/sec_tools directly.

Each of the three live API calls is wrapped individually: since these now
hit real rate-limited APIs (Alpha Vantage's free tier is a strict 25
requests/day, 5/minute — 2 calls per symbol here means that limit is easy
to hit), any single failure (rate limit, bad ticker, network blip) is
caught and recorded in state.data_gaps rather than crashing the whole
pipeline. A missing quote for one symbol shouldn't prevent a report from
using whatever data *did* come back for the others.
"""

from __future__ import annotations
from state import ResearchState
from tools import mcp_client


def market_agent_node(state: ResearchState) -> ResearchState:
    symbols = [
        step.split(":", 1)[1]
        for step in state.plan
        if step.startswith("fetch_market_data:")
    ]

    for symbol in symbols:
        try:
            quote = mcp_client.call_tool("get_stock_quote", symbol=symbol)
            state.market_data.append(quote)
            direction = "up" if quote["change_pct"] >= 0 else "down"
            claim = (
                f"{quote['symbol']} is trading at ${quote['price']:.2f}, "
                f"{direction} {abs(quote['change_pct']):.2f}% "
                f"on volume of {quote['volume']:,}"
            )
            state.add_evidence(
                source=quote["source"],
                claim=claim,
                raw_data=quote,
                url=quote.get("url"),
            )
        except Exception as e:
            state.data_gaps.append(f"Stock quote for {symbol} unavailable: {e}")

        try:
            profile = mcp_client.call_tool("get_company_profile", symbol=symbol)
            state.market_data.append(profile)
            profile_claim = (
                f"{profile['symbol']} ({profile['name']}) operates in the "
                f"{profile['sector']} sector with a market cap of "
                f"${profile['market_cap']:,}"
            )
            state.add_evidence(
                source=profile["source"],
                claim=profile_claim,
                raw_data=profile,
                url=profile.get("url"),
            )
        except Exception as e:
            state.data_gaps.append(f"Company profile for {symbol} unavailable: {e}")

        try:
            filings = mcp_client.call_tool("get_sec_filings", symbol=symbol)
            for filing in filings:
                state.add_evidence(
                    source="SEC EDGAR",
                    claim=(f"{filing['symbol']} filed {filing['form']} on "
                           f"{filing['filing_date']}."),
                    raw_data=filing,
                    url=filing["url"],
                )
        except Exception as e:
            state.data_gaps.append(f"SEC filings for {symbol} unavailable: {e}")

    return state
