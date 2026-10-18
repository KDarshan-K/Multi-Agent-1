"""
Market agent node for official LangGraph.
"""

from __future__ import annotations
from state import ResearchState, Evidence
from tools import mcp_client


def market_agent_node(state: ResearchState | dict) -> dict:
    if isinstance(state, dict):
        state = ResearchState.from_dict(state)

    symbols = [
        step.split(":", 1)[1]
        for step in state.plan
        if step.startswith("fetch_market_data:")
    ]

    market_data = []
    evidence = []
    data_gaps = []

    for symbol in symbols:
        try:
            quote = mcp_client.call_tool("get_stock_quote", symbol=symbol)
            market_data.append(quote)
            direction = "up" if quote["change_pct"] >= 0 else "down"
            claim = (
                f"{quote['symbol']} is trading at ${quote['price']:.2f}, "
                f"{direction} {abs(quote['change_pct']):.2f}% "
                f"on volume of {quote['volume']:,}"
            )
            evidence.append(Evidence(
                source=quote["source"],
                claim=claim,
                raw_data=quote,
                url=quote.get("url"),
            ))
        except Exception as e:
            data_gaps.append(f"Stock quote for {symbol} unavailable: {e}")

        try:
            profile = mcp_client.call_tool("get_company_profile", symbol=symbol)
            market_data.append(profile)
            profile_claim = (
                f"{profile['symbol']} ({profile['name']}) operates in the "
                f"{profile['sector']} sector with a market cap of "
                f"${profile['market_cap']:,}"
            )
            evidence.append(Evidence(
                source=profile["source"],
                claim=profile_claim,
                raw_data=profile,
                url=profile.get("url"),
            ))
        except Exception as e:
            data_gaps.append(f"Company profile for {symbol} unavailable: {e}")

        try:
            filings = mcp_client.call_tool("get_sec_filings", symbol=symbol)
            for filing in filings:
                evidence.append(Evidence(
                    source="SEC EDGAR",
                    claim=(f"{filing['symbol']} filed {filing['form']} on "
                           f"{filing['filing_date']}."),
                    raw_data=filing,
                    url=filing["url"],
                ))
        except Exception as e:
            data_gaps.append(f"SEC filings for {symbol} unavailable: {e}")

    return {
        "market_data": market_data,
        "evidence": evidence,
        "data_gaps": data_gaps,
    }
