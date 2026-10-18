from agents.orchestrator import _extract_symbols, _candidate_company_name


def test_extract_explicit_symbols():
    symbols = _extract_symbols("How is AAPL and MSFT doing today?")
    assert symbols == ["AAPL", "MSFT"]


def test_extract_company_names():
    symbols = _extract_symbols("Give me a report on Tesla and Costco's margins")
    assert "TSLA" in symbols
    assert "COST" in symbols


def test_ignore_non_ticker_abbreviations():
    symbols = _extract_symbols("How did NVDA perform in FY 2026 Q1?")
    assert "FY" not in symbols
    assert "Q1" not in symbols
    assert "NVDA" in symbols


def test_candidate_company_name_extraction():
    candidate = _candidate_company_name("Give me a report on Snowflake earnings")
    assert "Snowflake" in candidate
