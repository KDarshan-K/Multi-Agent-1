"""
Configuration.

Reads settings from environment variables so no secrets get committed to
the project. Copy .env.example to .env and fill in keys for whichever
provider(s) you want to use.

AUTO-LOADS .env from the project root (see _load_dotenv() below) — no
shell `export`/`$env:` command needed. This was a real reported issue:
`export` is bash-only syntax and silently does nothing on Windows
PowerShell (the correct PowerShell equivalent is `$env:VAR="value"`), so
a person on Windows following bash instructions would edit .env, see no
effect, and have no obvious reason why. Auto-loading removes that
platform-dependent step entirely. Real OS environment variables (if you
set one directly, on any OS) still take priority over .env — see
_load_dotenv()'s use of setdefault.

LLM_PROVIDER selects which one llm.py actually calls: "anthropic",
"gemini", or "openai". Only that provider's key needs to be set. If it's
missing or the call fails for any reason, agents/analyst.py and
agents/critic.py both fall back to template/heuristic behavior — the
pipeline never crashes for lack of a key.
"""

from __future__ import annotations
import os


def _load_dotenv(path: str | None = None) -> None:
    """Minimal .env loader — no python-dotenv dependency needed.

    Resolves .env relative to THIS file's directory (not the current
    working directory), so it's found whether you run `python main.py`
    from inside research_agent/ or `python research_agent/main.py` from
    one level up.

    Only sets a variable if it isn't already present in the real
    environment (os.environ.setdefault), so an actual `export`/`$env:`
    you set yourself always wins over the .env file. Silently does
    nothing if .env doesn't exist yet (e.g. before someone's copied
    .env.example) — that's expected, not an error.
    """
    if path is None:
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
    if not os.path.exists(path):
        return
    with open(path, encoding="utf-8-sig") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            os.environ.setdefault(key, value)


_load_dotenv()

LLM_PROVIDER = os.environ.get("LLM_PROVIDER", "anthropic").lower()

# --- Anthropic ---
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
ANTHROPIC_MODEL = os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-5")
ANTHROPIC_API_URL = "https://api.anthropic.com/v1/messages"
ANTHROPIC_API_VERSION = "2023-06-01"

# --- Gemini (Google AI Studio) ---
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")
GEMINI_API_URL = (
    f"https://generativelanguage.googleapis.com/v1beta/models/"
    f"{GEMINI_MODEL}:generateContent"
)

# --- OpenAI ---
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
OPENAI_API_URL = "https://api.openai.com/v1/chat/completions"

# --- Web search (Tavily) ---
# This is deliberately separate from LLM_PROVIDER: the research agents use
# Tavily to find source material regardless of which LLM writes the report.
TAVILY_API_KEY = os.environ.get("TAVILY_API_KEY", "")
TAVILY_SEARCH_URL = "https://api.tavily.com/search"

# --- Market, company, news, and filings data ---
ALPHA_VANTAGE_API_KEY = os.environ.get("ALPHA_VANTAGE_API_KEY", "")
ALPHA_VANTAGE_API_URL = "https://www.alphavantage.co/query"
FINNHUB_API_KEY = os.environ.get("FINNHUB_API_KEY", "")
FINNHUB_API_URL = "https://finnhub.io/api/v1/company-news"
# SEC requests that automated clients identify themselves with contact details.
SEC_USER_AGENT = os.environ.get("SEC_USER_AGENT", "")
SEC_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
SEC_SUBMISSIONS_URL = "https://data.sec.gov/submissions"
SEC_ARCHIVES_URL = "https://www.sec.gov/Archives/edgar/data"

ANALYST_MAX_TOKENS = int(os.environ.get("ANALYST_MAX_TOKENS", "1500"))

# Critic loop settings
MAX_REVISIONS = int(os.environ.get("MAX_REVISIONS", "2"))


def active_api_key() -> str:
    """Returns the key for whichever provider is selected, for easy
    'is anything configured at all' checks in llm.py."""
    return {
        "anthropic": ANTHROPIC_API_KEY,
        "gemini": GEMINI_API_KEY,
        "openai": OPENAI_API_KEY,
    }.get(LLM_PROVIDER, "")
