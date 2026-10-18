"""
LLM API wrapper — supports Anthropic, Gemini, and OpenAI.

Which provider gets called is controlled by config.LLM_PROVIDER ("anthropic"
/ "gemini" / "openai"). All three implementations use raw `requests` against
each provider's REST API rather than an official SDK, since this project
was built in a sandbox with no PyPI access (see README) — this is
functionally equivalent, just without the convenience wrapper.

VERIFICATION STATUS (be aware of this before trusting a provider blindly):
    - Anthropic path: REAL round-trip verified against the live API in this
      sandbox (confirmed reachable; got a proper 401 with an invalid key,
      not a connection failure).
    - Gemini and OpenAI paths: written correctly against each provider's
      documented REST API shape, but this sandbox's network allowlist
      blocks generativelanguage.googleapis.com and api.openai.com, so
      NEITHER was actually round-tripped here. Test these yourself on a
      machine with normal internet access before relying on them. If you
      hit an error, the shape of the request/response parsing is the
      first thing to double check against each provider's current docs.

REQUIRES: the API key for whichever provider is selected (see
config.active_api_key()). Without it, call_llm() raises
LLMNotConfiguredError immediately — no network call attempted. Callers
(agents/analyst.py, agents/critic.py) catch that and fall back to
template/heuristic behavior rather than crashing.
"""

from __future__ import annotations
import requests
import config


class LLMNotConfiguredError(RuntimeError):
    """Raised when no API key is available for the selected provider."""


class LLMCallError(RuntimeError):
    """Raised when the API call itself fails (auth, rate limit, network)."""


def call_llm(system: str, user_message: str, max_tokens: int | None = None) -> str:
    if not config.active_api_key():
        raise LLMNotConfiguredError(
            f"No API key set for LLM_PROVIDER={config.LLM_PROVIDER!r}. "
            f"Copy .env.example to .env, fill in the matching key "
            f"(ANTHROPIC_API_KEY / GEMINI_API_KEY / OPENAI_API_KEY), and "
            f"export it. Falling back to template-based synthesis."
        )

    max_tokens = max_tokens or config.ANALYST_MAX_TOKENS

    if config.LLM_PROVIDER == "anthropic":
        return _call_anthropic(system, user_message, max_tokens)
    elif config.LLM_PROVIDER == "gemini":
        return _call_gemini(system, user_message, max_tokens)
    elif config.LLM_PROVIDER == "openai":
        return _call_openai(system, user_message, max_tokens)
    else:
        raise LLMNotConfiguredError(
            f"Unknown LLM_PROVIDER={config.LLM_PROVIDER!r}. "
            f"Must be 'anthropic', 'gemini', or 'openai'."
        )


def _call_anthropic(system: str, user_message: str, max_tokens: int) -> str:
    headers = {
        "Content-Type": "application/json",
        "x-api-key": config.ANTHROPIC_API_KEY,
        "anthropic-version": config.ANTHROPIC_API_VERSION,
    }
    payload = {
        "model": config.ANTHROPIC_MODEL,
        "max_tokens": max_tokens,
        "system": system,
        "messages": [{"role": "user", "content": user_message}],
    }
    resp = _post(config.ANTHROPIC_API_URL, headers, payload)
    data = resp.json()
    blocks = [b["text"] for b in data.get("content", []) if b.get("type") == "text"]
    return "\n".join(blocks)


def _call_gemini(system: str, user_message: str, max_tokens: int) -> str:
    headers = {"Content-Type": "application/json"}
    payload = {
        "system_instruction": {"parts": [{"text": system}]},
        "contents": [{"role": "user", "parts": [{"text": user_message}]}],
        "generationConfig": {"maxOutputTokens": max_tokens},
    }
    url = f"{config.GEMINI_API_URL}?key={config.GEMINI_API_KEY}"
    resp = _post(url, headers, payload)
    data = resp.json()
    try:
        candidate = data["candidates"][0]
        parts = candidate["content"]["parts"]
        return "\n".join(p.get("text", "") for p in parts)
    except (KeyError, IndexError) as e:
        raise LLMCallError(
            f"Unexpected Gemini response shape: {data}"
        ) from e


def _call_openai(system: str, user_message: str, max_tokens: int) -> str:
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {config.OPENAI_API_KEY}",
    }
    payload = {
        "model": config.OPENAI_MODEL,
        "max_tokens": max_tokens,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user_message},
        ],
    }
    resp = _post(config.OPENAI_API_URL, headers, payload)
    data = resp.json()
    try:
        return data["choices"][0]["message"]["content"]
    except (KeyError, IndexError) as e:
        raise LLMCallError(
            f"Unexpected OpenAI response shape: {data}"
        ) from e


def _post(url: str, headers: dict, payload: dict) -> requests.Response:
    try:
        resp = requests.post(url, headers=headers, json=payload, timeout=90)
    except requests.RequestException as e:
        raise LLMCallError(f"Network error calling {url}: {e}") from e

    if resp.status_code != 200:
        raise LLMCallError(f"{url} returned {resp.status_code}: {resp.text}")

    return resp
