"""Provider-agnostic JSON LLM calls over Gemini, Groq, and OpenRouter.

Tiers ('fast', 'smart', 'adversary') map to (provider, model) in config.TIERS.
Groq and OpenRouter are called over their OpenAI-compatible REST APIs with httpx.
Gemini is called via the google-genai SDK.

Retry policy: only 429s and 5xx errors get retried. 404/401 raise immediately
(bad model ID or bad key; retrying won't help and wastes the token budget).
"""
import asyncio
import json
import logging
import os
import re

import httpx

from . import config

logger = logging.getLogger(__name__)

_clients = {}
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

# Status codes that are worth retrying
_RETRYABLE = {429, 500, 502, 503, 504}


def _parse(txt):
    m = re.search(r"\{.*\}", txt or "", re.S)
    return json.loads(m.group(0)) if m else {}


class RateLimited(Exception):
    def __init__(self, wait):
        self.wait = wait


class NoJsonReturned(Exception):
    """Raised (or recorded) when every provider attempt returned no usable JSON."""


async def _gemini(system, user, model, max_tokens):
    from google import genai
    from google.genai import types
    c = _clients.setdefault("gemini", genai.Client())  # reads GEMINI_API_KEY / GOOGLE_API_KEY
    try:
        r = await c.aio.models.generate_content(
            model=model, contents=user,
            config=types.GenerateContentConfig(system_instruction=system, response_mime_type="application/json",
                                               max_output_tokens=max_tokens * 2))  # headroom for thinking
        return r.text
    except Exception as e:
        msg = str(e)
        if "429" in msg or "RESOURCE_EXHAUSTED" in msg:
            import re as _re
            m = _re.search(r"retry in ([\d.]+)s", msg, _re.I)
            wait = float(m.group(1)) if m else 15.0
            raise RateLimited(min(wait, 60))
        raise


def _openai_compat_body(system, user, model, max_tokens):
    """Shared request body for any OpenAI-compatible endpoint."""
    return {
        "model": model,
        "temperature": 0.2,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "response_format": {"type": "json_object"},
        "max_tokens": max_tokens * 2,
    }


async def _call_openai_compat(c, url, headers, body, provider_name):
    """Shared HTTP call for Groq / OpenRouter (both OpenAI-compatible)."""
    r = await c.post(url, headers=headers, json=body)
    if r.status_code == 429:
        wait = min(float(r.headers.get("retry-after", 5)), 60)
        raise RateLimited(wait)
    if r.status_code not in _RETRYABLE and r.status_code >= 400:
        # 404 (bad model ID), 401 (bad key) — fail fast, don't retry
        r.raise_for_status()
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"]


async def _groq(system, user, model, max_tokens):
    key = os.getenv("GROQ_API_KEY")
    if not key:
        raise RuntimeError("GROQ_API_KEY is not set")
    body = _openai_compat_body(system, user, model, max_tokens)
    body["max_completion_tokens"] = body.pop("max_tokens")  # Groq uses max_completion_tokens
    if model.startswith("openai/gpt-oss"):
        body["reasoning_effort"] = "low"
    c = _clients.setdefault("groq_http", httpx.AsyncClient(timeout=90))
    return await _call_openai_compat(c, GROQ_URL, {"Authorization": f"Bearer {key}"}, body, "groq")


async def _openrouter(system, user, model, max_tokens):
    key = os.getenv("OPENROUTER_API_KEY")
    if not key:
        raise RuntimeError("OPENROUTER_API_KEY is not set")
    body = _openai_compat_body(system, user, model, max_tokens)
    headers = {
        "Authorization": f"Bearer {key}",
        "HTTP-Referer": "https://github.com/narrative-tracer",  # OpenRouter asks for a referer
        "X-Title": "NarrativeTracer",
    }
    c = _clients.setdefault("openrouter_http", httpx.AsyncClient(timeout=90))
    return await _call_openai_compat(c, OPENROUTER_URL, headers, body, "openrouter")


_CALLS = {"gemini": _gemini, "groq": _groq, "openrouter": _openrouter}

# Default fallback models per provider (used when switching providers in _plan)
_FALLBACK_MODELS = {
    "gemini": "gemini-3.8-flash",
    "groq": "openai/gpt-oss-120b",
    "openrouter": "google/gemini-2.0-flash-exp:free",
}


def _has_key(provider):
    if provider == "groq":
        return bool(os.getenv("GROQ_API_KEY"))
    if provider == "openrouter":
        return bool(os.getenv("OPENROUTER_API_KEY"))
    return bool(os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY"))


def _plan(tier, prompt_chars):
    """Ordered [(provider, model), ...] to try for this call."""
    prov, model = config.TIERS.get(tier, config.TIERS["fast"])
    plan = [(prov, model)]

    if config.LLM_FALLBACK:
        # Build a fallback list from the other configured providers that have keys
        for fb_prov in config.PROVIDERS:
            if fb_prov == prov:
                continue
            if not _has_key(fb_prov):
                continue
            # Use the tier's model if the fallback provider is already configured for it,
            # otherwise use its default fallback model
            fb_model = _FALLBACK_MODELS.get(fb_prov, "")
            plan.append((fb_prov, fb_model))

    # Groq free tier has a tiny tokens-per-minute budget: deprioritise it for big prompts.
    if prov == "groq" and prompt_chars > config.GROQ_MAX_INPUT_CHARS and len(plan) > 1:
        plan.reverse()

    return plan


def _is_retryable_http(exc):
    """Return True only for errors worth retrying (429 / 5xx / network / SDK transient)."""
    if isinstance(exc, RateLimited):
        return True
    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.status_code in _RETRYABLE
    # Network-level errors (connect timeout, read timeout…) are always retryable
    if isinstance(exc, (httpx.NetworkError, httpx.TimeoutException)):
        return True
    # Gemini SDK wraps 5xx as ServerError – always retryable
    exc_type = type(exc).__name__
    if exc_type in ("ServerError", "ServiceUnavailable", "InternalServerError"):
        return True
    # Any other exception whose string contains 5xx status codes
    msg = str(exc)
    for code in ("500", "502", "503", "504"):
        if code in msg:
            return True
    return False


async def _try(provider, model, system, user, max_tokens):
    call = _CALLS[provider]
    for attempt in range(3):
        try:
            return _parse(await call(system, user, model, max_tokens))
        except json.JSONDecodeError:
            continue
        except RateLimited as e:
            if attempt == 2:
                raise
            logger.warning("Rate-limited by %s (%s); sleeping %.1fs", provider, model, e.wait)
            await asyncio.sleep(e.wait)
        except Exception as e:
            if not _is_retryable_http(e):
                # 404, 401 etc. – no point retrying
                logger.error("Non-retryable error from %s (%s): %s", provider, model, e)
                raise
            if attempt == 2:
                raise
            await asyncio.sleep(2 ** attempt)
    return {}


async def ask_json(system: str, user: str, tier: str = "fast", max_tokens: int = 4000) -> dict:
    """Return a parsed JSON object.

    Raises NoJsonReturned when every provider returned empty/un-parseable JSON.
    Callers that catch broadly will still get a meaningful error rather than a
    silent {} that propagates bad data through the pipeline.
    """
    sys_prompt = system + "\nRespond with a single valid JSON object and nothing else."
    last = None
    for provider, model in _plan(tier, len(sys_prompt) + len(user)):
        try:
            out = await _try(provider, model, sys_prompt, user, max_tokens)
            if out:
                return out
            logger.warning("ask_json: %s/%s returned empty JSON for tier=%s", provider, model, tier)
        except Exception as e:
            logger.error("ask_json: %s/%s failed: %s", provider, model, e)
            last = e
    msg = f"ask_json tier={tier}: no provider returned usable JSON (last error: {last})"
    logger.error(msg)
    raise NoJsonReturned(msg)
