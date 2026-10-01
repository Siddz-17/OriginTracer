"""Provider-agnostic JSON LLM calls over Gemini and Groq.

Tiers ('fast', 'smart', 'adversary') map to (provider, model) in config.TIERS.
Groq is called over its OpenAI-compatible REST API with httpx (no extra SDK).
"""
import asyncio
import json
import os
import re

import httpx

from . import config

_clients = {}
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"


def _parse(txt):
    m = re.search(r"\{.*\}", txt or "", re.S)
    return json.loads(m.group(0)) if m else {}


class RateLimited(Exception):
    def __init__(self, wait):
        self.wait = wait


async def _gemini(system, user, model, max_tokens):
    from google import genai
    from google.genai import types
    c = _clients.setdefault("gemini", genai.Client())  # reads GEMINI_API_KEY / GOOGLE_API_KEY
    r = await c.aio.models.generate_content(
        model=model, contents=user,
        config=types.GenerateContentConfig(system_instruction=system, response_mime_type="application/json",
                                           max_output_tokens=max_tokens * 2))  # headroom for 2.5 thinking
    return r.text


async def _groq(system, user, model, max_tokens):
    key = os.getenv("GROQ_API_KEY")
    if not key:
        raise RuntimeError("GROQ_API_KEY is not set")
    body = {"model": model, "temperature": 0.2,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "response_format": {"type": "json_object"},
            "max_completion_tokens": max_tokens * 2}  # reasoning models spend tokens before the answer
    if model.startswith("openai/gpt-oss"):
        body["reasoning_effort"] = "low"
    c = _clients.setdefault("groq_http", httpx.AsyncClient(timeout=90))
    r = await c.post(GROQ_URL, headers={"Authorization": f"Bearer {key}"}, json=body)
    if r.status_code == 429:
        raise RateLimited(min(float(r.headers.get("retry-after", 5)), 60))
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"]


_CALLS = {"gemini": _gemini, "groq": _groq}


def _has_key(provider):
    if provider == "groq":
        return bool(os.getenv("GROQ_API_KEY"))
    return bool(os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY"))


def _plan(tier, prompt_chars):
    """Ordered [(provider, model), ...] to try for this call."""
    prov, model = config.TIERS.get(tier, config.TIERS["fast"])
    other = "gemini" if prov == "groq" else "groq"
    other_model = config.TIERS["fast"][1] if config.TIERS["fast"][0] == other else \
        {"gemini": "gemini-3.8-flash", "groq": "openai/gpt-oss-120b"}[other]
    plan = [(prov, model)]
    if config.LLM_FALLBACK and _has_key(other):
        plan.append((other, other_model))
    # Groq free tier has a tiny tokens-per-minute budget: send big prompts to Gemini first.
    if prov == "groq" and prompt_chars > config.GROQ_MAX_INPUT_CHARS and len(plan) > 1:
        plan.reverse()
    return plan


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
            await asyncio.sleep(e.wait)
        except Exception:  # transient network / 5xx
            if attempt == 2:
                raise
            await asyncio.sleep(2 ** attempt)
    return {}


async def ask_json(system: str, user: str, tier: str = "fast", max_tokens: int = 4000) -> dict:
    """Return a parsed JSON object ({} if no model produced valid JSON)."""
    sys_prompt = system + "\nRespond with a single valid JSON object and nothing else."
    last = None
    for provider, model in _plan(tier, len(sys_prompt) + len(user)):
        try:
            out = await _try(provider, model, sys_prompt, user, max_tokens)
            if out:
                return out
        except Exception as e:  # try the fallback provider, if any
            last = e
    if last:
        raise last
    return {}
