"""Provider-agnostic JSON LLM calls over OpenRouter, Groq, and Gemini.

Tiers ('fast', 'smart', 'adversary') map to (provider, model) in config.TIERS.
Groq and OpenRouter are called over their OpenAI-compatible REST APIs with httpx.
Gemini is called via the google-genai SDK.

Free OpenRouter models are rate limited per model (upstream) and per account (~20/min, 50/day without
credits), and get retired without notice. So one OpenRouter call walks config.OPENROUTER_MODELS:
  * 404 / 400 / empty or non-JSON output / 5xx / timeout / per-model 429  -> next model, no sleep
  * account daily quota exhausted / 401 / 402                            -> stop using OpenRouter
If every entry was rate limited, the whole plan is retried once after the shortest Retry-After.
Requests to one provider are spaced at least <PROVIDER>_MIN_INTERVAL seconds apart.
"""
import asyncio
import json
import logging
import os
import re
import time

import httpx

from . import config

logger = logging.getLogger(__name__)

_clients = {}
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
OPENROUTER_MODELS_URL = "https://openrouter.ai/api/v1/models"

_RETRYABLE = {429, 500, 502, 503, 504}
_MIN_INTERVAL = {"openrouter": float(os.getenv("OPENROUTER_MIN_INTERVAL", "3.2")),
                 "groq": float(os.getenv("GROQ_MIN_INTERVAL", "2.0")), "gemini": 0.0}
_last_call: dict = {}
_locks: dict = {}


class RateLimited(Exception):
    def __init__(self, wait):
        super().__init__(f"rate limited; retry after {wait}s")
        self.wait = wait


class ProviderUnavailable(Exception):
    """The provider cannot serve any more requests now (bad key, no credit, daily quota used up)."""


class ModelUnavailable(Exception):
    """This model cannot serve the request (retired, unsupported params, empty output); try another."""


class NoJsonReturned(Exception):
    """Raised when every provider/model attempt returned no usable JSON."""


# ---------------------------------------------------------------- JSON extraction
_THINK = re.compile(r"<(think|thinking|reasoning)>.*?</\1>", re.S | re.I)


def _parse(txt):
    """Best-effort: the largest JSON object in the reply (reasoning tags and code fences stripped)."""
    if not txt:
        return {}
    txt = _THINK.sub("", txt)
    txt = re.sub(r"```(?:json)?", "", txt)
    try:
        v = json.loads(txt.strip())
        if isinstance(v, dict):
            return v
    except ValueError:
        pass
    dec, best = json.JSONDecoder(), {}
    for m in re.finditer(r"\{", txt):
        try:
            v, end = dec.raw_decode(txt, m.start())
        except ValueError:
            continue
        if isinstance(v, dict) and end - m.start() > len(json.dumps(best)):
            best = v
    return best


# ---------------------------------------------------------------- providers
async def _throttle(provider):
    lock = _locks.setdefault(provider, asyncio.Lock())
    async with lock:
        wait = _last_call.get(provider, 0) + _MIN_INTERVAL.get(provider, 0) - time.monotonic()
        if wait > 0:
            await asyncio.sleep(wait)
        _last_call[provider] = time.monotonic()


def _http(name):
    c = _clients.get(name)
    if c is None or c.is_closed:
        c = _clients[name] = httpx.AsyncClient(timeout=httpx.Timeout(config.LLM_TIMEOUT, connect=10.0))
    return c


async def _gemini(system, user, model, max_tokens):
    try:
        from google import genai
        from google.genai import types
    except ImportError as e:
        raise ProviderUnavailable(f"google-genai is not installed ({e}); pip install google-genai")
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
            m = re.search(r"retry in ([\d.]+)s", msg, re.I)
            raise RateLimited(min(float(m.group(1)) if m else 15.0, 60))
        if "404" in msg or "NOT_FOUND" in msg:
            raise ModelUnavailable(msg[:200])
        raise


def _openai_compat_body(system, user, model, max_tokens, json_mode=True):
    body = {"model": model, "temperature": 0.1,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "max_tokens": max_tokens * 2}
    if json_mode:
        body["response_format"] = {"type": "json_object"}
    return body


def _error_text(r):
    try:
        e = r.json().get("error") or {}
        return f"{e.get('message', '')} {json.dumps(e.get('metadata') or {})}"[:600]
    except Exception:
        return r.text[:600]


async def _call_openai_compat(c, url, headers, body, provider_name):
    """Shared HTTP call for Groq / OpenRouter (both OpenAI-compatible)."""
    r = await c.post(url, headers=headers, json=body)
    if r.status_code == 429:
        err = _error_text(r)
        if re.search(r"per.?day|daily|free-models-per", err, re.I):
            raise ProviderUnavailable(f"{provider_name} daily free quota exhausted: {err[:200]}")
        try:
            wait = float(r.headers.get("retry-after", 8))
        except ValueError:
            wait = 8.0
        raise RateLimited(min(wait, 60))
    if r.status_code in (401, 402, 403):
        raise ProviderUnavailable(f"{provider_name} HTTP {r.status_code}: {_error_text(r)[:200]}")
    if r.status_code in (400, 404, 413, 422):
        raise ModelUnavailable(f"{provider_name} HTTP {r.status_code}: {_error_text(r)[:300]}")
    r.raise_for_status()
    data = r.json()
    if data.get("error"):  # OpenRouter sometimes reports upstream failures with HTTP 200
        err = data["error"]
        code = err.get("code") if isinstance(err, dict) else None
        msg = err.get("message", "") if isinstance(err, dict) else str(err)
        if code == 429:
            raise RateLimited(8.0)
        raise ModelUnavailable(f"{provider_name} upstream error {code}: {msg[:200]}")
    choices = data.get("choices") or []
    content = (choices[0].get("message") or {}).get("content") if choices else None
    if not content:
        reason = choices[0].get("finish_reason") if choices else "no choices"
        raise ModelUnavailable(f"{provider_name} empty content (finish_reason={reason})")
    return content


async def _groq(system, user, model, max_tokens):
    key = os.getenv("GROQ_API_KEY")
    if not key:
        raise ProviderUnavailable("GROQ_API_KEY is not set")
    body = _openai_compat_body(system, user, model, max_tokens)
    body["max_completion_tokens"] = body.pop("max_tokens")  # Groq uses max_completion_tokens
    if model.startswith("openai/gpt-oss"):
        body["reasoning_effort"] = "low"
    return await _call_openai_compat(_http("groq"), GROQ_URL, {"Authorization": f"Bearer {key}"}, body, "groq")


_or_params: dict = {}  # model id -> supported_parameters (from the public models endpoint)


async def _openrouter_supports(model, param):
    if not _or_params:
        try:
            r = await _http("openrouter").get(OPENROUTER_MODELS_URL, timeout=15.0)
            _or_params.update({m["id"]: set(m.get("supported_parameters") or []) for m in r.json()["data"]})
        except Exception as e:
            logger.debug("openrouter models list failed: %s", e)
            _or_params["__failed__"] = set()
    if model not in _or_params and "__failed__" not in _or_params:
        raise ModelUnavailable(f"openrouter: model {model} is not listed (retired or misspelled)")
    return param in _or_params.get(model, set())


async def _openrouter(system, user, model, max_tokens):
    key = os.getenv("OPENROUTER_API_KEY")
    if not key:
        raise ProviderUnavailable("OPENROUTER_API_KEY is not set")
    # Many free models reject response_format; only send it where the model advertises support.
    body = _openai_compat_body(system, user, model, max_tokens,
                               json_mode=await _openrouter_supports(model, "response_format"))
    if await _openrouter_supports(model, "reasoning"):
        body["reasoning"] = {"exclude": True}  # keep chain-of-thought out of `content`
    headers = {"Authorization": f"Bearer {key}", "HTTP-Referer": "https://github.com/Siddz-17/OriginTracer",
               "X-Title": "OriginTracer"}
    return await _call_openai_compat(_http("openrouter"), OPENROUTER_URL, headers, body, "openrouter")


_CALLS = {"gemini": _gemini, "groq": _groq, "openrouter": _openrouter}


def _has_key(provider):
    if provider == "groq":
        return bool(os.getenv("GROQ_API_KEY"))
    if provider == "openrouter":
        return bool(os.getenv("OPENROUTER_API_KEY"))
    return bool(os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY"))


def _models_for(provider, first=None):
    if provider == "openrouter":
        return list(dict.fromkeys([m for m in [first] + config.OPENROUTER_MODELS if m]))
    return [first or config._DEFAULT_MODEL[provider]]


def _plan(tier, prompt_chars):
    """Ordered [(provider, model), ...] to try for this call."""
    prov, model = config.TIERS.get(tier, config.TIERS["smart"])
    groups = [[(prov, m) for m in _models_for(prov, model)]]
    if config.LLM_FALLBACK:
        for fb in config.PROVIDERS:
            if fb != prov and _has_key(fb):
                groups.append([(fb, m) for m in _models_for(fb)])
    # Groq free tier has a tiny tokens-per-minute budget: deprioritise it for big prompts.
    if prompt_chars > config.GROQ_MAX_INPUT_CHARS and len(groups) > 1:
        groq = [g for g in groups if g[0][0] == "groq"]
        groups = [g for g in groups if g[0][0] != "groq"] + groq
    return [pm for g in groups for pm in g]


def _is_retryable_http(exc):
    """Return True only for errors worth retrying on the SAME model (5xx / network)."""
    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.status_code in _RETRYABLE
    if isinstance(exc, (httpx.NetworkError, httpx.TimeoutException)):
        return True
    if type(exc).__name__ in ("ServerError", "ServiceUnavailable", "InternalServerError"):
        return True
    return any(code in str(exc) for code in ("500", "502", "503", "504"))


async def _try(provider, model, system, user, max_tokens):
    """One model. Returns parsed JSON ({} if the reply had none). Raises on errors.
    OpenRouter rate limits are raised immediately (the next free model has its own upstream limit)."""
    call = _CALLS[provider]
    for attempt in range(2):
        await _throttle(provider)
        try:
            return _parse(await call(system, user, model, max_tokens))
        except RateLimited as e:
            if provider == "openrouter" or attempt == 1:
                raise
            logger.warning("Rate-limited by %s (%s); sleeping %.1fs", provider, model, e.wait)
            await asyncio.sleep(e.wait)
        except (ProviderUnavailable, ModelUnavailable):
            raise
        except Exception as e:
            if attempt == 1 or not _is_retryable_http(e) or provider == "openrouter":
                raise
            await asyncio.sleep(2)
    return {}


async def ask_json(system: str, user: str, tier: str = "fast", max_tokens: int = 4000) -> dict:
    """Return a parsed JSON object, or raise NoJsonReturned when every provider/model failed."""
    sys_prompt = system + "\nRespond with a single valid JSON object and nothing else (no markdown)."
    plan = _plan(tier, len(sys_prompt) + len(user))
    dead, skip, last, waits = set(), set(), None, []
    for sweep in range(2):
        waits = []
        for provider, model in plan:
            if provider in dead or (provider, model) in skip:
                continue
            try:
                out = await _try(provider, model, sys_prompt, user, max_tokens)
                if out:
                    if sweep or (provider, model) != plan[0]:
                        logger.warning("ask_json tier=%s answered by %s/%s", tier, provider, model)
                    return out
                skip.add((provider, model))
                last = ModelUnavailable(f"{provider}/{model} returned no JSON object")
                logger.warning("ask_json: %s/%s returned no JSON for tier=%s", provider, model, tier)
            except ProviderUnavailable as e:
                logger.error("ask_json: %s unavailable: %s", provider, e)
                dead.add(provider)
                last = e
            except RateLimited as e:
                logger.warning("ask_json: %s/%s rate limited (retry-after %.0fs)", provider, model, e.wait)
                waits.append(e.wait)
                last = e
            except Exception as e:
                skip.add((provider, model))
                logger.warning("ask_json: %s/%s failed: %s: %s", provider, model, type(e).__name__, str(e)[:200])
                last = e
        if not waits or sweep:
            break
        pause = min(max(min(waits), 5.0), 30.0)
        logger.warning("ask_json: every model was rate limited; waiting %.0fs before one more sweep", pause)
        await asyncio.sleep(pause)
    msg = f"ask_json tier={tier}: no provider returned usable JSON (last error: {type(last).__name__}: {last})"
    logger.error(msg)
    raise NoJsonReturned(msg)
