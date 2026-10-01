"""First-run checker: python check_setup.py
Loads .env, then tests each key with one tiny call and lists Groq models on your account."""
import asyncio
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def load_env(path=".env"):
    if not os.path.exists(path):
        alt = os.path.join(os.path.dirname(os.path.abspath(__file__)), path)
        if os.path.exists(alt):
            path = alt
        else:
            print("No .env file found. Run:  copy .env.example .env   then paste your keys into it.")
            sys.exit(1)
    for line in open(path):
        line = line.split("#", 1)[0].strip()
        if "=" in line:
            k, v = line.split("=", 1)
            if v.strip():
                os.environ.setdefault(k.strip(), v.strip())


async def main():
    load_env()
    from tracer import config, llm
    ok = True
    for tier, (prov, model) in config.TIERS.items():
        have = llm._has_key(prov)
        print(f"[{tier:9}] {prov}/{model}: key {'found' if have else 'MISSING'}")
        if not have:
            ok = False
            continue
        try:
            out = await llm._try(prov, model, 'Reply with JSON {"ok": true}', "ping", 200)
            print(f"           call {'OK' if out else 'returned no JSON'}")
        except Exception as e:
            ok = False
            print(f"           call FAILED: {type(e).__name__}: {str(e)[:200]}")
    if os.getenv("GROQ_API_KEY"):
        import httpx
        r = httpx.get("https://api.groq.com/openai/v1/models", headers={"Authorization": f"Bearer {os.environ['GROQ_API_KEY']}"})
        if r.status_code == 200:
            print("Groq models on your account:", ", ".join(sorted(m["id"] for m in r.json()["data"])))
            print("  (if a tier above failed with 404, set its *_MODEL in .env to one of these)")
    b = bool(os.getenv("BSKY_HANDLE") and os.getenv("BSKY_APP_PASSWORD"))
    if b:
        try:
            import httpx
            r = httpx.post("https://bsky.social/xrpc/com.atproto.server.createSession",
                           json={"identifier": os.environ["BSKY_HANDLE"], "password": os.environ["BSKY_APP_PASSWORD"]},
                           timeout=10)
            if r.status_code == 200:
                print(f"Bluesky credentials: found & authenticated as @{os.environ['BSKY_HANDLE']}")
            else:
                print(f"Bluesky credentials: found but login FAILED (HTTP {r.status_code}: {r.json().get('message', r.text)})")
        except Exception as ex:
            print(f"Bluesky credentials: check error: {ex}")
    else:
        print("Bluesky credentials: not set (Bluesky search will be unauthenticated/rate-limited)")
    print("\nAll good - try:  python -m tracer.cli \"your claim\" --rounds 1 --out report" if ok
          else "\nFix the items above, then run this again.")

asyncio.run(main())
