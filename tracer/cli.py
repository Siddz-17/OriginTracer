"""CLI:  python -m tracer.cli "claim" --rounds 2 --depth 3 --out report [--save] [--primary URL ...]"""
import argparse
import asyncio
import json

from .agents import jsonable
from .pipeline import run_pipeline
from .report import render_report


def main():
    ap = argparse.ArgumentParser(description="Narrative Origin Tracer")
    ap.add_argument("claim")
    ap.add_argument("--rounds", type=int, default=2)
    ap.add_argument("--depth", type=int, default=3)
    ap.add_argument("--primary", nargs="*", default=[])
    ap.add_argument("--no-archive", action="store_true")
    ap.add_argument("--no-langgraph", action="store_true")
    ap.add_argument("--mode", choices=["single", "agents"], default=None,
                    help="single (default): one LLM call per run; agents: legacy multi-agent loop")
    ap.add_argument("--save", action="store_true", help="also persist to DATABASE_URL")
    ap.add_argument("--out", default="report")
    a = ap.parse_args()
    st = asyncio.run(run_pipeline(a.claim, rounds=a.rounds, depth=a.depth, primary=a.primary,
                                  archive=not a.no_archive, use_langgraph=not a.no_langgraph, mode=a.mode))
    md = render_report(st)
    open(a.out + ".md", "w", encoding="utf-8").write(md)
    json.dump(jsonable(st.result), open(a.out + ".json", "w", encoding="utf-8"), indent=2)
    if a.save:
        from . import config
        from .store import Store
        s = Store(config.DATABASE_URL)
        rid = s.create_run(a.claim, vars(a))
        s.save_result(rid, st, md)
        print("saved run", rid)
    print(f"wrote {a.out}.md / {a.out}.json")


if __name__ == "__main__":
    main()
