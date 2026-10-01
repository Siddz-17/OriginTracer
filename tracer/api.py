"""FastAPI service.  Run:  uvicorn tracer.api:app --reload

POST /runs {claim, rounds, depth, primary[], archive}  -> 202 {id}
GET  /runs                      list
GET  /runs/{id}                 status + final result JSON
GET  /runs/{id}/events?after=N  live agent progress (poll, or wire to SSE later)
GET  /runs/{id}/report          markdown report
"""
import asyncio

import json
import os

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from . import config
from .pipeline import run_pipeline
from .report import render_report
from .store import Store

app = FastAPI(title="Narrative Origin Tracer", version="0.3.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
if os.path.exists(STATIC_DIR):
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

store = Store(config.DATABASE_URL)
_sem = asyncio.Semaphore(config.MAX_CONCURRENT_RUNS)
_tasks: set = set()


class RunRequest(BaseModel):
    claim: str = Field(min_length=3, max_length=1000)
    rounds: int = Field(2, ge=1, le=4)
    depth: int = Field(3, ge=1, le=4)
    primary: list[str] = []
    archive: bool = True


async def _execute(rid: str, req: RunRequest):
    async with _sem:
        store.set_status(rid, "running")
        try:
            st = await run_pipeline(req.claim, rounds=req.rounds, depth=req.depth, primary=req.primary,
                                    archive=req.archive, on_event=lambda a, m: store.add_event(rid, a, m))
            store.save_result(rid, st, render_report(st))
        except Exception as ex:  # surfaced to the client via GET /runs/{id}
            store.set_status(rid, "failed", f"{type(ex).__name__}: {ex}")


@app.post("/runs", status_code=202)
async def create_run(req: RunRequest):
    rid = store.create_run(req.claim, req.model_dump())
    t = asyncio.create_task(_execute(rid, req))
    _tasks.add(t)
    t.add_done_callback(_tasks.discard)
    return {"id": rid, "status": "queued"}


@app.get("/runs")
def list_runs():
    return store.list_runs()


@app.get("/runs/{rid}")
def get_run(rid: str):
    r = store.get_run(rid)
    if not r:
        raise HTTPException(404, "run not found")
    r.pop("report_md", None)
    return r


@app.get("/runs/{rid}/events")
def get_events(rid: str, after: int = 0):
    return store.get_events(rid, after)


@app.get("/runs/{rid}/report", response_class=PlainTextResponse)
def get_report(rid: str):
    r = store.get_run(rid)
    if not r or not r.get("report_md"):
        raise HTTPException(404, "report not ready")
    return r["report_md"]


@app.get("/health")
def health():
    return {"ok": True, "tiers": {t: {"provider": p, "model": m} for t, (p, m) in config.TIERS.items()}}


@app.get("/style.css")
def get_css():
    css_path = os.path.join(STATIC_DIR, "style.css")
    if os.path.exists(css_path):
        return FileResponse(css_path, media_type="text/css")
    raise HTTPException(404, "style.css not found")


@app.get("/app.js")
def get_js():
    js_path = os.path.join(STATIC_DIR, "app.js")
    if os.path.exists(js_path):
        return FileResponse(js_path, media_type="application/javascript")
    raise HTTPException(404, "app.js not found")


@app.get("/favicon.ico", include_in_schema=False)
def favicon():
    return PlainTextResponse("", status_code=204)


@app.get("/")
def index():
    idx_path = os.path.join(STATIC_DIR, "index.html")
    if os.path.exists(idx_path):
        return FileResponse(idx_path)
    return {"message": "Narrative Origin Tracer API"}


@app.get("/api/local-report")
def local_report():
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    json_path = os.path.join(base, "report.json")
    md_path = os.path.join(base, "report.md")
    data = {}
    if os.path.exists(json_path):
        try:
            data["result"] = json.load(open(json_path, encoding="utf-8"))
        except Exception:
            pass
    if os.path.exists(md_path):
        try:
            data["markdown"] = open(md_path, encoding="utf-8").read()
        except Exception:
            pass
    return data
