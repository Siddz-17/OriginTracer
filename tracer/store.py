"""Persistence via SQLAlchemy Core. Postgres in production (JSONB), SQLite for local dev.
Everything is plain relational tables + JSON columns - no graph DB needed for the MVP; the mutation/propagation
graphs are stored as rows (variants.parent, evidence.variant) and in the final result JSON."""
import uuid
from datetime import datetime, timezone

from sqlalchemy import (JSON, Column, DateTime, Float, Integer, MetaData, String, Table, Text, create_engine,
                        insert, select, update)
from sqlalchemy.dialects.postgresql import JSONB

JSONT = JSON().with_variant(JSONB(), "postgresql")
now = lambda: datetime.now(timezone.utc)

meta = MetaData()
runs = Table("runs", meta,
             Column("id", String(36), primary_key=True), Column("claim", Text, nullable=False),
             Column("status", String(16), nullable=False), Column("params", JSONT),
             Column("created_at", DateTime(timezone=True)), Column("finished_at", DateTime(timezone=True)),
             Column("verdict", String(32)), Column("confidence", Float),
             Column("result", JSONT), Column("report_md", Text), Column("error", Text))
events = Table("events", meta,
               Column("id", Integer, primary_key=True, autoincrement=True), Column("run_id", String(36), index=True),
               Column("ts", DateTime(timezone=True)), Column("agent", String(32)), Column("message", Text))
evidence = Table("evidence", meta,
                 Column("run_id", String(36), primary_key=True), Column("eid", String(16), primary_key=True),
                 Column("source", String(16)), Column("url", Text), Column("author", Text),
                 Column("ts", DateTime(timezone=True)), Column("eff_ts", DateTime(timezone=True)),
                 Column("text", Text), Column("engagement", Integer), Column("variant", String(16)),
                 Column("stance", String(16)), Column("meta", JSONT))
variants = Table("variants", meta,
                 Column("run_id", String(36), primary_key=True), Column("vid", String(16), primary_key=True),
                 Column("parent", String(16)), Column("label", String(32)), Column("text", Text),
                 Column("diff", JSONT), Column("n_items", Integer))
refs = Table("refs", meta,
             Column("run_id", String(36), primary_key=True), Column("label", String(16), primary_key=True),
             Column("kind", String(2)), Column("type", String(24)), Column("weight", Float),
             Column("url", Text), Column("title", Text))


class Store:
    def __init__(self, url):
        self.engine = create_engine(url, future=True, pool_pre_ping=True)
        meta.create_all(self.engine)

    def create_run(self, claim, params):
        rid = str(uuid.uuid4())
        with self.engine.begin() as cx:
            cx.execute(insert(runs).values(id=rid, claim=claim, status="queued", params=params, created_at=now()))
        return rid

    def set_status(self, rid, status, error=None):
        vals = {"status": status, "error": error}
        if status in ("done", "failed"):
            vals["finished_at"] = now()
        with self.engine.begin() as cx:
            cx.execute(update(runs).where(runs.c.id == rid).values(**vals))

    def add_event(self, rid, agent, message):
        with self.engine.begin() as cx:
            cx.execute(insert(events).values(run_id=rid, ts=now(), agent=agent, message=message))

    def save_result(self, rid, st, report_md):
        ev, res = st.by_id(), st.result
        with self.engine.begin() as cx:
            cx.execute(update(runs).where(runs.c.id == rid).values(
                status="done", finished_at=now(), verdict=res.get("verdict"), confidence=res.get("confidence"),
                result=res, report_md=report_md))
            if st.pool:
                cx.execute(insert(evidence), [dict(
                    run_id=rid, eid=e.id, source=e.source, url=e.url, author=e.author, ts=e.ts, eff_ts=e.eff_ts,
                    text=e.text, engagement=e.engagement, variant=(st.assign.get(e.id) or {}).get("variant"),
                    stance=(st.assign.get(e.id) or {}).get("stance"),
                    meta={k: v for k, v in e.meta.items() if k in ("links", "quotes", "at_uri", "subreddit", "resolved_url")})
                    for e in ev.values()])
            if st.variants:
                counts = {}
                for a in st.assign.values():
                    counts[a.get("variant")] = counts.get(a.get("variant"), 0) + 1
                cx.execute(insert(variants), [dict(run_id=rid, vid=v["id"], parent=v.get("parent"), label=v.get("mutation"),
                                                   text=v.get("text"), diff=v.get("diff", {}), n_items=counts.get(v["id"], 0))
                                              for v in st.variants])
            if st.refs:
                cx.execute(insert(refs), [dict(run_id=rid, label=r["label"], kind=r["kind"], type=r["type"], weight=r["weight"],
                                               url=r["url"], title=(r.get("title") or "")[:500]) for r in st.refs if r.get("label")])

    def get_run(self, rid):
        with self.engine.connect() as cx:
            row = cx.execute(select(runs).where(runs.c.id == rid)).mappings().first()
        return dict(row) if row else None

    def list_runs(self, limit=50):
        with self.engine.connect() as cx:
            rows = cx.execute(select(runs.c.id, runs.c.claim, runs.c.status, runs.c.verdict, runs.c.confidence,
                                     runs.c.created_at).order_by(runs.c.created_at.desc()).limit(limit)).mappings().all()
        return [dict(r) for r in rows]

    def get_events(self, rid, after=0):
        with self.engine.connect() as cx:
            rows = cx.execute(select(events).where(events.c.run_id == rid, events.c.id > after)
                              .order_by(events.c.id)).mappings().all()
        return [dict(r) for r in rows]
