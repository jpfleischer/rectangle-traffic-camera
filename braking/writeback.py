from __future__ import annotations
from typing import List

import numpy as np
import json

from .chio import ClickHouseHTTP
from .detect import BrakingEvent


def ensure_braking_events_schema(ch: ClickHouseHTTP) -> None:
    db = ch.db
    sql = f"""
    CREATE TABLE IF NOT EXISTS {db}.braking_events
    (
        intersection_id String,
        approach_id     String,
        video           String,
        track_id        UInt32,
        class           String,

        t_start         Float64,
        t_end           Float64,

        r_start         Float64,
        r_end           Float64,

        v_start         Float64,
        v_end           Float64,
        dv              Float64,

        a_min           Float64,
        avg_decel       Float64,
        severity        String,

        event_ts        Nullable(DateTime64(3)),
        created_at      DateTime DEFAULT now()
    )
    ENGINE = MergeTree
    ORDER BY (intersection_id, approach_id, video, track_id, t_start)
    """
    ch._post_sql(sql, use_db=False)


def _dt64ms_to_str(dt64) -> str:
    # "2026-01-02T23:45:40.123" -> "2026-01-02 23:45:40.123"
    s = np.datetime_as_string(dt64, unit="ms")
    return s.replace("T", " ")


def insert_braking_events(ch: ClickHouseHTTP, events: List[BrakingEvent], chunk_size: int = 5000) -> None:
    if not events:
        return

    db = ch.db

    # Insert using JSONEachRow to avoid massive VALUES(...) SQL strings
    for i in range(0, len(events), chunk_size):
        chunk = events[i : i + chunk_size]

        lines = []
        for ev in chunk:
            row = {
                "intersection_id": ev.intersection_id,
                "approach_id": ev.approach_id,
                "video": ev.video,
                "track_id": int(ev.track_id),
                "class": ev.cls,
                "t_start": float(ev.t_start),
                "t_end": float(ev.t_end),
                "r_start": float(ev.r_start),
                "r_end": float(ev.r_end),
                "v_start": float(ev.v_start),
                "v_end": float(ev.v_end),
                "dv": float(ev.dv),
                "a_min": float(ev.a_min),
                "avg_decel": float(ev.avg_decel),
                "severity": ev.severity,
                # Nullable(DateTime64(3)) accepts string or null in JSONEachRow
                "event_ts": None if ev.event_ts is None else _dt64ms_to_str(ev.event_ts),
            }
            lines.append(json.dumps(row, separators=(",", ":")))

        body = (
            f"INSERT INTO {db}.braking_events FORMAT JSONEachRow\n"
            + "\n".join(lines)
            + "\n"
        )

        ch._post_sql(body, use_db=False)
