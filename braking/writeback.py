from __future__ import annotations
from typing import List

import numpy as np

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


def insert_braking_events(ch: ClickHouseHTTP, events: List[BrakingEvent]) -> None:
    if not events:
        return

    db = ch.db
    values_lines = []
    for ev in events:
        isect = ev.intersection_id.replace("'", "\\'")
        appr = ev.approach_id.replace("'", "\\'")
        vid = ev.video.replace("'", "\\'")
        cls = ev.cls.replace("'", "\\'")

        if ev.event_ts is None:
            ev_ts_sql = "NULL"
        else:
            ev_ts_sql = f"toDateTime64('{np.datetime_as_string(ev.event_ts, unit='ms')}', 3)"

        values_lines.append(
            f"('{isect}','{appr}','{vid}',{int(ev.track_id)},'{cls}',"
            f"{ev.t_start},{ev.t_end},"
            f"{ev.r_start},{ev.r_end},"
            f"{ev.v_start},{ev.v_end},{ev.dv},"
            f"{ev.a_min},{ev.avg_decel},'{ev.severity}',"
            f"{ev_ts_sql})"
        )

    sql = f"""
    INSERT INTO {db}.braking_events
    (
        intersection_id, approach_id, video, track_id, class,
        t_start, t_end,
        r_start, r_end,
        v_start, v_end, dv,
        a_min, avg_decel, severity,
        event_ts
    )
    VALUES {", ".join(values_lines)}
    """
    ch._post_sql(sql, use_db=False)
