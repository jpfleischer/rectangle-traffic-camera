import json
from collections import defaultdict
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

from clickhouse_client import ClickHouseHTTP


def ch_query_json_each_row(ch: ClickHouseHTTP, sql: str, params: Optional[dict] = None) -> List[dict]:
    """Run a SQL query with FORMAT JSONEachRow and return list of dicts."""
    resp = ch._post_sql(sql, use_db=True, params=params)
    text = resp.text.strip()
    if not text:
        return []
    rows = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        rows.append(json.loads(line))
    return rows


def load_tracks_interval(
    ch: ClickHouseHTTP,
    start_dt: datetime,
    duration_s: float,
    transform,
    units_to_m: float,
    video_filter: Optional[str] = None,
) -> Tuple[Dict[object, List[dict]], List[float], Optional[datetime]]:
    """
    Load tracks from trajectories.raw in ClickHouse for a time window.

    Returns:
      tracks: dict[track_key] -> list of {'t','x','y','cls'}
      times: sorted list of unique relative seconds (float)
      t0:    earliest absolute timestamp (datetime) or None
    """
    end_dt = start_dt + timedelta(seconds=duration_s)

    start_str = start_dt.strftime("%Y-%m-%d %H:%M:%S")
    end_str = end_dt.strftime("%Y-%m-%d %H:%M:%S")

    db = ch.db

    where = (
        "WHERE timestamp >= {start_ts:DateTime} "
        "AND timestamp < {end_ts:DateTime} "
        "AND (map_m_x != 0 OR map_m_y != 0)"
    )

    params = {
        "start_ts": start_str,
        "end_ts": end_str,
    }

    if video_filter:
        where += " AND video = {video:String}"
        params["video"] = video_filter

    sql = f"""
    SELECT
        video,
        timestamp,
        track_id,
        class,
        map_m_x,
        map_m_y
    FROM {db}.raw
    {where}
    ORDER BY timestamp, video, track_id
    FORMAT JSONEachRow
    """

    rows = ch_query_json_each_row(ch, sql, params=params)
    if not rows:
        return {}, [], None

    samples = []
    for row in rows:
        try:
            video = row["video"]
            ts = datetime.fromisoformat(row["timestamp"])
            track_id = int(row["track_id"])

            x_m = float(row["map_m_x"])
            y_m = float(row["map_m_y"])

            x_u = x_m / units_to_m
            y_u = y_m / units_to_m
            col, rowpix = (~transform) * (x_u, y_u)  # DO NOT CHANGE

            x_px = float(col)
            y_px = float(rowpix)

            cls = row.get("class", "")
        except Exception:
            continue

        samples.append((video, track_id, ts, x_px, y_px, cls))

    if not samples:
        return {}, [], None

    t0 = min(s[2] for s in samples)

    tracks: Dict[object, List[dict]] = defaultdict(list)
    times_set = set()

    for video, track_id, ts, x_px, y_px, cls in samples:
        t = (ts - t0).total_seconds()
        key = f"{video}#{track_id}"
        tracks[key].append({"t": t, "x": x_px, "y": y_px, "cls": cls})
        times_set.add(t)

    for key in list(tracks.keys()):
        tracks[key].sort(key=lambda d: d["t"])

    times = sorted(times_set)
    return tracks, times, t0


def get_video_base_timestamp(ch: ClickHouseHTTP, cache: dict, video: str) -> Optional[datetime]:
    if video in cache:
        return cache[video]

    db = ch.db
    sql = f"""
    SELECT min(timestamp) AS ts_min
    FROM {db}.raw
    WHERE video = {{video:String}}
    FORMAT JSONEachRow
    """
    rows = ch_query_json_each_row(ch, sql, params={"video": video})
    if not rows:
        return None

    ts_str = rows[0].get("ts_min")
    if not ts_str:
        return None

    try:
        ts = datetime.fromisoformat(ts_str)
    except Exception:
        return None

    cache[video] = ts
    return ts
