from __future__ import annotations
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pyarrow as pa

from .chio import ClickHouseHTTP, ch_query_json_each_row, ch_query_arrow_table


@dataclass
class Sample:
    secs: float
    x_m: float
    y_m: float
    cls: str
    frame: Optional[int] = None
    cam_x: Optional[float] = None
    cam_y: Optional[float] = None
    ts: Optional[np.datetime64] = None


def _safe_float(v) -> Optional[float]:
    try:
        if v is None:
            return None
        f = float(v)
        return f if np.isfinite(f) else None
    except Exception:
        return None


def fetch_tracks(
    ch: ClickHouseHTTP,
    video_filter: Optional[str] = None,
    intersection_id: Optional[int] = None,
) -> Dict[Tuple[str, int], List[Sample]]:
    # Base WHERE: must have valid map coords
    where_clauses = ["(map_m_x != 0 OR map_m_y != 0)"]

    if video_filter:
        safe_video = video_filter.replace("'", "\\'")
        where_clauses.append(f"video = '{safe_video}'")

    if intersection_id is not None:
        # intersection_id is UInt8 in the table, so safe to cast to int
        safe_isect = int(intersection_id)
        where_clauses.append(f"intersection_id = {safe_isect}")

    where_sql = "WHERE " + " AND ".join(where_clauses)

    db = ch.db
    sql = f"""
    SELECT
        video,
        intersection_id,
        frame,
        secs,
        timestamp,
        track_id,
        class,
        cam_x,
        cam_y,
        map_m_x,
        map_m_y
    FROM {db}.raw
    {where_sql}
    ORDER BY video, track_id, frame
    FORMAT ArrowStream
    """

    table = ch_query_arrow_table(ch, sql)
    if table.num_rows == 0:
        return {}

    df = table.to_pandas()

    tracks: Dict[Tuple[str, int], List[Sample]] = {}
    for (video, track_id), g in df.groupby(["video", "track_id"], sort=False):
        samples: List[Sample] = []
        secs = g["secs"].to_numpy(dtype=float)
        x_m = g["map_m_x"].to_numpy(dtype=float)
        y_m = g["map_m_y"].to_numpy(dtype=float)
        cls_series = g.get("class")
        frame_series = g.get("frame")
        cam_x_series = g.get("cam_x")
        cam_y_series = g.get("cam_y")
        ts_series = g.get("timestamp")

        n = len(g)
        for i in range(n):
            sec_i = float(secs[i])
            x_i = float(x_m[i])
            y_i = float(y_m[i])

            cls = str(cls_series.iat[i]) if cls_series is not None else ""
            frame = int(frame_series.iat[i]) if frame_series is not None else None

            cam_x = cam_x_series.iat[i] if cam_x_series is not None else None
            cam_y = cam_y_series.iat[i] if cam_y_series is not None else None

            ts_val = ts_series.iat[i] if ts_series is not None else None
            ts = None
            if ts_val is not None and not (isinstance(ts_val, float) and np.isnan(ts_val)):
                try:
                    ts = np.datetime64(ts_val)
                except Exception:
                    ts = None

            samples.append(
                Sample(
                    secs=sec_i,
                    x_m=x_i,
                    y_m=y_i,
                    cls=cls,
                    frame=frame,
                    cam_x=_safe_float(cam_x),
                    cam_y=_safe_float(cam_y),
                    ts=ts,
                )
            )

        samples.sort(key=lambda s: s.secs)
        tracks[(str(video), int(track_id))] = samples

    return tracks
