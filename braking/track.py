from __future__ import annotations
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from .chio import ClickHouseHTTP, ch_query_json_each_row


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


def fetch_tracks(ch: ClickHouseHTTP, video_filter: Optional[str] = None) -> Dict[Tuple[str, int], List[Sample]]:
    where = "WHERE (map_m_x != 0 OR map_m_y != 0)"
    if video_filter:
        safe_video = video_filter.replace("'", "\\'")
        where += f" AND video = '{safe_video}'"

    db = ch.db
    sql = f"""
    SELECT
        video,
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
    {where}
    ORDER BY video, track_id, frame
    FORMAT JSONEachRow
    """

    rows = ch_query_json_each_row(ch, sql)
    tracks: Dict[Tuple[str, int], List[Sample]] = defaultdict(list)

    for row in rows:
        try:
            video = str(row["video"])
            track_id = int(row["track_id"])
            secs = float(row["secs"])
            x_m = float(row["map_m_x"])
            y_m = float(row["map_m_y"])
            frame = int(row["frame"])
            cls = str(row.get("class", "") or "")
        except Exception:
            continue

        ts = None
        ts_val = row.get("timestamp")
        if ts_val is not None:
            try:
                ts = np.datetime64(ts_val)
            except Exception:
                ts = None

        tracks[(video, track_id)].append(
            Sample(
                secs=secs,
                x_m=x_m,
                y_m=y_m,
                cls=cls,
                frame=frame,
                cam_x=_safe_float(row.get("cam_x")),
                cam_y=_safe_float(row.get("cam_y")),
                ts=ts,
            )
        )

    for key in list(tracks.keys()):
        tracks[key].sort(key=lambda s: s.secs)

    return tracks
