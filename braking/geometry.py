from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple

from .chio import ClickHouseHTTP, ch_query_json_each_row
from .units import units2_to_m_xy, units_to_m


@dataclass
class Geometry:
    intersection_id: str
    approach_id: str
    stopbar_center: Tuple[float, float]   # meters
    braking_window_m: float               # meters
    divider_p1: Tuple[float, float]       # meters
    divider_p2: Tuple[float, float]       # meters


def side_of_divider(p: Tuple[float, float], p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
    x, y = p
    x1, y1 = p1
    x2, y2 = p2
    return (x2 - x1) * (y - y1) - (y2 - y1) * (x - x1)


def fetch_geometry(
    ch: ClickHouseHTTP,
    intersection_id: str,
    approach_id: str,
    default_window_m: float = 80.0,
) -> Geometry:
    isect_sql = intersection_id.replace("'", "\\'")
    appr_sql = approach_id.replace("'", "\\'")
    db = ch.db

    sql = f"""
    SELECT
        intersection_id,
        approach_id,
        stopbar_u_x1, stopbar_u_y1,
        stopbar_u_x2, stopbar_u_y2,
        divider_u_x1, divider_u_y1,
        divider_u_x2, divider_u_y2,
        braking_window_u
    FROM {db}.stopbar_metadata
    WHERE intersection_id = '{isect_sql}'
      AND approach_id = '{appr_sql}'
    ORDER BY created_at DESC
    LIMIT 1
    FORMAT JSONEachRow
    """

    rows = ch_query_json_each_row(ch, sql)
    if not rows:
        raise RuntimeError(f"No stopbar_metadata for intersection={intersection_id!r}, approach={approach_id!r}")
    row = rows[0]

    sx1_m, sy1_m = units2_to_m_xy(float(row["stopbar_u_x1"]), float(row["stopbar_u_y1"]))
    sx2_m, sy2_m = units2_to_m_xy(float(row["stopbar_u_x2"]), float(row["stopbar_u_y2"]))
    cx_m = 0.5 * (sx1_m + sx2_m)
    cy_m = 0.5 * (sy1_m + sy2_m)

    bw_u = float(row.get("braking_window_u", 0.0) or 0.0)
    bw_m = default_window_m if bw_u <= 0 else units_to_m(bw_u)

    d1_m = units2_to_m_xy(float(row["divider_u_x1"]), float(row["divider_u_y1"]))
    d2_m = units2_to_m_xy(float(row["divider_u_x2"]), float(row["divider_u_y2"]))

    return Geometry(
        intersection_id=str(row["intersection_id"]),
        approach_id=str(row["approach_id"]),
        stopbar_center=(cx_m, cy_m),
        braking_window_m=bw_m,
        divider_p1=d1_m,
        divider_p2=d2_m,
    )
