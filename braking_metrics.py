#!/usr/bin/env python3
"""
braking_metrics.py — derive braking episodes from ClickHouse trajectories.

Pipeline:

1) Read latest geometry for (intersection_id, approach_id) from stopbar_metadata:
   - stopbar_m_*  (center of stop bar)
   - braking_window_m  (distance upstream where we look for braking)

2) Read trajectories from trajectories.raw:
   - video, frame, secs, track_id, class, map_m_x, map_m_y

3) For each (video, track_id):
   - Build time series t, p(t) = (map_m_x, map_m_y)
   - Compute distance r(t) = ||p(t) - stopbar_center||
   - Compute radial speed v ≈ -Δr/Δt (positive when approaching stop bar)
   - Compute radial decel a ≈ Δv/Δt

4) Detect braking episodes where:
   - r(t) within [0, braking_window_m]
   - v above a minimum "entry" speed
   - a drops below a negative threshold (e.g. <= -2 m/s²)
   - Δv over the episode exceeds a minimum

5) Classify severity:
   - mild:    a_min ∈ [ -mild,      -moderate )
   - moderate:a_min ∈ [ -moderate,  -severe   )
   - severe:  a_min ≤ -severe

6) Optionally write events into trajectories.braking_events.
"""

import argparse
import json
import logging
import os
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import dotenv

dotenv.load_dotenv()

import numpy as np

# --- make sure we can import the ClickHouse client from gui/ ---
HERE = Path(__file__).resolve().parent
GUI_DIR = HERE / "gui"
if str(GUI_DIR) not in map(str, map(Path, __import__("sys").path)):
    import sys as _sys
    _sys.path.append(str(GUI_DIR))

from clickhouse_client import ClickHouseHTTP  # type: ignore


@dataclass
class Geometry:
    intersection_id: str
    approach_id: str
    stopbar_center: Tuple[float, float]
    braking_window_m: float
    divider_p1: Tuple[float, float]        # (X,Y) meters
    divider_p2: Tuple[float, float]        # (X,Y) meters

def side_of_divider(p: Tuple[float, float],
                    p1: Tuple[float, float],
                    p2: Tuple[float, float]) -> float:
    """
    Signed distance proxy using 2D cross product w.r.t. the infinite line p1->p2.
    >0 = one side, <0 = the other, 0 ~ on the line.
    """
    x,  y  = p
    x1, y1 = p1
    x2, y2 = p2
    return (x2 - x1) * (y - y1) - (y2 - y1) * (x - x1)


@dataclass
class Sample:
    secs: float
    x_m: float
    y_m: float
    cls: str
    ts: Optional[np.datetime64] = None   # <-- NEW



@dataclass
class BrakingEvent:
    intersection_id: str
    approach_id: str
    video: str
    track_id: int
    cls: str

    t_start: float
    t_end: float

    r_start: float
    r_end: float

    v_start: float
    v_end: float
    dv: float
    a_min: float
    avg_decel: float
    severity: str
    event_ts: Optional[np.datetime64] = None  # <-- NEW


# ---------------- ClickHouse helpers ---------------- #

def ch_query_json_each_row(ch: ClickHouseHTTP, sql: str) -> List[dict]:
    """Run a SQL query with FORMAT JSONEachRow and return list of dicts."""
    resp = ch._post_sql(sql, use_db=True)  # use same DB as ch.db
    text = resp.text.strip()
    if not text:
        return []
    rows = []
    for line in text.splitlines():
        if not line.strip():
            continue
        rows.append(json.loads(line))
    return rows



def _infer_units_to_m_factor_from_ortho() -> float:
    """
    Returns multiplier to convert (GeoTIFF CRS linear units) -> meters.

    - If ortho CRS is meters: returns 1.0
    - If ortho CRS is US survey foot: returns ~0.3048006096
    - If we can't read ortho: fall back to env, else assume meters.
    """
    # Hard override if you want to force it:
    #   export ORTHO_UNITS_TO_M=0.30480060960121924
    v = os.getenv("ORTHO_UNITS_TO_M")
    if v:
        try:
            f = float(v)
            if f > 0:
                return f
        except Exception:
            pass

    ortho_path = os.getenv("ORTHO_PATH", "ortho_zoom.tif")
    try:
        import rasterio as rio
        with rio.open(ortho_path) as src:
            crs = src.crs
            if crs is None:
                return 1.0
            # rasterio gives you the linear units factor if it knows it
            try:
                # this is exactly what you printed earlier
                return float(crs.linear_units_factor[1])
            except Exception:
                # last-resort: common foot cases
                units = (getattr(crs, "linear_units", None) or "").lower()
                if "us survey foot" in units:
                    return 0.30480060960121924
                if units in ("foot", "feet"):
                    return 0.3048
                return 1.0
    except Exception:
        # If we can't read the ortho, do NOT guess wildly.
        # But treating as meters is safer than silently scaling wrong.
        return 1.0


_UNITS_TO_M = _infer_units_to_m_factor_from_ortho()


def units_to_m(x: float) -> float:
    """GeoTIFF CRS linear units -> meters."""
    return float(x) * _UNITS_TO_M


def units2_to_m_xy(x: float, y: float) -> tuple[float, float]:
    return (units_to_m(x), units_to_m(y))


def fetch_geometry(
    ch: ClickHouseHTTP,
    intersection_id: str,
    approach_id: str,
    default_window_m: float = 80.0,
) -> Geometry:
    """Fetch latest geometry row for intersection/approach from stopbar_metadata."""
    isect_sql = intersection_id.replace("'", "\\'")
    appr_sql  = approach_id.replace("'", "\\'")
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
        raise RuntimeError(
            f"No stopbar_metadata found for intersection={intersection_id!r}, "
            f"approach={approach_id!r}"
        )
    row = rows[0]

    sx1_u = float(row["stopbar_u_x1"]); sy1_u = float(row["stopbar_u_y1"])
    sx2_u = float(row["stopbar_u_x2"]); sy2_u = float(row["stopbar_u_y2"])

    sx1_m, sy1_m = units2_to_m_xy(sx1_u, sy1_u)
    sx2_m, sy2_m = units2_to_m_xy(sx2_u, sy2_u)
    cx_m = 0.5 * (sx1_m + sx2_m)
    cy_m = 0.5 * (sy1_m + sy2_m)

    bw_u = float(row.get("braking_window_u", 0.0) or 0.0)
    bw_m = default_window_m if bw_u <= 0 else units_to_m(bw_u)

    d1_u = (float(row["divider_u_x1"]), float(row["divider_u_y1"]))
    d2_u = (float(row["divider_u_x2"]), float(row["divider_u_y2"]))
    d1_m = units2_to_m_xy(*d1_u)
    d2_m = units2_to_m_xy(*d2_u)

    return Geometry(
        intersection_id=row["intersection_id"],
        approach_id=row["approach_id"],
        stopbar_center=(cx_m, cy_m),
        braking_window_m=bw_m,
        divider_p1=d1_m,
        divider_p2=d2_m,
    )


def fetch_tracks(
    ch: ClickHouseHTTP,
    video_filter: Optional[str] = None,
) -> Dict[Tuple[str, int], List[Sample]]:

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
        track_id,
        class,
        map_m_x,
        map_m_y,
        timestamp   -- make sure this column really exists
    FROM {db}.raw
    {where}
    ORDER BY video, track_id, frame
    FORMAT JSONEachRow
    """

    rows = ch_query_json_each_row(ch, sql)
    tracks: Dict[Tuple[str, int], List[Sample]] = defaultdict(list)

    for row in rows:
        try:
            video = row["video"]
            track_id = int(row["track_id"])
            secs = float(row["secs"])
            x_m = float(row["map_m_x"])
            y_m = float(row["map_m_y"])
            cls = row.get("class", "")
            ts_val = row.get("timestamp", None)
        except Exception:
            continue

        ts = None
        if ts_val is not None:
            try:
                ts = np.datetime64(ts_val)
            except Exception:
                ts = None

        key = (video, track_id)
        tracks[key].append(Sample(secs=secs, x_m=x_m, y_m=y_m, cls=cls, ts=ts))


    for key in list(tracks.keys()):
        tracks[key].sort(key=lambda s: s.secs)

    return tracks


# ---------------- Braking detection ---------------- #

def smooth_1d(arr: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return arr
    if len(arr) < window:
        return arr
    kernel = np.ones(window, dtype=float) / float(window)
    return np.convolve(arr, kernel, mode="same")

def split_into_plausible_segments(
    t: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    r: np.ndarray,
    side_vals: np.ndarray,
    *,
    braking_window_m: float,

    # ----- low-speed urban BASE thresholds (near stopbar) -----
    v_max_near: float = 18.0,          # m/s ≈ 40 mph
    max_backstep_near: float = 1.5,    # m
    max_heading_near: float = 55.0,    # deg
    a_min_near: float = -6.0,          # m/s^2
    a_max_near: float = 3.0,           # m/s^2
    j_max_near: float = 8.0,           # m/s^3
    outlier_k_near: float = 3.0,

    # ----- FAR thresholds (allow more jitter) -----
    v_max_far: float = 20.0,
    max_backstep_far: float = 2.3,
    max_heading_far: float = 70.0,
    a_min_far: float = -7.0,
    a_max_far: float = 3.5,
    j_max_far: float = 10.0,
    outlier_k_far: float = 3.5,

    far_start_frac: float = 0.60,
    min_segment_len: int = 3,          # <- allow short but real segments
) -> List[Tuple[int, int]]:

    n = len(t)
    if n < min_segment_len:
        return []

    # ---- distance-aware interpolation near<->far ----
    far_start_m = far_start_frac * braking_window_m
    alpha = np.clip(
        (r - far_start_m) / max(braking_window_m - far_start_m, 1e-6),
        0.0, 1.0
    )

    def lerp(a, b):
        return a + alpha * (b - a)

    v_max_phys      = lerp(v_max_near,        v_max_far)
    max_backstep_m  = lerp(max_backstep_near, max_backstep_far)
    max_heading_deg = lerp(max_heading_near,  max_heading_far)
    a_min_phys      = lerp(a_min_near,        a_min_far)
    a_max_phys      = lerp(a_max_near,        a_max_far)
    j_max_phys      = lerp(j_max_near,        j_max_far)
    outlier_k       = lerp(outlier_k_near,    outlier_k_far)

    # ---- step speed continuity ----
    dt = np.diff(t)
    good_dt = dt > 1e-3
    dx = np.diff(x); dy = np.diff(y)
    step_dist = np.hypot(dx, dy)

    v_step = np.full(n, np.nan)
    v_step[1:][good_dt] = step_dist[good_dt] / dt[good_dt]

    med_v = np.nanmedian(v_step)
    if not np.isfinite(med_v):
        med_v = 4.0  # or 3.0, something reasonable for near-stopbar traffic
    med_v = max(float(med_v), 2.0)


    # outlier threshold per-sample (same length as v_step)
    outlier_thr = outlier_k * med_v + 1.5
    speed_bad = (v_step > v_max_phys) | (v_step > outlier_thr)

    # ---- radial backsteps (only local bad points; NO global kill) ----
    dr = np.diff(r)
    back_bad = np.zeros(n, dtype=bool)
    if dr.size:
        # slice threshold to dr length
        back_bad[1:] |= (dr > max_backstep_m[1:])

    # ---- heading flips (local bad points) ----
    heading_bad = np.zeros(n, dtype=bool)
    u = np.stack([dx, dy], axis=1)  # length n-1
    if len(u) >= 2:
        u_next = u[1:]   # n-2
        u_prev = u[:-1]  # n-2
        dot = np.sum(u_next * u_prev, axis=1)
        norm = np.linalg.norm(u_next, axis=1) * np.linalg.norm(u_prev, axis=1)
        cosang = np.clip(dot / (norm + 1e-9), -1.0, 1.0)
        ang = np.degrees(np.arccos(cosang))  # length n-2

        # compare to matching slice
        heading_bad[2:] |= (ang > max_heading_deg[2:])

    # ---- radial accel + jerk momentum sanity (local bad points) ----
    v_rad = np.full(n, np.nan)
    v_rad[1:][good_dt] = -(dr[good_dt] / dt[good_dt])

    a_inst = np.full(n, np.nan)
    accel_bad = np.zeros(n, dtype=bool)
    if n >= 3:
        dt2 = np.diff(t[1:])
        good_dt2 = dt2 > 1e-3
        dv = np.diff(v_rad[1:])         # length n-2
        a_inst[2:][good_dt2] = dv[good_dt2] / dt2[good_dt2]

        # compare with matching slices
        accel_bad[2:] = (a_inst[2:] < a_min_phys[2:]) | (a_inst[2:] > a_max_phys[2:])

    jerk_bad = np.zeros(n, dtype=bool)
    if n >= 4:
        dt3 = np.diff(t[2:])
        good_dt3 = dt3 > 1e-3
        dj = np.diff(a_inst[2:])        # length n-3
        j_inst = np.full(n, np.nan)
        j_inst[3:][good_dt3] = dj[good_dt3] / dt3[good_dt3]

        jerk_bad[3:] = np.abs(j_inst[3:]) > j_max_phys[3:]

    # ---- divider instability rule REMOVED ----
    # You already hard-mask to left side upstream, so side flips here are redundant
    # and were causing total rejection.

    bad = speed_bad | back_bad | heading_bad | accel_bad | jerk_bad

    # ---- split at bad points ----
    segments: List[Tuple[int, int]] = []
    start = 0
    for i in range(n):
        if bad[i]:
            if i - start >= min_segment_len:
                segments.append((start, i))
            start = i + 1
    if n - start >= min_segment_len:
        segments.append((start, n))

    return segments


def compute_braking_for_track(
    geom: Geometry,
    samples: List[Sample],
    min_entry_speed: float,
    min_delta_v: float,
    accel_trigger: float,
    severe_thresh: float,
    moderate_thresh: float,
    mild_thresh: float,
    smoothing_window: int = 3,
    disable_window: bool = False,
    min_duration: float = 1.0,
    side_sign: float = 1.0,
) -> List[BrakingEvent]:
    """
    Low-speed urban braking detector with sanity filtering:
      1) mask to braking window + divider side
      2) split into physically plausible sub-segments
      3) run your conservative average-decel event logic per segment
      4) keep strongest (largest dv) event per track
    """
    if len(samples) < 5:
        return []

    cx, cy = geom.stopbar_center

    t_all = np.array([s.secs for s in samples], dtype=float)
    x_all = np.array([s.x_m for s in samples], dtype=float)
    y_all = np.array([s.y_m for s in samples], dtype=float)

    # radial distance to stopbar center
    r_all = np.hypot(x_all - cx, y_all - cy)

    # --- PRE-FLIGHT sanity sample (prints for ~0.05% tracks) ---
    if np.random.rand() < 0.0005:
        print(
            "DEBUG preflight",
            "x_range=", (float(np.min(x_all)), float(np.max(x_all))),
            "y_range=", (float(np.min(y_all)), float(np.max(y_all))),
            "stopbar=", geom.stopbar_center,
            "r_p10/p50/p90=",
            (float(np.percentile(r_all,10)),
             float(np.percentile(r_all,50)),
             float(np.percentile(r_all,90))),
        )


    # mask: within braking window
    mask_window = np.ones_like(r_all, dtype=bool) if disable_window else (r_all <= geom.braking_window_m)



    side_vals_all = np.array(
        [side_of_divider((x, y), geom.divider_p1, geom.divider_p2)
        for x, y in zip(x_all, y_all)],
        dtype=float,
    )

    # --- STRICT divider, but buffered + sign fallback ---
    # distance-aware buffer: allow a tiny "gray zone" around divider for jitter
    # (farther points get a little more slack)
    divider_buffer_all = 0.6 + 0.008 * r_all   # meters; ~0.6m near, ~1.2m at 80m

    def make_mask_for_sign(sign: float):
        # keep only the chosen side, but accept points up to buffer "over the line"
        return (side_vals_all * sign) > -divider_buffer_all

    mask_side = make_mask_for_sign(side_sign)

    # if almost nothing survives, the divider endpoints are probably reversed;
    # flip sign ONCE as a fallback (still keeps only ONE side)
    if mask_side.sum() < 3:
        side_sign = -side_sign
        mask_side = make_mask_for_sign(side_sign)

    mask = mask_window & mask_side
    if mask.sum() < 3:
        return []


    # masked arrays
    t = t_all[mask]
    x = x_all[mask]
    y = y_all[mask]
    r = r_all[mask]
    side_vals = side_vals_all[mask]

    # --- NEW: split at impossible motion (urban-tuned defaults) ---
    segments = split_into_plausible_segments(
        t, x, y, r, side_vals,
        braking_window_m=geom.braking_window_m,
    )

    if not segments:
        segments = [(0, len(t))]

    events: List[BrakingEvent] = []

    # run braking logic per plausible segment
    for s, e in segments:
        tt = t[s:e]
        rr = r[s:e]
        if len(tt) < 5:
            continue

        dt_total = float(tt[-1] - tt[0])
        if dt_total < min_duration:
            continue

        # --- v(t): approach speed series (positive when approaching) ---
        # Use gradient for better numerical behavior than chained diffs.
        # v_raw = -dr/dt  (since r decreases when approaching)
        v_raw = -np.gradient(rr, tt)

        # Smooth v(t) (keep this modest; too much smoothing will reduce peak decel)
        v = smooth_1d(v_raw, smoothing_window)

        # Optional: clamp physically impossible spikes (you already do segment plausibility, so this is just extra safety)
        v = np.where(np.isfinite(v), np.clip(v, 0.0, 40.0), np.nan)  # 40 m/s is generous

        # Entry/exit speeds from early/late samples (robust to a couple NaNs)
        k = 3
        v_entry_vals = v[: min(k, len(v))]
        v_exit_vals  = v[max(0, len(v) - k):]

        v_entry = float(np.nanmedian(v_entry_vals))
        v_exit  = float(np.nanmedian(v_exit_vals))
        if not np.isfinite(v_entry) or not np.isfinite(v_exit):
            continue

        dv_total = max(0.0, v_entry - v_exit)

        # Gate: must be actually approaching stopbar over the segment
        r_start = float(rr[0])
        r_end   = float(rr[-1])
        if (r_start - r_end) < 1.0:
            continue

        # Gate: ensure meaningful speed + speed drop
        if v_entry < min_entry_speed:
            continue
        if dv_total < min_delta_v:
            continue

        # --- a(t): instantaneous acceleration (negative for braking) ---
        a_raw = np.gradient(v, tt)

        # Smooth a(t) a bit (usually same or slightly larger than v smoothing)
        a = smooth_1d(a_raw, smoothing_window)

        # Extract the most negative accel (hardest braking)
        a_min = float(np.nanmin(a))
        if not np.isfinite(a_min):
            continue

        # Optional sanity: reject ridiculous spikes (units m/s^2)
        if a_min < -12.0:   # you used 12 as a magnitude cap before
            continue
        if a_min > 3.0:
            continue

        # Require at least some hard braking (accel_trigger is POSITIVE magnitude)
        # i.e., need a_min <= -accel_trigger
        if a_min > -accel_trigger:
            continue

        avg_decel = dv_total / dt_total   # positive magnitude

        if avg_decel >= -severe_thresh:        # thresholds passed as negative currently
            severity = "severe"
        elif avg_decel >= -moderate_thresh:
            severity = "moderate"
        elif avg_decel >= -mild_thresh:
            severity = "mild"
        else:
            continue


        cls = samples[0].cls

        # compute median wall-clock time for this braking segment if available
        event_ts = None
        ts_all = np.array([s.ts for s in samples], dtype=object)
        ts_seg = ts_all[mask][s:e]   # same mask + segment slice you used for tt/rr

        finite_ts = [t for t in ts_seg if t is not None and not np.isnat(t)]

        if finite_ts:
            # median to avoid single-frame weirdness
            event_ts = np.sort(np.array(finite_ts, dtype="datetime64[ms]"))[len(finite_ts)//2]

        events.append(
            BrakingEvent(
                intersection_id=geom.intersection_id,
                approach_id=geom.approach_id,
                video="",              # caller fills
                track_id=-1,           # caller fills
                cls=cls,
                t_start=float(tt[0]),
                t_end=float(tt[-1]),
                r_start=r_start,
                r_end=r_end,
                v_start=v_entry,
                v_end=v_exit,
                dv=dv_total,
                a_min=a_min,
                avg_decel=avg_decel,           # avg decel
                severity=severity,
                event_ts=event_ts,
            )
        )

    if not events:
        return []

    # keep strongest per track
    events.sort(key=lambda ev: (ev.avg_decel, ev.dv), reverse=True)
    return [events[0]]


# ---------------- Braking events table ---------------- #

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

        event_ts        Nullable(DateTime64(3)),  -- <-- NEW: when braking happened
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
        appr  = ev.approach_id.replace("'", "\\'")
        vid   = ev.video.replace("'", "\\'")
        cls   = ev.cls.replace("'", "\\'")

        # format Nullable(DateTime64)
        if ev.event_ts is None:
            ev_ts_sql = "NULL"
        else:
            ev_ts_sql = f"toDateTime64('{np.datetime_as_string(ev.event_ts, unit='ms')}', 3)"


        line = (
            f"('{isect}','{appr}','{vid}',{ev.track_id},'{cls}',"
            f"{ev.t_start},{ev.t_end},"
            f"{ev.r_start},{ev.r_end},"
            f"{ev.v_start},{ev.v_end},{ev.dv},"
            f"{ev.a_min},{ev.avg_decel},'{ev.severity}',"
            f"{ev_ts_sql})"
        )
        values_lines.append(line)

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




# ---------------- Main driver ---------------- #

def run(args: argparse.Namespace) -> None:
    logger = logging.getLogger("braking_metrics")

    # --- ClickHouse connection (same env vars as your other scripts) ---
    CH_HOST = os.getenv("CH_HOST", "example.com")
    CH_PORT = int(os.getenv("CH_PORT", "8123"))
    CH_USER = os.getenv("CH_USER", "default")
    CH_PASSWORD = os.getenv("CH_PASSWORD", "")
    CH_DB = os.getenv("CH_DB", "trajectories")

    ch = ClickHouseHTTP(
        host=CH_HOST,
        port=CH_PORT,
        user=CH_USER,
        password=CH_PASSWORD,
        database=CH_DB,
        logger=logger,
    )

    # Which half-plane of the infinite divider is "toward camera"?
    side_sign = 1.0 if args.divider_side == "positive" else -1.0


    logger.info("Fetching geometry for intersection=%s approach=%s",
                args.intersection_id, args.approach_id)
    geom = fetch_geometry(ch, args.intersection_id, args.approach_id)
    logger.info("GeoTIFF units->meters factor: %.12f", _UNITS_TO_M)


    if args.stopbar_x is not None and args.stopbar_y is not None:
        # CLI override is in METERS (consistent with the "real fix")
        geom.stopbar_center = (float(args.stopbar_x), float(args.stopbar_y))

        logger.info("Overriding stopbar center to (%.3f, %.3f)",
                    geom.stopbar_center[0], geom.stopbar_center[1])


    if args.window_m is not None and args.window_m > 0:
        geom.braking_window_m = args.window_m

    logger.info(
        "Using stopbar center at (%.3f, %.3f) and braking_window=%.1f m",
        geom.stopbar_center[0],
        geom.stopbar_center[1],
        geom.braking_window_m,
    )

    logger.info("Fetching tracks from ClickHouse...")
    tracks = fetch_tracks(ch, video_filter=args.video)
    logger.info("Loaded %d tracks", len(tracks))

    events: List[BrakingEvent] = []
    veh_with_any_event = set()

    for (video, track_id), samples in tracks.items():
        evs = compute_braking_for_track(
            geom=geom,
            samples=samples,
            min_entry_speed=args.min_entry_speed,
            min_delta_v=args.min_delta_v,
            # accel_trigger is now a POSITIVE magnitude in m/s^2
            accel_trigger=args.accel_trigger,
            severe_thresh=-args.severe_g,
            moderate_thresh=-args.moderate_g,
            mild_thresh=-args.mild_g,
            smoothing_window=args.smooth,
            disable_window=args.disable_window,
            side_sign=side_sign,
        )

        for ev in evs:
            ev.video = video
            ev.track_id = track_id
        if evs:
            veh_with_any_event.add((video, track_id))
            events.extend(evs)

    # --- Summary ---
    total_vehicles = len(tracks)
    logger.info("Total vehicles with BEV points: %d", total_vehicles)
    logger.info("Vehicles with >=1 braking event: %d", len(veh_with_any_event))
    logger.info("Total braking events: %d", len(events))

    sev_counts = {"mild": 0, "moderate": 0, "severe": 0}
    for ev in events:
        if ev.severity in sev_counts:
            sev_counts[ev.severity] += 1

    if events:
        logger.info(
            "Severity counts: mild=%d, moderate=%d, severe=%d",
            sev_counts["mild"],
            sev_counts["moderate"],
            sev_counts["severe"],
        )
    else:
        logger.info("No braking events detected with current thresholds.")

    # --- Optional: write events back to ClickHouse ---
    if args.write_events and events:
        logger.info("Ensuring braking_events schema and inserting events...")
        ensure_braking_events_schema(ch)
        insert_braking_events(ch, events)
        logger.info("Inserted %d events into %s.braking_events", len(events), ch.db)


def main():
    ap = argparse.ArgumentParser(
        description="Compute braking episodes from trajectories.raw + stopbar_metadata"
    )
    ap.add_argument("--intersection-id", required=True, help="intersection_id (String)")
    ap.add_argument("--approach-id", required=True, help="approach_id (String)")
    ap.add_argument(
        "--video",
        default=None,
        help="optional video basename filter (e.g. Hiv00454-encoded-SB.mp4)",
    )

    # thresholds in m/s and m/s^2
    ap.add_argument(
        "--min-entry-speed",
        type=float,
        default=3.0,
        help="min approach speed before braking (m/s) [default: 3.0]",
    )
    ap.add_argument(
        "--min-delta-v",
        type=float,
        default=3.0,
        help="min speed drop during event (m/s) [default: 3.0]",
    )
    ap.add_argument(
        "--accel-trigger",
        type=float,
        default=1.5,
        help="require min a <= -accel_trigger [default: 1.5]",
    )

    ap.add_argument(
        "--mild-g",
        type=float,
        default=1.0,
        help="mild threshold in m/s^2 (severity by a_min) [default: 1.0]",
    )
    ap.add_argument(
        "--moderate-g",
        type=float,
        default=2.0,
        help="moderate threshold in m/s^2 (severity by a_min) [default: 2.0]",
    )
    ap.add_argument(
        "--severe-g",
        type=float,
        default=3.5,
        help="severe threshold in m/s^2 (severity by a_min) [default: 3.5]",
    )
    ap.add_argument(
        "--window-m",
        type=float,
        default=None,
        help="override braking_window_m (meters upstream of stopbar)",
    )
    ap.add_argument(
        "--smooth",
        type=int,
        default=3,
        help="smoothing window for v,a (odd integer; 0/1 disables) [default: 3]",
    )

    ap.add_argument(
        "--write-events",
        action="store_true",
        help="insert braking_events into ClickHouse",
    )

    ap.add_argument(
        "--divider-side",
        choices=["positive", "negative"],
        default="positive",
        help="Which side of the divider line is toward-camera (by cross-product sign). Default: positive",
    )

    ap.add_argument("--stopbar-x", type=float, default=None,
                    help="Override stopbar center X in same coords as map_m_x.")
    ap.add_argument("--stopbar-y", type=float, default=None,
                    help="Override stopbar center Y in same coords as map_m_y.")
    ap.add_argument("--disable-window", action="store_true",
                    help="Ignore braking_window_m gating (debug).")



    ap.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="verbose logging",
    )

    args = ap.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)8s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    run(args)


if __name__ == "__main__":
    main()
