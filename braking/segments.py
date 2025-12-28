from __future__ import annotations
import os
from typing import List, Optional, Tuple

import numpy as np

from .config import SegmentSplitterConfig


def smooth_1d(arr: np.ndarray, window: int) -> np.ndarray:
    if window <= 1 or arr is None:
        return arr
    n = len(arr)
    if n < window:
        return arr
    kernel = np.ones(int(window), dtype=float) / float(window)
    return np.convolve(arr, kernel, mode="same")


def segment_is_approaching(
    r_seg: np.ndarray,
    *,
    min_approach_m: float,
    max_move_away_m: float,
) -> bool:
    if r_seg.size < 2:
        return False
    r0 = float(r_seg[0])
    r1 = float(r_seg[-1])
    net_approach = r0 - r1
    net_move_away = r1 - r0
    if net_approach >= min_approach_m:
        return True
    if net_move_away <= max_move_away_m:
        return True
    return False


def split_into_plausible_segments(
    t: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    r: np.ndarray,
    side_vals: np.ndarray,
    *,
    braking_window_m: float,
    cfg: SegmentSplitterConfig = SegmentSplitterConfig(),
    video: Optional[str] = None,
    track_id: Optional[int] = None,
) -> List[Tuple[int, int]]:
    debug = False
    dbg_video = os.getenv("DBG_VIDEO")
    dbg_track = os.getenv("DBG_TRACK")
    if dbg_video and dbg_track and video is not None and track_id is not None:
        try:
            debug = (dbg_video == video) and (int(dbg_track) == int(track_id))
        except Exception:
            debug = False

    t = np.asarray(t); x = np.asarray(x); y = np.asarray(y); r = np.asarray(r); side_vals = np.asarray(side_vals)
    n0 = len(t)
    if n0 < cfg.min_segment_len:
        return []

    dx0 = np.diff(x); dy0 = np.diff(y)
    same = np.zeros(n0, dtype=bool)
    same[1:] = (dx0 == 0.0) & (dy0 == 0.0)
    keep = ~same
    if np.count_nonzero(keep) >= cfg.min_segment_len:
        t = t[keep]; x = x[keep]; y = y[keep]; r = r[keep]; side_vals = side_vals[keep]

    n = len(t)
    if n < cfg.min_segment_len:
        return []

    far_start_m = cfg.far_start_frac * braking_window_m
    alpha = np.clip((r - far_start_m) / max(braking_window_m - far_start_m, 1e-6), 0.0, 1.0)

    def lerp(a, b):
        return a + alpha * (b - a)

    v_max_phys = lerp(cfg.v_max_near, cfg.v_max_far)
    max_backstep_m = lerp(cfg.max_backstep_near, cfg.max_backstep_far)
    max_heading_deg = lerp(cfg.max_heading_near, cfg.max_heading_far)
    a_min_phys = lerp(cfg.a_min_near, cfg.a_min_far)
    a_max_phys = lerp(cfg.a_max_near, cfg.a_max_far)
    j_max_phys = lerp(cfg.j_max_near, cfg.j_max_far)
    outlier_k = lerp(cfg.outlier_k_near, cfg.outlier_k_far)

    dt = np.diff(t)
    good_dt = dt > 1e-3
    dx = np.diff(x); dy = np.diff(y)
    step_dist = np.hypot(dx, dy)

    v_step = np.full(n, np.nan)
    v_step[1:][good_dt] = step_dist[good_dt] / dt[good_dt]

    med_v = np.nanmedian(v_step)
    if not np.isfinite(med_v):
        med_v = 4.0
    med_v = max(float(med_v), 2.0)

    outlier_thr = outlier_k * med_v + 1.5
    speed_bad = (v_step > v_max_phys) | (v_step > outlier_thr)

    moving = np.zeros(n, dtype=bool)
    moving[1:] = step_dist > float(cfg.moving_eps_m)

    dr = np.diff(r)
    back_bad = np.zeros(n, dtype=bool)
    if dr.size:
        back_bad[1:] |= (dr > max_backstep_m[1:])

    heading_bad = np.zeros(n, dtype=bool)
    u = np.stack([dx, dy], axis=1)
    if len(u) >= 2:
        u_next = u[1:]
        u_prev = u[:-1]
        dot = np.sum(u_next * u_prev, axis=1)
        norm = np.linalg.norm(u_next, axis=1) * np.linalg.norm(u_prev, axis=1)
        cosang = np.clip(dot / (norm + 1e-9), -1.0, 1.0)
        ang = np.degrees(np.arccos(cosang))
        valid_turn = moving[1:-1] & moving[2:]
        heading_bad[2:] |= valid_turn & (ang > max_heading_deg[2:])

    v_rad = np.full(n, np.nan)
    if dr.size:
        v_rad[1:][good_dt] = -(dr[good_dt] / dt[good_dt])

    a_inst = np.full(n, np.nan)
    accel_bad = np.zeros(n, dtype=bool)
    if n >= 3:
        dt2 = np.diff(t[1:])
        good_dt2 = dt2 > 1e-3
        dv = np.diff(v_rad[1:])
        a_inst[2:][good_dt2] = dv[good_dt2] / dt2[good_dt2]
        valid_a = np.zeros(n, dtype=bool)
        valid_a[2:] = moving[2:]
        accel_bad[2:] = valid_a[2:] & ((a_inst[2:] < a_min_phys[2:]) | (a_inst[2:] > a_max_phys[2:]))

    jerk_bad = np.zeros(n, dtype=bool)
    if n >= 4:
        dt3 = np.diff(t[2:])
        good_dt3 = dt3 > 1e-3
        dj = np.diff(a_inst[2:])
        j_inst = np.full(n, np.nan)
        j_inst[3:][good_dt3] = dj[good_dt3] / dt3[good_dt3]
        valid_j = np.zeros(n, dtype=bool)
        valid_j[3:] = moving[3:]
        jerk_bad[3:] = valid_j[3:] & (np.abs(j_inst[3:]) > j_max_phys[3:])

    bad = speed_bad | back_bad | heading_bad | accel_bad | jerk_bad

    if debug:
        print("SPLIT DIAG", "video", video, "track", track_id, "n", n, "bad_frac", float(np.mean(bad)), flush=True)

    segments: List[Tuple[int, int]] = []
    start = 0
    for i in range(n):
        if bad[i]:
            if i - start >= cfg.min_segment_len:
                segments.append((start, i))
            start = i + 1
    if n - start >= cfg.min_segment_len:
        segments.append((start, n))

    return segments
