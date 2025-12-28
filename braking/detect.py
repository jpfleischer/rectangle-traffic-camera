from __future__ import annotations
import os
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

from .config import DetectorConfig, EdgeVetoConfig, SegmentSplitterConfig
from .geometry import Geometry, side_of_divider
from .segments import smooth_1d, segment_is_approaching, split_into_plausible_segments
from .track import Sample


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
    event_ts: Optional[np.datetime64] = None


class DebugGate:
    def __init__(self, video: str, track_id: int):
        self._enabled = False
        dv = os.getenv("DBG_VIDEO")
        dt = os.getenv("DBG_TRACK")
        if dv and dt:
            try:
                self._enabled = (dv == video) and (int(dt) == int(track_id))
            except Exception:
                self._enabled = False

    @property
    def enabled(self) -> bool:
        return self._enabled

    def log(self, *args):
        if self._enabled:
            print(*args, flush=True)


def _contiguous_runs(idx: np.ndarray) -> List[Tuple[int, int]]:
    if idx.size == 0:
        return []
    runs: List[Tuple[int, int]] = []
    rs = int(idx[0]); rp = int(idx[0])
    for k in idx[1:]:
        k = int(k)
        if k == rp + 1:
            rp = k
        else:
            runs.append((rs, rp + 1))
            rs = k; rp = k
    runs.append((rs, rp + 1))
    return runs


def dbg_drop(dbg: DebugGate, why: str, **kv):
    if dbg.enabled:
        msg = "DROP " + why + " " + " ".join([f"{k}={v}" for k,v in kv.items()])
        dbg.log(msg)


@dataclass
class BestRun:
    ev_i0: int
    ev_i1: int
    v_entry: float
    v_exit: float
    dv: float
    score: Tuple[float, float, float]


def pick_best_brake_run(
    *,
    tt: np.ndarray,
    v: np.ndarray,
    a: np.ndarray,
    accel_trigger: float,
    min_delta_v: float,
    min_entry_speed: float,
    dbg: DebugGate,
) -> Optional[BestRun]:
    if dbg.enabled:
        dbg.log("STATS",
                "len", len(tt),
                "v_raw_min", float(np.nanmin(v)),
                "v_raw_med", float(np.nanmedian(v)),
                "a_min", float(np.nanmin(a)),
                "a_p05", float(np.nanpercentile(a, 5)),
                "accel_trigger", accel_trigger,
                "mask_ct", int(np.sum((a <= -accel_trigger) & np.isfinite(a) & np.isfinite(v))))

    # Don't consider "braking" when we're basically already stopped (jitter creates fake accel spikes)
    v_floor = max(0.25, 0.5 * float(min_entry_speed))   # tune: 0.25–0.5 m/s works well at 15 FPS
    brake_mask = (a <= -accel_trigger) & np.isfinite(a) & np.isfinite(v) & (v >= v_floor)

    if not np.any(brake_mask):
        return None

    runs = _contiguous_runs(np.flatnonzero(brake_mask))
    best: Optional[BestRun] = None

    for bs, be in runs:
        left = max(0, bs - 3)
        right = min(len(v), be + 3)

        ev_i1 = be - 1
        if ev_i1 < 0:
            continue

        pre = v[left:bs + 1]
        post = v[be:right]

        if pre.size == 0:
            continue

        v_entry = float(np.nanmedian(pre))

        if post.size > 0:
            v_exit = float(np.nanmedian(post))
        else:
            # Segment ends right after the braking run.
            # Use the tail of the run as exit estimate.
            tail0 = max(bs, be - 3)
            tail = v[tail0:be]
            if tail.size == 0:
                continue
            v_exit = float(np.nanmedian(tail))

            # Optional: if we can't see post-event frames, only accept if it "looks like a stop"
            # i.e., exit speed is low.
            # if v_exit > 1.0:   # m/s (~2.2 mph), tune as needed
            #     continue


        dv = max(0.0, max(0.0, v_entry) - max(0.0, v_exit))

        if dbg.enabled:
            dbg.log("RUN_TIMES", "tt[left..ev_i1]", tt[left:ev_i1+1].tolist())

        dt = float(tt[ev_i1] - tt[left])
        if dt <= 1e-6:
            continue

        if dbg.enabled:
            a_run_min = float(np.nanmin(a[bs:be])) if np.any(np.isfinite(a[bs:be])) else float("nan")
            dbg.log("  RUN", "bs/be", (bs, be), "dv", dv, "dt", dt, "dv/dt", dv/dt, "a_min_run", a_run_min)

        if dv < min_delta_v:
            continue
        if v_entry < min_entry_speed:
            continue


        score = (dv / dt, dv, -dt)
        cand = BestRun(ev_i0=left, ev_i1=ev_i1, v_entry=v_entry, v_exit=v_exit, dv=dv, score=score)
        if best is None or cand.score > best.score:
            best = cand

    return best


def segment_is_stationary(
    tt: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    cam_x: Optional[np.ndarray],
    cam_y: Optional[np.ndarray],
    *,
    min_disp_m: float = 1.0,          # total displacement over segment
    min_r_span_m: float = 0.8,         # r max-min over segment
    min_med_speed_mps: float = 0.5,    # robust median speed
    max_cam_jitter_px: float = 2.5,    # median frame-to-frame jitter in pixels
) -> bool:
    if len(tt) < 3:
        return True

    dt = np.diff(tt)
    ok = np.isfinite(dt) & (dt > 1e-6)
    if not np.any(ok):
        return True

    dx = np.diff(x); dy = np.diff(y)
    step_m = np.hypot(dx, dy)
    step_m = step_m[ok]; dt = dt[ok]
    speed = step_m / dt

    disp_m = float(np.hypot(x[-1] - x[0], y[-1] - y[0]))
    r_span = float(np.nanmax(np.hypot(x - x[0], y - y[0])) - np.nanmin(np.hypot(x - x[0], y - y[0])))
    med_speed = float(np.nanmedian(speed)) if speed.size else 0.0

    cam_ok = False
    cam_jitter = 0.0
    if cam_x is not None and cam_y is not None and len(cam_x) == len(tt):
        cx = cam_x; cy = cam_y
        m = np.isfinite(cx) & np.isfinite(cy)
        if np.sum(m) >= 3:
            dc = np.hypot(np.diff(cx[m]), np.diff(cy[m]))
            cam_jitter = float(np.nanmedian(dc)) if dc.size else 0.0
            cam_ok = True

    # “stationary” if it barely moved in map-space and speed is tiny;
    # optional pixel jitter gate catches “vibration-only” tracks.
    stationary_map = (disp_m < min_disp_m) and (r_span < min_r_span_m) and (med_speed < min_med_speed_mps)

    stationary_cam = cam_ok and (cam_jitter <= max_cam_jitter_px)

    return stationary_map and (stationary_cam or not cam_ok)



def compute_event_timestamp(samples: List[Sample], mask: np.ndarray, seg_s: int, seg_e: int) -> Optional[np.datetime64]:
    ts_all = np.array([s.ts for s in samples], dtype=object)
    ts_seg = ts_all[mask][seg_s:seg_e]
    finite_ts = [t for t in ts_seg if t is not None and not np.isnat(t)]
    if not finite_ts:
        return None
    arr = np.sort(np.array(finite_ts, dtype="datetime64[ms]"))
    return arr[len(arr) // 2]


def post_event_visibility_ok(
    *,
    samples: List[Sample],
    cam_x_all: np.ndarray,
    cam_y_all: np.ndarray,
    near_edge_all: np.ndarray,
    event_end_full_idx: int,
    min_post_event_frames: int,
) -> bool:
    N_full = len(samples)
    post = int(min_post_event_frames)
    post_start = event_end_full_idx + 1
    post_end = post_start + post
    if post_end > N_full:
        return False

    frames_all = np.array([(s.frame if s.frame is not None else -1) for s in samples], dtype=int)
    post_frames = frames_all[post_start:post_end]
    if post_frames.size != post or np.any(post_frames < 0):
        return False
    if not np.all(np.diff(post_frames) == 1):
        return False

    post_cam_ok = np.isfinite(cam_x_all[post_start:post_end]) & np.isfinite(cam_y_all[post_start:post_end])
    if int(np.sum(post_cam_ok)) < post:
        return False

    post_visible = int(np.sum(~near_edge_all[post_start:post_end]))
    if post_visible < post:
        return False

    return True

def split_on_time_gaps(tt: np.ndarray, *, max_gap_s: float = 0.6, min_len: int = 5):
    """
    Split [0..N) into subsegments when time gaps exceed max_gap_s.
    Returns list of (s,e) over tt indices.
    """
    N = len(tt)
    if N < min_len:
        return []
    cuts = [0]
    dt = np.diff(tt)
    bad = np.where(~np.isfinite(dt) | (dt > max_gap_s))[0]
    for i in bad:
        cuts.append(int(i + 1))
    cuts.append(N)
    out = []
    for s, e in zip(cuts[:-1], cuts[1:]):
        if e - s >= min_len:
            out.append((s, e))
    return out


def split_on_sustained_stop(
    tt: np.ndarray,
    v: np.ndarray,
    *,
    v_stop: float = 0.35,      # m/s (tune 0.25–0.5)
    stop_s: float = 1.5,       # seconds of continuous low speed to declare "stopped"
    fps: float = 15.0,
    min_len: int = 5,
):
    """
    Split segment when we observe a sustained stop. Returns list of (s,e) index ranges.
    This is the red-light jitter killer.
    """
    N = len(tt)
    if N < min_len:
        return []
    stop_len = max(3, int(stop_s * fps))
    below = np.isfinite(v) & (v <= v_stop)

    cuts = []
    i = 0
    while i <= N - stop_len:
        if np.all(below[i : i + stop_len]):
            cuts.append(i)  # cut at start of the stop
            j = i + stop_len
            while j < N and below[j]:
                j += 1
            i = j
        else:
            i += 1

    segs = []
    s = 0
    for c in cuts:
        if c - s >= min_len:
            segs.append((s, c))
        s = c
    if N - s >= min_len:
        segs.append((s, N))
    return segs


def events_dedupe_key(ev: BrakingEvent, *, tol_s: float = 0.6) -> tuple:
    """
    Dedupe events coming from multiple passes (plausible segments + raw fallback).
    """
    return (ev.video, ev.track_id, int(round(ev.t_start / tol_s)), int(round(ev.t_end / tol_s)))


def merge_close_segments(segments, tt, max_gap_s=0.4, min_len=5):
    if not segments:
        return []
    merged = []
    s0, e0 = segments[0]
    for s1, e1 in segments[1:]:
        gap = float(tt[s1] - tt[e0 - 1]) if (e0 > s0 and e1 > s1) else 1e9
        if gap <= max_gap_s:
            e0 = e1
        else:
            if (e0 - s0) >= min_len:
                merged.append((s0, e0))
            s0, e0 = s1, e1
    if (e0 - s0) >= min_len:
        merged.append((s0, e0))
    return merged



def compute_braking_for_track(
    *,
    geom: Geometry,
    video: str,
    track_id: int,
    samples: List[Sample],
    min_entry_speed: float,
    min_delta_v: float,
    accel_trigger: float,
    severe_thresh: float,
    moderate_thresh: float,
    mild_thresh: float,
    disable_window: bool = False,
    side_sign: float = 1.0,
    det_cfg: DetectorConfig = DetectorConfig(),
    seg_cfg: SegmentSplitterConfig = SegmentSplitterConfig(),
    edge_cfg: EdgeVetoConfig = EdgeVetoConfig(),
) -> List[BrakingEvent]:
    dbg = DebugGate(video, track_id)

    cx, cy = geom.stopbar_center
    t_all = np.array([s.secs for s in samples], dtype=float)
    x_all = np.array([s.x_m for s in samples], dtype=float)
    y_all = np.array([s.y_m for s in samples], dtype=float)

    cam_x_all = np.array([s.cam_x if s.cam_x is not None else np.nan for s in samples], dtype=float)
    cam_y_all = np.array([s.cam_y if s.cam_y is not None else np.nan for s in samples], dtype=float)

    r_all = np.hypot(x_all - cx, y_all - cy)

    # full-track edge proximity
    w = int(edge_cfg.frame_w)
    h = int(edge_cfg.frame_h)
    m = int(edge_cfg.edge_margin_px)
    near_edge_all = (
        np.isfinite(cam_x_all) & np.isfinite(cam_y_all) &
        (
            (cam_x_all <= m) | (cam_x_all >= (w - 1 - m)) |
            (cam_y_all <= m) | (cam_y_all >= (h - 1 - m))
        )
    )
    N_full = len(samples)

    mask_window = np.ones_like(r_all, dtype=bool) if disable_window else (r_all <= geom.braking_window_m)

    side_vals_all = np.array([side_of_divider((x, y), geom.divider_p1, geom.divider_p2) for x, y in zip(x_all, y_all)], dtype=float)
    dx = geom.divider_p2[0] - geom.divider_p1[0]
    dy = geom.divider_p2[1] - geom.divider_p1[1]
    den = float(np.hypot(dx, dy)) + 1e-9
    signed_dist_m_all = side_vals_all / den
    mask_side = (signed_dist_m_all * side_sign) > float(det_cfg.divider_buffer_m)

    if dbg.enabled:
        dbg.log("MASK_COUNTS",
                "N_full", len(mask_side),
                "mask_side", int(mask_side.sum()),
                "mask_window", int(mask_window.sum()))
        dbg.log("SIDE_DIST_STATS",
                "signed_p05", float(np.nanpercentile(signed_dist_m_all, 5)),
                "signed_p50", float(np.nanpercentile(signed_dist_m_all, 50)),
                "signed_p95", float(np.nanpercentile(signed_dist_m_all, 95)))


    mask = mask_window & mask_side
    if int(mask.sum()) < 3:
        dbg_drop(dbg, "mask<3", mask_sum=int(mask.sum()), mask_window=int(mask_window.sum()), mask_side=int(mask_side.sum()))
        return []
    mask_idx = np.flatnonzero(mask)

    t = t_all[mask]; x = x_all[mask]; y = y_all[mask]; r = r_all[mask]; side_vals = side_vals_all[mask]

    # --- existing: segments from plausibility splitter ---
    segments = split_into_plausible_segments(
        t, x, y, r, side_vals,
        braking_window_m=geom.braking_window_m,
        cfg=seg_cfg,
        video=video,
        track_id=track_id,
    )
    segments = merge_close_segments(segments, t, max_gap_s=0.4, min_len=5)

    # --- NEW: also consider "raw fallback" segments that keep ALL points ---
    # Split only on time gaps, so we don't merge disjoint chunks.
    fallback = split_on_time_gaps(t, max_gap_s=0.6, min_len=5)
    if not fallback:
        fallback = [(0, len(t))] if len(t) >= 5 else []

    # Combine (plausible + raw fallback). We'll dedupe events later.
    all_segments = segments + fallback

    events: List[BrakingEvent] = []
    seen = set()

    for seg_s, seg_e in all_segments:
        tt = t[seg_s:seg_e]
        xx = x[seg_s:seg_e]
        yy = y[seg_s:seg_e]
        rr = r[seg_s:seg_e]

        cam_x_seg = cam_x_all[mask][seg_s:seg_e]
        cam_y_seg = cam_y_all[mask][seg_s:seg_e]

        # If it's wholly stationary, no braking event here (but we also split stops below)
        if len(tt) < 5:
            continue
        if float(tt[-1] - tt[0]) < float(det_cfg.min_duration):
            continue

        # --- velocity/accel as before (radial) ---
        v_raw = -np.gradient(rr, tt)
        v = smooth_1d(v_raw, det_cfg.smoothing_window)
        v = np.where(np.isfinite(v), np.clip(v, -5.0, 40.0), np.nan)

        # --- true map speed (m/s): invariant to stopbar geometry ---
        vx_map = np.gradient(xx, tt)
        vy_map = np.gradient(yy, tt)
        v_map_raw = np.hypot(vx_map, vy_map)
        v_map = smooth_1d(v_map_raw, det_cfg.smoothing_window)
        v_map = np.where(np.isfinite(v_map), np.clip(v_map, 0.0, 40.0), np.nan)


        a_raw = np.gradient(v, tt)
        a = smooth_1d(a_raw, det_cfg.smoothing_window)

        # --- NEW: split this segment at sustained stops (red-light dwell) ---
        sub_segs = split_on_sustained_stop(
            tt, np.clip(v, 0.0, None),  # stop detection should use nonnegative speed
            v_stop=0.35,
            stop_s=1.5,
            fps=float(det_cfg.fps),
            min_len=5,
        )
        if not sub_segs:
            sub_segs = [(0, len(tt))]

        for sub_s, sub_e in sub_segs:
            ttt = tt[sub_s:sub_e]
            xxx = xx[sub_s:sub_e]
            yyy = yy[sub_s:sub_e]
            rrr = rr[sub_s:sub_e]
            vv  = v[sub_s:sub_e]
            aa  = a[sub_s:sub_e]

            if len(ttt) < 5:
                continue
            if float(ttt[-1] - ttt[0]) < float(det_cfg.min_duration):
                continue

            # Skip truly stationary subsegments (this is now safer because stop-tail is isolated)
            cam_x_sub = cam_x_seg[sub_s:sub_e] if cam_x_seg is not None else None
            cam_y_sub = cam_y_seg[sub_s:sub_e] if cam_y_seg is not None else None
            if segment_is_stationary(
                ttt, xxx, yyy, cam_x_sub, cam_y_sub,
                min_disp_m=1.0,
                min_r_span_m=0.8,
                min_med_speed_mps=0.5,
                max_cam_jitter_px=2.5,
            ):
                continue

            # Keep your "approaching" guards (still useful to reject sideways junk)
            if not segment_is_approaching(
                rrr,
                min_approach_m=det_cfg.seg_min_approach_m,
                max_move_away_m=det_cfg.seg_max_move_away_m,
            ):
                continue

            best = pick_best_brake_run(
                tt=ttt, v=vv, a=aa,
                accel_trigger=accel_trigger,
                min_delta_v=min_delta_v,
                min_entry_speed=min_entry_speed,
                dbg=dbg,
            )
            if best is None:
                continue

            # Convert best indices back into tt indices for this subsegment
            ev_i0 = sub_s + best.ev_i0
            ev_i1 = sub_s + best.ev_i1

            t_start = float(tt[ev_i0])
            t_end = float(tt[ev_i1])
            if (t_end - t_start) < float(det_cfg.min_duration):
                continue

            rr_evt = rr[ev_i0:ev_i1 + 1]
            if not segment_is_approaching(
                rr_evt,
                min_approach_m=det_cfg.evt_min_approach_m,
                max_move_away_m=det_cfg.evt_max_move_away_m,
            ):
                continue

            a_slice = a[ev_i0:ev_i1 + 1]
            if a_slice.size == 0 or not np.any(np.isfinite(a_slice)):
                continue
            a_robust = float(np.nanpercentile(a_slice, 5))
            robust_frac = 0.85
            if a_robust > -robust_frac * accel_trigger:
                continue
            a_min = a_robust

            # peak expand logic uses "a" indexes, unchanged
            i0 = ev_i0 + int(np.nanargmin(a[ev_i0:ev_i1 + 1]))
            keep_thr = -float(det_cfg.keep_thr_frac) * float(accel_trigger)

            L = i0
            while L > 0 and np.isfinite(a[L]) and a[L] <= keep_thr:
                L -= 1
            R = i0
            while R < len(a) - 1 and np.isfinite(a[R]) and a[R] <= keep_thr:
                R += 1

            evt_frames = (R - L + 1)
            if evt_frames < int(det_cfg.min_event_frames):
                continue

            # robust entry/exit speeds around the event — use TRUE speed, not radial
            k_pre  = 5
            k_post = 8

            pre0 = max(0, L - k_pre)
            pre1 = L + 1
            post0 = R
            post1 = min(len(v_map), R + 1 + k_post)

            v_entry = float(np.nanmedian(v_map[pre0:pre1]))
            v_exit  = float(np.nanmedian(v_map[post0:post1]))

            if dbg.enabled:
                v_entry_rad = float(np.nanmedian(np.clip(v[pre0:pre1], 0.0, None)))
                v_exit_rad  = float(np.nanmedian(np.clip(v[post0:post1], 0.0, None)))
                dbg.log("EV_SPEEDS_MAP", "v_entry", v_entry, "v_exit", v_exit,
                        "| radial", v_entry_rad, v_exit_rad)

            if not np.isfinite(v_entry) or not np.isfinite(v_exit):
                dbg_drop(dbg, "nonfinite_speed", v_entry=v_entry, v_exit=v_exit)
                continue

            if v_entry < float(min_entry_speed):
                dbg_drop(dbg, "evt_entry_too_slow", v_entry=v_entry)
                continue

            dv_evt = max(0.0, v_entry - v_exit)
            if dv_evt < float(min_delta_v):
                dbg_drop(dbg, "dv_evt<min", dv_evt=dv_evt, min_delta_v=min_delta_v)
                continue

            # stoplike must be based on TRUE speed
            if v_exit > 0.7:
                dbg_drop(dbg, "evt_not_stoplike_map", v_exit=v_exit)
                continue

            min_slowdown_frac = 0.50
            if v_exit > (1.0 - min_slowdown_frac) * v_entry:
                dbg_drop(dbg, "not_enough_slowdown_map", v_entry=v_entry, v_exit=v_exit, dv_evt=dv_evt)
                continue


            dt_evt = float(tt[R] - tt[L])
            dr_evt = float(r[L] - r[R])

            if dr_evt < -float(det_cfg.max_move_away_evt):
                continue
            if dr_evt < float(det_cfg.min_dr_evt):
                continue

            avg_decel = dv_evt / max(dt_evt, 1e-6)

            if avg_decel >= severe_thresh:
                severity = "severe"
            elif avg_decel >= moderate_thresh:
                severity = "moderate"
            elif avg_decel >= mild_thresh:
                severity = "mild"
            else:
                continue

            event_ts = compute_event_timestamp(samples, mask, seg_s, seg_e)

            event_end_full_idx = int(mask_idx[seg_s + ev_i1])  # note: ev_i1 is within [seg_s, seg_e)

            if not post_event_visibility_ok(
                samples=samples,
                cam_x_all=cam_x_all,
                cam_y_all=cam_y_all,
                near_edge_all=near_edge_all,
                event_end_full_idx=event_end_full_idx,
                min_post_event_frames=edge_cfg.min_post_event_frames,
            ):
                continue

            tail = int(edge_cfg.edge_tail_frames)
            ends_near_end = (N_full - 1 - event_end_full_idx) <= tail
            if ends_near_end:
                tail_start = max(0, N_full - tail)
                tail_edge_frac = float(np.mean(near_edge_all[tail_start:N_full])) if N_full > 0 else 0.0
                post_start = event_end_full_idx + 1
                post_end = post_start + int(edge_cfg.min_post_event_frames)
                post_visible = int(np.sum(~near_edge_all[post_start:post_end]))
                if tail_edge_frac >= float(edge_cfg.edge_tail_edge_frac) and post_visible < int(edge_cfg.min_post_event_frames):
                    continue

            cls = samples[0].cls
            ev = BrakingEvent(
                intersection_id=geom.intersection_id,
                approach_id=geom.approach_id,
                video=video,
                track_id=track_id,
                cls=cls,
                t_start=t_start,
                t_end=t_end,
                r_start=float(r[ev_i0]),
                r_end=float(r[ev_i1]),
                v_start=float(best.v_entry),
                v_end=float(best.v_exit),
                dv=float(best.dv),
                a_min=float(a_min),
                avg_decel=float(avg_decel),
                severity=severity,
                event_ts=event_ts,
            )

            key = events_dedupe_key(ev, tol_s=0.6)
            if key in seen:
                continue
            seen.add(key)
            events.append(ev)

    if not events:
        return []
    events.sort(key=lambda ev: (ev.avg_decel, ev.dv), reverse=True)
    return events
