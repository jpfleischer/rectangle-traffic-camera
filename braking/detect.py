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
    brake_mask = (a <= -accel_trigger) & np.isfinite(a) & np.isfinite(v)
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
        if pre.size == 0 or post.size == 0:
            continue

        v_entry = float(np.nanmedian(pre))
        v_exit = float(np.nanmedian(post))
        dv = max(0.0, max(0.0, v_entry) - max(0.0, v_exit))

        dt = float(tt[ev_i1] - tt[left])
        if dt <= 1e-6:
            continue

        if dbg.enabled:
            a_run_min = float(np.nanmin(a[bs:be])) if np.any(np.isfinite(a[bs:be])) else float("nan")
            dbg.log("  RUN", "bs/be", (bs, be), "dv", dv, "dt", dt, "dv/dt", dv/dt, "a_min_run", a_run_min)

        if (dv < min_delta_v or v_entry < min_entry_speed) and not dbg.enabled:
            continue

        score = (dv / dt, dv, -dt)
        cand = BestRun(ev_i0=left, ev_i1=ev_i1, v_entry=v_entry, v_exit=v_exit, dv=dv, score=score)
        if best is None or cand.score > best.score:
            best = cand

    return best


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

    mask = mask_window & mask_side
    if int(mask.sum()) < 3:
        return []
    mask_idx = np.flatnonzero(mask)

    t = t_all[mask]; x = x_all[mask]; y = y_all[mask]; r = r_all[mask]; side_vals = side_vals_all[mask]

    segments = split_into_plausible_segments(
        t, x, y, r, side_vals,
        braking_window_m=geom.braking_window_m,
        cfg=seg_cfg,
        video=video,
        track_id=track_id,
    )
    if not segments:
        # relaxed retry (same as before)
        relaxed = SegmentSplitterConfig(
            **{**seg_cfg.__dict__},
        )
        segments = split_into_plausible_segments(
            t, x, y, r, side_vals,
            braking_window_m=geom.braking_window_m,
            cfg=SegmentSplitterConfig(
                v_max_near=seg_cfg.v_max_near,
                v_max_far=seg_cfg.v_max_far,
                max_backstep_near=2.5,
                max_backstep_far=3.5,
                max_heading_near=85.0,
                max_heading_far=95.0,
                j_max_near=14.0,
                j_max_far=18.0,
                outlier_k_near=5.0,
                outlier_k_far=6.0,
                far_start_frac=seg_cfg.far_start_frac,
                min_segment_len=seg_cfg.min_segment_len,
                moving_eps_m=seg_cfg.moving_eps_m,
                a_min_near=seg_cfg.a_min_near,
                a_min_far=seg_cfg.a_min_far,
                a_max_near=seg_cfg.a_max_near,
                a_max_far=seg_cfg.a_max_far,
            ),
            video=video,
            track_id=track_id,
        )
    if not segments:
        return []

    events: List[BrakingEvent] = []

    for seg_s, seg_e in segments:
        tt = t[seg_s:seg_e]
        rr = r[seg_s:seg_e]
        if len(tt) < 5:
            continue
        if float(tt[-1] - tt[0]) < float(det_cfg.min_duration):
            continue

        v_raw = -np.gradient(rr, tt)
        v = smooth_1d(v_raw, det_cfg.smoothing_window)
        v = np.where(np.isfinite(v), np.clip(v, -5.0, 40.0), np.nan)

        a_raw = np.gradient(v, tt)
        a = smooth_1d(a_raw, det_cfg.smoothing_window)

        if not segment_is_approaching(
            rr,
            min_approach_m=det_cfg.seg_min_approach_m,
            max_move_away_m=det_cfg.seg_max_move_away_m,
        ):
            continue

        best = pick_best_brake_run(
            tt=tt, v=v, a=a,
            accel_trigger=accel_trigger,
            min_delta_v=min_delta_v,
            min_entry_speed=min_entry_speed,
            dbg=dbg,
        )
        if best is None:
            continue

        t_start = float(tt[best.ev_i0])
        t_end = float(tt[best.ev_i1])
        if (t_end - t_start) < float(det_cfg.min_duration):
            continue

        rr_evt = rr[best.ev_i0:best.ev_i1 + 1]
        if not segment_is_approaching(
            rr_evt,
            min_approach_m=det_cfg.evt_min_approach_m,
            max_move_away_m=det_cfg.evt_max_move_away_m,
        ):
            continue

        a_slice = a[best.ev_i0:best.ev_i1 + 1]
        a_robust = float(np.nanpercentile(a_slice, 5))
        if a_robust > -accel_trigger:
            continue
        a_min = a_robust

        a_evt = a[best.ev_i0:best.ev_i1 + 1]
        if a_evt.size == 0 or not np.any(np.isfinite(a_evt)):
            continue
        i0 = best.ev_i0 + int(np.nanargmin(a_evt))

        keep_thr = -float(det_cfg.keep_thr_frac) * float(accel_trigger)
        L = i0
        while L > 0 and np.isfinite(a[L]) and a[L] <= keep_thr:
            L -= 1
        R = i0
        while R < len(a) - 1 and np.isfinite(a[R]) and a[R] <= keep_thr:
            R += 1

        dt_evt = float(tt[R] - tt[L])
        dr_evt = float(rr[L] - rr[R])

        if dr_evt < -float(det_cfg.max_move_away_evt):
            continue
        if dr_evt < float(det_cfg.min_dr_evt):
            continue
        if dt_evt < float(det_cfg.min_duration):
            continue

        dv_evt = max(0.0, float(max(0.0, v[L]) - max(0.0, v[R])))
        if dv_evt < float(min_delta_v):
            continue

        avg_decel = dv_evt / dt_evt

        if avg_decel >= severe_thresh:
            severity = "severe"
        elif avg_decel >= moderate_thresh:
            severity = "moderate"
        elif avg_decel >= mild_thresh:
            severity = "mild"
        else:
            continue

        event_ts = compute_event_timestamp(samples, mask, seg_s, seg_e)

        if seg_e <= 0 or seg_e > len(mask_idx):
            continue
        event_end_full_idx = int(mask_idx[seg_s + best.ev_i1])

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
        events.append(
            BrakingEvent(
                intersection_id=geom.intersection_id,
                approach_id=geom.approach_id,
                video="",
                track_id=-1,
                cls=cls,
                t_start=t_start,
                t_end=t_end,
                r_start=float(rr[best.ev_i0]),
                r_end=float(rr[best.ev_i1]),
                v_start=float(best.v_entry),
                v_end=float(best.v_exit),
                dv=float(best.dv),
                a_min=float(a_min),
                avg_decel=float(avg_decel),
                severity=severity,
                event_ts=event_ts,
            )
        )

    if not events:
        return []
    events.sort(key=lambda ev: (ev.avg_decel, ev.dv), reverse=True)
    return [events[0]]
