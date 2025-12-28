from __future__ import annotations
from dataclasses import dataclass

G_STD = 9.80665  # m/s^2 per g


@dataclass(frozen=True)
class EdgeVetoConfig:
    frame_w: int = 1920
    frame_h: int = 1080
    edge_margin_px: int = 40
    edge_tail_frames: int = 8
    min_post_event_frames: int = 15
    edge_tail_edge_frac: float = 0.6


@dataclass(frozen=True)
class SegmentSplitterConfig:
    # near
    v_max_near: float = 18.0
    max_backstep_near: float = 1.5
    max_heading_near: float = 55.0
    a_min_near: float = -6.0
    a_max_near: float = 3.0
    j_max_near: float = 8.0
    outlier_k_near: float = 3.0

    # far
    v_max_far: float = 20.0
    max_backstep_far: float = 2.3
    max_heading_far: float = 70.0
    a_min_far: float = -7.0
    a_max_far: float = 3.5
    j_max_far: float = 10.0
    outlier_k_far: float = 3.5

    far_start_frac: float = 0.60
    min_segment_len: int = 3
    moving_eps_m: float = 0.10


@dataclass(frozen=True)
class DetectorConfig:
    smoothing_window: int = 3
    min_duration: float = 0.2
    divider_buffer_m: float = 0.5

    # local braking-window expansion around peak decel
    keep_thr_frac: float = 0.3   # keep_thr = -keep_thr_frac * accel_trigger

    # approach guards for event-local window
    min_dr_evt: float = 0.03
    max_move_away_evt: float = 0.05

    # segment-level approach guard (looser)
    seg_min_approach_m: float = 0.05
    seg_max_move_away_m: float = 0.20

    # event-level approach guard (stronger)
    evt_min_approach_m: float = 0.25
    evt_max_move_away_m: float = 0.40
