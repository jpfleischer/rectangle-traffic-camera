from __future__ import annotations
import argparse
import logging
import os

import dotenv

from .chio import ClickHouseHTTP
from .config import G_STD, DetectorConfig, EdgeVetoConfig, SegmentSplitterConfig
from .detect import compute_braking_for_track
from .geometry import fetch_geometry
from .track import fetch_tracks
from .units import units_to_m_factor
from .writeback import ensure_braking_events_schema, insert_braking_events


def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Compute braking episodes from trajectories.raw + stopbar_metadata")
    ap.add_argument("--intersection-id", required=True)
    ap.add_argument("--approach-id", required=True)
    ap.add_argument("--video", default=None)

    ap.add_argument("--min-entry-speed", type=float, default=0.5)
    ap.add_argument("--min-delta-v", type=float, default=3.0)
    ap.add_argument("--accel-trigger", type=float, default=1.5, help="in g")

    ap.add_argument("--mild-g", type=float, default=0.10)
    ap.add_argument("--moderate-g", type=float, default=0.20)
    ap.add_argument("--severe-g", type=float, default=0.40)

    ap.add_argument("--window-m", type=float, default=None)
    ap.add_argument("--smooth", type=int, default=3)
    ap.add_argument("--min-duration", type=float, default=0.1)

    ap.add_argument("--write-events", action="store_true")
    ap.add_argument("--divider-side", choices=["positive", "negative"], default="positive")

    ap.add_argument("--stopbar-x", type=float, default=None)
    ap.add_argument("--stopbar-y", type=float, default=None)
    ap.add_argument("--disable-window", action="store_true")

    ap.add_argument("--frame-w", type=int, default=1920)
    ap.add_argument("--frame-h", type=int, default=1080)
    ap.add_argument("--fps", type=float, default=15.0)
    ap.add_argument("--min-event-frames", type=int, default=4)   # 4 frames ~= 0.2s at 15 fps

    ap.add_argument("--edge-margin-px", type=int, default=40)
    ap.add_argument("--edge-tail-frames", type=int, default=8)
    ap.add_argument("--min-post-event-frames", type=int, default=15)
    ap.add_argument("--edge-tail-edge-frac", type=float, default=0.6)

    ap.add_argument("-v", "--verbose", action="store_true")
    return ap


def main() -> None:
    dotenv.load_dotenv()

    ap = build_argparser()
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)8s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    logger = logging.getLogger("braking")

    ch = ClickHouseHTTP(
        host=os.getenv("CH_HOST", "example.com"),
        port=int(os.getenv("CH_PORT", "8123")),
        user=os.getenv("CH_USER", "default"),
        password=os.getenv("CH_PASSWORD", ""),
        database=os.getenv("CH_DB", "trajectories"),
        logger=logger,
    )

    side_sign = 1.0 if args.divider_side == "positive" else -1.0

    geom = fetch_geometry(ch, args.intersection_id, args.approach_id)
    logger.info("GeoTIFF units->meters factor: %.12f", units_to_m_factor())

    if args.stopbar_x is not None and args.stopbar_y is not None:
        geom.stopbar_center = (float(args.stopbar_x), float(args.stopbar_y))
        logger.info("Overriding stopbar center to (%.3f, %.3f)", *geom.stopbar_center)

    if args.window_m is not None and args.window_m > 0:
        geom.braking_window_m = float(args.window_m)

    logger.info("Using stopbar center=%s braking_window=%.1f m", geom.stopbar_center, geom.braking_window_m)

    tracks = fetch_tracks(
        ch,
        video_filter=args.video,
        intersection_id=int(args.intersection_id),
    )
    logger.info("Loaded %d tracks", len(tracks))

    accel_trigger_mps2 = float(args.accel_trigger) * G_STD
    mild_mps2 = float(args.mild_g) * G_STD
    moderate_mps2 = float(args.moderate_g) * G_STD
    severe_mps2 = float(args.severe_g) * G_STD

    det_cfg = DetectorConfig(
        smoothing_window=int(args.smooth),
        min_duration=float(args.min_duration),
        fps=float(args.fps),
        min_event_frames=int(args.min_event_frames),
    )
    seg_cfg = SegmentSplitterConfig()
    edge_cfg = EdgeVetoConfig(
        frame_w=int(args.frame_w),
        frame_h=int(args.frame_h),
        edge_margin_px=int(args.edge_margin_px),
        edge_tail_frames=int(args.edge_tail_frames),
        min_post_event_frames=int(args.min_post_event_frames),
        edge_tail_edge_frac=float(args.edge_tail_edge_frac),
    )

    events = []
    veh_with_any_event = set()

    for (video, track_id), samples in tracks.items():
        evs = compute_braking_for_track(
            geom=geom,
            video=video,
            track_id=track_id,
            samples=samples,
            min_entry_speed=float(args.min_entry_speed),
            min_delta_v=float(args.min_delta_v),
            accel_trigger=accel_trigger_mps2,
            severe_thresh=severe_mps2,
            moderate_thresh=moderate_mps2,
            mild_thresh=mild_mps2,
            disable_window=bool(args.disable_window),
            side_sign=side_sign,
            det_cfg=det_cfg,
            seg_cfg=seg_cfg,
            edge_cfg=edge_cfg,
        )

        for ev in evs:
            ev.video = video
            ev.track_id = track_id

        if evs:
            veh_with_any_event.add((video, track_id))
            events.extend(evs)

    logger.info("Total vehicles with BEV points: %d", len(tracks))
    logger.info("Vehicles with >=1 braking event: %d", len(veh_with_any_event))
    logger.info("Total braking events: %d", len(events))

    sev_counts = {"mild": 0, "moderate": 0, "severe": 0}
    for ev in events:
        if ev.severity in sev_counts:
            sev_counts[ev.severity] += 1
    logger.info("Severity counts: mild=%d, moderate=%d, severe=%d",
                sev_counts["mild"], sev_counts["moderate"], sev_counts["severe"])

    if args.write_events and events:
        ensure_braking_events_schema(ch)
        insert_braking_events(ch, events)
        logger.info("Inserted %d events into %s.braking_events", len(events), ch.db)


if __name__ == "__main__":
    main()
