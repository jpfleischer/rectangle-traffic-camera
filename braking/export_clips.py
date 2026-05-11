from __future__ import annotations

import argparse
import concurrent.futures as cf
import logging
import os
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence

from .chio import ClickHouseHTTP, ch_query_json_each_row

try:
    import dotenv
except ModuleNotFoundError:  # pragma: no cover - optional at runtime
    dotenv = None


@dataclass
class ClipEvent:
    intersection_id: str
    approach_id: str
    video: str
    track_id: int
    cls: str
    t_start: float
    t_end: float
    severity: str
    event_ts: Optional[str]


def _safe_slug(text: str) -> str:
    text = re.sub(r"[^A-Za-z0-9._=-]+", "_", str(text).strip())
    return text.strip("._") or "unknown"


def _secs_slug(value: float) -> str:
    return f"{value:09.3f}".replace(".", "p")


def _derive_video_candidates(video_name: str, prefer_smooth: bool) -> List[str]:
    if video_name.lower().endswith(".mp4"):
        smooth = video_name[:-4] + "_track_smooth.mp4"
    else:
        smooth = video_name + "_track_smooth.mp4"

    if prefer_smooth:
        return [smooth, video_name]
    return [video_name, smooth]


def _intersection_match_rank(path: Path, event: ClipEvent) -> int:
    """
    Lower is better.
    0: explicit match for this intersection
    1: ambiguous path that doesn't mention any intersection
    2: explicit mention of some other intersection
    """
    parts_lower = [p.lower() for p in path.parts]
    target_tokens = [
        f"intersection_{event.intersection_id}".lower(),
        f"intersection-{event.intersection_id}".lower(),
        f"intersection{event.intersection_id}".lower(),
    ]
    if any(tok in part for tok in target_tokens for part in parts_lower):
        return 0
    if any("intersection_" in part or "intersection-" in part or "intersection" in part for part in parts_lower):
        return 2
    return 1


def _score_video_path(path: Path, event: ClipEvent, candidate_rank: int) -> tuple[int, int, int, int, str]:
    """
    Lower scores are better.
    Prefer:
    1. paths that explicitly match the event's intersection
    2. candidate preference order (raw vs smooth depending on CLI)
    3. shallower paths
    4. deterministic lexical order
    """
    intersection_rank = _intersection_match_rank(path, event)
    smooth_penalty = 0 if path.name.endswith("_track_smooth.mp4") else 1
    return (intersection_rank, candidate_rank, smooth_penalty, len(path.parts), str(path))


def _resolve_input_video(mp4_root: Path, event: ClipEvent, prefer_smooth: bool) -> Optional[Path]:
    ranked_matches = []
    candidates = _derive_video_candidates(event.video, prefer_smooth)
    for candidate_rank, candidate in enumerate(candidates):
        direct = (mp4_root / candidate).resolve()
        if direct.exists():
            ranked_matches.append((candidate_rank, direct))

        matches = [p.resolve() for p in mp4_root.rglob(candidate) if p.is_file()]
        for path in matches:
            ranked_matches.append((candidate_rank, path))

    if not ranked_matches:
        return None

    uniq = {}
    for candidate_rank, path in ranked_matches:
        prev = uniq.get(path)
        if prev is None or candidate_rank < prev:
            uniq[path] = candidate_rank

    best = min(uniq.items(), key=lambda item: _score_video_path(item[0], event, item[1]))
    return best[0]


def _ffmpeg_clip_cmd(
    input_path: Path,
    output_path: Path,
    clip_start_s: float,
    clip_end_s: float,
    overwrite: bool,
    ffmpeg_threads: Optional[int],
) -> List[str]:
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y" if overwrite else "-n",
        "-ss",
        f"{clip_start_s:.3f}",
        "-to",
        f"{clip_end_s:.3f}",
        "-i",
        str(input_path),
        "-map",
        "0:v:0",
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "fast",
        "-crf",
        "18",
    ]
    if ffmpeg_threads is not None and int(ffmpeg_threads) > 0:
        cmd.extend(["-threads", str(int(ffmpeg_threads))])
    cmd.append(str(output_path))
    return cmd


def _default_jobs() -> int:
    cpu = os.cpu_count() or 1
    return max(1, min(2, cpu))


def fetch_braking_events_for_export(
    ch: ClickHouseHTTP,
    *,
    intersection_id: Optional[str] = None,
    approach_id: Optional[str] = None,
    video: Optional[str] = None,
    severity: Optional[str] = None,
    limit: Optional[int] = None,
) -> List[ClipEvent]:
    db = ch.db
    where: List[str] = []
    params = {}

    if intersection_id:
        where.append("intersection_id = {intersection_id:String}")
        params["intersection_id"] = str(intersection_id)
    if approach_id:
        where.append("approach_id = {approach_id:String}")
        params["approach_id"] = str(approach_id)
    if video:
        where.append("video = {video:String}")
        params["video"] = str(video)
    if severity:
        where.append("severity = {severity:String}")
        params["severity"] = str(severity)

    where_sql = ""
    if where:
        where_sql = "WHERE " + " AND ".join(where)

    limit_sql = ""
    if limit is not None and int(limit) > 0:
        limit_sql = f"LIMIT {int(limit)}"

    sql = f"""
    SELECT
        intersection_id,
        approach_id,
        video,
        track_id,
        class,
        t_start,
        t_end,
        severity,
        event_ts
    FROM {db}.braking_events
    {where_sql}
    ORDER BY intersection_id, approach_id, video, t_start, track_id
    {limit_sql}
    FORMAT JSONEachRow
    """

    rows = ch_query_json_each_row(ch, sql if not params else _inject_params(sql, params))
    events: List[ClipEvent] = []
    for row in rows:
        events.append(
            ClipEvent(
                intersection_id=str(row["intersection_id"]),
                approach_id=str(row["approach_id"]),
                video=str(row["video"]),
                track_id=int(row["track_id"]),
                cls=str(row.get("class", "")),
                t_start=float(row["t_start"]),
                t_end=float(row["t_end"]),
                severity=str(row.get("severity", "")),
                event_ts=(None if row.get("event_ts") in (None, "") else str(row["event_ts"])),
            )
        )
    return events


def _inject_params(sql: str, params: dict) -> str:
    """
    Keep export logic simple by inlining safe string parameters into SQL.
    """
    out = sql
    for key, value in params.items():
        escaped = str(value).replace("\\", "\\\\").replace("'", "\\'")
        out = out.replace("{" + f"{key}:String" + "}", f"'{escaped}'")
    return out


def build_output_path(output_root: Path, event: ClipEvent) -> Path:
    folder = (
        output_root
        / f"intersection={_safe_slug(event.intersection_id)}"
        / f"approach={_safe_slug(event.approach_id)}"
        / f"video={_safe_slug(Path(event.video).stem)}"
    )
    name = (
        f"track_{event.track_id:06d}"
        f"__tstart_{_secs_slug(event.t_start)}"
        f"__tend_{_secs_slug(event.t_end)}"
        f"__{_safe_slug(event.severity or 'unknown')}.mp4"
    )
    return folder / name


def _process_one_event(
    *,
    event: ClipEvent,
    mp4_root: Path,
    output_root: Path,
    pre_seconds: float,
    post_seconds: float,
    prefer_smooth: bool,
    overwrite: bool,
    dry_run: bool,
    ffmpeg_threads: Optional[int],
    logger: logging.Logger,
) -> tuple[str, str]:
    input_path = _resolve_input_video(mp4_root, event, prefer_smooth=prefer_smooth)
    if input_path is None:
        logger.warning(
            "Skipping event video=%s intersection=%s track=%s: no matching MP4 under %s",
            event.video,
            event.intersection_id,
            event.track_id,
            mp4_root,
        )
        return ("skipped", "missing-input")

    output_path = build_output_path(output_root, event)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    clip_start_s = max(0.0, float(event.t_start) - float(pre_seconds))
    clip_end_s = max(clip_start_s, float(event.t_end) + float(post_seconds))
    cmd = _ffmpeg_clip_cmd(
        input_path=input_path,
        output_path=output_path,
        clip_start_s=clip_start_s,
        clip_end_s=clip_end_s,
        overwrite=overwrite,
        ffmpeg_threads=ffmpeg_threads,
    )

    if dry_run:
        logger.info(
            "[dry-run] %s -> %s  start=%.3fs end=%.3fs",
            input_path.name,
            output_path,
            clip_start_s,
            clip_end_s,
        )
        return ("written", "dry-run")

    if output_path.exists() and not overwrite:
        logger.info("Skipping existing clip: %s", output_path)
        return ("skipped", "exists")

    try:
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
        logger.info(
            "Wrote clip: %s  source=%s start=%.3fs end=%.3fs",
            output_path,
            input_path.name,
            clip_start_s,
            clip_end_s,
        )
        return ("written", "ok")
    except subprocess.CalledProcessError as exc:
        logger.error(
            "ffmpeg failed for video=%s track=%s start=%.3f end=%.3f: %s",
            event.video,
            event.track_id,
            clip_start_s,
            clip_end_s,
            (exc.stderr or str(exc)).strip(),
        )
        return ("skipped", "ffmpeg-error")


def export_event_clips(
    *,
    events: Sequence[ClipEvent],
    mp4_root: Path,
    output_root: Path,
    pre_seconds: float,
    post_seconds: float,
    prefer_smooth: bool,
    overwrite: bool,
    dry_run: bool,
    jobs: int,
    ffmpeg_threads: Optional[int],
    logger: logging.Logger,
) -> tuple[int, int]:
    written = 0
    skipped = 0

    jobs = max(1, int(jobs))
    if jobs == 1:
        for event in events:
            status, _reason = _process_one_event(
                event=event,
                mp4_root=mp4_root,
                output_root=output_root,
                pre_seconds=pre_seconds,
                post_seconds=post_seconds,
                prefer_smooth=prefer_smooth,
                overwrite=overwrite,
                dry_run=dry_run,
                ffmpeg_threads=ffmpeg_threads,
                logger=logger,
            )
            if status == "written":
                written += 1
            else:
                skipped += 1
        return written, skipped

    with cf.ThreadPoolExecutor(max_workers=jobs, thread_name_prefix="clip-export") as pool:
        futures = [
            pool.submit(
                _process_one_event,
                event=event,
                mp4_root=mp4_root,
                output_root=output_root,
                pre_seconds=pre_seconds,
                post_seconds=post_seconds,
                prefer_smooth=prefer_smooth,
                overwrite=overwrite,
                dry_run=dry_run,
                ffmpeg_threads=ffmpeg_threads,
                logger=logger,
            )
            for event in events
        ]
        for fut in cf.as_completed(futures):
            status, _reason = fut.result()
            if status == "written":
                written += 1
            else:
                skipped += 1

    return written, skipped


def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Export reviewable MP4 clips for braking events.")
    ap.add_argument("--intersection-id", default=None)
    ap.add_argument("--approach-id", default=None)
    ap.add_argument("--video", default=None)
    ap.add_argument("--severity", choices=["mild", "moderate", "severe"], default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--mp4-root", required=True, help="Folder containing raw or *_track_smooth.mp4 videos.")
    ap.add_argument(
        "--output-root",
        default="output/braking_clips",
        help="Destination root for exported review clips.",
    )
    ap.add_argument("--pre-seconds", type=float, default=3.0)
    ap.add_argument("--post-seconds", type=float, default=3.0)
    ap.add_argument(
        "--jobs",
        type=int,
        default=_default_jobs(),
        help="Number of clip exports to run concurrently. Defaults to a bounded parallel value.",
    )
    ap.add_argument(
        "--ffmpeg-threads",
        type=int,
        default=None,
        help="Optional thread count to pass through to each ffmpeg process.",
    )
    ap.add_argument(
        "--prefer-raw",
        action="store_true",
        help="Look for the raw video name before *_track_smooth.mp4.",
    )
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("-v", "--verbose", action="store_true")
    return ap


def main() -> None:
    if dotenv is not None:
        dotenv.load_dotenv()
    ap = build_argparser()
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)8s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    logger = logging.getLogger("braking.export_clips")

    ch = ClickHouseHTTP(
        host=os.getenv("CH_HOST", "example.com"),
        port=int(os.getenv("CH_PORT", "8123")),
        user=os.getenv("CH_USER", "default"),
        password=os.getenv("CH_PASSWORD", ""),
        database=os.getenv("CH_DB", "trajectories"),
        logger=logger,
    )

    events = fetch_braking_events_for_export(
        ch,
        intersection_id=args.intersection_id,
        approach_id=args.approach_id,
        video=args.video,
        severity=args.severity,
        limit=args.limit,
    )
    logger.info("Loaded %d braking events for export", len(events))

    written, skipped = export_event_clips(
        events=events,
        mp4_root=Path(args.mp4_root).resolve(),
        output_root=Path(args.output_root).resolve(),
        pre_seconds=float(args.pre_seconds),
        post_seconds=float(args.post_seconds),
        prefer_smooth=not bool(args.prefer_raw),
        overwrite=bool(args.overwrite),
        dry_run=bool(args.dry_run),
        jobs=int(args.jobs),
        ffmpeg_threads=args.ffmpeg_threads,
        logger=logger,
    )
    logger.info("Clip export complete: wrote=%d skipped=%d", written, skipped)


if __name__ == "__main__":
    main()
