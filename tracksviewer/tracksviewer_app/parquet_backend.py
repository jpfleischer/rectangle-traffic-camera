from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, date
from pathlib import Path
from typing import Optional, List, Dict, Tuple

import json
import pandas as pd


EventId = Tuple[str, int, float, float, str]  # (video, track_id, t_start, t_end, created_at)


@dataclass
class ParquetStore:
    root: Path  # export root that contains raw/ and braking_events/

    # ---- in-memory caches ----
    _video_base_index: Dict[str, datetime] = field(default_factory=dict, init=False)
    _tombstones: set[EventId] = field(default_factory=set, init=False)
    _tombstones_loaded: bool = field(default=False, init=False)

    @property
    def raw_dir(self) -> Path:
        return self.root / "raw"

    @property
    def braking_path(self) -> Path:
        return self.root / "braking_events" / "braking_events.parquet"

    # ---- NEW: tombstone file ----
    @property
    def tombstone_path(self) -> Path:
        # Keep next to braking_events parquet
        return self.root / "braking_events" / "braking_events.tombstones.jsonl"

    # ---- NEW: base timestamp index ----
    @property
    def video_base_index_path(self) -> Path:
        return self.root / "video_base_ts.parquet"

    # ---------------------------
    # Tombstones
    # ---------------------------
    def _ensure_tombstones_loaded(self) -> None:
        if self._tombstones_loaded:
            return
        self._tombstones_loaded = True
        p = self.tombstone_path
        if not p.exists():
            return
        try:
            with p.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    obj = json.loads(line)
                    ev_id = self._norm_event_id(obj.get("event_id"))
                    if ev_id is not None:
                        self._tombstones.add(ev_id)
        except Exception:
            # If tombstones are corrupt, fail "softly": treat as none loaded.
            self._tombstones = set()

    def _norm_event_id(self, event_id) -> Optional[EventId]:
        if event_id is None:
            return None
        if isinstance(event_id, list):
            event_id = tuple(event_id)
        if not (isinstance(event_id, tuple) and len(event_id) == 5):
            return None
        video, track_id, t_start, t_end, created_at = event_id
        try:
            return (str(video), int(track_id), float(t_start), float(t_end), str(created_at))
        except Exception:
            return None

    def tombstone_event(self, ev_or_event_id) -> Optional[EventId]:
        """
        Record a deletion for parquet mode.
        Accepts either:
          - event_id tuple/list, or
          - event dict containing video/track_id/t_start/t_end/created_at (your GUI rows)
        """
        if isinstance(ev_or_event_id, (tuple, list)):
            event_id = self._norm_event_id(ev_or_event_id)
        elif isinstance(ev_or_event_id, dict):
            event_id = (
                str(ev_or_event_id.get("video", "")),
                int(ev_or_event_id.get("track_id", 0) or 0),
                float(ev_or_event_id.get("t_start", 0.0) or 0.0),
                float(ev_or_event_id.get("t_end", 0.0) or 0.0),
                str(ev_or_event_id.get("created_at", "")),
            )
            event_id = self._norm_event_id(event_id)
        else:
            event_id = None

        if event_id is None:
            return None

        self._ensure_tombstones_loaded()
        if event_id in self._tombstones:
            return event_id  # already deleted

        # Append-only write
        self.tombstone_path.parent.mkdir(parents=True, exist_ok=True)
        rec = {"event_id": list(event_id), "deleted_at": datetime.utcnow().isoformat(timespec="seconds")}
        with self.tombstone_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")

        self._tombstones.add(event_id)
        return event_id

    def is_tombstoned(self, event_id: EventId) -> bool:
        self._ensure_tombstones_loaded()
        return event_id in self._tombstones

    # ---------------------------
    # Braking events
    # ---------------------------
    def load_braking_events(self) -> pd.DataFrame:
        p = self.braking_path
        if not p.exists():
            raise FileNotFoundError(f"Missing braking_events parquet: {p}")

        df = pd.read_parquet(p)

        # normalize types
        if "track_id" in df.columns:
            df["track_id"] = pd.to_numeric(df["track_id"], errors="coerce").fillna(0).astype(int)
        if "t_start" in df.columns:
            df["t_start"] = pd.to_numeric(df["t_start"], errors="coerce").fillna(0.0).astype(float)
        if "t_end" in df.columns:
            df["t_end"] = pd.to_numeric(df["t_end"], errors="coerce").fillna(0.0).astype(float)
        if "created_at" in df.columns:
            df["created_at"] = df["created_at"].astype(str)

        # Apply tombstones (this is the important part you noticed missing)
        self._ensure_tombstones_loaded()
        if self._tombstones and not df.empty:
            # build event_id columns to filter quickly
            vid = df.get("video", "").astype(str)
            tid = df.get("track_id", 0).astype(int)
            ts = df.get("t_start", 0.0).astype(float)
            te = df.get("t_end", 0.0).astype(float)
            ca = df.get("created_at", "").astype(str)

            # tuple per row (fast enough; braking_events is usually not enormous)
            ev_ids = list(zip(vid, tid, ts, te, ca))
            mask = [eid not in self._tombstones for eid in ev_ids]
            df = df.loc[mask].copy()

        return df

    # ---------------------------
    # Raw interval loading
    # ---------------------------
    def _days_in_range(self, start: datetime, end: datetime) -> List[date]:
        d0 = start.date()
        d1 = (end - timedelta(microseconds=1)).date()
        days = []
        cur = d0
        while cur <= d1:
            days.append(cur)
            cur = cur + timedelta(days=1)
        return days

    def load_raw_interval(
        self,
        start_dt: datetime,
        duration_s: float,
        video_filter: Optional[str] = None,
    ) -> pd.DataFrame:
        end_dt = start_dt + timedelta(seconds=float(duration_s))

        frames: List[pd.DataFrame] = []
        cols = ["video", "timestamp", "track_id", "class", "map_m_x", "map_m_y"]

        def _safe_video_name(v: str) -> str:
            # must match exporter’s folder-safe transform
            return v.replace("/", "_").replace("\\", "_")

        for day in self._days_in_range(start_dt, end_dt):
            # ---- NEW layout: raw/day=YYYY-MM-DD/video=.../part.parquet ----
            day_dir = self.raw_dir / f"day={day.isoformat()}"
            if day_dir.exists() and day_dir.is_dir():
                if video_filter:
                    vdir = day_dir / f"video={_safe_video_name(video_filter)}"
                    part = vdir / "part.parquet"
                    if part.exists():
                        frames.append(pd.read_parquet(part, columns=cols))
                else:
                    # load all videos for that day
                    for vdir in sorted(day_dir.glob("video=*/")):
                        part = vdir / "part.parquet"
                        if part.exists():
                            frames.append(pd.read_parquet(part, columns=cols))

            # ---- OLD layout fallback: raw/part-YYYY-MM-DD.parquet ----
            old_part = self.raw_dir / f"part-{day.isoformat()}.parquet"
            if old_part.exists():
                frames.append(pd.read_parquet(old_part, columns=cols))

        if not frames:
            return pd.DataFrame(columns=cols)

        df = pd.concat(frames, ignore_index=True)

        # normalize + filter interval
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        df = df[df["timestamp"].notna()]

        df = df[(df["timestamp"] >= start_dt) & (df["timestamp"] < end_dt)]
        df = df[(df["map_m_x"] != 0) | (df["map_m_y"] != 0)]

        if video_filter:
            df = df[df["video"] == video_filter]

        df = df.sort_values(["timestamp", "video", "track_id"], kind="mergesort")
        return df

    # ---------------------------
    # Video base timestamp index
    # ---------------------------
    def _ensure_video_base_index(self) -> None:
        if self._video_base_index:
            return

        p = self.video_base_index_path
        if p.exists():
            df = pd.read_parquet(p)
            df["ts_min"] = pd.to_datetime(df["ts_min"], errors="coerce")
            df = df[df["ts_min"].notna()]
            self._video_base_index = {
                str(v): ts.to_pydatetime()
                for v, ts in zip(df["video"], df["ts_min"])
            }
            return

        # Build once (can take time)
        mins = []

        # NEW layout: raw/day=.../video=.../part.parquet
        for part in sorted(self.raw_dir.glob("day=*/video=*/part.parquet")):
            df = pd.read_parquet(part, columns=["video", "timestamp"])
            df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
            df = df[df["timestamp"].notna()]
            if df.empty:
                continue
            g = df.groupby("video", sort=False)["timestamp"].min()
            mins.append(g)

        # OLD layout: raw/part-YYYY-MM-DD.parquet
        for part in sorted(self.raw_dir.glob("part-*.parquet")):
            df = pd.read_parquet(part, columns=["video", "timestamp"])
            df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
            df = df[df["timestamp"].notna()]
            if df.empty:
                continue
            g = df.groupby("video", sort=False)["timestamp"].min()
            mins.append(g)


        if not mins:
            self._video_base_index = {}
            return

        gmin = pd.concat(mins).groupby(level=0).min()
        out = gmin.reset_index()
        out.columns = ["video", "ts_min"]
        out.to_parquet(p, index=False)

        self._video_base_index = {
            str(v): ts.to_pydatetime()
            for v, ts in zip(out["video"], out["ts_min"])
        }

    def get_video_base_timestamp(self, video: str) -> Optional[datetime]:
        self._ensure_video_base_index()
        return self._video_base_index.get(video)

    def load_deleted_event_ids(self):
        """
        Compatibility helper for older GUI code.
        Returns the set of tombstoned EventIds.
        """
        self._ensure_tombstones_loaded()
        return set(self._tombstones)
