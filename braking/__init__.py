"""
braking — braking episode detection from ClickHouse trajectories.
"""
from .detect import compute_braking_for_track
from .geometry import fetch_geometry, Geometry
from .track import fetch_tracks, Sample
from .writeback import ensure_braking_events_schema, insert_braking_events

__all__ = [
    "compute_braking_for_track",
    "fetch_geometry",
    "Geometry",
    "fetch_tracks",
    "Sample",
    "ensure_braking_events_schema",
    "insert_braking_events",
]
