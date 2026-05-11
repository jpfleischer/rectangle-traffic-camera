"""
braking — braking episode detection from ClickHouse trajectories.
"""

__all__ = [
    "compute_braking_for_track",
    "export_event_clips",
    "fetch_braking_events_for_export",
    "fetch_geometry",
    "Geometry",
    "fetch_tracks",
    "Sample",
    "ensure_braking_events_schema",
    "insert_braking_events",
]


try:
    from .detect import compute_braking_for_track
    from .geometry import fetch_geometry, Geometry
    from .track import fetch_tracks, Sample
    from .writeback import ensure_braking_events_schema, insert_braking_events
except ModuleNotFoundError:
    # Some entrypoints only need a subset of the package and shouldn't fail
    # just because an optional dependency for another module is missing.
    pass

from .export_clips import export_event_clips, fetch_braking_events_for_export
