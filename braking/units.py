from __future__ import annotations
import os
from typing import Tuple


def infer_units_to_m_factor_from_ortho() -> float:
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
            try:
                return float(crs.linear_units_factor[1])
            except Exception:
                units = (getattr(crs, "linear_units", None) or "").lower()
                if "us survey foot" in units:
                    return 0.30480060960121924
                if units in ("foot", "feet"):
                    return 0.3048
                return 1.0
    except Exception:
        return 1.0


_UNITS_TO_M = infer_units_to_m_factor_from_ortho()


def units_to_m(x: float) -> float:
    return float(x) * _UNITS_TO_M


def units2_to_m_xy(x: float, y: float) -> Tuple[float, float]:
    return (units_to_m(x), units_to_m(y))


def units_to_m_factor() -> float:
    return float(_UNITS_TO_M)
