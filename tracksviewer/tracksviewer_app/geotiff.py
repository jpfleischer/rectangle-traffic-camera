from typing import Tuple

import numpy as np
import rasterio as rio


def ortho_units_to_m(src) -> float:
    try:
        return float(src.crs.linear_units_factor[1])
    except Exception:
        return 1.0


def read_ortho_rgb(path: str) -> Tuple[np.ndarray, float, object, float]:
    """
    Returns:
      ortho_rgb_uint8, m_per_px, transform, units_to_m
    """
    with rio.open(path) as src:
        units_to_m = ortho_units_to_m(src)
        m_per_px = abs(src.transform.a) * units_to_m
        transform = src.transform

        arr = src.read([1, 2, 3]) if src.count >= 3 else src.read(1)
        if arr.ndim == 3:
            ortho = np.moveaxis(arr, 0, 2)
        else:
            ortho = np.stack([arr, arr, arr], axis=2)

        if ortho.dtype != np.uint8:
            ortho = np.clip(ortho, 0, 255).astype(np.uint8)

        return ortho, m_per_px, transform, units_to_m
