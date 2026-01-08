#!/usr/bin/env python3
"""
ortho_overlay.py

Utilities to build a publication-friendly overlay:
darkened ortho (GeoTIFF) + warped camera image blended on top.

No Qt dependencies. Uses rasterio + OpenCV + NumPy.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import cv2
import rasterio as rio


def to_8bit_rgb(arr: np.ndarray, nodata=None) -> np.ndarray:
    """
    Convert (H,W) or (H,W,C) array to uint8 RGB via per-channel percentile stretch.
    - If nodata is provided, pixels equal to nodata (in any channel) are excluded from percentiles.
    """
    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=2)
    elif arr.shape[2] == 1:
        arr = np.repeat(arr, 3, axis=2)

    if arr.dtype == np.uint8:
        # assume already 0..255
        return arr

    o = arr.astype(np.float32, copy=True)
    mask = None
    if nodata is not None:
        # mask out nodata pixels (any channel)
        mask = np.any(arr == nodata, axis=2)

    def pct(ch: np.ndarray) -> tuple[float, float]:
        if mask is not None:
            vals = ch[~mask]
            if vals.size == 0:
                return 0.0, 1.0
            lo, hi = np.percentile(vals, (1, 99))
        else:
            lo, hi = np.percentile(ch, (1, 99))
        if hi <= lo:
            hi = lo + 1.0
        return float(lo), float(hi)

    for c in range(o.shape[2]):
        lo, hi = pct(o[..., c])
        o[..., c] = np.clip((o[..., c] - lo) / (hi - lo) * 255.0, 0, 255)

    return o.astype(np.uint8, copy=False)


def load_ortho_bgr(ortho_path: str | Path) -> np.ndarray:
    """
    Load a GeoTIFF ortho as 8-bit BGR for OpenCV blending.
    Reads bands 1..3 if available, else single band replicated.
    """
    ortho_path = str(ortho_path)
    with rio.open(ortho_path) as src:
        nodata = src.nodata
        bands = [1, 2, 3] if src.count >= 3 else [1]
        arr = src.read(bands)            # (C,H,W)
        arr = np.moveaxis(arr, 0, 2)     # (H,W,C)
        rgb8 = to_8bit_rgb(arr, nodata=nodata)
        bgr8 = cv2.cvtColor(rgb8, cv2.COLOR_RGB2BGR)
        return bgr8


def infer_mask_from_bev(bev_bgr: np.ndarray) -> np.ndarray:
    """
    Infer a 0/255 mask from non-black pixels in the warped camera image.
    """
    m = np.any(bev_bgr != 0, axis=2).astype(np.uint8) * 255
    return m


def overlay_warp_on_ortho(
    *,
    ortho_bgr: np.ndarray,
    bev_bgr: np.ndarray,
    mask_u8: Optional[np.ndarray] = None,
    alpha_strength: float = 0.70,
    background_darken: float = 0.65,
    draw_outline: bool = True,
    outline_thickness: int = 3,
) -> np.ndarray:
    """
    Return blended overlay image (uint8 BGR).

    ortho_bgr: (H,W,3) uint8
    bev_bgr:   (H,W,3) uint8 (warped camera already in ortho pixel space)
    mask_u8:   (H,W) uint8 0/255 where overlay applies. If None, inferred from bev.
    """
    if ortho_bgr.shape[:2] != bev_bgr.shape[:2]:
        raise ValueError(
            f"Shape mismatch: ortho {ortho_bgr.shape[:2]} vs bev {bev_bgr.shape[:2]}"
        )

    if mask_u8 is None:
        mask_u8 = infer_mask_from_bev(bev_bgr)
    if mask_u8.shape[:2] != ortho_bgr.shape[:2]:
        raise ValueError(
            f"Mask mismatch: mask {mask_u8.shape[:2]} vs ortho {ortho_bgr.shape[:2]}"
        )

    base_f = ortho_bgr.astype(np.float32) * float(background_darken)
    bev_f = bev_bgr.astype(np.float32)

    a = (mask_u8.astype(np.float32) / 255.0) * float(alpha_strength)
    a3 = a[..., None]

    out = base_f * (1.0 - a3) + bev_f * a3
    out = np.clip(out, 0, 255).astype(np.uint8)

    if draw_outline:
        edges = cv2.Canny(mask_u8, 50, 150)
        # thickness control via dilation
        iters = max(1, int(round(outline_thickness / 2)))
        edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=iters)
        out[edges > 0] = (255, 255, 255)

    return out


def save_overlay_figure(
    *,
    ortho_path: str | Path,
    bev_bgr: np.ndarray,
    out_path: str | Path,
    mask_u8: Optional[np.ndarray] = None,
    alpha_strength: float = 0.70,
    background_darken: float = 0.65,
    draw_outline: bool = True,
    outline_thickness: int = 3,
) -> str:
    """
    Convenience: load ortho from GeoTIFF, blend with bev, and write PNG/JPG.
    Returns out_path as string.
    """
    ortho_bgr = load_ortho_bgr(ortho_path)
    if ortho_bgr.shape[:2] != bev_bgr.shape[:2]:
        # Robust fallback: resize ortho to match bev (should rarely be needed)
        ortho_bgr = cv2.resize(ortho_bgr, (bev_bgr.shape[1], bev_bgr.shape[0]), interpolation=cv2.INTER_LINEAR)

    out = overlay_warp_on_ortho(
        ortho_bgr=ortho_bgr,
        bev_bgr=bev_bgr,
        mask_u8=mask_u8,
        alpha_strength=alpha_strength,
        background_darken=background_darken,
        draw_outline=draw_outline,
        outline_thickness=outline_thickness,
    )
    out_path = str(out_path)
    cv2.imwrite(out_path, out)
    return out_path
