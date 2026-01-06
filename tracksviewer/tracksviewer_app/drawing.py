from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np


def color_for_track(track_key) -> Tuple[int, int, int]:
    seed = (abs(hash(track_key)) % (2**31 - 1)) or 1
    rng = np.random.default_rng(seed)
    rgb = rng.integers(60, 255, size=3).tolist()
    return int(rgb[2]), int(rgb[1]), int(rgb[0])  # BGR


def draw_scale_bar(img: np.ndarray, m_per_px: float, meters: float = 10.0):
    if not (m_per_px and m_per_px > 0):
        return
    px = int(round(meters / m_per_px))
    h, _w = img.shape[:2]
    x0 = 40
    y0 = h - 40
    cv2.line(img, (x0, y0), (x0 + px, y0), (255, 255, 255), 6, cv2.LINE_AA)
    cv2.line(img, (x0, y0), (x0 + px, y0), (0, 0, 0), 2, cv2.LINE_AA)
    label = f"{int(meters)} m"
    cv2.putText(img, label, (x0, y0 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(img, label, (x0, y0 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)


def put_hud(img: np.ndarray, text: str, y: int = 26, scale: float = 0.6):
    cv2.putText(img, text, (12, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(img, text, (12, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), 1, cv2.LINE_AA)


def render_frame(
    base_img: np.ndarray,
    tracks: Dict[object, List[dict]],
    times: List[float],
    idx: int,
    scale: float,
    m_per_px: float,
    trail_len: int,
    show_trails: bool = True,
    highlight_keys: Optional[set] = None,
) -> np.ndarray:
    if not times:
        return base_img.copy()

    t_now = times[idx]
    frame = base_img.copy()

    start_idx = max(0, idx - trail_len) if (show_trails and trail_len > 0) else idx
    t_start = times[start_idx]

    for tid, pts in tracks.items():
        sub = [p for p in pts if t_start <= p["t"] <= t_now]
        if not sub:
            continue

        is_highlight = bool(highlight_keys and tid in highlight_keys)

        if is_highlight:
            col = (0, 255, 255)
            thickness = 3
            radius = 7
        else:
            col = (0, 255, 0)  # or color_for_track(tid)
            thickness = 2
            radius = 5

        poly = np.array([[int(p["x"] * scale), int(p["y"] * scale)] for p in sub], dtype=np.int32)
        if len(poly) >= 2 and show_trails:
            cv2.polylines(frame, [poly], False, col, thickness, cv2.LINE_AA)

        cx, cy = int(sub[-1]["x"] * scale), int(sub[-1]["y"] * scale)
        cv2.circle(frame, (cx, cy), radius, col, -1, cv2.LINE_AA)

    draw_scale_bar(frame, m_per_px, meters=10.0)
    put_hud(frame, f"t = {t_now:.2f}s  ({idx + 1}/{len(times)})", y=26)
    return frame
