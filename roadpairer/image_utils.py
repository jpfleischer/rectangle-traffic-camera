import numpy as np
import cv2
from PySide6 import QtGui


def to_8bit_rgb(arr, nodata=None):
    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=2)
    elif arr.shape[2] == 1:
        arr = np.repeat(arr, 3, axis=2)
    if arr.dtype == np.uint8:
        return arr
    o = arr.astype(np.float32, copy=True)
    mask = None
    if nodata is not None:
        mask = np.any(arr == nodata, axis=2)

    def pct(ch):
        if mask is not None:
            vals = ch[~mask]
            if vals.size == 0:
                return 0.0, 1.0
            lo, hi = np.percentile(vals, (1, 99))
        else:
            lo, hi = np.percentile(ch, (1, 99))
        if hi <= lo:
            hi = lo + 1.0
        return lo, hi

    for c in range(o.shape[2]):
        lo, hi = pct(o[..., c])
        o[..., c] = np.clip((o[..., c] - lo) / (hi - lo) * 255.0, 0, 255)
    return o.astype(np.uint8, copy=False)


def cv_bgr_to_qimage(bgr: np.ndarray) -> QtGui.QImage:
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    h, w, ch = rgb.shape
    return QtGui.QImage(rgb.data, w, h, ch * w, QtGui.QImage.Format_RGB888).copy()


def clamp(v, lo, hi):
    return max(lo, min(hi, v))