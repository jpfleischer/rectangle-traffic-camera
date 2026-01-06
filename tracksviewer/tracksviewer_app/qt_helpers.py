import cv2
import numpy as np
from PySide6 import QtGui


def numpy_to_qimage(frame: np.ndarray) -> QtGui.QImage:
    if frame.ndim == 2:
        h, w = frame.shape
        return QtGui.QImage(frame.data, w, h, w, QtGui.QImage.Format_Grayscale8).copy()

    h, w, ch = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    bytes_per_line = ch * w
    return QtGui.QImage(rgb.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888).copy()
