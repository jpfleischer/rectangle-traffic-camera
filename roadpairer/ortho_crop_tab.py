#!/usr/bin/env python3
# ortho_crop_tab.py — interactive crop of ortho into ortho_zoom.tif

import os
import cv2
import numpy as np
import rasterio as rio
from rasterio.windows import Window

from PySide6 import QtCore, QtGui, QtWidgets


DEFAULT_IN = "ortho_monroe_2024.tif"
DEFAULT_OUT = "ortho_zoom.tif"

MAX_W, MAX_H = 1500, 800   # preview bounds, similar to your script


def to_8bit(img: np.ndarray) -> np.ndarray:
    if img.dtype == np.uint8:
        return img
    o = img.astype(np.float32)
    for c in range(o.shape[2]):
        lo, hi = np.percentile(o[..., c], (1, 99))
        if hi <= lo:
            hi = lo + 1.0
        o[..., c] = np.clip((o[..., c] - lo) / (hi - lo) * 255.0, 0, 255)
    return o.astype(np.uint8)


class DragLabel(QtWidgets.QLabel):
    """
    QLabel that lets the user drag a rectangle.
    Emits dragChanged(x0, y0, x1, y1, dragging).
    """
    dragChanged = QtCore.Signal(int, int, int, int, bool)

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.setMouseTracking(True)
        self._drag = False
        self._x0 = self._y0 = self._x1 = self._y1 = -1

    def mousePressEvent(self, e: QtGui.QMouseEvent) -> None:
        if e.button() == QtCore.Qt.LeftButton:
            pos = e.position() if hasattr(e, "position") else e.localPos()
            self._x0 = self._x1 = int(pos.x())
            self._y0 = self._y1 = int(pos.y())
            self._drag = True
            self.dragChanged.emit(self._x0, self._y0, self._x1, self._y1, True)
        super().mousePressEvent(e)

    def mouseMoveEvent(self, e: QtGui.QMouseEvent) -> None:
        if self._drag:
            pos = e.position() if hasattr(e, "position") else e.localPos()
            self._x1 = int(pos.x())
            self._y1 = int(pos.y())
            self.dragChanged.emit(self._x0, self._y0, self._x1, self._y1, True)
        super().mouseMoveEvent(e)

    def mouseReleaseEvent(self, e: QtGui.QMouseEvent) -> None:
        if e.button() == QtCore.Qt.LeftButton and self._drag:
            pos = e.position() if hasattr(e, "position") else e.localPos()
            self._x1 = int(pos.x())
            self._y1 = int(pos.y())
            self._drag = False
            self.dragChanged.emit(self._x0, self._y0, self._x1, self._y1, False)
        super().mouseReleaseEvent(e)


class OrthoCropTab(QtWidgets.QWidget):
    """
    Tab for interactively cropping a larger ortho TIFF into ortho_zoom.tif.
    Workflow:
      1. Load big ortho (default ortho_monroe_2024.tif)
      2. Drag rectangle
      3. Click "Save crop" -> writes ortho_zoom.tif
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Crop Ortho")

        # state
        self.src_path = DEFAULT_IN
        self.out_path = DEFAULT_OUT

        self.disp_img = None   # preview (uint8, HxWx3)
        self.scale = 1.0       # display_px = full_px * scale
        self.full_w = 0
        self.full_h = 0

        self.sel = None        # (x0d, y0d, x1d, y1d) in display coords

        self._build_ui()
        self._connect_signals()

        # try auto-load default
        if os.path.exists(self.src_path):
            self._load_src(self.src_path)

    # ---------------- UI ---------------- #
    def _build_ui(self):
        root = QtWidgets.QHBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(8)

        # left: preview
        self.preview = DragLabel()
        # Align image to top-left so mouse coords match image coords
        self.preview.setAlignment(QtCore.Qt.AlignLeft | QtCore.Qt.AlignTop)
        self.preview.setStyleSheet("background-color: black;")

        root.addWidget(self.preview, stretch=1)

        # right: controls
        right_panel = QtWidgets.QWidget()
        rp = QtWidgets.QVBoxLayout(right_panel)
        rp.setContentsMargins(0, 0, 0, 0)
        rp.setSpacing(8)

        rp.addWidget(QtWidgets.QLabel("<b>Step 2: Crop ortho to working zoom</b>"))

        form = QtWidgets.QFormLayout()
        self.edit_src = QtWidgets.QLineEdit(self.src_path)
        self.btn_browse_src = QtWidgets.QPushButton("Browse…")
        src_box = QtWidgets.QHBoxLayout()
        src_box.addWidget(self.edit_src)
        src_box.addWidget(self.btn_browse_src)

        self.edit_out = QtWidgets.QLineEdit(self.out_path)
        self.btn_browse_out = QtWidgets.QPushButton("Browse…")
        out_box = QtWidgets.QHBoxLayout()
        out_box.addWidget(self.edit_out)
        out_box.addWidget(self.btn_browse_out)

        form.addRow("Input GeoTIFF:", src_box)
        form.addRow("Output GeoTIFF:", out_box)
        rp.addLayout(form)

        self.lbl_dims = QtWidgets.QLabel("No image loaded.")
        rp.addWidget(self.lbl_dims)

        self.lbl_sel = QtWidgets.QLabel("Selection: none")
        rp.addWidget(self.lbl_sel)

        self.btn_reset_sel = QtWidgets.QPushButton("Clear selection")
        self.btn_save = QtWidgets.QPushButton("Save crop to ortho_zoom.tif")
        rp.addWidget(self.btn_reset_sel)
        rp.addWidget(self.btn_save)

        self.status = QtWidgets.QLabel("")
        self.status.setWordWrap(True)
        rp.addWidget(self.status)

        rp.addStretch(1)

        root.addWidget(right_panel, stretch=0)

    def _connect_signals(self):
        self.btn_browse_src.clicked.connect(self._on_browse_src)
        self.btn_browse_out.clicked.connect(self._on_browse_out)
        self.btn_reset_sel.clicked.connect(self._on_reset_selection)
        self.btn_save.clicked.connect(self._on_save_crop)
        self.preview.dragChanged.connect(self._on_drag_changed)

    # ---------------- helpers ---------------- #
    def _set_status(self, text: str):
        self.status.setText(text)

    def _on_browse_src(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Choose input ortho GeoTIFF",
            self.edit_src.text() or self.src_path,
            "GeoTIFF (*.tif *.tiff);;All files (*)",
        )
        if path:
            self.edit_src.setText(path)
            self._load_src(path)

    def _on_browse_out(self):
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Choose output crop GeoTIFF",
            self.edit_out.text() or self.out_path,
            "GeoTIFF (*.tif *.tiff);;All files (*)",
        )
        if path:
            self.edit_out.setText(path)
            self.out_path = path

    def _load_src(self, path: str):
        try:
            with rio.open(path) as src:
                rgb = (
                    src.read([1, 2, 3])
                    if src.count >= 3
                    else np.repeat(src.read(1)[None, ...], 3, axis=0)
                )
                img = np.moveaxis(rgb, 0, 2)  # (H,W,C)
                disp = to_8bit(img)

                H, W = disp.shape[:2]
                self.full_h, self.full_w = H, W

                # scale preview to fit
                scale = min(1.0, min(MAX_W / W, MAX_H / H))
                disp_sz = (max(1, int(W * scale)), max(1, int(H * scale)))
                disp_img = cv2.resize(disp, disp_sz, interpolation=cv2.INTER_AREA)

            self.disp_img = disp_img
            self.scale = scale
            self.src_path = path
            self.sel = None

            self.lbl_dims.setText(f"Loaded {os.path.basename(path)}  ({self.full_w} x {self.full_h} px)")
            self.lbl_sel.setText("Selection: none")

            self._update_preview()
            self._set_status("Ready: drag a rectangle on the image, then click 'Save crop'.")
        except Exception as e:
            self._set_status(f"Error loading ortho: {e}")
            QtWidgets.QMessageBox.critical(self, "Load failed", str(e))

    def _update_preview(self):
        if self.disp_img is None:
            self.preview.setPixmap(QtGui.QPixmap())
            return

        frame = self.disp_img.copy()
        if self.sel is not None:
            x0d, y0d, x1d, y1d = self.sel
            cv2.rectangle(
                frame,
                (x0d, y0d),
                (x1d, y1d),
                (0, 255, 255),
                2,
            )
        qimg = QtGui.QImage(
            frame.data,
            frame.shape[1],
            frame.shape[0],
            frame.strides[0],
            QtGui.QImage.Format_BGR888,
        )
        self.preview.setPixmap(QtGui.QPixmap.fromImage(qimg))

    # ---------------- interaction ---------------- #
    def _on_drag_changed(self, x0, y0, x1, y1, dragging: bool):
        if self.disp_img is None:
            return
        # clamp to image bounds
        h, w = self.disp_img.shape[:2]
        x0 = max(0, min(w - 1, x0))
        x1 = max(0, min(w - 1, x1))
        y0 = max(0, min(h - 1, y0))
        y1 = max(0, min(h - 1, y1))
        self.sel = (x0, y0, x1, y1)
        self._update_preview()

        if self.sel is not None:
            x0d, y0d, x1d, y1d = self.sel
            sel_w = abs(x1d - x0d)
            sel_h = abs(y1d - y0d)
            self.lbl_sel.setText(f"Selection (display px): {sel_w} x {sel_h}")
        else:
            self.lbl_sel.setText("Selection: none")

    def _on_reset_selection(self):
        self.sel = None
        self.lbl_sel.setText("Selection: none")
        self._update_preview()

    def _on_save_crop(self):
        if self.disp_img is None or self.src_path is None:
            QtWidgets.QMessageBox.warning(self, "No image", "Load an input GeoTIFF first.")
            return
        if self.sel is None:
            QtWidgets.QMessageBox.warning(self, "No selection", "Drag a rectangle on the image first.")
            return

        x0d, y0d, x1d, y1d = self.sel
        x0d, x1d = sorted([x0d, x1d])
        y0d, y1d = sorted([y0d, y1d])

        if x1d <= x0d or y1d <= y0d:
            QtWidgets.QMessageBox.warning(self, "Bad selection", "Selection area is empty.")
            return

        # map display coords -> original pixel coords
        x0 = int(round(x0d / self.scale))
        x1 = int(round(x1d / self.scale))
        y0 = int(round(y0d / self.scale))
        y1 = int(round(y1d / self.scale))
        wpx = max(1, x1 - x0)
        hpx = max(1, y1 - y0)

        out_path = self.edit_out.text().strip() or self.out_path
        self.out_path = out_path

        try:
            with rio.open(self.src_path) as src:
                win = Window(x0, y0, wpx, hpx)
                data = src.read(window=win)
                transform = src.window_transform(win)
                profile = src.profile.copy()
                profile.update(width=wpx, height=hpx, transform=transform)
                with rio.open(out_path, "w", **profile) as dst:
                    dst.write(data)

            self._set_status(
                f"Wrote {out_path}  ({wpx} x {hpx} px)  "
                f"bounds={transform * (0, 0)}..{transform * (wpx, hpx)}"
            )
            QtWidgets.QMessageBox.information(self, "Crop saved", f"Saved crop to:\n{out_path}")
        except Exception as e:
            self._set_status(f"Error writing crop: {e}")
            QtWidgets.QMessageBox.critical(self, "Save failed", str(e))
