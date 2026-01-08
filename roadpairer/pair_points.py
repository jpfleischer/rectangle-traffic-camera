#!/usr/bin/env python3
# pair_points.py — resizable panes, hover-targeted pan/zoom, robust homography
# NEW:
#  - USAC/MAGSAC++ upgrade (fallback to RANSAC)
#  - "Load Points" button; Solve&Save also saves points to JSON
#  - Result panel with inlier stats (count, %, RMSE, median, P95, method, thr)
#  - Camera pane colored by inlier/outlier after solve (like ortho)


import json, time
import numpy as np
import cv2
import rasterio as rio
from rasterio.enums import Resampling
from PySide6 import QtCore, QtGui, QtWidgets

from roadpairer.ortho_overlay import save_overlay_figure


# ---- I/O paths ----
CAM_PATH   = "camera_frame.png"
ORTHO_PATH = "ortho_zoom.tif"
H_OUT      = "H_cam_to_map.npy"
WARP_OUT   = "camera_warped_to_ortho.png"
PTS_OUT    = "pairs_cam_to_map.json"   # <-- points saved/loaded here

# ---- robust estimator settings ----
H_METHOD = getattr(cv2, "USAC_MAGSAC", cv2.RANSAC)  # prefer MAGSAC++
RANSAC_THRESH   = 10.0   # reprojection threshold in *ortho pixels*
MAX_ITERS       = 10000
CONFIDENCE      = 0.999
REFINE_ITERS    = 10

ACTIVE_BORDER   = "QLabel { border: 2px solid #4da3ff; }"
INACTIVE_BORDER = "QLabel { border: 1px solid #555; }"


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


class ClickLabel(QtWidgets.QLabel):
    clicked = QtCore.Signal(int, int)   # x, y in widget coords
    hovered = QtCore.Signal(bool)       # True on enter, False on leave

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.setMouseTracking(True)
        self.setMinimumSize(150, 120)

    def mousePressEvent(self, e: QtGui.QMouseEvent) -> None:
        if e.button() == QtCore.Qt.LeftButton:
            pos = e.position() if hasattr(e, "position") else e.localPos()
            self.clicked.emit(int(pos.x()), int(pos.y()))
        super().mousePressEvent(e)

    def enterEvent(self, e: QtCore.QEvent) -> None:
        self.hovered.emit(True)
        super().enterEvent(e)

    def leaveEvent(self, e: QtCore.QEvent) -> None:
        self.hovered.emit(False)
        super().leaveEvent(e)


class PairingTab(QtWidgets.QWidget):
    """
    This is your old PairPointsGUI, refactored into a QWidget so it can be used
    as a tab inside the main app.
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("RoadPairer")
        self.resize(1400, 900)

        # ---- load camera ----
        cam_bgr = cv2.imread(CAM_PATH, cv2.IMREAD_COLOR)
        if cam_bgr is None:
            raise SystemExit(f"Could not read {CAM_PATH}")
        self.cam_bgr = cam_bgr
        self.cam_h, self.cam_w = cam_bgr.shape[:2]

        # ---- load ortho preview ----
        with rio.open(ORTHO_PATH) as src:
            self.ortho_w, self.ortho_h = src.width, src.height
            nodata = src.nodata
            bands = [1, 2, 3] if src.count >= 3 else [1]
            self.prev_w = min(3000, self.ortho_w)
            self.prev_h = max(1, int(round(self.ortho_h * (self.prev_w / self.ortho_w))))
            prev = src.read(
                bands,
                out_shape=(len(bands), self.prev_h, self.prev_w),
                resampling=Resampling.bilinear,
            )
            prev = np.moveaxis(prev, 0, 2)  # HWC
            o8 = to_8bit_rgb(prev, nodata=nodata)
            self.ortho_bgr_base = cv2.cvtColor(o8, cv2.COLOR_RGB2BGR)
            self.prev_scale = self.prev_w / float(self.ortho_w)

        # ---- pairing/interaction state ----
        self.cam_pts, self.map_pts = [], []
        self.inlier_mask = None
        self.cam_zoom, self.cam_pan_x, self.cam_pan_y = 1.0, 0, 0
        self.ortho_zoom, self.ortho_pan_x, self.ortho_pan_y = 2.0, 0, 0
        self.active_pane = "ortho"
        self._shortcuts = []
        self.cam_path = CAM_PATH

        self._build_ui()
        self._connect_signals()
        self._redraw_timer = QtCore.QTimer(self)
        self._redraw_timer.setSingleShot(True)
        self._redraw_timer.timeout.connect(self.redraw_all)

        self._update_status()
        self.redraw_all()


    def _build_ui(self):
        root = QtWidgets.QHBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(8)

        self.vsplit = QtWidgets.QSplitter(QtCore.Qt.Vertical)
        self.vsplit.setChildrenCollapsible(False)

        self.cam_label = ClickLabel()
        self.cam_label.setStyleSheet(INACTIVE_BORDER)
        self.cam_label.setSizePolicy(
            QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding
        )

        self.ortho_label = ClickLabel()
        self.ortho_label.setStyleSheet(ACTIVE_BORDER)
        self.ortho_label.setSizePolicy(
            QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding
        )

        self.vsplit.addWidget(self.cam_label)
        self.vsplit.addWidget(self.ortho_label)
        self.vsplit.setStretchFactor(0, 2)
        self.vsplit.setStretchFactor(1, 3)

        self.right_panel = QtWidgets.QWidget()
        self.right_panel.setMinimumWidth(340)
        self.right_panel.setMaximumWidth(700)
        self.right_panel.setSizePolicy(
            QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Expanding
        )

        right = QtWidgets.QVBoxLayout(self.right_panel)
        right.setContentsMargins(0, 0, 0, 0)
        right.setSpacing(10)
        right.addWidget(QtWidgets.QLabel("<b>Controls</b>"))

        # Zoom
        zoom_box = QtWidgets.QGroupBox("Zoom (pane under mouse)")
        zl = QtWidgets.QGridLayout(zoom_box)
        self.btn_zoom_in = QtWidgets.QPushButton("Zoom In  (+/=)")
        self.btn_zoom_out = QtWidgets.QPushButton("Zoom Out (-)")
        zl.addWidget(self.btn_zoom_in, 0, 0)
        zl.addWidget(self.btn_zoom_out, 0, 1)
        right.addWidget(zoom_box)

        # Pan
        pan_box = QtWidgets.QGroupBox("Pan (pane under mouse)")
        pl = QtWidgets.QGridLayout(pan_box)
        self.btn_pan_up = QtWidgets.QPushButton("Up    (W/↑)")
        self.btn_pan_left = QtWidgets.QPushButton("Left  (A/←)")
        self.btn_pan_right = QtWidgets.QPushButton("Right (D/→)")
        self.btn_pan_down = QtWidgets.QPushButton("Down  (S/↓)")
        pl.addWidget(self.btn_pan_up, 0, 1)
        pl.addWidget(self.btn_pan_left, 1, 0)
        pl.addWidget(self.btn_pan_right, 1, 2)
        pl.addWidget(self.btn_pan_down, 2, 1)
        right.addWidget(pan_box)

        # Actions
        act_box = QtWidgets.QGroupBox("Actions")
        al = QtWidgets.QVBoxLayout(act_box)
        self.btn_load = QtWidgets.QPushButton("Load Points…")
        self.btn_undo = QtWidgets.QPushButton("Undo last point (U)")
        self.btn_solve = QtWidgets.QPushButton("Solve (least-squares)")
        self.btn_solve_pwa = QtWidgets.QPushButton("Solve (piecewise, exact)")
        self.btn_solve_robust = QtWidgets.QPushButton("Solve (robust, MAGSAC++)")
        self.btn_swap_cam = QtWidgets.QPushButton("Swap Camera Image…")
        self.btn_quit = QtWidgets.QPushButton("Quit (Q)")

        al.addWidget(self.btn_load)
        al.addWidget(self.btn_undo)
        al.addWidget(self.btn_solve)
        al.addWidget(self.btn_solve_pwa)
        al.addWidget(self.btn_solve_robust)
        al.addWidget(self.btn_swap_cam)
        al.addWidget(self.btn_quit)
        right.addWidget(act_box)

        self.status_lbl = QtWidgets.QLabel()
        self.status_lbl.setWordWrap(True)
        right.addWidget(self.status_lbl)

        self.results_box = QtWidgets.QGroupBox("Last solve: details")
        rl = QtWidgets.QVBoxLayout(self.results_box)
        self.results_lbl = QtWidgets.QLabel("—")
        self.results_lbl.setWordWrap(True)
        rl.addWidget(self.results_lbl)
        self.results_box.setVisible(False)
        right.addWidget(self.results_box)

        note_box = QtWidgets.QGroupBox("Estimator note")
        nl = QtWidgets.QVBoxLayout(note_box)
        self.magsac_note_lbl = QtWidgets.QLabel(
            "For our intents and purposes, <b>MAGSAC++</b> (USAC) tends to be the most "
            "accurate overall on planar pavement when points are well-spread and mostly on road. "
            "This app prefers USAC/MAGSAC++ automatically and falls back to USAC_ACCURATE or "
            "classic RANSAC if unavailable."
        )
        self.magsac_note_lbl.setWordWrap(True)
        nl.addWidget(self.magsac_note_lbl)
        right.addWidget(note_box)

        help_box = QtWidgets.QGroupBox("Shortcuts")
        hl = QtWidgets.QVBoxLayout(help_box)
        hl.addWidget(
            QtWidgets.QLabel(
                "Hover decides target pane (highlighted).\n"
                "Zoom: +/= in, - out.  Pan: arrows or WASD.\n"
                "Click top (CAM) then bottom (ORTHO) to pair.\n"
                "U undo, Q quit."
            )
        )
        right.addWidget(help_box)
        right.addStretch(1)

        self.hsplit = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        self.hsplit.setChildrenCollapsible(False)
        self.hsplit.addWidget(self.vsplit)
        self.hsplit.addWidget(self.right_panel)
        self.hsplit.setStretchFactor(0, 1)
        self.hsplit.setStretchFactor(1, 0)
        self.hsplit.setSizes([1000, 360])

        root.addWidget(self.hsplit)



    def _connect_signals(self):
        # Hover selects active
        self.cam_label.hovered.connect(lambda on: self._set_active('cam' if on else None))
        self.ortho_label.hovered.connect(lambda on: self._set_active('ortho' if on else None))

        # Clicks
        self.cam_label.clicked.connect(self._on_cam_click)
        self.ortho_label.clicked.connect(self._on_ortho_click)

        # Buttons
        self.btn_zoom_in.clicked.connect(lambda: self._zoom_active(1.25))
        self.btn_zoom_out.clicked.connect(lambda: self._zoom_active(1/1.25))
        self.btn_pan_left.clicked.connect(lambda: self._pan_active(-1, 0))
        self.btn_pan_right.clicked.connect(lambda: self._pan_active(1, 0))
        self.btn_pan_up.clicked.connect(lambda: self._pan_active(0, -1))
        self.btn_pan_down.clicked.connect(lambda: self._pan_active(0, 1))
        self.btn_undo.clicked.connect(self._undo)
        self.btn_solve.clicked.connect(self._solve_and_save)
        self.btn_solve_pwa.clicked.connect(self._solve_piecewise_and_save)
        self.btn_solve_robust.clicked.connect(self._solve_robust_and_save)

        self.btn_swap_cam.clicked.connect(self._swap_camera_image)

        self.btn_quit.clicked.connect(self.close)
        self.btn_load.clicked.connect(self._load_points_dialog)

        # Redraw on splitter moves / resizes
        self.hsplit.splitterMoved.connect(lambda *_: self.redraw_all())
        self.vsplit.splitterMoved.connect(lambda *_: self.redraw_all())
        self.installEventFilter(self)

        # Keyboard shortcuts (keep refs)
        def sc(key, slot):
            s = QtGui.QShortcut(QtGui.QKeySequence(key), self)
            s.activated.connect(slot); self._shortcuts.append(s)
        def sc_pan(key, dx, dy):
            s = QtGui.QShortcut(QtGui.QKeySequence(key), self)
            s.activated.connect(lambda dx=dx, dy=dy: self._pan_active(dx, dy))
            self._shortcuts.append(s)

        sc("Q", self.close)
        for k in ["+", "=", "]"]: sc(k, lambda f=1.25: self._zoom_active(f))
        for k in ["-", "["]:      sc(k, lambda f=1/1.25: self._zoom_active(f))
        for ks, dx, dy in [
            ("Left", -1, 0), ("A", -1, 0),
            ("Right", 1, 0), ("D", 1, 0),
            ("Up", 0, -1), ("W", 0, -1),
            ("Down", 0, 1), ("S", 0, 1),
        ]: sc_pan(ks, dx, dy)
        sc("U", self._undo)
        # sc("S", self._solve_and_save)

    # ---- event filter: redraw on window resize ----
    def eventFilter(self, obj, ev):
        # Coalesce rapid resize events: redraw once after a short delay (16 ms)
        if ev.type() == QtCore.QEvent.Resize:
            if self._redraw_timer.isActive():
                self._redraw_timer.stop()
            self._redraw_timer.start(16)  # ~60 FPS cap
            return False  # let Qt continue normal processing
        return super().eventFilter(obj, ev)
    

    def _swap_camera_image(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Choose new camera image", ".", "Images (*.png *.jpg *.jpeg *.bmp *.tif *.tiff)"
        )
        if not path:
            return
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is None:
            QtWidgets.QMessageBox.critical(self, "Load failed", f"Could not read image:\n{path}")
            return
        h, w = img.shape[:2]
        if (h, w) != (self.cam_h, self.cam_w):
            QtWidgets.QMessageBox.warning(
                self, "Resolution mismatch",
                f"Expected {self.cam_w}×{self.cam_h}, got {w}×{h}. "
                "Swap cancelled to preserve existing point coordinates."
            )
            return

        # Accept swap (points retained because resolution matches)
        self.cam_bgr = img
        self.cam_path = path
        # keep cam_pts/map_pts as-is; invalidate any inlier coloring until next solve
        self.inlier_mask = None

        self._update_status(f"Swapped camera image to: {path}")
        self._draw_cam()


    def _find_homography_compat(self, cam_arr, map_arr, thr_px=3.0):
        """
        Try USAC/MAGSAC if available, otherwise fall back to classic RANSAC.
        Handles OpenCV builds that don't accept extra keyword args.
        Returns (H, inlier_mask_bool) or (None, None).
        """
        # Prefer USAC_MAGSAC if present
        method_codes = []
        if hasattr(cv2, "USAC_MAGSAC"):
            method_codes.append(cv2.USAC_MAGSAC)
        if hasattr(cv2, "USAC_ACCURATE"):
            method_codes.append(cv2.USAC_ACCURATE)
        # Always include classic RANSAC as fallback
        method_codes.append(cv2.RANSAC)

        last_err = None
        for m in method_codes:
            try:
                # Try the *modern* signature first (some builds accept keywords)
                H, inliers = cv2.findHomography(
                    cam_arr, map_arr,
                    method=m,
                    ransacReprojThreshold=float(thr_px)
                )
                if H is not None and inliers is not None:
                    return H, inliers.ravel().astype(bool), m
            except TypeError as e:
                last_err = e
                # Fall back to the older positional signature (max 4 args)
                try:
                    H, inliers = cv2.findHomography(cam_arr, map_arr, m, float(thr_px))
                    if H is not None and inliers is not None:
                        return H, inliers.ravel().astype(bool), m
                except Exception as e2:
                    last_err = e2
            except Exception as e:
                last_err = e
                continue

        # Nothing worked
        print("[homography] failed:", last_err)
        return None, None, None
    

    def _solve_robust_and_save(self):
        n = min(len(self.cam_pts), len(self.map_pts))
        if n < 4:
            QtWidgets.QMessageBox.warning(self, "Not enough points", "Need at least 4 matched pairs.")
            return

        cam_arr = np.asarray(self.cam_pts[:n], dtype=np.float64)
        map_arr = np.asarray(self.map_pts[:n], dtype=np.float64)

        # Robust estimate (USAC/MAGSAC if available, else classic RANSAC)
        H0, inliers_bool, method_used = self._find_homography_compat(
            cam_arr, map_arr, thr_px=RANSAC_THRESH
        )
        if H0 is None or inliers_bool is None or inliers_bool.sum() < 4:
            QtWidgets.QMessageBox.warning(
                self, "Homography failed",
                "Robust estimation could not find a stable model. "
                "Try more/better-spread *pavement* points."
            )
            return

        # Optional LSQ refit on inliers for best accuracy
        H, _ = cv2.findHomography(cam_arr[inliers_bool], map_arr[inliers_bool], 0)

        # Residuals on inliers
        errs_all = self._reproj_errors(H, cam_arr.astype(np.float32), map_arr.astype(np.float32))
        inl = inliers_bool
        rmse = float(np.sqrt(np.mean(errs_all[inl] ** 2)))
        med  = float(np.median(errs_all[inl]))
        p95  = float(np.percentile(errs_all[inl], 95))

        # Save H
        np.save(H_OUT, H)

        # Warp to full ortho canvas with constant borders
        bev = cv2.warpPerspective(
            self.cam_bgr, H, (self.ortho_w, self.ortho_h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0)
        )


        # Full projected footprint of the entire camera frame (not just inlier hull)
        cam_mask = np.full((self.cam_h, self.cam_w), 255, dtype=np.uint8)  # all pixels valid in camera
        footprint = cv2.warpPerspective(
            cam_mask, H, (self.ortho_w, self.ortho_h),
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0
        )

        # Optional: clean up jaggies / tiny holes
        kernel = np.ones((3, 3), np.uint8)
        footprint = cv2.morphologyEx(footprint, cv2.MORPH_CLOSE, kernel, iterations=1)

        # Save warp (use a distinct filename to not clobber LSQ result)
        out_path = WARP_OUT.replace(".png", "_robust.png")
        cv2.imwrite(out_path, bev)

        overlay_path = out_path.replace(".png", "_overlay.png")
        save_overlay_figure(
            ortho_path=ORTHO_PATH,
            bev_bgr=bev,          # <-- keep full bev
            mask_u8=footprint,    # <-- use full footprint mask
            out_path=overlay_path,
            alpha_strength=0.70,
            background_darken=0.65,
            draw_outline=True,
            outline_thickness=3,
        )


        # Persist points JSON
        try:
            pts_path = self._save_points()
        except Exception as e:
            pts_path = f"(save failed: {e})"

        # Update UI state (color points by inlier/outlier)
        self.inlier_mask = inliers_bool

        # Method label
        if method_used == getattr(cv2, "USAC_MAGSAC", -1):
            method_name = "USAC_MAGSAC++"
        elif method_used == getattr(cv2, "USAC_ACCURATE", -2):
            method_name = "USAC_ACCURATE"
        elif method_used == cv2.RANSAC:
            method_name = "RANSAC"
        else:
            method_name = f"method={method_used}"

        summary = (
            f"<b>Method</b>: {method_name}  |  <b>thr</b>: {RANSAC_THRESH:.1f} px<br>"
            f"<b>Pairs</b>: {n}  |  <b>Inliers</b>: {int(inl.sum())} ({100.0 * inl.mean():.1f}%)<br>"
            f"<b>RMSE@inliers</b>: {rmse:.2f}px  |  <b>Median</b>: {med:.2f}px  |  <b>P95</b>: {p95:.2f}px<br>"
            f"Saved H → <code>{H_OUT}</code><br>"
            f"Saved warp → <code>{out_path}</code><br>"
            f"Saved points → <code>{pts_path}</code>"
        )
        self.results_lbl.setText(summary)
        self.results_box.setVisible(True)

        self._update_status("Solved with robust homography (inliers highlighted).")
        self.redraw_all()




    # ---------- active pane ----------
    def _set_active(self, pane_or_none):
        if pane_or_none is None: return
        self.active_pane = pane_or_none
        self.cam_label.setStyleSheet(ACTIVE_BORDER if self.active_pane == 'cam' else INACTIVE_BORDER)
        self.ortho_label.setStyleSheet(ACTIVE_BORDER if self.active_pane == 'ortho' else INACTIVE_BORDER)
        self._update_status()

    def _active_desc(self):
        return "Camera (top)" if self.active_pane == 'cam' else "Ortho (bottom)"

    def _update_status(self, extra: str = ""):
        pairs = min(len(self.cam_pts), len(self.map_pts))
        txt = (f"Active: {self._active_desc()} | "
               f"Pairs: {pairs} | "
               f"Cam zoom {self.cam_zoom:.2f} pan({self.cam_pan_x},{self.cam_pan_y}) | "
               f"Ortho zoom {self.ortho_zoom:.2f} pan({self.ortho_pan_x},{self.ortho_pan_y})")
        if extra: txt += "\n" + extra
        self.status_lbl.setText(txt)

    # ---------- drawing ----------
    def redraw_all(self):
        self._draw_cam()
        self._draw_ortho()

    def _draw_cam(self):
        view_w = max(1, self.cam_label.width())
        view_h = max(1, self.cam_label.height())
        base_scale = min(view_w / self.cam_w, view_h / self.cam_h)
        eff = base_scale * self.cam_zoom
        big_w = max(1, int(round(self.cam_w * eff)))
        big_h = max(1, int(round(self.cam_h * eff)))
        self.cam_pan_x = clamp(self.cam_pan_x, 0, max(0, big_w - view_w))
        self.cam_pan_y = clamp(self.cam_pan_y, 0, max(0, big_h - view_h))
        big = cv2.resize(self.cam_bgr, (big_w, big_h), interpolation=cv2.INTER_LINEAR)
        view = big[self.cam_pan_y:self.cam_pan_y+view_h, self.cam_pan_x:self.cam_pan_x+view_w].copy()

        # draw points (color by inlier mask if available)
        for i, (x, y) in enumerate(self.cam_pts, 1):
            xd = int(round(x * eff)) - self.cam_pan_x
            yd = int(round(y * eff)) - self.cam_pan_y
            if 0 <= xd < view_w and 0 <= yd < view_h:
                color = (0,255,0)  # default green
                if self.inlier_mask is not None and i-1 < len(self.inlier_mask):
                    color = (0,200,255) if self.inlier_mask[i-1] else (0,0,255)  # same scheme as bottom
                cv2.circle(view, (xd, yd), 5, color, -1)
                cv2.putText(view, str(i), (xd+6, yd-6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1, cv2.LINE_AA)
        self.cam_label.setPixmap(QtGui.QPixmap.fromImage(cv_bgr_to_qimage(view)))

    def _draw_ortho(self):
        view_w = max(1, self.ortho_label.width())
        view_h = max(1, self.ortho_label.height())
        base_scale = min(view_w / self.ortho_w, view_h / self.ortho_h)
        eff = base_scale * self.ortho_zoom
        big_w = max(1, int(round(self.ortho_w * eff)))
        big_h = max(1, int(round(self.ortho_h * eff)))
        self.ortho_pan_x = clamp(self.ortho_pan_x, 0, max(0, big_w - view_w))
        self.ortho_pan_y = clamp(self.ortho_pan_y, 0, max(0, big_h - view_h))

        rel_scale = eff / self.prev_scale
        bw = max(1, int(round(self.prev_w * rel_scale)))
        bh = max(1, int(round(self.prev_h * rel_scale)))
        big = cv2.resize(self.ortho_bgr_base, (bw, bh), interpolation=cv2.INTER_LINEAR)
        if big.shape[1] < view_w or big.shape[0] < view_h:
            pad_w = max(0, view_w - big.shape[1]); pad_h = max(0, view_h - big.shape[0])
            big = cv2.copyMakeBorder(big, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=(0,0,0))
        view = big[self.ortho_pan_y:self.ortho_pan_y+view_h,
                   self.ortho_pan_x:self.ortho_pan_x+view_w].copy()

        # draw points (inlier=orange/yellow, outlier=red)
        for i, (x, y) in enumerate(self.map_pts, 1):
            xd = int(round(x * eff)) - self.ortho_pan_x
            yd = int(round(y * eff)) - self.ortho_pan_y
            if 0 <= xd < view_w and 0 <= yd < view_h:
                color = (0,200,255)
                if self.inlier_mask is not None and i-1 < len(self.inlier_mask):
                    color = (0,200,255) if self.inlier_mask[i-1] else (0,0,255)
                cv2.circle(view, (xd, yd), 5, color, -1)
                cv2.putText(view, str(i), (xd+6, yd-6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1, cv2.LINE_AA)
        self.ortho_label.setPixmap(QtGui.QPixmap.fromImage(cv_bgr_to_qimage(view)))

    # ---------- clicks ----------
    def _on_cam_click(self, x, y):
        view_w = max(1, self.cam_label.width())
        view_h = max(1, self.cam_label.height())
        base_scale = min(view_w / self.cam_w, view_h / self.cam_h)
        eff = base_scale * self.cam_zoom
        x_px = int(round((self.cam_pan_x + x) / eff))
        y_px = int(round((self.cam_pan_y + y) / eff))
        x_px = clamp(x_px, 0, self.cam_w - 1)
        y_px = clamp(y_px, 0, self.cam_h - 1)
        self.cam_pts.append([x_px, y_px])
        self.inlier_mask = None
        self._update_status(); self._draw_cam()

    def _on_ortho_click(self, x, y):
        view_w = max(1, self.ortho_label.width())
        view_h = max(1, self.ortho_label.height())
        base_scale = min(view_w / self.ortho_w, view_h / self.ortho_h)
        eff = base_scale * self.ortho_zoom
        x_px = int(round((self.ortho_pan_x + x) / eff))
        y_px = int(round((self.ortho_pan_y + y) / eff))
        x_px = clamp(x_px, 0, self.ortho_w - 1)
        y_px = clamp(y_px, 0, self.ortho_h - 1)
        self.map_pts.append([x_px, y_px])
        self.inlier_mask = None
        self._update_status(); self._draw_ortho()

    # ---------- pan/zoom targeting the hovered pane ----------
    def _zoom_active(self, factor):
        if self.active_pane == 'cam': self._zoom_cam(factor)
        else:                          self._zoom_ortho(factor)
    def _pan_active(self, dx, dy):
        if self.active_pane == 'cam': self._pan_cam(dx, dy)
        else:                          self._pan_ortho(dx, dy)

    # Camera pan/zoom
    def _zoom_cam(self, factor):
        view_w = max(1, self.cam_label.width())
        view_h = max(1, self.cam_label.height())
        base_scale = min(view_w / self.cam_w, view_h / self.cam_h)
        old_eff = base_scale * self.cam_zoom
        self.cam_zoom = clamp(self.cam_zoom * factor, 0.25, 16.0)
        new_eff = base_scale * self.cam_zoom
        cx = self.cam_pan_x + view_w // 2; cy = self.cam_pan_y + view_h // 2
        cx_px = cx / old_eff; cy_px = cy / old_eff
        self.cam_pan_x = int(round(cx_px * new_eff)) - view_w // 2
        self.cam_pan_y = int(round(cy_px * new_eff)) - view_h // 2
        self._update_status(); self._draw_cam()

    def _pan_cam(self, dx, dy):
        step = max(5, int(50 / max(self.cam_zoom, 1e-6)))
        self.cam_pan_x += dx * step; self.cam_pan_y += dy * step
        self._update_status(); self._draw_cam()

    # Ortho pan/zoom
    def _zoom_ortho(self, factor):
        view_w = max(1, self.ortho_label.width())
        view_h = max(1, self.ortho_label.height())
        base_scale = min(view_w / self.ortho_w, view_h / self.ortho_h)
        old_eff = base_scale * self.ortho_zoom
        self.ortho_zoom = clamp(self.ortho_zoom * factor, 0.25, 16.0)
        new_eff = base_scale * self.ortho_zoom
        cx = self.ortho_pan_x + view_w // 2; cy = self.ortho_pan_y + view_h // 2
        cx_px = cx / old_eff; cy_px = cy / old_eff
        self.ortho_pan_x = int(round(cx_px * new_eff)) - view_w // 2
        self.ortho_pan_y = int(round(cy_px * new_eff)) - view_h // 2
        self._update_status(); self._draw_ortho()

    def _pan_ortho(self, dx, dy):
        step = max(5, int(50 / max(self.ortho_zoom, 1e-6)))
        self.ortho_pan_x += dx * step; self.ortho_pan_y += dy * step
        self._update_status(); self._draw_ortho()

    # ---------- points I/O ----------
    def _save_points(self):
        data = {
            "created": time.strftime("%Y-%m-%d %H:%M:%S"),
            "camera": self.cam_path,          # <- use current path
            "ortho": ORTHO_PATH,
            "cam_pts": self.cam_pts,
            "map_pts": self.map_pts,
        }
        with open(PTS_OUT, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        return PTS_OUT

    def _load_points_dialog(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Load points JSON", ".", "JSON (*.json)")
        if not path: return
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            cam = data.get("cam_pts") or []
            mpp = data.get("map_pts") or []

            k = min(len(cam), len(mpp))
            if k < 1:
                raise ValueError("Invalid or empty point arrays.")

            if len(cam) != len(mpp):
                print(f"[load] WARNING: cam_pts={len(cam)} map_pts={len(mpp)}; trimming to {k}")

            cam = cam[:k]
            mpp = mpp[:k]

            self.cam_pts = [list(map(int, p)) for p in cam]
            self.map_pts = [list(map(int, p)) for p in mpp]

            self.inlier_mask = None
            self._update_status(f"Loaded {len(self.cam_pts)} pairs from {path}")
            self.redraw_all()
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Load failed", str(e))

    # ---------- undo / solve ----------
    def _undo(self):
        if len(self.cam_pts) > len(self.map_pts): self.cam_pts.pop()
        elif len(self.map_pts) > len(self.cam_pts): self.map_pts.pop()
        elif self.cam_pts:
            self.cam_pts.pop(); self.map_pts.pop()
        self.inlier_mask = None
        self._update_status(); self.redraw_all()

    def _solve_and_save(self):
        n = min(len(self.cam_pts), len(self.map_pts))
        if n < 4:
            QtWidgets.QMessageBox.warning(self, "Not enough points", "Need at least 4 matched pairs.")
            return

        cam_arr = np.array(self.cam_pts[:n], np.float64)
        map_arr = np.array(self.map_pts[:n], np.float64)

        # Plain DLT homography: use ALL points (no RANSAC, no outliers)
        H, _ = cv2.findHomography(cam_arr, map_arr, 0)
        if H is None:
            QtWidgets.QMessageBox.warning(self, "Homography failed", "Numerical issue; try adding/redistributing points.")
            return

        # Everyone is an "inlier" now
        self.inlier_mask = np.ones(n, dtype=bool)

        # Residual stats (so you can still judge fit quality)
        errs = self._reproj_errors(H, cam_arr.astype(np.float32), map_arr.astype(np.float32))
        rmse = float(np.sqrt(np.mean(errs**2)))
        med  = float(np.median(errs))
        p95  = float(np.percentile(errs, 95))

        # Save outputs
        np.save(H_OUT, H)
        bev = cv2.warpPerspective(
            self.cam_bgr, H, (self.ortho_w, self.ortho_h),
            flags=cv2.INTER_LINEAR
        )
        cv2.imwrite(WARP_OUT, bev)

        # Save points (use your helper; preserve points even if solve already done)
        try:
            pts_path = self._save_points()
        except Exception as e:
            pts_path = f"(save failed: {e})"

        summary = (f"<b>Method</b>: LSQ Homography (no RANSAC)<br>"
                f"<b>Pairs</b>: {n}<br>"
                f"<b>RMSE</b>: {rmse:.2f}px  |  <b>Median</b>: {med:.2f}px  |  <b>P95</b>: {p95:.2f}px<br>"
                f"Saved H → <code>{H_OUT}</code><br>"
                f"Saved warp → <code>{WARP_OUT}</code><br>"
                f"Saved points → <code>{pts_path}</code>")
        self.results_lbl.setText(summary)
        self.results_box.setVisible(True)

        self._update_status("Solved with all points (no rejections).")
        self.redraw_all()


    def _solve_piecewise_and_save(self):
        """Piecewise-affine (exact-through-points) with convex-hull masking (no .mesh needed)."""
        try:
            from skimage.transform import PiecewiseAffineTransform, warp
        except Exception:
            QtWidgets.QMessageBox.critical(self, "Missing dependency",
                "scikit-image is required:\n  pip install scikit-image")
            return

        n = min(len(self.cam_pts), len(self.map_pts))
        if n < 3:
            QtWidgets.QMessageBox.warning(self, "Not enough points",
                                        "Need at least 3 non-collinear pairs.")
            return

        # Points are (x, y) already; scikit-image expects (x, y) as well.
        src = np.asarray(self.cam_pts[:n], dtype=np.float64)  # camera points
        dst = np.asarray(self.map_pts[:n], dtype=np.float64)  # ortho/map points

        # Guard against duplicates / degeneracy
        if np.unique(dst.round(3), axis=0).shape[0] < 3:
            QtWidgets.QMessageBox.warning(self, "Bad points",
                "Destination points contain duplicates/collinearity; add more spread points.")
            return

        tform = PiecewiseAffineTransform()
        if not tform.estimate(src, dst):
            QtWidgets.QMessageBox.warning(self, "Solve failed",
                                        "Triangulation failed; add/redistribute points (avoid collinear sets).")
            return

        # Warp with constant fill (avoid edge streaks)
        img_rgb = cv2.cvtColor(self.cam_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        out = warp(
            img_rgb,
            tform,                                    # correct direction: cam -> map
            output_shape=(self.ortho_h, self.ortho_w),
            order=1, mode="constant", cval=0.0, preserve_range=True
        )
        out_bgr = cv2.cvtColor((out * 255.0).astype(np.uint8), cv2.COLOR_RGB2BGR)

        # ---- Mask to convex hull of destination points (eliminate extrapolated wedges) ----
        hull = cv2.convexHull(dst.astype(np.float32)).astype(np.int32)  # shape (k,1,2) or (k,2)
        hull = hull.reshape(-1, 2)
        mask = np.zeros((self.ortho_h, self.ortho_w), np.uint8)
        if len(hull) >= 3:
            cv2.fillConvexPoly(mask, hull, 255)
            out_bgr = cv2.bitwise_and(out_bgr, out_bgr, mask=mask)

            # Optional: crop to mask bbox
            ys, xs = np.where(mask > 0)
            if ys.size and xs.size:
                y0, y1 = ys.min(), ys.max() + 1
                x0, x1 = xs.min(), xs.max() + 1
                out_bgr = out_bgr[y0:y1, x0:x1]

        # Treat all as "inliers" (no rejection)
        self.inlier_mask = np.ones(n, dtype=bool)

        # Report exact interpolation error at control points (should be ~0)
        pred = tform(src)
        errs = np.linalg.norm(pred - dst, axis=1)
        rmse = float(np.sqrt(np.mean(errs**2)))
        med  = float(np.median(errs))
        p95  = float(np.percentile(errs, 95))

        out_path = WARP_OUT.replace(".png", "_pwa.png")
        cv2.imwrite(out_path, out_bgr)

        summary = (f"<b>Method</b>: Piecewise Affine (exact), convex-hull masked<br>"
                f"<b>Pairs</b>: {n}<br>"
                f"<b>RMSE@ctrl</b>: {rmse:.3f}px  |  <b>Median</b>: {med:.3f}px  |  <b>P95</b>: {p95:.3f}px<br>"
                f"Saved warp → <code>{out_path}</code>")
        self.results_lbl.setText(summary)
        self.results_box.setVisible(True)
        self._update_status("Solved with piecewise affine (no rejections), convex-hull masked.")
        self.redraw_all()


    @staticmethod
    def _reproj_errors(H, src_pts, dst_pts):
        src_h = np.hstack([src_pts, np.ones((len(src_pts), 1), np.float32)])
        proj = (H @ src_h.T).T
        proj = proj[:, :2] / proj[:, 2:3]
        return np.linalg.norm(proj - dst_pts, axis=1)
