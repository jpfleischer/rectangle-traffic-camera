#!/usr/bin/env python3
# intersection_tab.py – "Intersection Geometry" tab (stop bar + divider line)

import os
import json

import numpy as np
import cv2
import rasterio as rio
from rasterio.enums import Resampling
from affine import Affine
from PySide6 import QtCore, QtGui, QtWidgets

from .clickhouse_client import ClickHouseHTTP
from .ch_config import (
    load_clickhouse_config,
    save_clickhouse_config,
)

from .image_utils import to_8bit_rgb, cv_bgr_to_qimage, clamp


ORTHO_PATH = "ortho_zoom.tif"   # must match what you use elsewhere

ACTIVE_BORDER   = "QLabel { border: 2px solid #4da3ff; }"
INACTIVE_BORDER = "QLabel { border: 1px solid #555; }"

US_SURVEY_FT_TO_M = 0.30480060960121924


class ClickLabel(QtWidgets.QLabel):
    clicked = QtCore.Signal(int, int)   # x, y in widget coords

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.setMouseTracking(True)
        self.setMinimumSize(150, 120)

    def mousePressEvent(self, e: QtGui.QMouseEvent) -> None:
        if e.button() == QtCore.Qt.LeftButton:
            pos = e.position() if hasattr(e, "position") else e.localPos()
            self.clicked.emit(int(pos.x()), int(pos.y()))
        super().mousePressEvent(e)


class ClickHouseConfigDialog(QtWidgets.QDialog):
    def __init__(self, parent=None, host="", port=8123, user="default", password="", db="trajectories"):
        super().__init__(parent)
        self.setWindowTitle("Configure ClickHouse")

        form = QtWidgets.QFormLayout(self)

        self.host_edit = QtWidgets.QLineEdit(host)
        self.port_spin = QtWidgets.QSpinBox()
        self.port_spin.setRange(1, 65535)
        self.port_spin.setValue(int(port) if port else 8123)

        self.user_edit = QtWidgets.QLineEdit(user)
        self.pw_edit = QtWidgets.QLineEdit(password)
        self.pw_edit.setEchoMode(QtWidgets.QLineEdit.Password)
        self.db_edit = QtWidgets.QLineEdit(db)

        form.addRow("Host:", self.host_edit)
        form.addRow("Port:", self.port_spin)
        form.addRow("User:", self.user_edit)
        form.addRow("Password:", self.pw_edit)
        form.addRow("Database:", self.db_edit)

        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        form.addWidget(buttons)

    def values(self) -> dict:
        return {
            "host": self.host_edit.text().strip(),
            "port": int(self.port_spin.value()),
            "user": self.user_edit.text().strip() or "default",
            "password": self.pw_edit.text(),
            "db": self.db_edit.text().strip() or "trajectories",
        }


class IntersectionGeometryTab(QtWidgets.QWidget):
    """
    Tab for defining per-intersection geometry on the ortho:
      - stop bar (short line across lanes)
      - divider line (line that separates “toward camera” vs “away” side)

    Both are stored in ClickHouse (pixels + map meters).
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Intersection Geometry")

        # ---- load ortho (tolerant) ----
        self.ortho_ok = False
        self._startup_warning = ""

        try:
            with rio.open(ORTHO_PATH) as src:
                self.ortho_w, self.ortho_h = src.width, src.height
                self.ortho_transform: Affine = src.transform
                nodata = src.nodata
                bands = [1, 2, 3] if src.count >= 3 else [1]
                self.prev_w = min(3000, self.ortho_w)
                self.prev_h = max(
                    1, int(round(self.ortho_h * (self.prev_w / self.ortho_w)))
                )
                prev = src.read(
                    bands,
                    out_shape=(len(bands), self.prev_h, self.prev_w),
                    resampling=Resampling.bilinear,
                )
                prev = np.moveaxis(prev, 0, 2)  # HWC
                o8 = to_8bit_rgb(prev, nodata=nodata)
                self.ortho_bgr_base = cv2.cvtColor(o8, cv2.COLOR_RGB2BGR)
                self.prev_scale = self.prev_w / float(self.ortho_w)
                self.ortho_ok = True
        except Exception as e:
            # Fallback: black canvas, identity transform, editing disabled
            self.ortho_w, self.ortho_h = 2048, 2048
            self.ortho_transform = Affine.identity()
            self.prev_w = self.ortho_w
            self.prev_h = self.ortho_h
            self.prev_scale = 1.0
            self.ortho_bgr_base = np.zeros(
                (self.ortho_h, self.ortho_w, 3), dtype=np.uint8
            )
            self._startup_warning = (
                f"Ortho TIFF '{ORTHO_PATH}' not found; showing blank map. "
                "Geometry editing and saving are disabled until a real ortho is available."
            )

        # pan/zoom
        self.ortho_zoom = 2.0
        self.ortho_pan_x = 0
        self.ortho_pan_y = 0

        # geometry state (in pixel coords)
        self.stopbar_pts_px = []    # list of two [x, y]
        self.divider_pts_px = []    # list of two [x, y]

        # current mode: "idle", "draw_stopbar", "draw_divider"
        self.mode = "idle"

        self._shortcuts = []

        # --- ClickHouse config (lazy, like TracksViewer) ---
        try:
            cfg = load_clickhouse_config()
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Keyring error", str(e))
            cfg = {
                "host": os.getenv("CH_HOST", ""),
                "port": int(os.getenv("CH_PORT", "8123")),
                "user": os.getenv("CH_USER", "default"),
                "password": os.getenv("CH_PASSWORD", ""),
                "db": os.getenv("CH_DB", "trajectories"),
            }

        self.ch_host = cfg["host"]
        self.ch_port = int(cfg["port"])
        self.ch_user = cfg["user"]
        self.ch_password = cfg["password"]
        self.ch_db = cfg["db"]
        self.ch = None  # <-- do not connect yet

        self._build_ui()
        self._connect_signals()
        self._redraw_timer = QtCore.QTimer(self)
        self._redraw_timer.setSingleShot(True)
        self._redraw_timer.timeout.connect(self.redraw)

        if self._startup_warning:
            self.status_lbl.setText(self._startup_warning)

        self.redraw()

    # ---------------------- UI ---------------------- #

    def _build_ui(self):
        root = QtWidgets.QHBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(8)

        # Left: ortho view
        self.ortho_label = ClickLabel()
        self.ortho_label.setSizePolicy(
            QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding
        )
        self.ortho_label.setStyleSheet(ACTIVE_BORDER)

        # Right: controls
        self.right_panel = QtWidgets.QWidget()
        self.right_panel.setMinimumWidth(340)
        self.right_panel.setMaximumWidth(600)
        rp = QtWidgets.QVBoxLayout(self.right_panel)
        rp.setContentsMargins(0, 0, 0, 0)
        rp.setSpacing(8)

        rp.addWidget(QtWidgets.QLabel("<b>Intersection Geometry</b>"))

        form = QtWidgets.QFormLayout()
        self.edit_intersection_id = QtWidgets.QLineEdit()
        self.edit_approach_id = QtWidgets.QLineEdit()
        self.edit_brake_window = QtWidgets.QLineEdit()
        self.edit_brake_window.setPlaceholderText("e.g. 80  (meters upstream)")
        form.addRow("Intersection ID:", self.edit_intersection_id)
        form.addRow("Approach ID:", self.edit_approach_id)
        form.addRow("Braking window (m):", self.edit_brake_window)
        rp.addLayout(form)

        btn_box = QtWidgets.QGroupBox("Geometry editing")
        bl = QtWidgets.QVBoxLayout(btn_box)
        self.btn_draw_stopbar = QtWidgets.QPushButton("Draw Stop Bar (2 clicks)")
        self.btn_draw_divider = QtWidgets.QPushButton("Draw Divider Line (2 clicks)")
        self.btn_clear = QtWidgets.QPushButton("Clear Current Geometry")
        bl.addWidget(self.btn_draw_stopbar)
        bl.addWidget(self.btn_draw_divider)
        bl.addWidget(self.btn_clear)
        rp.addWidget(btn_box)

        zoom_box = QtWidgets.QGroupBox("Pan/Zoom")
        zl = QtWidgets.QGridLayout(zoom_box)
        self.btn_zoom_in = QtWidgets.QPushButton("Zoom In")
        self.btn_zoom_out = QtWidgets.QPushButton("Zoom Out")
        self.btn_pan_up = QtWidgets.QPushButton("Up")
        self.btn_pan_down = QtWidgets.QPushButton("Down")
        self.btn_pan_left = QtWidgets.QPushButton("Left")
        self.btn_pan_right = QtWidgets.QPushButton("Right")
        zl.addWidget(self.btn_zoom_in, 0, 0)
        zl.addWidget(self.btn_zoom_out, 0, 1)
        zl.addWidget(self.btn_pan_up, 1, 1)
        zl.addWidget(self.btn_pan_left, 2, 0)
        zl.addWidget(self.btn_pan_right, 2, 2)
        zl.addWidget(self.btn_pan_down, 3, 1)
        rp.addWidget(zoom_box)

        db_box = QtWidgets.QGroupBox("Database")
        dl = QtWidgets.QVBoxLayout(db_box)

        self.btn_config_ch = QtWidgets.QPushButton("Configure ClickHouse…")
        dl.addWidget(self.btn_config_ch)

        self.btn_load_db = QtWidgets.QPushButton("Load from DB")
        self.btn_save_db = QtWidgets.QPushButton("Save to DB")
        dl.addWidget(self.btn_load_db)
        dl.addWidget(self.btn_save_db)
        rp.addWidget(db_box)

        self.status_lbl = QtWidgets.QLabel("")
        self.status_lbl.setWordWrap(True)
        rp.addWidget(self.status_lbl)
        rp.addStretch(1)

        splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        splitter.setChildrenCollapsible(False)
        splitter.addWidget(self.ortho_label)
        splitter.addWidget(self.right_panel)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 0)
        root.addWidget(splitter)

    def _connect_signals(self):
        self.ortho_label.clicked.connect(self._on_ortho_click)

        self.btn_draw_stopbar.clicked.connect(self._start_draw_stopbar)
        self.btn_draw_divider.clicked.connect(self._start_draw_divider)
        self.btn_clear.clicked.connect(self._clear_geometry)

        self.btn_zoom_in.clicked.connect(lambda: self._zoom_ortho(1.25))
        self.btn_zoom_out.clicked.connect(lambda: self._zoom_ortho(1.0 / 1.25))
        self.btn_pan_left.clicked.connect(lambda: self._pan_ortho(-1, 0))
        self.btn_pan_right.clicked.connect(lambda: self._pan_ortho(1, 0))
        self.btn_pan_up.clicked.connect(lambda: self._pan_ortho(0, -1))
        self.btn_pan_down.clicked.connect(lambda: self._pan_ortho(0, 1))

        self.btn_load_db.clicked.connect(self.load_from_db)
        self.btn_save_db.clicked.connect(self.save_to_db)
        self.btn_config_ch.clicked.connect(self.on_configure_clickhouse)

        # Keyboard shortcuts for panning (WASD + arrows)
        def _make_shortcut(key, dx, dy):
            sc = QtGui.QShortcut(QtGui.QKeySequence(key), self)
            sc.activated.connect(lambda dx=dx, dy=dy: self._pan_ortho(dx, dy))
            self._shortcuts.append(sc)

        # Left / right
        _make_shortcut("A", -1, 0)
        _make_shortcut("Left", -1, 0)
        _make_shortcut("D", 1, 0)
        _make_shortcut("Right", 1, 0)

        # Up / down
        _make_shortcut("W", 0, -1)
        _make_shortcut("Up", 0, -1)
        _make_shortcut("S", 0, 1)
        _make_shortcut("Down", 0, 1)

        self.installEventFilter(self)

    # ---------------------- event filter for resize ---------------------- #

    def eventFilter(self, obj, ev):
        if ev.type() == QtCore.QEvent.Resize:
            if self._redraw_timer.isActive():
                self._redraw_timer.stop()
            self._redraw_timer.start(16)
            return False
        return super().eventFilter(obj, ev)

    # ---------------------- pan/zoom + drawing ---------------------- #

    def redraw(self):
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
        big = cv2.resize(
            self.ortho_bgr_base, (bw, bh), interpolation=cv2.INTER_LINEAR
        )
        if big.shape[1] < view_w or big.shape[0] < view_h:
            pad_w = max(0, view_w - big.shape[1])
            pad_h = max(0, view_h - big.shape[0])
            big = cv2.copyMakeBorder(
                big, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=(0, 0, 0)
            )

        view = big[
            self.ortho_pan_y : self.ortho_pan_y + view_h,
            self.ortho_pan_x : self.ortho_pan_x + view_w,
        ].copy()

        def draw_line(px_pts, color):
            if len(px_pts) != 2:
                return
            (x1, y1), (x2, y2) = px_pts
            x1d = int(round(x1 * eff)) - self.ortho_pan_x
            y1d = int(round(y1 * eff)) - self.ortho_pan_y
            x2d = int(round(x2 * eff)) - self.ortho_pan_x
            y2d = int(round(y2 * eff)) - self.ortho_pan_y
            cv2.line(view, (x1d, y1d), (x2d, y2d), color, 2)

        # stop bar = yellow
        draw_line(self.stopbar_pts_px, (0, 255, 255))
        # divider = cyan
        draw_line(self.divider_pts_px, (255, 255, 0))

        self.ortho_label.setPixmap(
            QtGui.QPixmap.fromImage(cv_bgr_to_qimage(view))
        )
        self._update_status()

    def _zoom_ortho(self, factor):
        view_w = max(1, self.ortho_label.width())
        view_h = max(1, self.ortho_label.height())
        base_scale = min(view_w / self.ortho_w, view_h / self.ortho_h)
        old_eff = base_scale * self.ortho_zoom
        self.ortho_zoom = clamp(self.ortho_zoom * factor, 0.25, 16.0)
        new_eff = base_scale * self.ortho_zoom
        cx = self.ortho_pan_x + view_w // 2
        cy = self.ortho_pan_y + view_h // 2
        cx_px = cx / old_eff
        cy_px = cy / old_eff
        self.ortho_pan_x = int(round(cx_px * new_eff)) - view_w // 2
        self.ortho_pan_y = int(round(cy_px * new_eff)) - view_h // 2
        self.redraw()

    def _pan_ortho(self, dx, dy):
        step = max(5, int(50 / max(self.ortho_zoom, 1e-6)))
        self.ortho_pan_x += dx * step
        self.ortho_pan_y += dy * step
        self.redraw()

    # ---------------------- click handling ---------------------- #

    def _screen_to_px(self, x, y):
        """Convert click in widget coords -> ortho pixel coords."""
        view_w = max(1, self.ortho_label.width())
        view_h = max(1, self.ortho_label.height())
        base_scale = min(view_w / self.ortho_w, view_h / self.ortho_h)
        eff = base_scale * self.ortho_zoom
        x_px = int(round((self.ortho_pan_x + x) / eff))
        y_px = int(round((self.ortho_pan_y + y) / eff))
        x_px = clamp(x_px, 0, self.ortho_w - 1)
        y_px = clamp(y_px, 0, self.ortho_h - 1)
        return x_px, y_px

    def _on_ortho_click(self, x, y):
        if not self.ortho_ok:
            self.status_lbl.setText(
                f"No ortho TIFF loaded (expected '{ORTHO_PATH}'); "
                "geometry editing is disabled."
            )
            return

        x_px, y_px = self._screen_to_px(x, y)

        if self.mode == "draw_stopbar":
            self.stopbar_pts_px.append([x_px, y_px])
            if len(self.stopbar_pts_px) == 2:
                self.mode = "idle"
                self.status_lbl.setText("Stop bar set.")
            self.redraw()
        elif self.mode == "draw_divider":
            self.divider_pts_px.append([x_px, y_px])
            if len(self.divider_pts_px) == 2:
                self.mode = "idle"
                self.status_lbl.setText("Divider line set.")
            self.redraw()
        else:
            # idle – no special behavior
            pass

    # ---------------------- mode helpers ---------------------- #

    def _start_draw_stopbar(self):
        self.mode = "draw_stopbar"
        self.stopbar_pts_px = []
        self.status_lbl.setText("Click two points on the stop bar (across lanes).")

    def _start_draw_divider(self):
        self.mode = "draw_divider"
        self.divider_pts_px = []
        self.status_lbl.setText(
            "Click two points to draw the divider line (splits flows)."
        )

    def _clear_geometry(self):
        self.mode = "idle"
        self.stopbar_pts_px = []
        self.divider_pts_px = []
        self.status_lbl.setText("Geometry cleared.")
        self.redraw()


    def _init_clickhouse(self) -> None:
        self.ch = ClickHouseHTTP(
            host=self.ch_host,
            port=self.ch_port,
            user=self.ch_user,
            password=self.ch_password,
            database=self.ch_db,
        )

    def on_configure_clickhouse(self) -> None:
        try:
            cur = load_clickhouse_config()
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Keyring error", str(e))
            return

        dlg = ClickHouseConfigDialog(
            self,
            host=cur.get("host", ""),
            port=cur.get("port", 8123),
            user=cur.get("user", "default"),
            password=cur.get("password", ""),
            db=cur.get("db", "trajectories"),
        )
        if dlg.exec() != QtWidgets.QDialog.Accepted:
            return

        cfg = dlg.values()
        if not cfg["host"]:
            QtWidgets.QMessageBox.warning(self, "Invalid config", "Host is required.")
            return
        if not cfg["db"]:
            QtWidgets.QMessageBox.warning(self, "Invalid config", "Database is required.")
            return

        try:
            save_clickhouse_config(cfg)
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Keyring error", str(e))
            return

        self.ch_host = cfg["host"]
        self.ch_port = int(cfg["port"])
        self.ch_user = cfg["user"]
        self.ch_password = cfg["password"]
        self.ch_db = cfg["db"]

        try:
            self._init_clickhouse()
            self.ensure_metadata_schema()
        except Exception as e:
            QtWidgets.QMessageBox.warning(self, "ClickHouse init failed", str(e))
            return

        self.status_lbl.setText("Saved ClickHouse config (password stored in keyring).")

    # ---------------------- DB schema + save/load ---------------------- #

    def ensure_metadata_schema(self):
        """
        Create a simple metadata table if it doesn't exist yet.
        Stores stop bar + divider line in both pixel and meter space,
        plus a braking window for analysis.
        """
        if self.ch is None:
            # Not configured yet; nothing to do
            return

        db = self.ch.db
        sql = f"""
        CREATE TABLE IF NOT EXISTS {db}.stopbar_metadata
        (
            intersection_id String,
            approach_id String,

            -- stop bar in ortho pixels
            stopbar_px_x1 Float64,
            stopbar_px_y1 Float64,
            stopbar_px_x2 Float64,
            stopbar_px_y2 Float64,

            -- stop bar in GeoTIFF CRS linear units (EPSG units; e.g. US survey foot)
            stopbar_u_x1 Float64,
            stopbar_u_y1 Float64,
            stopbar_u_x2 Float64,
            stopbar_u_y2 Float64,

            -- divider line in ortho pixels
            divider_px_x1 Float64,
            divider_px_y1 Float64,
            divider_px_x2 Float64,
            divider_px_y2 Float64,

            -- divider line in GeoTIFF CRS linear units
            divider_u_x1 Float64,
            divider_u_y1 Float64,
            divider_u_x2 Float64,
            divider_u_y2 Float64,

            -- braking window in CRS linear units
            braking_window_u Float64,
            created_at DateTime DEFAULT now()
        )
        ENGINE = MergeTree
        ORDER BY (intersection_id, approach_id, created_at)
        """
        self.ch._post_sql(sql, use_db=False)


    def _px_to_u(self, x_px, y_px):
        """
        Convert ortho pixel coords -> GeoTIFF CRS linear units (EPSG units).
        For EPSG:6438 this is US survey feet.
        """
        X_u, Y_u = self.ortho_transform * (x_px, y_px)
        return float(X_u), float(Y_u)

    def _m_to_u(self, m: float) -> float:
        return float(m) / US_SURVEY_FT_TO_M


    def save_to_db(self):
        """
        Save current geometry as one row in stopbar_metadata.
        Requires:
          - 2 stop bar points
          - 2 divider points
          - intersection_id, approach_id
        """
        if self.ch is None:
            self.status_lbl.setText("ClickHouse not configured. Use 'Configure ClickHouse…' first.")
            return
        
        if not self.ortho_ok:
            self.status_lbl.setText(
                f"Cannot save geometry: no ortho TIFF loaded (expected '{ORTHO_PATH}')."
            )
            return

        if len(self.stopbar_pts_px) != 2 or len(self.divider_pts_px) != 2:
            self.status_lbl.setText(
                "Need both stop bar (2 pts) and divider (2 pts) before saving."
            )
            return

        intersection_id = self.edit_intersection_id.text().strip()
        approach_id = self.edit_approach_id.text().strip()
        if not intersection_id or not approach_id:
            self.status_lbl.setText("Intersection ID and approach ID are required.")
            return

        try:
            braking_window_m = float(self.edit_brake_window.text().strip() or "0")
            braking_window_u = self._m_to_u(braking_window_m)

        except ValueError:
            self.status_lbl.setText("Braking window must be a number (meters).")
            return

        (sx1, sy1), (sx2, sy2) = self.stopbar_pts_px
        (dx1, dy1), (dx2, dy2) = self.divider_pts_px

        su_x1, su_y1 = self._px_to_u(sx1, sy1)
        su_x2, su_y2 = self._px_to_u(sx2, sy2)

        du_x1, du_y1 = self._px_to_u(dx1, dy1)
        du_x2, du_y2 = self._px_to_u(dx2, dy2)

        # escape quotes in IDs
        isect_sql = intersection_id.replace("'", "\\'")
        appr_sql = approach_id.replace("'", "\\'")

        db = self.ch.db
        sql = f"""
        INSERT INTO {db}.stopbar_metadata
        (
            intersection_id, approach_id,
            stopbar_px_x1, stopbar_px_y1, stopbar_px_x2, stopbar_px_y2,
            stopbar_u_x1, stopbar_u_y1, stopbar_u_x2, stopbar_u_y2,
            divider_px_x1, divider_px_y1, divider_px_x2, divider_px_y2,
            divider_u_x1, divider_u_y1, divider_u_x2, divider_u_y2,
            braking_window_u
        )
        VALUES
        (
            '{isect_sql}', '{appr_sql}',
            {sx1}, {sy1}, {sx2}, {sy2},
            {su_x1}, {su_y1}, {su_x2}, {su_y2},
            {dx1}, {dy1}, {dx2}, {dy2},
            {du_x1}, {du_y1}, {du_x2}, {du_y2},
            {braking_window_u}
        )
        """
        try:
            self.ch._post_sql(sql, use_db=False)
            self.status_lbl.setText("Saved geometry to ClickHouse.")
        except Exception as e:
            self.status_lbl.setText(f"Error saving to ClickHouse: {e}")

    def load_from_db(self):
        """
        Load latest geometry for the given intersection & approach.
        """
        if self.ch is None:
            self.status_lbl.setText("ClickHouse not configured. Use 'Configure ClickHouse…' first.")
            return
        intersection_id = self.edit_intersection_id.text().strip()
        approach_id = self.edit_approach_id.text().strip()
        if not intersection_id or not approach_id:
            self.status_lbl.setText("Intersection ID and approach ID are required.")
            return

        isect_sql = intersection_id.replace("'", "\\'")
        appr_sql = approach_id.replace("'", "\\'")

        db = self.ch.db
        sql = f"""
        SELECT
            stopbar_px_x1, stopbar_px_y1, stopbar_px_x2, stopbar_px_y2,
            divider_px_x1, divider_px_y1, divider_px_x2, divider_px_y2,
            braking_window_u
        FROM {db}.stopbar_metadata
        WHERE intersection_id = '{isect_sql}'
          AND approach_id = '{appr_sql}'
        ORDER BY created_at DESC
        LIMIT 1
        FORMAT JSONEachRow
        """
        try:
            resp = self.ch._post_sql(sql, use_db=False)
        except Exception as e:
            self.status_lbl.setText(f"Error loading from ClickHouse: {e}")
            return

        text = resp.text.strip()
        if not text:
            self.status_lbl.setText("No metadata found for that intersection/approach.")
            return

        row = json.loads(text.splitlines()[0])

        sx1 = row["stopbar_px_x1"]; sy1 = row["stopbar_px_y1"]
        sx2 = row["stopbar_px_x2"]; sy2 = row["stopbar_px_y2"]
        dx1 = row["divider_px_x1"]; dy1 = row["divider_px_y1"]
        dx2 = row["divider_px_x2"]; dy2 = row["divider_px_y2"]

        self.stopbar_pts_px = [[sx1, sy1], [sx2, sy2]]
        self.divider_pts_px = [[dx1, dy1], [dx2, dy2]]

        bw_u = float(row["braking_window_u"])
        self.edit_brake_window.setText(str(bw_u * US_SURVEY_FT_TO_M))


        self.status_lbl.setText("Loaded geometry from ClickHouse.")
        self.redraw()

    # ---------------------- status ---------------------- #

    def _update_status(self):
        mode_txt = {
            "idle": "Idle",
            "draw_stopbar": "Drawing stop bar",
            "draw_divider": "Drawing divider line",
        }.get(self.mode, self.mode)
        sb = "yes" if len(self.stopbar_pts_px) == 2 else "no"
        dv = "yes" if len(self.divider_pts_px) == 2 else "no"
        self.status_lbl.setText(
            f"Mode: {mode_txt} | Stop bar set: {sb} | Divider set: {dv}"
        )
