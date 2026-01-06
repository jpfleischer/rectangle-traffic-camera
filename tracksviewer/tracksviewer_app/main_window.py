import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np

import pandas as pd
from collections import defaultdict
import keyring
from keyring.errors import KeyringError
from PySide6 import QtCore, QtGui, QtWidgets

from mp4_player import Mp4PlayerTab

from tracksviewer_app.constants import (
    APP_ORG, APP_NAME,
    SETTINGS_LAST_TIF, SETTINGS_MP4_DIR, SETTINGS_PARQUET_ROOT,
    SETTINGS_CH_HOST, SETTINGS_CH_PORT, SETTINGS_CH_USER, SETTINGS_CH_DB,
    KEYRING_SERVICE, KEYRING_ACCOUNT,
)
from tracksviewer_app.dialogs import ClickHouseConfigDialog
from tracksviewer_app.workers import DeleteWorker
from tracksviewer_app.geotiff import read_ortho_rgb
from tracksviewer_app.drawing import render_frame, put_hud
from tracksviewer_app.qt_helpers import numpy_to_qimage
from tracksviewer_app.clickhouse_api import (
    load_tracks_interval,
    ch_query_json_each_row,
    get_video_base_timestamp,
)
from tracksviewer_app.parquet_backend import ParquetStore

from clickhouse_client import ClickHouseHTTP


class TracksPlayer(QtWidgets.QMainWindow):
    def __init__(self, tif_path: str, repo_root: Path, parent=None):
        super().__init__(parent)

        self.repo_root = Path(repo_root).resolve()
        self.settings = QtCore.QSettings(APP_ORG, APP_NAME)

        saved_mp4_dir = self.settings.value(SETTINGS_MP4_DIR, "", type=str)
        if saved_mp4_dir and Path(saved_mp4_dir).exists():
            self.mp4_dir = Path(saved_mp4_dir)
        else:
            self.mp4_dir = (self.repo_root / "mp4s") if (self.repo_root / "mp4s").exists() else self.repo_root

        if not tif_path:
            saved = self.settings.value(SETTINGS_LAST_TIF, "", type=str)
            self.tif_path = saved if (saved and Path(saved).exists()) else ""
        else:
            self.tif_path = tif_path

        self.setWindowTitle("Tracks Viewer")

        # --- State ---
        self.ortho_img: Optional[np.ndarray] = None
        self.disp_img: Optional[np.ndarray] = None
        self.m_per_px: float = 1.0
        self.scale: float = 1.0

        self.tracks: Dict[object, List[dict]] = {}
        self.times: List[float] = []
        self.t0: Optional[datetime] = None
        self.highlight_key: Optional[str] = None

        self.playing = False
        self.trail_len = 30
        self.show_trails = True
        self.current_idx = 0
        self.gui_tick_ms = 40  # ~25 FPS

                
        self.parquet_store = None
        self.data_mode = "clickhouse"  # or "parquet"
        saved = self.settings.value(SETTINGS_PARQUET_ROOT, "", type=str)
        if saved and Path(saved).exists():
            self.parquet_store = ParquetStore(Path(saved))
            # don’t auto-switch unless you want to

        # ClickHouse config
        try:
            cfg = self._load_ch_config()
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
        self.ch: Optional[ClickHouseHTTP] = None

        # Braking events state
        self.braking_events: List[dict] = []
        self._video_base_ts_cache: Dict[str, datetime] = {}
        self.pre_event_seconds = 5.0
        self.post_event_seconds = 15.0
        self.jump_playback_lead_s = 1.5

        self._build_ui()

        if self.tif_path and Path(self.tif_path).exists():
            self._load_ortho()
        else:
            self.status_label.setText("No GeoTIFF selected yet. Click 'Choose TIF…'")

        self._init_clickhouse()

        # Shortcuts
        self.delete_shortcut = QtGui.QShortcut(QtGui.QKeySequence("Shift+D"), self)
        self.delete_shortcut.setContext(QtCore.Qt.ApplicationShortcut)
        self.delete_shortcut.activated.connect(self.on_shift_d_delete)

        self.prev_event_shortcut = QtGui.QShortcut(QtGui.QKeySequence(QtCore.Qt.Key_Left), self)
        self.prev_event_shortcut.setContext(QtCore.Qt.ApplicationShortcut)
        self.prev_event_shortcut.activated.connect(self.on_prev_braking_event)

        self.next_event_shortcut = QtGui.QShortcut(QtGui.QKeySequence(QtCore.Qt.Key_Right), self)
        self.next_event_shortcut.setContext(QtCore.Qt.ApplicationShortcut)
        self.next_event_shortcut.activated.connect(self.on_next_braking_event)

        # Timer for playback
        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.on_timer)
        self.timer.start(self.gui_tick_ms)

    # -------- Config / keyring --------
    def _load_ch_config(self) -> dict:
        s = self.settings
        host = s.value(SETTINGS_CH_HOST, os.getenv("CH_HOST", ""), type=str)
        port = int(s.value(SETTINGS_CH_PORT, int(os.getenv("CH_PORT", "8123")), type=int))
        user = s.value(SETTINGS_CH_USER, os.getenv("CH_USER", "default"), type=str)
        db = s.value(SETTINGS_CH_DB, os.getenv("CH_DB", "trajectories"), type=str)

        pw = os.getenv("CH_PASSWORD", "")
        try:
            kr = keyring.get_password(KEYRING_SERVICE, KEYRING_ACCOUNT)
            if kr is not None:
                pw = kr
        except KeyringError as e:
            raise RuntimeError(f"Keyring error reading password: {e}") from e

        return {"host": host, "port": port, "user": user, "db": db, "password": pw}

    def _save_ch_config(self, cfg: dict) -> None:
        self.settings.setValue(SETTINGS_CH_HOST, cfg.get("host", ""))
        self.settings.setValue(SETTINGS_CH_PORT, int(cfg.get("port", 8123)))
        self.settings.setValue(SETTINGS_CH_USER, cfg.get("user", "default"))
        self.settings.setValue(SETTINGS_CH_DB, cfg.get("db", "trajectories"))

        pw = cfg.get("password", "")
        try:
            keyring.set_password(KEYRING_SERVICE, KEYRING_ACCOUNT, pw)
        except KeyringError as e:
            raise RuntimeError(f"Keyring error saving password: {e}") from e

        self.settings.sync()

    def on_configure_clickhouse(self) -> None:
        try:
            cur = self._load_ch_config()
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
            self._save_ch_config(cfg)
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
        except Exception as e:
            QtWidgets.QMessageBox.warning(self, "ClickHouse init failed", str(e))
            return

        self.status_label.setText("Saved ClickHouse config (password stored in keyring).")


    def on_choose_parquet_root(self):
        saved = self.settings.value(SETTINGS_PARQUET_ROOT, "", type=str)
        start_dir = saved if (saved and Path(saved).exists()) else str(self.repo_root)

        path = QtWidgets.QFileDialog.getExistingDirectory(
            self,
            "Select Parquet export folder (contains raw/ and braking_events/)",
            start_dir,
        )
        if not path:
            return

        root = Path(path).resolve()
        if not (root / "raw").exists() or not (root / "braking_events").exists():
            self.status_label.setText("That folder doesn't look like an export root (missing raw/ or braking_events/).")
            return

        # Use the SAME ParquetStore import path you already use at the top
        self.parquet_store = ParquetStore(root)
        self.data_mode = "parquet"

        self.settings.setValue(SETTINGS_PARQUET_ROOT, str(root))
        self.settings.sync()

        # Optional: update button text so it's obvious
        self.load_button.setText("Load trajectories from Parquet")
        self.status_label.setText(f"Parquet mode enabled: {root}")

    def _tracks_from_raw_df(self, df):
        """
        Vectorized conversion of a raw dataframe (video,timestamp,track_id,class,map_m_x,map_m_y)
        into (tracks, times, t0) exactly like load_tracks_interval returns.

        Big speed win vs itertuples loop.
        """
        if df is None or len(df) == 0:
            return {}, [], None


        # ---- Normalize + filter ----
        df = df.copy()

        # Ensure timestamp is datetime
        if not pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
            df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        df = df[df["timestamp"].notna()]
        if df.empty:
            return {}, [], None

        # earliest timestamp is t0
        t0_ts = df["timestamp"].min()
        t0 = t0_ts.to_pydatetime() if hasattr(t0_ts, "to_pydatetime") else t0_ts

        # Relative time in seconds (vectorized)
        # Use pandas Timedelta -> seconds
        df["t"] = (df["timestamp"] - t0_ts).dt.total_seconds().astype("float64")

        # ---- Compute pixel coords (vectorized) ----
        # meters -> ortho units
        # (match your original: x_u = x_m / self.units_to_m)
        x_u = pd.to_numeric(df["map_m_x"], errors="coerce").astype("float64").to_numpy() / float(self.units_to_m)
        y_u = pd.to_numeric(df["map_m_y"], errors="coerce").astype("float64").to_numpy() / float(self.units_to_m)

        # Apply the SAME transform as your original:
        # col,rowpix = (~self.transform) * (x_u, y_u)
        invA = ~self.transform  # rasterio Affine
        col = invA.a * x_u + invA.b * y_u + invA.c
        rowpix = invA.d * x_u + invA.e * y_u + invA.f

        df["x_px"] = col
        df["y_px"] = rowpix

        # ---- Build outputs ----
        # times: sorted unique t
        # (use numpy for speed)
        times = np.unique(df["t"].to_numpy()).astype("float64")
        times.sort()
        times_list = times.tolist()

        # tracks: key -> list of dicts sorted by t
        tracks = defaultdict(list)

        # Ensure stable order within each track
        df.sort_values(["video", "track_id", "t"], kind="mergesort", inplace=True)

        # Group by (video, track_id) and build per-track lists
        # This loop is over tracks (usually far fewer than rows).
        has_class = "class" in df.columns
        for (video, track_id), g in df.groupby(["video", "track_id"], sort=False, dropna=False):
            # class: keep first non-null if present
            cls = ""
            if has_class:
                # keep exact string value like before (empty if missing)
                v = g["class"].iloc[0]
                cls = "" if (v is None or (isinstance(v, float) and np.isnan(v))) else str(v)

            key = f"{video}#{int(track_id)}"

            # Build dict list with python-level loop over arrays (fast)
            t_arr = g["t"].to_numpy(dtype="float64", copy=False)
            x_arr = g["x_px"].to_numpy(dtype="float64", copy=False)
            y_arr = g["y_px"].to_numpy(dtype="float64", copy=False)

            tracks[key] = [
                {"t": float(t), "x": float(x), "y": float(y), "cls": cls}
                for t, x, y in zip(t_arr, x_arr, y_arr)
            ]

        return dict(tracks), times_list, t0

    # -------- MP4 dir --------
    def _resolve_mp4_dir(self, chosen: Path) -> Path:
        p = Path(chosen).resolve()
        if p.is_dir() and p.name.lower() in ("mp4s", "mp4"):
            return p
        for name in ("mp4s", "mp4"):
            child = p / name
            if child.exists() and child.is_dir():
                return child
        return p

    def on_choose_mp4_dir(self):
        start_dir = str(self.mp4_dir) if hasattr(self, "mp4_dir") else str(self.repo_root)
        path = QtWidgets.QFileDialog.getExistingDirectory(
            self,
            "Select mp4s/ folder (or a folder that contains mp4s/)",
            start_dir,
        )
        if not path:
            return
        self.set_mp4_dir(path)

    def set_mp4_dir(self, chosen_dir: str) -> None:
        chosen = Path(chosen_dir).resolve()
        if not chosen.exists() or not chosen.is_dir():
            self.status_label.setText(f"Not a directory: {chosen}")
            return

        resolved = self._resolve_mp4_dir(chosen)
        if not resolved.exists() or not resolved.is_dir():
            self.status_label.setText(f"Could not resolve mp4 folder from: {chosen}")
            return

        self.mp4_dir = resolved
        self.settings.setValue(SETTINGS_MP4_DIR, str(self.mp4_dir))
        self.settings.sync()

        if hasattr(self, "mp4_label"):
            self.mp4_label.setText(self.mp4_dir.name)
            self.mp4_label.setToolTip(str(self.mp4_dir))

        if hasattr(self, "mp4_widget") and self.mp4_widget is not None:
            if hasattr(self.mp4_widget, "set_base_dir"):
                self.mp4_widget.set_base_dir(self.mp4_dir)
            else:
                self._recreate_mp4_widget()

        self.status_label.setText(f"MP4 folder set to: {self.mp4_dir}")

    def _recreate_mp4_widget(self) -> None:
        old = self.mp4_widget
        self.mp4_widget = Mp4PlayerTab(base_dir=self.mp4_dir, error_parent=self)

        if hasattr(self, "left_splitter"):
            if old is not None:
                old.setParent(None)
                old.deleteLater()
            self.left_splitter.insertWidget(0, self.mp4_widget)

    # -------- GeoTIFF --------
    def on_choose_tif(self):
        start_dir = str(Path(self.tif_path).parent) if self.tif_path else str(Path.cwd())
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Select ortho GeoTIFF",
            start_dir,
            "GeoTIFF (*.tif *.tiff);;All files (*)",
        )
        if not path:
            return
        self.set_tif(path)

    def set_tif(self, tif_path: str) -> None:
        self.tif_path = tif_path
        self.settings.setValue(SETTINGS_LAST_TIF, str(tif_path))
        self.settings.sync()

        if hasattr(self, "tif_label"):
            p = Path(tif_path)
            self.tif_label.setText(p.name)
            self.tif_label.setToolTip(str(p))

        self.status_label.setText(f"Loading GeoTIFF: {tif_path}")
        QtWidgets.QApplication.processEvents()

        try:
            self._load_ortho()
        except Exception as e:
            self.status_label.setText(f"Failed to load GeoTIFF: {e}")
            return

        self._redraw_current_frame()
        self.status_label.setText(f"Loaded GeoTIFF: {tif_path}")

    def _load_ortho(self):
        ortho, m_per_px, transform, units_to_m = read_ortho_rgb(self.tif_path)
        self.m_per_px = m_per_px
        self.transform = transform
        self.units_to_m = units_to_m

        h0, w0 = ortho.shape[:2]
        target_h = 900
        self.scale = min(1.0, target_h / float(h0))
        disp = cv2.resize(ortho, (int(w0 * self.scale), int(h0 * self.scale)), interpolation=cv2.INTER_AREA)

        self.ortho_img = ortho
        self.disp_img = disp
        self._update_image_label(disp)

    # -------- ClickHouse client --------
    def _init_clickhouse(self):
        self.ch = ClickHouseHTTP(
            host=self.ch_host,
            port=self.ch_port,
            user=self.ch_user,
            password=self.ch_password,
            database=self.ch_db,
        )

    # -------- Qt image display --------
    def _update_image_label(self, frame: np.ndarray):
        qimg = numpy_to_qimage(frame)
        pix = QtGui.QPixmap.fromImage(qimg)
        self.image_label.setPixmap(pix)

    # -------- UI construction --------
    def _build_ui(self):
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)

        main_layout = QtWidgets.QHBoxLayout(central)

        left_layout = QtWidgets.QVBoxLayout()
        main_layout.addLayout(left_layout, stretch=3)

        self.left_splitter = QtWidgets.QSplitter(QtCore.Qt.Vertical)
        left_layout.addWidget(self.left_splitter, stretch=1)

        # Ortho widget
        ortho_widget = QtWidgets.QWidget()
        ortho_layout = QtWidgets.QVBoxLayout(ortho_widget)
        ortho_layout.setContentsMargins(0, 0, 0, 0)

        self.image_label = QtWidgets.QLabel("No data loaded yet")
        self.image_label.setAlignment(QtCore.Qt.AlignCenter)
        self.image_label.setMinimumSize(640, 240)
        self.image_label.setStyleSheet("background-color: black; color: white;")
        ortho_layout.addWidget(self.image_label, stretch=1)

        self.time_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.time_slider.setMinimum(0)
        self.time_slider.setMaximum(0)
        self.time_slider.valueChanged.connect(self.on_slider_changed)
        ortho_layout.addWidget(self.time_slider)

        controls_layout = QtWidgets.QHBoxLayout()
        ortho_layout.addLayout(controls_layout)

        self.play_button = QtWidgets.QPushButton("Play")
        self.play_button.clicked.connect(self.on_play_clicked)
        controls_layout.addWidget(self.play_button)

        self.pause_button = QtWidgets.QPushButton("Pause")
        self.pause_button.clicked.connect(self.on_pause_clicked)
        controls_layout.addWidget(self.pause_button)

        self.trails_checkbox = QtWidgets.QCheckBox("Show trails")
        self.trails_checkbox.setChecked(True)
        self.trails_checkbox.toggled.connect(self.on_trails_toggled)
        controls_layout.addWidget(self.trails_checkbox)

        controls_layout.addStretch()

        # MP4 widget
        self.mp4_widget = Mp4PlayerTab(base_dir=self.mp4_dir, error_parent=self)

        self.left_splitter.addWidget(self.mp4_widget)  # TOP: video
        self.left_splitter.addWidget(ortho_widget)     # BOTTOM: ortho
        self.left_splitter.setStretchFactor(0, 3)
        self.left_splitter.setStretchFactor(1, 2)

        # Right panel
        right_layout = QtWidgets.QVBoxLayout()
        main_layout.addLayout(right_layout, stretch=2)

        right_layout.addWidget(QtWidgets.QLabel("Start datetime:"))

        default_dt = QtCore.QDateTime(2025, 2, 13, 9, 56, 0)
        self.start_dt_edit = QtWidgets.QDateTimeEdit(default_dt)
        self.start_dt_edit.setCalendarPopup(True)
        self.start_dt_edit.setDisplayFormat("yyyy-MM-dd HH:mm:ss")
        right_layout.addWidget(self.start_dt_edit)

        right_layout.addWidget(QtWidgets.QLabel("Window length (seconds):"))
        self.duration_spin = QtWidgets.QSpinBox()
        self.duration_spin.setRange(5, 3600)
        self.duration_spin.setValue(60)
        right_layout.addWidget(self.duration_spin)

        right_layout.addWidget(QtWidgets.QLabel("Video filter (optional exact name):"))
        self.video_line = QtWidgets.QLineEdit()
        self.video_line.setPlaceholderText("e.g. Hiv00454-encoded-SB.mp4")
        right_layout.addWidget(self.video_line)

        right_layout.addWidget(QtWidgets.QLabel("Trail length (samples):"))
        self.trail_spin = QtWidgets.QSpinBox()
        self.trail_spin.setRange(0, 2000)
        self.trail_spin.setValue(self.trail_len)
        self.trail_spin.valueChanged.connect(self.on_trail_changed)
        right_layout.addWidget(self.trail_spin)

        self.load_button = QtWidgets.QPushButton("Load trajectories from ClickHouse")
        self.load_button.clicked.connect(self.on_load_clicked)
        right_layout.addWidget(self.load_button)

        tif_row = QtWidgets.QHBoxLayout()
        right_layout.addLayout(tif_row)

        self.tif_label = QtWidgets.QLabel(Path(self.tif_path).name if self.tif_path else "(no tif)")
        self.tif_label.setToolTip(str(self.tif_path))
        self.tif_label.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
        tif_row.addWidget(self.tif_label, stretch=1)

        self.choose_tif_button = QtWidgets.QPushButton("Choose TIF…")
        self.choose_tif_button.clicked.connect(self.on_choose_tif)
        tif_row.addWidget(self.choose_tif_button)

        mp4_row = QtWidgets.QHBoxLayout()
        right_layout.addLayout(mp4_row)

        self.mp4_label = QtWidgets.QLabel(self.mp4_dir.name if hasattr(self, "mp4_dir") else "(mp4s)")
        self.mp4_label.setToolTip(str(getattr(self, "mp4_dir", "")))
        self.mp4_label.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
        mp4_row.addWidget(self.mp4_label, stretch=1)

        self.choose_mp4_button = QtWidgets.QPushButton("Choose MP4 folder…")
        self.choose_mp4_button.clicked.connect(self.on_choose_mp4_dir)
        mp4_row.addWidget(self.choose_mp4_button)

        ch_row = QtWidgets.QHBoxLayout()
        right_layout.addLayout(ch_row)

        self.choose_ch_button = QtWidgets.QPushButton("Configure ClickHouse…")
        self.choose_ch_button.clicked.connect(self.on_configure_clickhouse)
        ch_row.addWidget(self.choose_ch_button)

        self.choose_parquet_button = QtWidgets.QPushButton("Load Parquet…")
        self.choose_parquet_button.clicked.connect(self.on_choose_parquet_root)
        ch_row.addWidget(self.choose_parquet_button)


        self.status_label = QtWidgets.QLabel("")
        self.status_label.setWordWrap(True)
        right_layout.addWidget(self.status_label)

        sep = QtWidgets.QFrame()
        sep.setFrameShape(QtWidgets.QFrame.HLine)
        sep.setFrameShadow(QtWidgets.QFrame.Sunken)
        right_layout.addWidget(sep)

        right_layout.addWidget(QtWidgets.QLabel("Braking events (double-click to jump):"))

        form = QtWidgets.QFormLayout()
        self.intersection_edit = QtWidgets.QLineEdit()
        self.approach_edit = QtWidgets.QLineEdit()
        self.intersection_edit.setText("1")
        self.approach_edit.setText("toward_cam_main")
        form.addRow("Intersection ID:", self.intersection_edit)
        form.addRow("Approach ID:", self.approach_edit)
        right_layout.addLayout(form)

        self.load_brake_button = QtWidgets.QPushButton("Load braking events")
        self.load_brake_button.clicked.connect(self.on_load_braking_clicked)
        right_layout.addWidget(self.load_brake_button)

        self.brake_table = QtWidgets.QTableWidget(0, 8)
        self.brake_table.setHorizontalHeaderLabels([
            "Video", "Track", "Class",
            "dv", "a_min", "Severity", "Event time",
            ""
        ])
        self.brake_table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.brake_table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.brake_table.setSortingEnabled(True)
        self.brake_table.doubleClicked.connect(self.on_braking_event_activated)

        header = self.brake_table.horizontalHeader()
        header.setSectionResizeMode(QtWidgets.QHeaderView.Interactive)
        header.setStretchLastSection(False)

        self.brake_table.setColumnWidth(0, 180)
        self.brake_table.setColumnWidth(1, 60)
        self.brake_table.setColumnWidth(2, 90)
        self.brake_table.setColumnWidth(3, 70)
        self.brake_table.setColumnWidth(4, 80)
        self.brake_table.setColumnWidth(5, 90)
        self.brake_table.setColumnWidth(6, 260)
        self.brake_table.setColumnWidth(7, 40)

        right_layout.addWidget(self.brake_table, stretch=1)
        right_layout.addStretch()

    # -------- Trajectory controls --------
    def on_play_clicked(self):
        if self.times:
            self.playing = True

    def on_pause_clicked(self):
        self.playing = False

    def on_trails_toggled(self, checked: bool):
        self.show_trails = checked
        self._redraw_current_frame()

    def on_trail_changed(self, val: int):
        self.trail_len = val
        self._redraw_current_frame()

    def on_slider_changed(self, value: int):
        if not self.times:
            return
        self.current_idx = max(0, min(value, len(self.times) - 1))
        self._redraw_current_frame()

    def on_load_clicked(self):
        if not hasattr(self, "transform") or not hasattr(self, "units_to_m"):
            self.status_label.setText("Choose a GeoTIFF first (needed for map->pixel projection).")
            return

        start_qdt = self.start_dt_edit.dateTime()
        try:
            start_dt = start_qdt.toPython()
        except AttributeError:
            secs = start_qdt.toSecsSinceEpoch()
            start_dt = datetime.fromtimestamp(secs)

        duration_s = float(self.duration_spin.value())
        video_filter = self.video_line.text().strip() or None

        # -------- PARQUET MODE --------
        if self.data_mode == "parquet":
            if self.parquet_store is None:
                self.status_label.setText("Parquet mode enabled, but no parquet folder selected.")
                return

            self.status_label.setText("Loading tracks from Parquet…")
            QtWidgets.QApplication.processEvents()

            try:
                df = self.parquet_store.load_raw_interval(
                    start_dt=start_dt,
                    duration_s=duration_s,
                    video_filter=video_filter,
                )
                tracks, times, t0 = self._tracks_from_raw_df(df)
            except Exception as e:
                self.status_label.setText(f"Error loading from Parquet: {e}")
                self.tracks, self.times, self.t0 = {}, [], None
                self.time_slider.setMaximum(0)
                self._redraw_current_frame()
                return

        # -------- CLICKHOUSE MODE --------
        else:
            if self.ch is None:
                self.status_label.setText("ClickHouse client not initialized.")
                return

            self.status_label.setText("Loading tracks from ClickHouse…")
            QtWidgets.QApplication.processEvents()

            try:
                tracks, times, t0 = load_tracks_interval(
                    self.ch,
                    start_dt=start_dt,
                    duration_s=duration_s,
                    transform=self.transform,
                    units_to_m=self.units_to_m,
                    video_filter=video_filter,
                )
            except Exception as e:
                self.status_label.setText(f"Error loading from ClickHouse: {e}")
                self.tracks, self.times, self.t0 = {}, [], None
                self.time_slider.setMaximum(0)
                self._redraw_current_frame()
                return

        # Common post-load
        self.tracks, self.times, self.t0 = tracks, times, t0
        self.current_idx = 0
        self.highlight_key = None

        self.time_slider.setMaximum(max(0, len(self.times) - 1))
        self.time_slider.setValue(0)

        if not self.times:
            self.status_label.setText("No map points found in this window.")
        else:
            n_tracks = len(self.tracks)
            n_samples = sum(len(v) for v in self.tracks.values())
            info = (
                f"Loaded {n_tracks} tracks / {n_samples} samples\n"
                f"Time span: {self.times[0]:.2f}s → {self.times[-1]:.2f}s"
            )
            if self.t0:
                info += f"\nFirst timestamp: {self.t0.isoformat(sep=' ')}"
            self.status_label.setText(info)

        self._redraw_current_frame()

    def on_timer(self):
        if not self.playing or not self.times:
            return
        self.current_idx += 1
        if self.current_idx >= len(self.times):
            self.current_idx = len(self.times) - 1
            self.playing = False
        self.time_slider.blockSignals(True)
        self.time_slider.setValue(self.current_idx)
        self.time_slider.blockSignals(False)
        self._redraw_current_frame()

    def _redraw_current_frame(self):
        if self.disp_img is None:
            return
        if not self.times or not self.tracks:
            self._update_image_label(self.disp_img)
            return

        idx = max(0, min(self.current_idx, len(self.times) - 1))
        highlight_set = {self.highlight_key} if self.highlight_key is not None else None

        frame = render_frame(
            self.disp_img,
            self.tracks,
            self.times,
            idx,
            scale=self.scale,
            m_per_px=self.m_per_px / self.scale,
            trail_len=self.trail_len,
            show_trails=self.show_trails,
            highlight_keys=highlight_set,
        )

        if self.t0 is not None:
            t_now = self.times[idx]
            abs_ts = self.t0 + timedelta(seconds=t_now)
            put_hud(frame, f"{abs_ts.isoformat(sep=' ')}", y=52, scale=0.6)

        self._update_image_label(frame)

    # -------- Braking events --------
    def _norm_event_id(self, event_id):
        if event_id is None:
            return None
        if isinstance(event_id, list):
            event_id = tuple(event_id)
        if isinstance(event_id, tuple) and len(event_id) == 5:
            video, track_id, t_start, t_end, created_at = event_id
            return (str(video), int(track_id), float(t_start), float(t_end), str(created_at))
        return event_id

    def on_load_braking_clicked(self):
        intersection_id = self.intersection_edit.text().strip()
        approach_id = self.approach_edit.text().strip()
        if not intersection_id or not approach_id:
            self.status_label.setText("Enter intersection and approach IDs to load braking events.")
            return

        # -------- PARQUET MODE --------
        if self.data_mode == "parquet":
            if self.parquet_store is None:
                self.status_label.setText("Parquet mode enabled, but no parquet folder selected.")
                return

            self.status_label.setText("Loading braking events from Parquet…")
            QtWidgets.QApplication.processEvents()

            try:
                df = self.parquet_store.load_braking_events()
                deleted = self.parquet_store.load_deleted_event_ids()
                if deleted:
                    # build event_id for each row and filter
                    ids = list(zip(
                        df["video"].astype(str),
                        df["track_id"].astype(int),
                        df["t_start"].astype(float),
                        df["t_end"].astype(float),
                        df["created_at"].astype(str),
                    ))
                    mask = [eid not in deleted for eid in ids]
                    df = df[mask]

                # filter same as SQL
                df = df[
                    (df["intersection_id"].astype(str) == str(intersection_id)) &
                    (df["approach_id"].astype(str) == str(approach_id))
                ]
                # match SQL order
                if "created_at" in df.columns:
                    df = df.sort_values("created_at", ascending=False, kind="mergesort")

                rows = df.to_dict(orient="records")
            except Exception as e:
                self.status_label.setText(f"Error loading braking_events from Parquet: {e}")
                return

        # -------- CLICKHOUSE MODE --------
        else:
            if self.ch is None:
                self.status_label.setText("ClickHouse client not initialized.")
                return

            db = self.ch.db
            sql = f"""
            SELECT
                intersection_id,
                approach_id,
                video,
                track_id,
                class,
                t_start,
                t_end,
                v_start,
                v_end,
                dv,
                a_min,
                severity,
                created_at
            FROM {db}.braking_events
            WHERE intersection_id = {{intersection_id:String}}
            AND approach_id = {{approach_id:String}}
            ORDER BY created_at DESC
            FORMAT JSONEachRow
            """

            try:
                rows = ch_query_json_each_row(
                    self.ch,
                    sql,
                    params={"intersection_id": intersection_id, "approach_id": approach_id},
                )
            except Exception as e:
                self.status_label.setText(f"Error loading braking_events: {e}")
                return

        self.braking_events = rows

        # build stable id -> event mapping
        self.ev_by_id = {}
        for ev in self.braking_events:
            event_id = (
                ev.get("video", ""),
                int(ev.get("track_id", 0) or 0),
                float(ev.get("t_start", 0.0) or 0.0),
                float(ev.get("t_end", 0.0) or 0.0),
                str(ev.get("created_at", "")),
            )
            ev["_event_id"] = event_id
            self.ev_by_id[event_id] = ev

        self._refresh_brake_table_from_cache()

        self.status_label.setText(
            f"Loaded {len(self.braking_events)} braking events for {intersection_id}/{approach_id}. "
            "Double-click a row to jump."
        )

    def _get_video_base_timestamp(self, video: str) -> Optional[datetime]:
        # cache
        if video in self._video_base_ts_cache:
            return self._video_base_ts_cache[video]

        # parquet mode
        if self.data_mode == "parquet":
            if self.parquet_store is None:
                return None
            try:
                ts = self.parquet_store.get_video_base_timestamp(video)  # you'll add this
            except Exception:
                ts = None
            if ts is not None:
                self._video_base_ts_cache[video] = ts
            return ts

        # clickhouse mode
        if self.ch is None:
            return None
        ts = get_video_base_timestamp(self.ch, self._video_base_ts_cache, video)
        return ts

    def _refresh_brake_table_from_cache(self):
        self.brake_table.setUpdatesEnabled(False)
        try:
            rows = self.braking_events

            self.brake_table.setSortingEnabled(False)
            self.brake_table.setRowCount(0)
            for i, row in enumerate(rows):
                video = row.get("video", "")
                track_id = row.get("track_id", "")
                t_start = float(row.get("t_start", 0.0) or 0.0)
                dv = float(row.get("dv", 0.0) or 0.0)
                a_min = float(row.get("a_min", 0.0) or 0.0)
                cls = row.get("class", "")
                severity = row.get("severity", "")

                base_ts = self._get_video_base_timestamp(video)
                if base_ts is not None:
                    event_ts = (base_ts + timedelta(seconds=t_start)).isoformat(sep=" ")
                else:
                    event_ts = ""

                self.brake_table.insertRow(i)

                values = [
                    video,
                    str(track_id),
                    cls,
                    f"{dv:.2f}",
                    f"{a_min:.2f}",
                    severity,
                    event_ts,
                ]

                for j, val in enumerate(values):
                    item = QtWidgets.QTableWidgetItem(val)
                    if j == 0:
                        item.setData(QtCore.Qt.UserRole, tuple(row["_event_id"]))
                    self.brake_table.setItem(i, j, item)

                btn = QtWidgets.QToolButton()
                btn.setToolTip("Delete this event")
                btn.setCursor(QtCore.Qt.PointingHandCursor)
                btn.setIcon(self.style().standardIcon(QtWidgets.QStyle.SP_TrashIcon))
                btn.clicked.connect(self.on_delete_event_clicked)
                self.brake_table.setCellWidget(i, 7, btn)

            self.brake_table.setSortingEnabled(True)
            self.brake_table.sortItems(6, QtCore.Qt.AscendingOrder)
        finally:
            self.brake_table.setUpdatesEnabled(True)

    def on_braking_event_activated(self, index: QtCore.QModelIndex):
        row = index.row()
        item0 = self.brake_table.item(row, 0)
        if item0 is None:
            return

        event_id = self._norm_event_id(item0.data(QtCore.Qt.UserRole))
        if event_id is None:
            return

        ev = getattr(self, "ev_by_id", {}).get(event_id)
        if ev is None:
            return

        self.jump_to_braking_event(ev)

    # --- table navigation + delete ---
    def _jump_to_event_at_view_row(self, view_row: int) -> None:
        if view_row < 0 or view_row >= self.brake_table.rowCount():
            return

        item0 = self.brake_table.item(view_row, 0)
        if item0 is None:
            return

        event_id = self._norm_event_id(item0.data(QtCore.Qt.UserRole))
        if event_id is None:
            return

        ev = getattr(self, "ev_by_id", {}).get(event_id)
        if ev is None:
            return

        self.brake_table.blockSignals(True)
        try:
            self.brake_table.setFocus(QtCore.Qt.OtherFocusReason)
            self.brake_table.clearSelection()
            self.brake_table.setCurrentItem(item0)
            sm = self.brake_table.selectionModel()
            idx0 = self.brake_table.model().index(view_row, 0)
            sm.select(idx0, QtCore.QItemSelectionModel.ClearAndSelect | QtCore.QItemSelectionModel.Rows)
            self.brake_table.scrollToItem(item0, QtWidgets.QAbstractItemView.PositionAtCenter)
        finally:
            self.brake_table.blockSignals(False)

        self.jump_to_braking_event(ev)

    def on_next_braking_event(self) -> None:
        nrows = self.brake_table.rowCount()
        if nrows <= 0:
            return
        cur = self.brake_table.currentRow()
        if cur < 0:
            cur = 0
        next_row = min(cur + 1, nrows - 1)
        if next_row == cur and cur != 0:
            return
        self._jump_to_event_at_view_row(next_row)

    def on_prev_braking_event(self) -> None:
        nrows = self.brake_table.rowCount()
        if nrows <= 0:
            return
        cur = self.brake_table.currentRow()
        if cur < 0:
            cur = 0
        prev_row = max(cur - 1, 0)
        if prev_row == cur:
            return
        self._jump_to_event_at_view_row(prev_row)

    def on_shift_d_delete(self) -> None:
        view_row = self.brake_table.currentRow()
        if view_row < 0:
            return
        self._delete_event_at_view_row(view_row, jump_after=True)

    def on_delete_event_clicked(self) -> None:
        btn = self.sender()
        if btn is None:
            return
        idx = self.brake_table.indexAt(btn.mapTo(self.brake_table.viewport(), QtCore.QPoint(1, 1)))
        view_row = idx.row()
        if view_row < 0:
            return
        self._delete_event_at_view_row(view_row, jump_after=True)

    def _build_delete_sql(self, ev: dict):
        db = self.ch.db
        sql = f"""
        ALTER TABLE {db}.braking_events
        DELETE WHERE
            intersection_id = {{intersection_id:String}}
            AND approach_id = {{approach_id:String}}
            AND video = {{video:String}}
            AND track_id = {{track_id:UInt32}}
            AND t_start = {{t_start:Float64}}
            AND t_end = {{t_end:Float64}}
            AND created_at = {{created_at:String}}
        """
        params = {
            "intersection_id": str(ev.get("intersection_id", "")),
            "approach_id": str(ev.get("approach_id", "")),
            "video": str(ev.get("video", "")),
            "track_id": int(ev.get("track_id", 0) or 0),
            "t_start": float(ev.get("t_start", 0.0) or 0.0),
            "t_end": float(ev.get("t_end", 0.0) or 0.0),
            "created_at": str(ev.get("created_at", "")),
        }
        return sql, params

    def _start_delete_worker(self, sql: str, params: Optional[dict] = None) -> None:
        if self.ch is None:
            self.status_label.setText("ClickHouse client not initialized.")
            return

        thread = QtCore.QThread(self)
        worker = DeleteWorker(self.ch, sql, params=params)
        worker.moveToThread(thread)

        thread.started.connect(worker.run)
        worker.finished.connect(lambda ok, err: self._on_delete_finished(ok, err, thread, worker))

        worker.finished.connect(thread.quit)
        worker.finished.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)

        thread.start()

    def _on_delete_finished(self, ok: bool, err: str, _thread, _worker):
        self.status_label.setText("Delete submitted to ClickHouse (background)." if ok else f"Delete failed: {err}")

    def _delete_event_at_view_row(self, view_row: int, jump_after: bool = True) -> None:
        nrows = self.brake_table.rowCount()
        if view_row < 0 or view_row >= nrows:
            return

        item0 = self.brake_table.item(view_row, 0)
        if item0 is None:
            return

        event_id = self._norm_event_id(item0.data(QtCore.Qt.UserRole))
        if event_id is None:
            return

        ev = getattr(self, "ev_by_id", {}).get(event_id)
        if ev is None:
            return

        # ---- PARQUET MODE: tombstone ----
        if self.data_mode == "parquet":
            if self.parquet_store is None:
                self.status_label.setText("Parquet mode enabled, but no parquet folder selected.")
                return

            # write tombstone (append-only)
            eid = self.parquet_store.tombstone_event(ev)
            self.status_label.setText(f"Tombstoned={bool(eid)} -> {self.parquet_store.tombstone_path}")


            # UI removal (same as before)
            self.ev_by_id.pop(event_id, None)
            self.braking_events = [e for e in self.braking_events if e.get("_event_id") != event_id]
            self.brake_table.removeRow(view_row)
            self.status_label.setText("Deleted (parquet tombstone).")
            
            # QtWidgets.QMessageBox.information(
            #     self,
            #     "Parquet delete",
            #     f"tombstoned={bool(eid)}\npath={self.parquet_store.tombstone_path}",
            # )

        # ---- CLICKHOUSE MODE: background delete ----
        else:
            if self.ch is None:
                return

            sql, params = self._build_delete_sql(ev)

            self.ev_by_id.pop(event_id, None)
            self.braking_events = [e for e in self.braking_events if e.get("_event_id") != event_id]
            self.brake_table.removeRow(view_row)
            self.status_label.setText("Deleting… (queued in background)")

            self._start_delete_worker(sql, params=params)

        # ---- select+jump to next row ----
        if not jump_after:
            return

        nrows2 = self.brake_table.rowCount()
        if nrows2 <= 0:
            self.highlight_key = None
            self.playing = False
            self.status_label.setText("Deleted. No more events.")
            return

        next_view_row = min(view_row, nrows2 - 1)
        QtCore.QTimer.singleShot(0, lambda r=next_view_row: self._jump_to_event_at_view_row(r))

    # --- jump-to-event ---
    def jump_to_braking_event(self, ev: dict):
        if not hasattr(self, "transform") or not hasattr(self, "units_to_m"):
            self.status_label.setText("Choose a GeoTIFF first (needed for map->pixel projection).")
            return
        if self.data_mode == "clickhouse" and self.ch is None:
            self.status_label.setText("ClickHouse client not initialized.")
            return
        if self.data_mode == "parquet" and self.parquet_store is None:
            self.status_label.setText("Parquet mode enabled, but no parquet folder selected.")
            return

        video = ev.get("video", "")
        video_smooth = video.replace(".mp4", "_track_smooth.mp4")
        track_id = ev.get("track_id", None)
        t_start = float(ev.get("t_start", 0.0))

        if hasattr(self, "mp4_widget"):
            lead = float(getattr(self, "jump_playback_lead_s", 1.5))
            smooth_path = self.mp4_dir / video_smooth
            if smooth_path.exists():
                self.mp4_widget.jump_to(video=video_smooth, t_event_s=t_start, lead_s=lead, autoplay=True)
            else:
                self.mp4_widget.close()
                self.status_label.setText(f"Missing smooth MP4: {video_smooth}")
                return

        self.highlight_key = f"{video}#{track_id}" if track_id is not None else None

        base_ts = self._get_video_base_timestamp(video)
        if base_ts is None:
            self.status_label.setText(f"Could not determine base timestamp for video {video}")
            return

        event_abs_start = base_ts + timedelta(seconds=t_start)
        pre = self.pre_event_seconds
        post = self.post_event_seconds
        window_start = event_abs_start - timedelta(seconds=pre)
        duration_s = pre + post

        qdt = QtCore.QDateTime(
            window_start.year, window_start.month, window_start.day,
            window_start.hour, window_start.minute, window_start.second,
        )
        self.start_dt_edit.setDateTime(qdt)
        self.duration_spin.setValue(int(duration_s))
        self.video_line.setText(video)

        try:
            if self.data_mode == "parquet":
                df = self.parquet_store.load_raw_interval(
                    start_dt=window_start,
                    duration_s=duration_s,
                    video_filter=video,
                )
                tracks, times, t0 = self._tracks_from_raw_df(df)
            else:
                tracks, times, t0 = load_tracks_interval(
                    self.ch,
                    start_dt=window_start,
                    duration_s=duration_s,
                    transform=self.transform,
                    units_to_m=self.units_to_m,
                    video_filter=video,
                )
        except Exception as e:
            self.status_label.setText(f"Error loading tracks for event: {e}")
            return

        self.tracks, self.times, self.t0 = tracks, times, t0
        self.current_idx = 0
        self.time_slider.setMaximum(max(0, len(self.times) - 1))

        if not self.times:
            self.status_label.setText("No trajectory points found around this braking event.")
            self._redraw_current_frame()
            return

        if self.t0 is not None:
            lead = float(getattr(self, "jump_playback_lead_s", 3.0))
            target_abs = max(self.t0, event_abs_start - timedelta(seconds=lead))
            target_rel = (target_abs - self.t0).total_seconds()
            diffs = [abs(t - target_rel) for t in self.times]
            if diffs:
                self.current_idx = int(np.argmin(diffs))

        self.time_slider.setValue(self.current_idx)
        self.status_label.setText(
            f"Jumped to braking event in {video}, track {ev.get('track_id', '')}, "
            f"severity {ev.get('severity', '')}."
        )
        self.playing = True
        self._redraw_current_frame()
