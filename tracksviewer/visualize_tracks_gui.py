#!/usr/bin/env python3
# ClickHouse-backed trajectories viewer with datetime picker GUI

import os, sys, json
from collections import defaultdict
from typing import Dict, List, Tuple, Optional
from pathlib import Path
from datetime import datetime, timedelta

import numpy as np
import cv2
import rasterio as rio

from PySide6 import QtCore, QtGui, QtWidgets

import keyring
from keyring.errors import KeyringError

from mp4_player import Mp4PlayerTab

# --- ClickHouse client import (from ../roadpairer/) ---
HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent

dotenv_path = REPO_ROOT / ".env"

GUI_DIR = REPO_ROOT / "roadpairer"
if str(GUI_DIR) not in sys.path:
    sys.path.insert(0, str(GUI_DIR))

from clickhouse_client import ClickHouseHTTP

import dotenv
dotenv.load_dotenv(str(dotenv_path), override=False)


APP_ORG = "rectangle-traffic-camera"
APP_NAME = "TracksViewer"
SETTINGS_LAST_TIF = "paths/last_tif"
SETTINGS_MP4_DIR = "paths/mp4_dir_resolved"
SETTINGS_CH_HOST = "clickhouse/host"
SETTINGS_CH_PORT = "clickhouse/port"
SETTINGS_CH_USER = "clickhouse/user"
SETTINGS_CH_DB   = "clickhouse/db"

KEYRING_SERVICE  = "rectangle-traffic-camera/TracksViewer"
KEYRING_ACCOUNT  = "clickhouse_password"   # entry name inside the service



def ortho_units_to_m(src) -> float:
    try:
        return float(src.crs.linear_units_factor[1])
    except Exception:
        return 1.0
    

class ClickHouseConfigDialog(QtWidgets.QDialog):
    def __init__(self, parent=None, host="", port=8123, user="default", password="", db="trajectories"):
        super().__init__(parent)
        self.setWindowTitle("Configure ClickHouse")
        self.setModal(True)

        layout = QtWidgets.QVBoxLayout(self)

        form = QtWidgets.QFormLayout()
        layout.addLayout(form)

        self.host_edit = QtWidgets.QLineEdit(host)
        self.port_spin = QtWidgets.QSpinBox()
        self.port_spin.setRange(1, 65535)
        self.port_spin.setValue(int(port) if port else 8123)

        self.user_edit = QtWidgets.QLineEdit(user)
        self.db_edit   = QtWidgets.QLineEdit(db)

        self.pass_edit = QtWidgets.QLineEdit(password)
        self.pass_edit.setEchoMode(QtWidgets.QLineEdit.Password)

        self.show_pass = QtWidgets.QCheckBox("Show password")
        self.show_pass.toggled.connect(
            lambda on: self.pass_edit.setEchoMode(
                QtWidgets.QLineEdit.Normal if on else QtWidgets.QLineEdit.Password
            )
        )

        form.addRow("Host:", self.host_edit)
        form.addRow("Port:", self.port_spin)
        form.addRow("User:", self.user_edit)
        form.addRow("Database:", self.db_edit)

        pw_row = QtWidgets.QHBoxLayout()
        pw_row.addWidget(self.pass_edit, 1)
        pw_row.addWidget(self.show_pass)
        form.addRow("Password:", pw_row)

        info = QtWidgets.QLabel("Password is stored in the OS keychain (keyring).")
        info.setWordWrap(True)
        layout.addWidget(info)

        btns = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel
        )
        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)
        layout.addWidget(btns)

    def values(self) -> dict:
        return {
            "host": self.host_edit.text().strip(),
            "port": int(self.port_spin.value()),
            "user": self.user_edit.text().strip(),
            "db":   self.db_edit.text().strip(),
            "password": self.pass_edit.text(),
        }


class DeleteWorker(QtCore.QObject):
    finished = QtCore.Signal(bool, str)  # ok, err

    def __init__(self, ch: ClickHouseHTTP, sql: str, params: Optional[dict] = None):
        super().__init__()
        self.ch = ch
        self.sql = sql
        self.params = params or {}

    @QtCore.Slot()
    def run(self):
        try:
            self.ch._post_sql(self.sql, use_db=True, params=self.params)
            self.finished.emit(True, "")
        except Exception as e:
            self.finished.emit(False, str(e))


# ---------- GeoTIFF -> RGB8 (robust) ----------
def read_ortho_rgb(path: str):
    with rio.open(path) as src:
        units_to_m = ortho_units_to_m(src)
        m_per_px = abs(src.transform.a) * units_to_m
        transform = src.transform

        # TEMP: just read 3 bands as uint8-ish display (replace with your robust block)
        arr = src.read([1, 2, 3]) if src.count >= 3 else src.read(1)
        if arr.ndim == 3:
            ortho = np.moveaxis(arr, 0, 2)
        else:
            ortho = arr
            ortho = np.stack([ortho, ortho, ortho], axis=2)

        if ortho.dtype != np.uint8:
            ortho = np.clip(ortho, 0, 255).astype(np.uint8)

        return ortho, m_per_px, transform, units_to_m


# ---------- Drawing helpers ----------
def color_for_track(track_key) -> Tuple[int, int, int]:
    # Stable color based on hash of track_key (int, str, or tuple)
    seed = (abs(hash(track_key)) % (2**31 - 1)) or 1
    rng = np.random.default_rng(seed)
    rgb = rng.integers(60, 255, size=3).tolist()
    # OpenCV uses BGR
    return int(rgb[2]), int(rgb[1]), int(rgb[0])


def draw_scale_bar(img: np.ndarray, m_per_px: float, meters: float = 10.0):
    if not (m_per_px and m_per_px > 0):
        return
    px = int(round(meters / m_per_px))
    h, w = img.shape[:2]
    x0 = 40
    y0 = h - 40
    cv2.line(img, (x0, y0), (x0 + px, y0), (255, 255, 255), 6, cv2.LINE_AA)
    cv2.line(img, (x0, y0), (x0 + px, y0), (0, 0, 0), 2, cv2.LINE_AA)
    label = f"{int(meters)} m"
    cv2.putText(img, label, (x0, y0 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(img, label, (x0, y0 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (255, 255, 255), 1, cv2.LINE_AA)


def put_hud(img, text, y=26, scale=0.6):
    cv2.putText(img, text, (12, y), cv2.FONT_HERSHEY_SIMPLEX,
                scale, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(img, text, (12, y), cv2.FONT_HERSHEY_SIMPLEX,
                scale, (255, 255, 255), 1, cv2.LINE_AA)


def render_frame(base_img: np.ndarray,
                 tracks: Dict[object, List[dict]],
                 times: List[float],
                 idx: int,
                 scale: float,
                 m_per_px: float,
                 trail_len: int,
                 show_trails: bool = True,
                 highlight_keys: Optional[set] = None) -> np.ndarray:
    """
    Render a frame at times[idx] on top of base_img.

    tracks: key -> list of dicts with 't','x','y','cls'
    highlight_keys: optional set of track keys to draw in a special style
                    (e.g. {"video.mp4#123"}).
    """
    if not times:
        return base_img.copy()

    t_now = times[idx]
    frame = base_img.copy()

    if show_trails and trail_len > 0:
        start_idx = max(0, idx - trail_len)
    else:
        start_idx = idx

    t_start = times[start_idx]

    for tid, pts in tracks.items():
        # Subset points in the visible time window
        sub = [p for p in pts if t_start <= p["t"] <= t_now]
        if not sub:
            continue

        is_highlight = bool(highlight_keys and tid in highlight_keys)

        if is_highlight:
            col = (0, 255, 255)      # bright yellow-ish in BGR
            thickness = 3
            radius = 7
        else:
            # col = color_for_track(tid)
            col = (0, 255, 0)
            thickness = 2
            radius = 5

        poly = np.array([[int(p["x"] * scale), int(p["y"] * scale)]
                         for p in sub], dtype=np.int32)
        if len(poly) >= 2 and show_trails:
            cv2.polylines(frame, [poly], False, col, thickness, cv2.LINE_AA)
        cx, cy = int(sub[-1]["x"] * scale), int(sub[-1]["y"] * scale)
        cv2.circle(frame, (cx, cy), radius, col, -1, cv2.LINE_AA)

    draw_scale_bar(frame, m_per_px, meters=10.0)
    put_hud(frame, f"t = {t_now:.2f}s  ({idx + 1}/{len(times)})", y=26)
    return frame


def median_dt(times: List[float]) -> float:
    if len(times) < 2:
        return 1.0
    dts = [b - a for a, b in zip(times[:-1], times[1:])]
    dts = [dt for dt in dts if dt > 0]
    if not dts:
        return 1.0
    return float(np.median(dts))


# ---------- ClickHouse helpers ----------
def ch_query_json_each_row(ch: ClickHouseHTTP, sql: str, params: Optional[dict] = None) -> List[dict]:
    """Run a SQL query with FORMAT JSONEachRow and return list of dicts. Supports ClickHouse HTTP params."""
    resp = ch._post_sql(sql, use_db=True, params=params)
    text = resp.text.strip()
    if not text:
        return []
    rows = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        rows.append(json.loads(line))
    return rows


def load_tracks_interval(ch, start_dt, duration_s, transform, units_to_m, video_filter=None):
    """
    Load tracks from trajectories.raw in ClickHouse for a time window.

    Returns:
      tracks: dict[track_key] -> list of {'t','x','y','cls'}
      times: sorted list of unique relative seconds (float)
      t0:    earliest absolute timestamp (datetime) or None
    """
    end_dt = start_dt + timedelta(seconds=duration_s)

    # Use params (no quoting/escaping)
    start_str = start_dt.strftime("%Y-%m-%d %H:%M:%S")
    end_str = end_dt.strftime("%Y-%m-%d %H:%M:%S")

    db = ch.db

    where = (
        "WHERE timestamp >= {start_ts:DateTime} "
        "AND timestamp < {end_ts:DateTime} "
        "AND (map_m_x != 0 OR map_m_y != 0)"
    )

    params = {
        "start_ts": start_str,
        "end_ts": end_str,
    }

    if video_filter:
        where += " AND video = {video:String}"
        params["video"] = video_filter

    sql = f"""
    SELECT
        video,
        timestamp,
        track_id,
        class,
        map_m_x,
        map_m_y
    FROM {db}.raw
    {where}
    ORDER BY timestamp, video, track_id
    FORMAT JSONEachRow
    """

    rows = ch_query_json_each_row(ch, sql, params=params)
    if not rows:
        return {}, [], None

    samples = []
    for row in rows:
        try:
            video = row["video"]
            ts_str = row["timestamp"]
            ts = datetime.fromisoformat(ts_str)
            track_id = int(row["track_id"])

            x_m = float(row["map_m_x"])
            y_m = float(row["map_m_y"])

            x_u = x_m / units_to_m
            y_u = y_m / units_to_m
            col, rowpix = (~transform) * (x_u, y_u) # DO NOT CHANGE.

            x_px = float(col)
            y_px = float(rowpix)

            cls = row.get("class", "")
        except Exception:
            continue

        samples.append((video, track_id, ts, x_px, y_px, cls))

    if not samples:
        return {}, [], None

    t0 = min(s[2] for s in samples)

    tracks: Dict[object, List[dict]] = defaultdict(list)
    times_set = set()

    for video, track_id, ts, x_px, y_px, cls in samples:
        t = (ts - t0).total_seconds()
        key = f"{video}#{track_id}"
        tracks[key].append({"t": t, "x": x_px, "y": y_px, "cls": cls})
        times_set.add(t)

    for key in list(tracks.keys()):
        tracks[key].sort(key=lambda d: d["t"])
    times = sorted(times_set)
    return tracks, times, t0


# ---------- Qt GUI ----------
def numpy_to_qimage(frame: np.ndarray) -> QtGui.QImage:
    """
    Convert a BGR uint8 image to QImage for display.
    """
    if frame.ndim == 2:
        h, w = frame.shape
        bytes_per_line = w
        return QtGui.QImage(frame.data, w, h, bytes_per_line,
                            QtGui.QImage.Format_Grayscale8).copy()

    h, w, ch = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    bytes_per_line = ch * w
    return QtGui.QImage(rgb.data, w, h, bytes_per_line,
                        QtGui.QImage.Format_RGB888).copy()


class TracksPlayer(QtWidgets.QMainWindow):
    def __init__(self, tif_path: str, parent=None):
        super().__init__(parent)

        self.settings = QtCore.QSettings(APP_ORG, APP_NAME)

        # MP4 base dir: where the mp4s/ folder is
        saved_mp4_dir = self.settings.value(SETTINGS_MP4_DIR, "", type=str)
        if saved_mp4_dir and Path(saved_mp4_dir).exists():
            self.mp4_dir = Path(saved_mp4_dir)
        else:
            # default: next to app
            self.mp4_dir = (REPO_ROOT / "mp4s") if (REPO_ROOT / "mp4s").exists() else REPO_ROOT


        # If caller didn't provide a tif (or provided empty), try restoring last used
        if not tif_path:
            saved = self.settings.value(SETTINGS_LAST_TIF, "", type=str)
            if saved and Path(saved).exists():
                self.tif_path = saved
            else:
                self.tif_path = ""
        else:
            self.tif_path = tif_path

        self.setWindowTitle("Tracks Viewer (ClickHouse)")

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

        # ClickHouse config (QSettings + keyring; falls back to env/defaults)
        try:
            cfg = self._load_ch_config()
        except Exception as e:
            # Keyring is enforced — show once, keep env values as last resort for this run
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
            # keep image area black until user picks one

        self._init_clickhouse()

        # Shift+D: delete selected braking event (no confirmation)
        self.delete_shortcut = QtGui.QShortcut(QtGui.QKeySequence("Shift+D"), self)
        self.delete_shortcut.setContext(QtCore.Qt.ApplicationShortcut)
        self.delete_shortcut.activated.connect(self.on_shift_d_delete)

        # Left/Right: prev/next braking event (table order)
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


    def _load_ch_config(self) -> dict:
        s = self.settings
        # Prefer stored settings; else fall back to env; else defaults
        host = s.value(SETTINGS_CH_HOST, os.getenv("CH_HOST", ""), type=str)
        port = int(s.value(SETTINGS_CH_PORT, int(os.getenv("CH_PORT", "8123")), type=int))
        user = s.value(SETTINGS_CH_USER, os.getenv("CH_USER", "default"), type=str)
        db   = s.value(SETTINGS_CH_DB,   os.getenv("CH_DB", "trajectories"), type=str)

        # ENFORCE keyring for password
        pw = os.getenv("CH_PASSWORD", "")
        try:
            kr = keyring.get_password(KEYRING_SERVICE, KEYRING_ACCOUNT)
            if kr is not None:
                pw = kr
        except KeyringError as e:
            # If keyring is broken, treat as fatal for password retrieval
            raise RuntimeError(f"Keyring error reading password: {e}") from e

        return {"host": host, "port": port, "user": user, "db": db, "password": pw}


    def _save_ch_config(self, cfg: dict) -> None:
        # Store non-secrets in QSettings
        self.settings.setValue(SETTINGS_CH_HOST, cfg.get("host", ""))
        self.settings.setValue(SETTINGS_CH_PORT, int(cfg.get("port", 8123)))
        self.settings.setValue(SETTINGS_CH_USER, cfg.get("user", "default"))
        self.settings.setValue(SETTINGS_CH_DB,   cfg.get("db", "trajectories"))

        # ENFORCE keyring for password
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

        # Apply immediately
        self.ch_host = cfg["host"]
        self.ch_port = int(cfg["port"])
        self.ch_user = cfg["user"]
        self.ch_password = cfg["password"]
        self.ch_db = cfg["db"]

        # Re-init client so subsequent queries use new credentials
        try:
            self._init_clickhouse()
        except Exception as e:
            QtWidgets.QMessageBox.warning(self, "ClickHouse init failed", str(e))
            return

        self.status_label.setText("Saved ClickHouse config (password stored in keyring).")


    def _resolve_mp4_dir(self, chosen: Path) -> Path:
        """
        Accept either:
        - a folder that CONTAINS mp4s/ or mp4/
        - the mp4s/ folder itself (or mp4/)
        Returns the folder that directly contains .mp4 files.
        """
        p = Path(chosen).resolve()

        # If they selected mp4s/ or mp4/ directly
        if p.is_dir() and p.name.lower() in ("mp4s", "mp4"):
            return p

        # If they selected the parent folder that contains mp4s/ or mp4/
        for name in ("mp4s", "mp4"):
            child = p / name
            if child.exists() and child.is_dir():
                return child

        # Otherwise assume they selected the actual folder containing mp4s
        return p


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
        """Load a new GeoTIFF and refresh the ortho display."""
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

        # Re-render current frame using new ortho (if tracks already loaded)
        self._redraw_current_frame()
        self.status_label.setText(f"Loaded GeoTIFF: {tif_path}")


    def _jump_to_event_at_view_row(self, view_row: int) -> None:
        if not hasattr(self, "brake_table"):
            return
        if view_row < 0 or view_row >= self.brake_table.rowCount():
            return

        item0 = self.brake_table.item(view_row, 0)
        if item0 is None:
            return

        event_id = item0.data(QtCore.Qt.UserRole)
        event_id = self._norm_event_id(event_id)

        if event_id is None:
            return

        ev = getattr(self, "ev_by_id", {}).get(event_id)
        if ev is None:
            # fallback scan
            for e in self.braking_events:
                if e.get("_event_id") == event_id:
                    ev = e
                    break
            if ev is None:
                return

        # ---- FORCE visible table highlight ----
        self.brake_table.blockSignals(True)
        try:
            self.brake_table.setFocus(QtCore.Qt.OtherFocusReason)  # <-- key
            self.brake_table.clearSelection()

            # set "current" item (important for currentRow()/keyboard nav)
            self.brake_table.setCurrentItem(item0)

            # force-select the whole row via selectionModel (more reliable than selectRow)
            sm = self.brake_table.selectionModel()
            idx0 = self.brake_table.model().index(view_row, 0)
            sm.select(
                idx0,
                QtCore.QItemSelectionModel.ClearAndSelect
                | QtCore.QItemSelectionModel.Rows
            )

            self.brake_table.scrollToItem(item0, QtWidgets.QAbstractItemView.PositionAtCenter)
        finally:
            self.brake_table.blockSignals(False)

        # ---- Jump + autoplay ----
        self.jump_to_braking_event(ev)


    def on_next_braking_event(self) -> None:
        """Right arrow: go to next braking event (table order)."""
        if not hasattr(self, "brake_table"):
            return
        nrows = self.brake_table.rowCount()
        if nrows <= 0:
            return

        cur = self.brake_table.currentRow()
        if cur < 0:
            cur = 0

        next_row = min(cur + 1, nrows - 1)
        if next_row == cur and cur != 0:
            return  # already at end

        self._jump_to_event_at_view_row(next_row)


    def on_prev_braking_event(self) -> None:
        """Left arrow: go to previous braking event (table order)."""
        if not hasattr(self, "brake_table"):
            return
        nrows = self.brake_table.rowCount()
        if nrows <= 0:
            return

        cur = self.brake_table.currentRow()
        if cur < 0:
            cur = 0

        prev_row = max(cur - 1, 0)
        if prev_row == cur:
            return  # already at beginning

        self._jump_to_event_at_view_row(prev_row)


    def _delete_event_at_view_row(self, view_row: int, jump_after: bool = True) -> None:
        """
        Delete the braking event represented by the visible row `view_row`.
        UI removes immediately (fast), ClickHouse mutation runs in background,
        then optionally selects+jumps to next row.
        """
        if not hasattr(self, "brake_table"):
            return

        nrows = self.brake_table.rowCount()
        if view_row < 0 or view_row >= nrows:
            return

        item0 = self.brake_table.item(view_row, 0)
        if item0 is None:
            return

        event_id = item0.data(QtCore.Qt.UserRole)
        event_id = self._norm_event_id(event_id)

        if event_id is None:
            return

        ev = getattr(self, "ev_by_id", {}).get(event_id)
        if ev is None:
            # fallback scan (should be rare)
            for e in getattr(self, "braking_events", []):
                if e.get("_event_id") == event_id:
                    ev = e
                    break
            if ev is None:
                return

        sql, params = self._build_delete_sql(ev)

        # --- 1) UI: remove immediately (FAST) ---
        if hasattr(self, "ev_by_id"):
            self.ev_by_id.pop(event_id, None)

        self.braking_events = [e for e in self.braking_events if e.get("_event_id") != event_id]

        # Remove visible row (no full rebuild)
        self.brake_table.removeRow(view_row)
        self.status_label.setText("Deleting… (queued in background)")

        # --- 2) Background delete (async) ---
        self._start_delete_worker(sql, params=params)

        # --- 3) Select + jump to next row (table order) ---
        if not jump_after:
            return

        nrows2 = self.brake_table.rowCount()
        if nrows2 <= 0:
            self.highlight_key = None
            self.playing = False
            self.status_label.setText("Deleted. No more events.")
            return

        next_view_row = min(view_row, nrows2 - 1)

        # Let table update after removeRow, then force-select + jump (your robust method)
        QtCore.QTimer.singleShot(0, lambda r=next_view_row: self._jump_to_event_at_view_row(r))


    def on_shift_d_delete(self) -> None:
        """
        Shift+D: delete selected braking event immediately, submit CH delete in background,
        then select + jump to the next row (table order).
        """
        if not hasattr(self, "brake_table"):
            return

        view_row = self.brake_table.currentRow()
        if view_row < 0:
            return

        self._delete_event_at_view_row(view_row, jump_after=True)


    def on_delete_event_clicked(self) -> None:
        """
        Trash icon: delete that row immediately, submit CH delete in background,
        then select + jump to the next row (table order).
        """
        btn = self.sender()
        if btn is None or not hasattr(self, "brake_table"):
            return

        idx = self.brake_table.indexAt(
            btn.mapTo(self.brake_table.viewport(), QtCore.QPoint(1, 1))
        )
        view_row = idx.row()
        if view_row < 0:
            return

        self._delete_event_at_view_row(view_row, jump_after=True)



    def _build_delete_sql(self, ev: dict) -> tuple[str, dict]:
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

        # prevent leaks
        worker.finished.connect(thread.quit)
        worker.finished.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)

        thread.start()

    def _on_delete_finished(self, ok: bool, err: str, thread: QtCore.QThread, worker: QtCore.QObject):
        if ok:
            self.status_label.setText("Delete submitted to ClickHouse (background).")
        else:
            # we already removed it from UI, so just warn
            self.status_label.setText(f"Delete failed: {err}")

            
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


    def on_choose_mp4_dir(self):
        start_dir = str(self.mp4_dir) if hasattr(self, "mp4_dir") else str(REPO_ROOT)
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

        # Persist resolved folder
        self.settings.setValue(SETTINGS_MP4_DIR, str(self.mp4_dir))
        self.settings.sync()

        # Update label
        if hasattr(self, "mp4_label"):
            self.mp4_label.setText(self.mp4_dir.name)
            self.mp4_label.setToolTip(str(self.mp4_dir))

        # ✅ CRITICAL: apply to the already-created mp4 widget
        if hasattr(self, "mp4_widget") and self.mp4_widget is not None:
            # Option A: Mp4PlayerTab supports changing base_dir dynamically
            if hasattr(self.mp4_widget, "set_base_dir"):
                self.mp4_widget.set_base_dir(self.mp4_dir)
            else:
                # Option B: recreate the widget in-place
                self._recreate_mp4_widget()

        self.status_label.setText(f"MP4 folder set to: {self.mp4_dir}")


    def _recreate_mp4_widget(self) -> None:
        old = self.mp4_widget
        self.mp4_widget = Mp4PlayerTab(base_dir=self.mp4_dir, error_parent=self)

        if hasattr(self, "left_splitter"):
            # remove old widget from splitter cleanly
            if old is not None:
                old.setParent(None)
                old.deleteLater()

            # mp4 should be at index 0
            self.left_splitter.insertWidget(0, self.mp4_widget)


    # ---- UI setup ----
    def _build_ui(self):
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)

        main_layout = QtWidgets.QHBoxLayout(central)
        # Left: Ortho (top) + MP4 (bottom)
        left_layout = QtWidgets.QVBoxLayout()
        main_layout.addLayout(left_layout, stretch=3)

        self.left_splitter = QtWidgets.QSplitter(QtCore.Qt.Vertical)
        left_layout.addWidget(self.left_splitter, stretch=1)

        # -----------------------
        # Top widget: Ortho overlay
        # -----------------------
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

        # -----------------------
        # Bottom widget: MP4 player
        # -----------------------
        self.mp4_widget = Mp4PlayerTab(base_dir=self.mp4_dir, error_parent=self)
        

        self.left_splitter.addWidget(self.mp4_widget)   # TOP: video
        self.left_splitter.addWidget(ortho_widget)      # BOTTOM: ortho

        # Give initial split (top bigger than bottom, tweak as you like)
        self.left_splitter.setStretchFactor(0, 3)  # video
        self.left_splitter.setStretchFactor(1, 2)  # ortho


        # Optional: start with a specific pixel split after window shows
        # (can be finicky pre-show; stretch factors usually enough)
        # self.left_splitter.setSizes([600, 400])

        # Right: control panel (give it more space)
        right_layout = QtWidgets.QVBoxLayout()
        main_layout.addLayout(right_layout, stretch=2)

        # Start datetime picker
        right_layout.addWidget(QtWidgets.QLabel("Start datetime:"))

        # Default: 2025-02-13 09:56:00
        default_dt = QtCore.QDateTime(2025, 2, 13, 9, 56, 0)
        self.start_dt_edit = QtWidgets.QDateTimeEdit(default_dt)
        self.start_dt_edit.setCalendarPopup(True)
        self.start_dt_edit.setDisplayFormat("yyyy-MM-dd HH:mm:ss")
        right_layout.addWidget(self.start_dt_edit)

        # Duration (seconds)
        right_layout.addWidget(QtWidgets.QLabel("Window length (seconds):"))
        self.duration_spin = QtWidgets.QSpinBox()
        self.duration_spin.setRange(5, 3600)
        self.duration_spin.setValue(60)
        right_layout.addWidget(self.duration_spin)

        # Video filter (optional)
        right_layout.addWidget(QtWidgets.QLabel("Video filter (optional exact name):"))
        self.video_line = QtWidgets.QLineEdit()
        self.video_line.setPlaceholderText("e.g. Hiv00454-encoded-SB.mp4")
        right_layout.addWidget(self.video_line)

        # Trail length
        right_layout.addWidget(QtWidgets.QLabel("Trail length (samples):"))
        self.trail_spin = QtWidgets.QSpinBox()
        self.trail_spin.setRange(0, 2000)
        self.trail_spin.setValue(self.trail_len)
        self.trail_spin.valueChanged.connect(self.on_trail_changed)
        right_layout.addWidget(self.trail_spin)

        # Load trajectories button
        self.load_button = QtWidgets.QPushButton("Load trajectories from ClickHouse")
        self.load_button.clicked.connect(self.on_load_clicked)
        right_layout.addWidget(self.load_button)

        # --- Ortho / GeoTIFF chooser ---
        tif_row = QtWidgets.QHBoxLayout()
        right_layout.addLayout(tif_row)

        self.tif_label = QtWidgets.QLabel(Path(self.tif_path).name if self.tif_path else "(no tif)")
        self.tif_label.setToolTip(str(self.tif_path))
        self.tif_label.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
        tif_row.addWidget(self.tif_label, stretch=1)

        self.choose_tif_button = QtWidgets.QPushButton("Choose TIF…")
        self.choose_tif_button.clicked.connect(self.on_choose_tif)
        tif_row.addWidget(self.choose_tif_button)

        # --- MP4 folder chooser ---
        mp4_row = QtWidgets.QHBoxLayout()
        right_layout.addLayout(mp4_row)

        self.mp4_label = QtWidgets.QLabel(self.mp4_dir.name if hasattr(self, "mp4_dir") else "(mp4s)")
        self.mp4_label.setToolTip(str(getattr(self, "mp4_dir", "")))
        self.mp4_label.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
        mp4_row.addWidget(self.mp4_label, stretch=1)

        self.choose_mp4_button = QtWidgets.QPushButton("Choose MP4 folder…")
        self.choose_mp4_button.clicked.connect(self.on_choose_mp4_dir)
        mp4_row.addWidget(self.choose_mp4_button)


        self.choose_ch_button = QtWidgets.QPushButton("Configure ClickHouse…")
        self.choose_ch_button.clicked.connect(self.on_configure_clickhouse)
        right_layout.addWidget(self.choose_ch_button)


        # Status label
        self.status_label = QtWidgets.QLabel("")
        self.status_label.setWordWrap(True)
        right_layout.addWidget(self.status_label)

        # Separator
        sep = QtWidgets.QFrame()
        sep.setFrameShape(QtWidgets.QFrame.HLine)
        sep.setFrameShadow(QtWidgets.QFrame.Sunken)
        right_layout.addWidget(sep)

        # --- Braking events panel ---
        right_layout.addWidget(QtWidgets.QLabel("Braking events (double-click to jump):"))

        form = QtWidgets.QFormLayout()
        self.intersection_edit = QtWidgets.QLineEdit()
        self.approach_edit = QtWidgets.QLineEdit()

        
        # ✅ Prefill defaults
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

        # Column widths: Track narrow, Event time wide
        self.brake_table.setColumnWidth(0, 180)  # Video
        self.brake_table.setColumnWidth(1, 60)   # Track (about half width)
        self.brake_table.setColumnWidth(2, 90)   # Class
        self.brake_table.setColumnWidth(3, 70)   # dv
        self.brake_table.setColumnWidth(4, 80)   # a_min
        self.brake_table.setColumnWidth(5, 90)   # Severity
        self.brake_table.setColumnWidth(6, 260)  # Event time (nice and wide)
        self.brake_table.setColumnWidth(7, 40)   # delete icon

        right_layout.addWidget(self.brake_table, stretch=1)



        right_layout.addStretch()

    def _load_ortho(self):
        ortho, m_per_px, transform, units_to_m = read_ortho_rgb(self.tif_path)
        self.m_per_px = m_per_px
        self.transform = transform
        self.units_to_m = units_to_m



        # Scale to fit in a reasonable window height
        h0, w0 = ortho.shape[:2]
        target_h = 900
        self.scale = min(1.0, target_h / float(h0))
        disp = cv2.resize(ortho, (int(w0 * self.scale), int(h0 * self.scale)),
                          interpolation=cv2.INTER_AREA)

        self.ortho_img = ortho
        self.disp_img = disp
        self._update_image_label(disp)

    def _init_clickhouse(self):
        self.ch = ClickHouseHTTP(
            host=self.ch_host,
            port=self.ch_port,
            user=self.ch_user,
            password=self.ch_password,
            database=self.ch_db,
        )

    def _update_image_label(self, frame: np.ndarray):
        qimg = numpy_to_qimage(frame)
        pix = QtGui.QPixmap.fromImage(qimg)
        self.image_label.setPixmap(pix)

    # ---- Trajectory controls ----
    def on_play_clicked(self):
        if not self.times:
            return
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


        if self.ch is None:
            self.status_label.setText("ClickHouse client not initialized.")
            return

        start_qdt = self.start_dt_edit.dateTime()
        # PySide6: QDateTime.toPython usually exists; fallback if not
        try:
            start_dt = start_qdt.toPython()
        except AttributeError:
            secs = start_qdt.toSecsSinceEpoch()
            start_dt = datetime.fromtimestamp(secs)

        duration_s = float(self.duration_spin.value())
        video_filter = self.video_line.text().strip() or None

        self.status_label.setText("Loading tracks from ClickHouse...")
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
            self.tracks = {}
            self.times = []
            self.t0 = None
            self.time_slider.setMaximum(0)
            self._redraw_current_frame()
            return

        self.tracks = tracks
        self.times = times
        self.t0 = t0
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
        # Simple fixed-step playback: advance one index every tick
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
            # Just show the ortho
            self._update_image_label(self.disp_img)
            return

        idx = max(0, min(self.current_idx, len(self.times) - 1))

        # Build highlight set if we have a specific trajectory to emphasize
        highlight_set = {self.highlight_key} if self.highlight_key is not None else None

        frame = render_frame(
            self.disp_img,
            self.tracks,
            self.times,
            idx,
            scale=self.scale,              # <-- scale point coords down to match disp_img
            m_per_px=self.m_per_px / self.scale,  # <-- this part was already correct
            trail_len=self.trail_len,
            show_trails=self.show_trails,
            highlight_keys=highlight_set,
        )

        # Add absolute time HUD if we know it
        if self.t0 is not None:
            t_now = self.times[idx]
            abs_ts = self.t0 + timedelta(seconds=t_now)
            put_hud(frame, f"{abs_ts.isoformat(sep=' ')}", y=52, scale=0.6)

        self._update_image_label(frame)

    def on_load_braking_clicked(self):
        if self.ch is None:
            self.status_label.setText("ClickHouse client not initialized.")
            return

        intersection_id = self.intersection_edit.text().strip()
        approach_id = self.approach_edit.text().strip()
        if not intersection_id or not approach_id:
            self.status_label.setText("Enter intersection and approach IDs to load braking events.")
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


    def _norm_event_id(self, event_id):
        if event_id is None:
            return None

        # PySide may give back list even if you stored tuple
        if isinstance(event_id, list):
            event_id = tuple(event_id)

        # Normalize element types to match how you build ev_by_id keys
        if isinstance(event_id, tuple) and len(event_id) == 5:
            video, track_id, t_start, t_end, created_at = event_id
            return (str(video), int(track_id), float(t_start), float(t_end), str(created_at))

        return event_id


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


    def _get_video_base_timestamp(self, video: str) -> Optional[datetime]:
        if video in self._video_base_ts_cache:
            return self._video_base_ts_cache[video]
        if self.ch is None:
            return None

        db = self.ch.db
        sql = f"""
        SELECT min(timestamp) AS ts_min
        FROM {db}.raw
        WHERE video = {{video:String}}
        FORMAT JSONEachRow
        """
        rows = ch_query_json_each_row(self.ch, sql, params={"video": video})
        if not rows:
            return None
        ts_str = rows[0].get("ts_min")
        if not ts_str:
            return None
        try:
            ts = datetime.fromisoformat(ts_str)
        except Exception:
            return None
        self._video_base_ts_cache[video] = ts
        return ts


    def jump_to_braking_event(self, ev: dict):
        if not hasattr(self, "transform") or not hasattr(self, "units_to_m"):
            self.status_label.setText("Choose a GeoTIFF first (needed for map->pixel projection).")
            return

        if self.ch is None:
            self.status_label.setText("ClickHouse client not initialized.")
            return

        video = ev.get("video", "")
        video_smooth = video.replace(".mp4", "_track_smooth.mp4")
        track_id = ev.get("track_id", None)
        t_start = float(ev.get("t_start", 0.0))

        if hasattr(self, "mp4_widget"):
            lead = float(getattr(self, "jump_playback_lead_s", 1.5))

            mp4_dir = self.mp4_dir
            smooth_path = mp4_dir / video_smooth

            if smooth_path.exists():
                self.mp4_widget.jump_to(video=video_smooth, t_event_s=t_start, lead_s=lead, autoplay=True)
            else:
                # black screen: close unloads the cap and resets the label to the black background
                self.mp4_widget.close()
                self.status_label.setText(f"Missing smooth MP4: {video_smooth}")
                return


        # Highlight this specific trajectory
        if track_id is not None:
            self.highlight_key = f"{video}#{track_id}"
        else:
            self.highlight_key = None


        base_ts = self._get_video_base_timestamp(video)
        if base_ts is None:
            self.status_label.setText(f"Could not determine base timestamp for video {video}")
            return

        event_abs_start = base_ts + timedelta(seconds=t_start)
        pre = self.pre_event_seconds
        post = self.post_event_seconds
        window_start = event_abs_start - timedelta(seconds=pre)
        duration_s = pre + post

        # Update controls to reflect window
        qdt = QtCore.QDateTime(
            window_start.year,
            window_start.month,
            window_start.day,
            window_start.hour,
            window_start.minute,
            window_start.second,
        )
        self.start_dt_edit.setDateTime(qdt)
        self.duration_spin.setValue(int(duration_s))
        self.video_line.setText(video)

        # Load tracks around this event
        try:
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

        self.tracks = tracks
        self.times = times
        self.t0 = t0
        self.current_idx = 0
        self.time_slider.setMaximum(max(0, len(self.times) - 1))

        if not self.times:
            self.status_label.setText("No trajectory points found around this braking event.")
            self._redraw_current_frame()
            return

        # Find closest index to (event time - lead)
        if self.t0 is not None:
            lead = float(getattr(self, "jump_playback_lead_s", 3.0))
            target_abs = event_abs_start - timedelta(seconds=lead)

            # Clamp so we don't target before the loaded window
            if target_abs < self.t0:
                target_abs = self.t0

            target_rel = (target_abs - self.t0).total_seconds()
            diffs = [abs(t - target_rel) for t in self.times]
            if diffs:
                self.current_idx = int(np.argmin(diffs))
        else:
            self.current_idx = 0

        self.time_slider.setValue(self.current_idx)
        self.status_label.setText(
            f"Jumped to braking event in {video}, track {ev.get('track_id', '')}, "
            f"severity {ev.get('severity', '')}."
        )
        self.playing = True
        self._redraw_current_frame()


def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="ClickHouse-backed tracks viewer with datetime picker"
    )
    parser.add_argument(
        "--tif", required=False, default="",
        help="Path to ortho GeoTIFF used for pairing (optional; will restore last used)"
    )
    args = parser.parse_args()

    app = QtWidgets.QApplication(sys.argv)
    win = TracksPlayer(tif_path=args.tif)
    win.showMaximized()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
