#!/usr/bin/env python3
"""
mp4_player.py

A self-contained MP4 playback tab widget for PySide6 using OpenCV VideoCapture.

Expected folder layout:
  app.py
  mp4_player.py
  mp4s/
    hiv00425.mp4
    hiv00426.mp4
    ...

Usage (later, in app.py):
  from mp4_player import Mp4PlayerTab
  self.mp4_tab = Mp4PlayerTab(base_dir=HERE, error_parent=self)
  self.left_tabs.addTab(self.mp4_tab, "MP4")

Then, on double-click:
  self.mp4_tab.jump_to(video="hiv00425.mp4", t_event_s=t_start, lead_s=3.0, autoplay=True)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets


def bgr_to_qimage(frame_bgr: np.ndarray) -> QtGui.QImage:
    """
    Convert BGR uint8 image (OpenCV) to QImage for display.
    Returns a deep copy (safe after frame is freed).
    """
    if frame_bgr is None or frame_bgr.size == 0:
        return QtGui.QImage()
    if frame_bgr.ndim == 2:
        h, w = frame_bgr.shape
        return QtGui.QImage(frame_bgr.data, w, h, w, QtGui.QImage.Format_Grayscale8).copy()

    h, w, ch = frame_bgr.shape
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    bytes_per_line = ch * w
    return QtGui.QImage(rgb.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888).copy()


@dataclass
class Mp4Info:
    path: Path
    fps: float
    frame_count: int
    width: int
    height: int


class Mp4PlayerTab(QtWidgets.QWidget):
    """
    A QWidget that can be placed as a tab in a QTabWidget.

    - Loads MP4 from base_dir/mp4s/<video_name>
    - Slider scrubs by frame
    - jump_to(video, t_event_s, lead_s) seeks to (t_event_s - lead_s) and optionally autoplays
    """

    def __init__(self, base_dir: Path, error_parent: Optional[QtWidgets.QWidget] = None, parent=None):
        super().__init__(parent)
        self.base_dir = Path(base_dir).resolve()
        self.mp4_dir = self.base_dir / "mp4s"
        self.error_parent = error_parent or self

        # Playback state
        self.cap: Optional[cv2.VideoCapture] = None
        self.info: Optional[Mp4Info] = None
        self.playing: bool = False
        self.frame_idx: int = 0

        # Timing: separate from app.py timer; you can also drive tick() from app.py if preferred
        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self._on_tick)
        self.timer.start(33)  # ~30 FPS visual tick; actual frame stepping still uses +1 frame per tick

        self._build_ui()
        self._set_status("MP4 tab: no video loaded")

    # ---------------- UI ----------------
    def _build_ui(self):
        layout = QtWidgets.QVBoxLayout(self)

        self.video_label = QtWidgets.QLabel("MP4 tab: no video loaded")
        self.video_label.setAlignment(QtCore.Qt.AlignCenter)
                
        # ✅ allow splitter to shrink it
        self.video_label.setMinimumSize(0, 0)
        self.video_label.setSizePolicy(QtWidgets.QSizePolicy.Expanding,
                                    QtWidgets.QSizePolicy.Expanding)

        self.video_label.setStyleSheet("background-color: black; color: white;")
        layout.addWidget(self.video_label, stretch=1)

        self.slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.slider.setMinimum(0)
        self.slider.setMaximum(0)
        self.slider.valueChanged.connect(self._on_slider_changed)
        layout.addWidget(self.slider)

        controls = QtWidgets.QHBoxLayout()
        layout.addLayout(controls)

        self.btn_play = QtWidgets.QPushButton("Play")
        self.btn_play.clicked.connect(self.play)
        controls.addWidget(self.btn_play)

        self.btn_pause = QtWidgets.QPushButton("Pause")
        self.btn_pause.clicked.connect(self.pause)
        controls.addWidget(self.btn_pause)

        controls.addStretch()

        self.status = QtWidgets.QLabel("")
        self.status.setWordWrap(True)
        layout.addWidget(self.status)

    def _set_status(self, msg: str):
        self.status.setText(msg)

    def _popup_error(self, title: str, msg: str):
        QtWidgets.QMessageBox.critical(self.error_parent, title, msg)

    # ---------------- Public API ----------------
    def open_video(self, video_name: str) -> bool:
        """
        Open base_dir/mp4s/video_name and prepare slider.
        """
        path = (self.mp4_dir / video_name).resolve()
        if not path.exists():
            self._popup_error(
                "Missing MP4",
                f"Could not find:\n{path}\n\nExpected MP4s in a folder named 'mp4s' next to app.py."
            )
            return False

        self._close_cap()

        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            self._popup_error("MP4 Error", f"Could not open video:\n{path}")
            return False

        fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)

        self.cap = cap
        self.info = Mp4Info(path=path, fps=fps, frame_count=frame_count, width=w, height=h)

        # Match GUI tick to video fps (clamp to sane range)
        ms = int(round(1000.0 / max(1e-6, fps)))
        ms = max(10, min(ms, 200))  # 5–100 FPS range clamp
        self.timer.setInterval(ms)


        self.playing = False
        self.frame_idx = 0

        self.slider.blockSignals(True)
        self.slider.setMinimum(0)
        self.slider.setMaximum(max(0, frame_count - 1) if frame_count > 0 else 0)
        self.slider.setValue(0)
        self.slider.blockSignals(False)

        self._render_frame(self.frame_idx)
        self._set_status(f"Opened: {path.name}  fps={fps:.2f}  frames={frame_count}")
        return True

    def jump_to(self, video: str, t_event_s: float, lead_s: float = 3.0, autoplay: bool = True) -> bool:
        """
        Ensure video is open, seek to (t_event_s - lead_s), render, and optionally start playback.
        """
        if self.info is None or self.info.path.name != video:
            if not self.open_video(video):
                return False

        assert self.info is not None
        fps = self.info.fps or 30.0
        t_seek = max(0.0, float(t_event_s) - float(lead_s))
        idx = int(round(t_seek * fps))

        if self.info.frame_count > 0:
            idx = max(0, min(idx, self.info.frame_count - 1))

        self.seek_frame(idx)
        self._set_status(f"Jumped to t≈{t_seek:.2f}s (frame {idx}) for event t={t_event_s:.2f}s")

        if autoplay:
            self.play()
        return True

    def seek_frame(self, idx: int):
        if self.cap is None:
            return
        self.frame_idx = max(0, int(idx))

        # Clamp if known
        if self.info and self.info.frame_count > 0:
            self.frame_idx = min(self.frame_idx, self.info.frame_count - 1)

        self.slider.blockSignals(True)
        if self.info and self.info.frame_count > 0:
            self.slider.setValue(self.frame_idx)
        else:
            self.slider.setValue(0)
        self.slider.blockSignals(False)

        self._render_frame(self.frame_idx)

    def play(self):
        if self.cap is None:
            return
        self.playing = True

    def pause(self):
        self.playing = False

    def close(self):
        self._close_cap()
        self._set_status("MP4 tab: no video loaded")
        self.video_label.setText("MP4 tab: no video loaded")
        self.slider.setMaximum(0)
        self.slider.setValue(0)

    # ---------------- Internals ----------------
    def _close_cap(self):
        if self.cap is not None:
            try:
                self.cap.release()
            except Exception:
                pass
        self.cap = None
        self.info = None
        self.playing = False
        self.frame_idx = 0

    def _on_slider_changed(self, value: int):
        if self.cap is None:
            return
        self.playing = False  # scrubbing pauses by default
        self.frame_idx = max(0, int(value))
        self._render_frame(self.frame_idx)

    def _render_frame_image(self, frame: np.ndarray):
        if frame is None:
            return

        lw = max(1, self.video_label.width())
        lh = max(1, self.video_label.height())
        h, w = frame.shape[:2]
        s = min(lw / w, lh / h)
        new_w = max(1, int(w * s))
        new_h = max(1, int(h * s))
        if new_w != w or new_h != h:
            frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)

        qimg = bgr_to_qimage(frame)
        self.video_label.setPixmap(QtGui.QPixmap.fromImage(qimg))


    def _on_tick(self):
        if not self.playing or self.cap is None:
            return

        ok, frame = self.cap.read()
        if not ok or frame is None:
            self.playing = False
            return

        # Update our frame index from capture (more accurate than +1)
        pos = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES) or (self.frame_idx + 1))
        self.frame_idx = max(0, pos - 1)

        # Clamp/stop at end if known
        if self.info and self.info.frame_count > 0 and self.frame_idx >= self.info.frame_count - 1:
            self.frame_idx = self.info.frame_count - 1
            self.playing = False

        # Update slider
        if self.info and self.info.frame_count > 0:
            self.slider.blockSignals(True)
            self.slider.setValue(self.frame_idx)
            self.slider.blockSignals(False)

        # Render the frame we just read (no seeking)
        self._render_frame_image(frame)

    def _render_frame(self, idx: int):
        if self.cap is None:
            return
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, float(idx))
        ok, frame = self.cap.read()
        if not ok or frame is None:
            return
        self._render_frame_image(frame)
