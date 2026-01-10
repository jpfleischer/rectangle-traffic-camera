#!/usr/bin/env python3
# ortho_download_tab.py — Download ortho GeoTIFF from ImageServer

import os
import json
import requests
import urllib3
import rasterio as rio
import numpy as np

from PySide6 import QtCore, QtWidgets, QtGui

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


# --- timeouts (seconds) ---
REQUEST_TIMEOUT = 30   # main exportImage + TIFF download
JSON_TIMEOUT    = 20   # fallback f=json request

class OrthoDownloadWorker(QtCore.QObject):
    # Signals to talk to the GUI thread
    log = QtCore.Signal(str)
    progress_range = QtCore.Signal(int, int)   # (min, max)
    progress_value = QtCore.Signal(int)        # current value
    finished = QtCore.Signal(str)              # path to TIFF
    error = QtCore.Signal(str)                 # error message

    def __init__(self, imageserver: str, bbox_str: str, out_sr: int,
                 size_w: int, size_h: int, out_path: str, verify_first: bool = True,
                 parent=None):
        super().__init__(parent)
        self.imageserver = imageserver
        self.bbox_str = bbox_str
        self.out_sr = out_sr
        self.size_w = size_w
        self.size_h = size_h
        self.out_path = out_path
        self.verify_first = verify_first

    @QtCore.Slot()
    def run(self):
        """Entry point for the worker thread."""
        try:
            try:
                path = self._export_once(
                    verify=self.verify_first,
                    imageserver=self.imageserver,
                    bbox_str=self.bbox_str,
                    out_sr=self.out_sr,
                    size_w=self.size_w,
                    size_h=self.size_h,
                    out_path=self.out_path,
                )
            except requests.exceptions.SSLError:
                self.log.emit("TLS verify failed — retrying INSECURELY (verify=False)...")
                path = self._export_once(
                    verify=False,
                    imageserver=self.imageserver,
                    bbox_str=self.bbox_str,
                    out_sr=self.out_sr,
                    size_w=self.size_w,
                    size_h=self.size_h,
                    out_path=self.out_path,
                )

            self.finished.emit(path)
        except Exception as e:
            self.error.emit(str(e))

    # ---------------- core logic (same as before, but using signals) ------------- #
    def _export_once(self, verify: bool, imageserver: str, bbox_str: str,
                     out_sr: int, size_w: int, size_h: int, out_path: str) -> str:
        # parse bbox string
        try:
            parts = [float(x.strip()) for x in bbox_str.split(",")]
            if len(parts) != 4:
                raise ValueError
        except Exception:
            raise RuntimeError("BBox must be 4 comma-separated numbers: minLon,minLat,maxLon,maxLat")

        bbox = parts

        params = {
            "bbox": f"{bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]}",
            "bboxSR": 4326,
            "outSR": out_sr,
            "size": f"{size_w},{size_h}",
            "format": "tiff",
            "pixelType": "U8",
            "noData": 0,
            "f": "image",
        }
        url = f"{imageserver.rstrip('/')}/exportImage"

        self.log.emit(f"Requesting: {url}")
        self.log.emit(f"Params: {params}")
        self.log.emit(f"verify TLS: {verify}")

        r = requests.get(url, params=params, stream=True, timeout=REQUEST_TIMEOUT, verify=verify)
        ctype = r.headers.get("Content-Type", "")

        # Case A: server streams TIFF directly
        if "image/tiff" in ctype or "application/octet-stream" in ctype:
            self.log.emit("Server returned TIFF directly.")
            return self._fetch_tiff(r.url, verify, out_path)

        # Case B: returned JSON (even though f=image)
        text = r.text
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            rj = requests.get(
                url, params={**params, "f": "json"}, timeout=JSON_TIMEOUT, verify=verify
            )
            try:
                data = rj.json()
            except Exception:
                self.log.emit("Unexpected response, first 1200 chars:")
                self.log.emit(text[:1200])
                raise RuntimeError("Server gave unexpected non-JSON response")

        if "error" in data:
            err = data["error"] or {}
            code = err.get("code", "unknown")
            msg = err.get("message", "Unknown server error")
            details = err.get("details") or []
            detail_str = "\n".join(str(d) for d in details) if details else ""
            full_msg = f"Server-side export error {code}: {msg}"
            if detail_str:
                full_msg += f"\nDetails:\n{detail_str}"
            raise RuntimeError(full_msg)

        href = data.get("href")
        if not href:
            self.log.emit("No 'href' in server JSON, response snippet:")
            self.log.emit(str(data)[:1200])
            raise RuntimeError("JSON response did not contain 'href' to TIFF")

        self.log.emit(f"Server returned JSON with href to TIFF:\n{href}")
        return self._fetch_tiff(href, verify, out_path)

    def _fetch_tiff(self, url: str, verify: bool, out_path: str) -> str:
        """Download URL to out_path with a magic-byte TIFF check, with progress signals."""
        r = requests.get(url, stream=True, timeout=REQUEST_TIMEOUT, verify=verify)

        total = r.headers.get("Content-Length")
        try:
            total = int(total) if total is not None else 0
        except ValueError:
            total = 0

        if total > 0:
            self.progress_range.emit(0, total)
            self.progress_value.emit(0)
        else:
            self.progress_range.emit(0, 0)  # indeterminate

        tmp = out_path + ".tmp"
        downloaded = 0
        chunk_size = 1 << 20  # 1 MiB

        with open(tmp, "wb") as f:
            for ch in r.iter_content(chunk_size):
                if not ch:
                    continue
                f.write(ch)
                downloaded += len(ch)
                if total > 0:
                    self.progress_value.emit(downloaded)

        if total > 0:
            self.progress_value.emit(total)
        else:
            self.progress_range.emit(0, 1)
            self.progress_value.emit(1)

        # TIFF magic-byte check
        with open(tmp, "rb") as f:
            magic = f.read(4)
        if magic not in (b"II*\x00", b"MM\x00*"):
            os.remove(tmp)
            raise RuntimeError(f"Downloaded file is not a TIFF (magic={magic!r})")

        os.replace(tmp, out_path)
        return out_path


class OrthoDownloadTab(QtWidgets.QWidget):
    """
    Tab for downloading an ortho GeoTIFF from an ArcGIS ImageServer.
    Mostly a GUI wrapper around your download_ortho.py script.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Download Ortho")

        # --- defaults (from your script) ---
        self.default_imageserver = (
            "https://mcgis4.monroecounty-fl.gov/public/rest/services/Images/Orthos2025/ImageServer"
        )
        self.default_bbox = "-81.753611,24.568889,-81.749722,24.570556"
        # 24.55659595916668, -81.7920786259649
        # -81.79207, 24.55659, -81.79062, 24.55582
        self.default_out_sr = "32617"
        self.default_size_w = 3500
        self.default_size_h = 3500
        self.default_out = "ortho_monroe_2025.tif"

        self.worker_thread = None
        self.worker = None


        self._build_ui()

    # ---------------- UI ---------------- #
    def _build_ui(self):
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        layout.addWidget(QtWidgets.QLabel("<b>Step 1: Download source ortho GeoTIFF</b>"))

        form = QtWidgets.QFormLayout()
        self.edit_imageserver = QtWidgets.QLineEdit(self.default_imageserver)
        self.edit_bbox = QtWidgets.QLineEdit(self.default_bbox)
        self.edit_out_sr = QtWidgets.QLineEdit(self.default_out_sr)

        size_box = QtWidgets.QHBoxLayout()
        self.edit_size_w = QtWidgets.QSpinBox()
        self.edit_size_w.setRange(1, 20000)
        self.edit_size_w.setValue(self.default_size_w)
        self.edit_size_h = QtWidgets.QSpinBox()
        self.edit_size_h.setRange(1, 20000)
        self.edit_size_h.setValue(self.default_size_h)
        size_box.addWidget(QtWidgets.QLabel("Width"))
        size_box.addWidget(self.edit_size_w)
        size_box.addWidget(QtWidgets.QLabel("Height"))
        size_box.addWidget(self.edit_size_h)

        self.edit_out = QtWidgets.QLineEdit(self.default_out)
        self.btn_browse_out = QtWidgets.QPushButton("Browse…")
        out_box = QtWidgets.QHBoxLayout()
        out_box.addWidget(self.edit_out)
        out_box.addWidget(self.btn_browse_out)

        form.addRow("ImageServer URL:", self.edit_imageserver)
        form.addRow("BBox (minLon,minLat,maxLon,maxLat):", self.edit_bbox)
        form.addRow("Output SR (EPSG):", self.edit_out_sr)
        form.addRow("Pixel size:", size_box)
        form.addRow("Output TIFF:", out_box)

        layout.addLayout(form)

        self.btn_download = QtWidgets.QPushButton("Download Ortho")
        layout.addWidget(self.btn_download)

        # --- NEW: download progress bar ---
        self.progress = QtWidgets.QProgressBar()
        self.progress.setTextVisible(True)
        self.progress.setValue(0)
        self.progress.setMinimum(0)
        self.progress.setMaximum(100)   # placeholder, will be set properly in _fetch_tiff
        layout.addWidget(self.progress)

        self.status = QtWidgets.QTextEdit()
        self.status.setReadOnly(True)
        self.status.setMinimumHeight(120)
        layout.addWidget(self.status)

        # --- NEW: preview area ---
        self.preview_label = QtWidgets.QLabel("No image loaded")
        self.preview_label.setAlignment(QtCore.Qt.AlignCenter)
        self.preview_label.setMinimumHeight(250)
        self.preview_label.setStyleSheet(
            "background-color: black; color: white; border: 1px solid #444;"
        )
        layout.addWidget(self.preview_label, stretch=1)

        layout.addStretch(1)

        # signals
        self.btn_download.clicked.connect(self._on_download_clicked)
        self.btn_browse_out.clicked.connect(self._on_browse_out)

    # ---------------- helpers ---------------- #
    def _log(self, msg: str):
        self.status.append(msg)
        self.status.ensureCursorVisible()

    def _on_browse_out(self):
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Choose output GeoTIFF",
            self.edit_out.text() or self.default_out,
            "GeoTIFF (*.tif *.tiff);;All files (*)",
        )
        if path:
            self.edit_out.setText(path)

    def _show_preview(self, tiff_path: str) -> None:
        """
        Load the GeoTIFF and show a downscaled RGB preview in preview_label.
        Assumes U8 pixelType (0–255) from the exportImage call.
        """
        try:
            with rio.open(tiff_path) as src:
                arr = src.read()  # shape: (bands, height, width)

            # Handle 1-band or multi-band
            if arr.ndim == 3:
                bands, h, w = arr.shape
                if bands >= 3:
                    # Use first three bands as RGB
                    rgb = np.stack([arr[0], arr[1], arr[2]], axis=-1)
                else:
                    # Single band -> replicate to RGB
                    single = arr[0]
                    rgb = np.stack([single, single, single], axis=-1)
            elif arr.ndim == 2:
                # 2D (H,W) -> replicate to RGB
                h, w = arr.shape
                rgb = np.stack([arr, arr, arr], axis=-1)
            else:
                raise RuntimeError("Unexpected array shape for TIFF preview")

            # Ensure uint8
            if rgb.dtype != np.uint8:
                rgb = np.clip(rgb, 0, 255).astype(np.uint8)

            # Build QImage (expects row-major HxW)
            h, w, _ = rgb.shape
            bytes_per_line = 3 * w
            qimg = QtGui.QImage(
                rgb.data,
                w,
                h,
                bytes_per_line,
                QtGui.QImage.Format_RGB888,
            )

            # Scale to fit the label while keeping aspect ratio
            if self.preview_label.width() > 0 and self.preview_label.height() > 0:
                pix = QtGui.QPixmap.fromImage(qimg)
                scaled = pix.scaled(
                    self.preview_label.size(),
                    QtCore.Qt.KeepAspectRatio,
                    QtCore.Qt.SmoothTransformation,
                )
                self.preview_label.setPixmap(scaled)
            else:
                # Fallback: show at original size
                self.preview_label.setPixmap(QtGui.QPixmap.fromImage(qimg))

        except Exception as e:
            self._log(f"Preview failed: {e}")
            self.preview_label.setText("Failed to load preview")


    def _on_download_finished(self, path: str):
        mb = os.path.getsize(path) / 1e6
        self._log(f"\nSaved {path}  ({mb:.2f} MB)")
        self._log("Done.")

        #  Tip: if OUT_SR=32617, the raster is in meters (nice for distances).

        # mark progress complete
        self.progress.setRange(0, 1)
        self.progress.setValue(1)

        # show preview
        self._show_preview(path)

        # re-enable button
        self.btn_download.setEnabled(True)

    def _on_download_error(self, msg: str):
        self._log(f"\nError: {msg}")
        QtWidgets.QMessageBox.critical(self, "Download failed", msg)
        self.progress.setRange(0, 1)
        self.progress.setValue(0)
        self.btn_download.setEnabled(True)

    def _on_thread_finished(self):
        # Optional: clear references
        self.worker_thread = None
        self.worker = None

    # ---------------- slot ---------------- #
    def _on_download_clicked(self):
        imageserver = self.edit_imageserver.text().strip()
        bbox_str = self.edit_bbox.text().strip()
        out_sr_str = self.edit_out_sr.text().strip()
        out_path = self.edit_out.text().strip() or self.default_out
        size_w = int(self.edit_size_w.value())
        size_h = int(self.edit_size_h.value())

        if not imageserver:
            QtWidgets.QMessageBox.warning(self, "Missing URL", "ImageServer URL is required.")
            return

        try:
            out_sr = int(out_sr_str)
        except ValueError:
            QtWidgets.QMessageBox.warning(
                self, "Bad SR", "Output SR must be an integer EPSG code (e.g., 32617)."
            )
            return

        self.status.clear()
        self._log("Starting download...")

        # Disable button while running
        self.btn_download.setEnabled(False)

        # Indeterminate until worker sets range
        self.progress.setRange(0, 0)
        self.progress.setValue(0)

        # --- set up worker + thread ---
        self.worker_thread = QtCore.QThread(self)
        self.worker = OrthoDownloadWorker(
            imageserver=imageserver,
            bbox_str=bbox_str,
            out_sr=out_sr,
            size_w=size_w,
            size_h=size_h,
            out_path=out_path,
        )
        self.worker.moveToThread(self.worker_thread)

        # Connect worker signals to GUI slots
        self.worker.log.connect(self._log)
        self.worker.progress_range.connect(self.progress.setRange)
        self.worker.progress_value.connect(self.progress.setValue)
        self.worker.finished.connect(self._on_download_finished)
        self.worker.error.connect(self._on_download_error)

        # Thread cleanup
        self.worker_thread.started.connect(self.worker.run)
        self.worker.finished.connect(self.worker_thread.quit)
        self.worker.error.connect(self.worker_thread.quit)
        self.worker_thread.finished.connect(self.worker.deleteLater)
        self.worker_thread.finished.connect(self._on_thread_finished)

        self.worker_thread.start()
