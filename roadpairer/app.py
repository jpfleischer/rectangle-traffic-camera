#!/usr/bin/env python3
import sys
from PySide6 import QtWidgets

from roadpairer.ortho_download_tab import OrthoDownloadTab
from roadpairer.ortho_crop_tab import OrthoCropTab
from roadpairer.pair_points import PairingTab
from roadpairer.intersection_tab import IntersectionGeometryTab
from pathlib import Path
from dotenv import load_dotenv


# Load environment variables from .env next to this file
load_dotenv(Path(__file__).with_name(".env"))

class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("RoadPairer")
        self.resize(1500, 900)

        tabs = QtWidgets.QTabWidget(self)

        # --- Workflow order tabs ---
        self.download_tab = OrthoDownloadTab()
        self.crop_tab = OrthoCropTab()
        self.pairing_tab = PairingTab()
        self.geometry_tab = IntersectionGeometryTab()

        tabs.addTab(self.download_tab, "1. Download Ortho")
        tabs.addTab(self.crop_tab, "2. Crop Ortho")
        tabs.addTab(self.pairing_tab, "3. Calibration (Pair Points)")
        tabs.addTab(self.geometry_tab, "4. Intersection Geometry")

        self.setCentralWidget(tabs)


def main():
    app = QtWidgets.QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
