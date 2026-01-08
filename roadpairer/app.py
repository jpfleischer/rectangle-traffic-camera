#!/usr/bin/env python3
import sys
from PySide6 import QtWidgets

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

        self.pairing_tab = PairingTab()
        self.geometry_tab = IntersectionGeometryTab()

        tabs.addTab(self.pairing_tab, "Calibration (Pair Points)")
        tabs.addTab(self.geometry_tab, "Intersection Geometry")

        self.setCentralWidget(tabs)


def main():
    app = QtWidgets.QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
