import sys
from pathlib import Path

from PySide6 import QtWidgets

from tracksviewer_app.main_window import TracksPlayer


def run(tif_path: str, repo_root: Path) -> int:
    app = QtWidgets.QApplication(sys.argv)
    win = TracksPlayer(tif_path=tif_path, repo_root=repo_root)
    win.showMaximized()
    return app.exec()
