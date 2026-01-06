from typing import Optional

from PySide6 import QtCore

from clickhouse_client import ClickHouseHTTP


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
