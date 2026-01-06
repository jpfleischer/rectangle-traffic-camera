from PySide6 import QtWidgets


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
        self.db_edit = QtWidgets.QLineEdit(db)

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
            "db": self.db_edit.text().strip(),
            "password": self.pass_edit.text(),
        }
