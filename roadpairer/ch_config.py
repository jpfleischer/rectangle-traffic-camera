#!/usr/bin/env python3
"""
ch_config.py — shared ClickHouse config helpers for RoadPairer.

Single source of truth for:
  - QSettings org/app names
  - settings keys
  - keyring service/account
  - load/save helpers that both tabs can use
"""

import os
from typing import Dict

from PySide6 import QtCore
import keyring
from keyring.errors import KeyringError

# --- ClickHouse config storage (local to RoadPairer) ---
APP_ORG = "RectangleTraffic"
APP_NAME = "RoadPairer"

SETTINGS_CH_HOST = "ch_host"
SETTINGS_CH_PORT = "ch_port"
SETTINGS_CH_USER = "ch_user"
SETTINGS_CH_DB   = "ch_db"

KEYRING_SERVICE = "rectangle_traffic_clickhouse"
KEYRING_ACCOUNT = "roadpairer_password"


def load_clickhouse_config() -> Dict[str, object]:
    """
    Load ClickHouse config from QSettings + keyring, with env-var fallback.

    Returns dict:
        {
            "host": str,
            "port": int,
            "user": str,
            "password": str,
            "db": str,
        }
    """
    s = QtCore.QSettings(APP_ORG, APP_NAME)

    host = s.value(SETTINGS_CH_HOST, os.getenv("CH_HOST", ""), type=str)
    port = int(s.value(SETTINGS_CH_PORT, int(os.getenv("CH_PORT", "8123")), type=int))
    user = s.value(SETTINGS_CH_USER, os.getenv("CH_USER", "default"), type=str)
    db   = s.value(SETTINGS_CH_DB, os.getenv("CH_DB", "trajectories"), type=str)

    pw = os.getenv("CH_PASSWORD", "")
    try:
        kr = keyring.get_password(KEYRING_SERVICE, KEYRING_ACCOUNT)
        if kr is not None:
            pw = kr
    except KeyringError as e:
        # Let caller decide what to do with the error
        raise RuntimeError(f"Keyring error reading ClickHouse password: {e}") from e

    return {"host": host, "port": port, "user": user, "db": db, "password": pw}


def save_clickhouse_config(cfg: Dict[str, object]) -> None:
    """
    Save ClickHouse config to QSettings + keyring.

    Expects keys:
        host, port, user, password, db
    """
    s = QtCore.QSettings(APP_ORG, APP_NAME)
    s.setValue(SETTINGS_CH_HOST, cfg.get("host", ""))
    s.setValue(SETTINGS_CH_PORT, int(cfg.get("port", 8123)))
    s.setValue(SETTINGS_CH_USER, cfg.get("user", "default"))
    s.setValue(SETTINGS_CH_DB, cfg.get("db", "trajectories"))

    pw = cfg.get("password", "")
    try:
        keyring.set_password(KEYRING_SERVICE, KEYRING_ACCOUNT, pw)
    except KeyringError as e:
        raise RuntimeError(f"Keyring error saving ClickHouse password: {e}") from e

    s.sync()
