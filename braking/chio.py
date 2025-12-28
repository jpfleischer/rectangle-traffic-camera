from __future__ import annotations
import json
from pathlib import Path
from typing import List

# Ensure we can import clickhouse_client from ../gui relative to this package (repo layout)
HERE = Path(__file__).resolve().parent
GUI_DIR = (HERE.parent / "gui").resolve()
if str(GUI_DIR) not in __import__("sys").path:
    import sys as _sys
    _sys.path.append(str(GUI_DIR))

from clickhouse_client import ClickHouseHTTP  # type: ignore


def ch_query_json_each_row(ch: ClickHouseHTTP, sql: str) -> List[dict]:
    resp = ch._post_sql(sql, use_db=True)
    text = resp.text.strip()
    if not text:
        return []
    out: List[dict] = []
    for line in text.splitlines():
        line = line.strip()
        if line:
            out.append(json.loads(line))
    return out
