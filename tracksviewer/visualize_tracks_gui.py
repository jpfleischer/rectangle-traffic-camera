#!/usr/bin/env python3
import argparse

from tracksviewer_app.paths import REPO_ROOT, init_env_and_sys_path
from tracksviewer_app.entrypoint import run


def main():
    init_env_and_sys_path()

    parser = argparse.ArgumentParser(description="ClickHouse-backed tracks viewer with datetime picker")
    parser.add_argument(
        "--tif", required=False, default="",
        help="Path to ortho GeoTIFF used for pairing (optional; will restore last used)"
    )
    args = parser.parse_args()

    raise SystemExit(run(tif_path=args.tif, repo_root=REPO_ROOT))


if __name__ == "__main__":
    main()
