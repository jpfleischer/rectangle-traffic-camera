# rectangle_tools.spec
# Build TracksViewer.exe and RoadPairer.exe into a single shared dist folder.

block_cipher = None

from PyInstaller.utils.hooks import collect_submodules, collect_data_files
from PyInstaller.building.build_main import Analysis, PYZ, EXE, COLLECT

# --- Common hidden imports / data ---

hiddenimports = []

# Things you previously hid via CLI
hiddenimports += [
    "clickhouse_client",
    "mp4_player",
    "tracksviewer_app",
    "roadpairer.ortho_overlay",
    "roadpairer.pair_points",
    "roadpairer.intersection_tab",
    "roadpairer.clickhouse_client",
]

# Collect full package submodules similar to --collect-submodules
hiddenimports += collect_submodules("tracksviewer_app")
hiddenimports += collect_submodules("roadpairer")
hiddenimports += collect_submodules("PySide6")
hiddenimports += collect_submodules("shiboken6")
hiddenimports += collect_submodules("rasterio")

# Like --collect-data rasterio
datas = collect_data_files("rasterio")

# Optional: trim heavy Qt pieces you don't use (like your CLI --exclude-module)
excludes = [
    "PySide6.QtWebEngineCore",
    "PySide6.QtWebEngineWidgets",
    "PySide6.QtWebEngine",
    "PySide6.QtQml",
    "PySide6.QtQuick",
    "PySide6.QtMultimedia",
    "PySide6.QtMultimediaWidgets",
]

# ---------------------------------------------------------------------------
# Single Analysis with BOTH entry scripts
#   0: tracksviewer/visualize_tracks_gui.py
#   1: roadpairer/app.py
# ---------------------------------------------------------------------------

a = Analysis(
    [
        "tracksviewer/visualize_tracks_gui.py",
        "roadpairer/app.py",
    ],
    pathex=[".", "tracksviewer", "roadpairer"],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=excludes,
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    noarchive=False,
)

pyz = PYZ(
    a.pure,
    a.zipped_data,
    cipher=block_cipher,
)

# a.scripts[0] is visualize_tracks_gui.py
exe_tv = EXE(
    pyz,
    a.scripts[0],
    name="TracksViewer",
    icon=None,
    console=False,   # windowed
    disable_windowed_traceback=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    bootloader_ignore_signals=False,
)

# a.scripts[1] is roadpairer/app.py
exe_rp = EXE(
    pyz,
    a.scripts[1],
    name="RoadPairer",
    icon=None,
    console=False,   # windowed
    disable_windowed_traceback=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    bootloader_ignore_signals=False,
)

# Single COLLECT: both EXEs share the same DLLs / libs in one folder
coll = COLLECT(
    exe_tv,
    exe_rp,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name="RectangleTrafficTools",
)
