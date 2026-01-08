# rectangle_tools.spec — builds TracksViewer.exe and RoadPairer.exe
# into a single dist/RectangleTrafficTools/ folder.

# -*- mode: python ; coding: utf-8 -*-

block_cipher = None

# --- TracksViewer analysis ---
a_tv = Analysis(
    ['tracksviewer/visualize_tracks_gui.py'],
    pathex=['.', 'tracksviewer', 'roadpairer'],
    binaries=[],
    datas=[],  # let hooks handle data
    hiddenimports=[
        'mp4_player',
        'clickhouse_client',
        'tracksviewer_app',
        # rasterio runtime deps pulled in from C-extensions:
        'rasterio.serde',
        'rasterio.sample',
        'rasterio.tools',
        'rasterio.vrt',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        'PySide6.QtWebEngineCore',
        'PySide6.QtWebEngineWidgets',
        'PySide6.QtWebEngine',
        'PySide6.QtQml',
        'PySide6.QtQuick',
        'PySide6.QtMultimedia',
        'PySide6.QtMultimediaWidgets',
    ],
    noarchive=False,
)

pyz_tv = PYZ(
    a_tv.pure,
    a_tv.zipped_data,
    cipher=block_cipher,
)

exe_tv = EXE(
    pyz_tv,
    a_tv.scripts,
    a_tv.binaries,
    a_tv.zipfiles,
    a_tv.datas,
    [],
    name='TracksViewer',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,  # GUI app
)


# --- RoadPairer analysis ---
a_rp = Analysis(
    ['roadpairer/app.py'],
    pathex=['.', 'roadpairer'],
    binaries=[],
    datas=[],  # let hooks handle data
    hiddenimports=[
        'roadpairer.ortho_overlay',
        'roadpairer.pair_points',
        'roadpairer.intersection_tab',
        'roadpairer.clickhouse_client',
        # rasterio runtime deps:
        'rasterio.serde',
        'rasterio.sample',
        'rasterio.tools',
        'rasterio.vrt',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        'PySide6.QtWebEngineCore',
        'PySide6.QtWebEngineWidgets',
        'PySide6.QtWebEngine',
        'PySide6.QtQml',
        'PySide6.QtQuick',
        'PySide6.QtMultimedia',
        'PySide6.QtMultimediaWidgets',
    ],
    noarchive=False,
)

pyz_rp = PYZ(
    a_rp.pure,
    a_rp.zipped_data,
    cipher=block_cipher,
)

exe_rp = EXE(
    pyz_rp,
    a_rp.scripts,
    a_rp.binaries,
    a_rp.zipfiles,
    a_rp.datas,
    [],
    name='RoadPairer',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,  # GUI app
)


# --- Final COLLECT (single folder with shared DLLs/libs) ---
coll = COLLECT(
    exe_tv,
    exe_rp,
    a_tv.binaries,
    a_rp.binaries,
    a_tv.zipfiles,
    a_rp.zipfiles,
    a_tv.datas,
    a_rp.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='RectangleTrafficTools',
)
