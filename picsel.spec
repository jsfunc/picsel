# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller build spec for picSel.

Build a standalone, single-file executable with:
    pyinstaller picsel.spec

Must be run separately on each target OS (Linux, Windows, macOS) --
PyInstaller does not cross-compile. See .github/workflows/release.yml for
the automated multi-platform build.
"""

from PyInstaller.utils.hooks import collect_all

datas = [("docs/face_recognition.html", "docs")]
binaries = []
hiddenimports = []

# pillow-heif bundles a native libheif; collect_all pulls in its shared
# libraries and data files that PyInstaller's static analysis can't see.
for pkg in ("pillow_heif",):
    pkg_datas, pkg_binaries, pkg_hiddenimports = collect_all(pkg)
    datas += pkg_datas
    binaries += pkg_binaries
    hiddenimports += pkg_hiddenimports

a = Analysis(
    ["main.py"],
    pathex=[],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    # picSel only uses QtCore/QtGui/QtWidgets; excluding the rest keeps the
    # build smaller and skips their PyInstaller hooks (e.g. QtNetwork's hook
    # probes OpenSSL support at build time, which is unnecessary here and can
    # be flaky on systems with multiple conflicting OpenSSL/libbrotli builds).
    excludes=[
        "PySide6.QtNetwork",
        "PySide6.QtQml",
        "PySide6.QtQuick",
        "PySide6.QtWebEngineCore",
        "PySide6.QtWebEngineWidgets",
        "PySide6.QtPdf",
        "PySide6.QtMultimedia",
    ],
    noarchive=False,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name="picSel",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
