# -*- mode: python ; coding: utf-8 -*-
from PyInstaller.utils.hooks import collect_data_files
from PyInstaller.utils.hooks import collect_submodules

datas = []
hiddenimports = []
datas += collect_data_files('pipelines')
datas += collect_data_files('sv_ttk')
datas += collect_data_files('tkinterdnd2')
datas += [('EyeFlow_logo.png', '.')]
datas += [('EyeFlow.ico', '.')]
datas += [('default_settings.json', '.')]
datas += [('pyproject.toml', '.')]
hiddenimports += collect_submodules('pipelines')
hiddenimports += collect_submodules('tkinterdnd2')


a = Analysis(
    ['src\\eye_flow.py'],
    pathex=['src'],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=['hooks'],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name='EyeFlow',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon='EyeFlow.ico',
)
