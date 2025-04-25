# -*- mode: python ; coding: utf-8 -*-

block_cipher = None

# 定义资源文件路径
added_files = [
    ('msjh.ttc', '.'),      # 字体文件
    ('nfu_logo.png', '.'),  # 图片文件
    ('best.pt', '.')        # 模型文件
]

a = Analysis(
    ['SteelPipeDetectorApp.py'],
    pathex=[],
    binaries=[],
    datas=added_files,      # 直接引用资源文件列表
    hiddenimports=[
        'kivy',
        'cv2',
        'ultralytics',
        'win32timezone',
        'win32file',
        'pypiwin32'
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

# 强制将资源文件打包到EXE内部
pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    name='GoodJA',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,          # 隐藏控制台
    icon='nfu_icon.ico'     # 应用图标
)