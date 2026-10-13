# -*- mode: python ; coding: utf-8 -*-
import os
import sys

block_cipher = None

a = Analysis(
    ['run_node.py'],
    pathex=['.'],
    binaries=[],
    datas=[
        ('templates', 'templates'),
        ('static', 'static'),
        ('models', 'models'),
        ('config_node.json', '.'),
        ('attendance.db', '.')
    ],
    hiddenimports=[
        'cv2',
        'numpy',
        'insightface',
        'onnxruntime',
        'flask',
        'werkzeug',
        'sqlite3',
        'PIL'
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        'tensorflow',
        'keras',
        'tensorboard',
        'bitsandbytes',
        'diffusers',
        'sentencepiece',
        'transformers',
        'torchaudio',
        'torchvision',
        'nltk',
        'spacy',
        'yt_dlp',
        'IPython',
        'pytest'
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='QuantEye_Live_Attendance_Node',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='QuantEye_Live_Attendance_Node',
)
