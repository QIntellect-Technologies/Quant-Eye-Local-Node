# -*- mode: python ; coding: utf-8 -*-
# QuantEye Client AI Node - GPU Edition Spec File
# GPU-Accelerated build with NVIDIA CUDA/cuDNN hardware inference support.
# Designed for client computers with dedicated NVIDIA GPUs (RTX / GTX).

import os
import sys

block_cipher = None

extra_binaries = []
conda_prefix = sys.prefix
lib_bin = os.path.join(conda_prefix, 'Library', 'bin')
torch_lib = os.path.join(conda_prefix, 'Lib', 'site-packages', 'torch', 'lib')

if os.path.exists(lib_bin):
    for f in os.listdir(lib_bin):
        fl = f.lower()
        if fl.endswith('.dll') and (fl.startswith('cudnn64') or 'infer' in fl):
            extra_binaries.append((os.path.join(lib_bin, f), '.'))

if os.path.exists(torch_lib):
    for f in os.listdir(torch_lib):
        fl = f.lower()
        if fl.endswith('.dll') and fl.startswith(('cublas', 'cufft', 'cudart')):
            extra_binaries.append((os.path.join(torch_lib, f), '.'))

a = Analysis(
    ['run_node.py'],
    pathex=['.'],
    binaries=extra_binaries,
    datas=[
        ('templates', 'templates'),
        ('static', 'static'),
        ('models', 'models')
    ],
    hiddenimports=[
        'cv2',
        'numpy',
        'insightface',
        'onnxruntime',
        'flask',
        'werkzeug',
        'sqlite3',
        'certifi',
        'PIL'
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        'torch',
        'torchvision',
        'torchaudio',
        'ultralytics',
        'catboost',
        'polars',
        '_polars_runtime_32',
        'pyarrow',
        'llvmlite',
        'numba',
        'faiss',
        'faiss_cpu',
        'boto3',
        'botocore',
        'pandas',
        'plotly',
        'tensorflow',
        'keras',
        'tensorboard',
        'bitsandbytes',
        'diffusers',
        'sentencepiece',
        'transformers',
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

# Strip only duplicate model zip, retain all CUDA & GPU binaries for NVIDIA acceleration
a.datas = [
    d for d in a.datas
    if not d[0].replace('\\', '/').lower().endswith('buffalo_l.zip')
    and os.path.basename(d[0]).lower() not in {
        'config_node.json', 'config_node_dev_backup.json', 'config_clean.json'
    }
]

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='QuantEye_Client_Node_GPU',
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
    name='QuantEye_Client_Node_GPU',
)
