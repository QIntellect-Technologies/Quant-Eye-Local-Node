# -*- mode: python ; coding: utf-8 -*-
# QuantEye Client AI Node - CPU Edition Spec File
# Ultra-compact standalone build (~825 MB uncompressed, ~470 MB ZIP)
# Designed for 100% universal compatibility on any Windows PC without NVIDIA GPU.

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

# Strip all CUDA/cuDNN DLLs and duplicate model zip to guarantee ultra-compact CPU build
_CUDA_DLL_PREFIXES = ('cublas', 'cufft', 'cudart', 'cudnn', 'curand', 'cusparse',
                      'cusolver', 'nvrtc', 'nvinfer', 'nvonnxparser')
_CUDA_EXACT = ('onnxruntime_providers_cuda.dll', 'onnxruntime_providers_tensorrt.dll')

def _is_cuda_bin(dest):
    name = os.path.basename(dest).lower()
    return name in _CUDA_EXACT or name.startswith(_CUDA_DLL_PREFIXES)

a.binaries = [b for b in a.binaries if not _is_cuda_bin(b[0])]
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
    name='QuantEye_Client_Node_CPU',
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
    name='QuantEye_Client_Node_CPU',
)
