# -*- mode: python ; coding: utf-8 -*-
"""
PyInstaller spec file for Hospital Muscle Segmentation App
"""

import os
import sys
from PyInstaller.utils.hooks import collect_data_files, collect_submodules

block_cipher = None

# アプリのルートディレクトリ
app_root = os.path.dirname(os.path.abspath(SPEC))

# nnU-Net モデルファイルのパス
model_base = os.path.join(app_root, '..', 'nnUNet_results', 'Dataset119_EyeMuscleSegmentation')
model_2d = os.path.join(model_base, 'nnUNetTrainer__nnUNetPlans__2d')

# 追加するデータファイル
datas = [
    # nnU-Net モデル
    (model_2d, 'nnUNet_results/Dataset119_EyeMuscleSegmentation/nnUNetTrainer__nnUNetPlans__2d'),
    # 設定ファイル
    (os.path.join(app_root, 'config'), 'config'),
]

# Hidden imports (nnU-Net と PyTorch 関連)
hiddenimports = [
    'nnunetv2',
    'nnunetv2.inference',
    'nnunetv2.inference.predict_from_raw_data',
    'torch',
    'torch.nn',
    'torch.cuda',
    'SimpleITK',
    'pydicom',
    'cv2',
    'numpy',
    'scipy',
    'skimage',
    'batchgenerators',
    'acvl_utils',
]

# nnU-Net のサブモジュールを収集
hiddenimports += collect_submodules('nnunetv2')
hiddenimports += collect_submodules('batchgenerators')

a = Analysis(
    ['gui/MuscleGUI.py'],
    pathex=[app_root],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        'matplotlib',  # 不要なら除外でサイズ削減
        'tkinter',
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
    name='MuscleSegmentation',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,  # GUIアプリなのでコンソール非表示
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=None,  # アイコンファイルがあれば指定
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='MuscleSegmentation',
)
