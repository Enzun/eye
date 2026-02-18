# dicom_handler.py
"""
DICOM処理モジュール
DICOMフォルダからNIfTI形式への変換を行う
"""

import os
import re
import numpy as np
import pydicom
import SimpleITK as sitk
from pathlib import Path


def load_and_preprocess_dicom(dicom_path):
    """DICOMを読み込み、Windowing + 8bit化を適用"""
    ds = pydicom.dcmread(dicom_path)
    image_array = ds.pixel_array
    
    # Windowing処理
    if 'WindowCenter' in ds and 'WindowWidth' in ds:
        wc = ds.WindowCenter if isinstance(ds.WindowCenter, (float, int)) else ds.WindowCenter[0]
        ww = ds.WindowWidth if isinstance(ds.WindowWidth, (float, int)) else ds.WindowWidth[0]
    else:
        wc = np.mean(image_array)
        ww = np.max(image_array) - np.min(image_array)
    
    min_val = wc - ww // 2
    max_val = wc + ww // 2
    
    # クリッピングと正規化 (0-255)
    image_8bit = np.clip(image_array, min_val, max_val)
    image_8bit = ((image_8bit - min_val) / (max_val - min_val) * 255).astype(np.uint8)
    
    return image_8bit


def get_folder_name(folder_path):
    """DICOMフォルダ名を取得（常に親フォルダ名も含む形式: 親フォルダ名SE○）
    
    例: 
        /path/to/うんたらかんたらEX100/SE3 → うんたらかんたらEX100SE3
        /path/to/Patient001/SE3 → Patient001SE3
    """
    path = Path(folder_path)
    se_name = path.name  # SE3 など
    parent_name = path.parent.name  # うんたらかんたらEX100 など
    
    # 常に親フォルダ名とSEフォルダ名を結合（アンダースコアなしで結合）
    return f"{parent_name}{se_name}"


def convert_dicom_folder_to_nifti(folder_path):
    """
    DICOMフォルダをNIfTI形式に変換
    
    Args:
        folder_path: DICOMファイルが入ったフォルダパス (SE○フォルダ)
    
    Returns:
        tuple: (SimpleITKイメージ, フォルダ識別名)
    
    Raises:
        FileNotFoundError: DICOMファイルが見つからない場合
        Exception: 変換エラー
    """
    # IMG番号順にソート
    dicom_files = [f for f in os.listdir(folder_path) if f.startswith('IMG')]
    
    def extract_number(filename):
        match = re.search(r'IMG(\d+)', filename)
        return int(match.group(1)) if match else 0
    
    dicom_files.sort(key=extract_number)
    
    if not dicom_files:
        raise FileNotFoundError(f"DICOMファイルが見つかりません: {folder_path}")
    
    # 各スライスを読み込み
    image_slices = []
    for filename in dicom_files:
        file_path = os.path.join(folder_path, filename)
        img_array = load_and_preprocess_dicom(file_path)
        image_slices.append(img_array)
    
    # 3D化
    image_3d = np.stack(image_slices, axis=0)
    
    # SimpleITK変換
    sitk_image = sitk.GetImageFromArray(image_3d)
    sitk_image.SetSpacing((1.0, 1.0, 1.0))
    
    folder_name = get_folder_name(folder_path)
    
    return sitk_image, folder_name


def save_temp_nifti(sitk_image, temp_dir, case_name="case"):
    """
    一時NIfTIファイルを保存
    
    Args:
        sitk_image: SimpleITKイメージ
        temp_dir: 一時ディレクトリパス
        case_name: ケース名
    
    Returns:
        str: 保存したファイルパス
    """
    os.makedirs(temp_dir, exist_ok=True)
    output_path = os.path.join(temp_dir, f"{case_name}_0000.nii.gz")
    sitk.WriteImage(sitk_image, output_path)
    return output_path
