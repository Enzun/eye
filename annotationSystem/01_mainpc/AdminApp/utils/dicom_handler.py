"""
dicom_handler.py
DICOM → NIfTI 変換（HospitalMuscleBatch から流用）
"""

import os
import re
from pathlib import Path

import numpy as np
import pydicom
import SimpleITK as sitk


# ── シングルフレーム DICOM ─────────────────────────────────


def load_and_preprocess_dicom(dicom_path):
    """DICOM を読み込み Windowing + 8bit 化を適用する"""
    ds = pydicom.dcmread(dicom_path)
    image_array = ds.pixel_array

    if "WindowCenter" in ds and "WindowWidth" in ds:
        wc = ds.WindowCenter if isinstance(ds.WindowCenter, (float, int)) else ds.WindowCenter[0]
        ww = ds.WindowWidth if isinstance(ds.WindowWidth, (float, int)) else ds.WindowWidth[0]
    else:
        wc = np.mean(image_array)
        ww = np.max(image_array) - np.min(image_array)

    min_val = wc - ww // 2
    max_val = wc + ww // 2

    image_8bit = np.clip(image_array, min_val, max_val)
    image_8bit = ((image_8bit - min_val) / (max_val - min_val) * 255).astype(np.uint8)
    return image_8bit


def get_folder_name(folder_path):
    """DICOMフォルダ名を取得（親フォルダ名 + SE番号）"""
    path = Path(folder_path)
    return f"{path.parent.name}{path.name}"


def convert_dicom_folder_to_nifti(folder_path):
    """IMG番号順の DICOM フォルダを NIfTI に変換する。

    Returns:
        tuple: (SimpleITK.Image, folder_name)
    """
    dicom_files = [f for f in os.listdir(folder_path) if f.startswith("IMG")]

    def _extract_number(fn):
        m = re.search(r"IMG(\d+)", fn)
        return int(m.group(1)) if m else 0

    dicom_files.sort(key=_extract_number)

    if not dicom_files:
        raise FileNotFoundError(f"DICOMファイルが見つかりません: {folder_path}")

    slices = [load_and_preprocess_dicom(os.path.join(folder_path, f)) for f in dicom_files]
    image_3d = np.stack(slices, axis=0)

    sitk_image = sitk.GetImageFromArray(image_3d)
    sitk_image.SetSpacing((1.0, 1.0, 1.0))

    return sitk_image, get_folder_name(folder_path)


# ── マルチフレーム DICOM（Enhanced DICOM） ────────────────


def load_multiframe_dicom(dicom_path, apply_windowing=True):
    """マルチフレーム DICOM (IM_xxxx 形式) を読み込む。

    Returns:
        numpy.ndarray shape=(Z, Y, X) or (Y, X)
    """
    ds = pydicom.dcmread(dicom_path)
    pixel_array = ds.pixel_array

    if apply_windowing:
        if "WindowCenter" in ds and "WindowWidth" in ds:
            wc = ds.WindowCenter if isinstance(ds.WindowCenter, (float, int)) else ds.WindowCenter[0]
            ww = ds.WindowWidth if isinstance(ds.WindowWidth, (float, int)) else ds.WindowWidth[0]
        else:
            wc = np.mean(pixel_array)
            ww = np.max(pixel_array) - np.min(pixel_array)

        lo = wc - ww // 2
        hi = wc + ww // 2
        image_8bit = np.clip(pixel_array, lo, hi)
        image_8bit = ((image_8bit - lo) / (hi - lo) * 255).astype(np.uint8)
        return image_8bit

    return pixel_array


def convert_multiframe_dicom_to_nifti(dicom_path, output_path=None):
    """マルチフレーム DICOM を NIfTI に変換する。

    Returns:
        SimpleITK.Image
    """
    if not os.path.exists(dicom_path):
        raise FileNotFoundError(f"DICOMファイルが見つかりません: {dicom_path}")

    image_3d = load_multiframe_dicom(dicom_path, apply_windowing=True)
    if image_3d.ndim == 2:
        image_3d = image_3d[np.newaxis, :, :]

    sitk_image = sitk.GetImageFromArray(image_3d)

    try:
        ds = pydicom.dcmread(dicom_path, stop_before_pixels=True)
        sx = float(ds.PixelSpacing[1]) if hasattr(ds, "PixelSpacing") else 1.0
        sy = float(ds.PixelSpacing[0]) if hasattr(ds, "PixelSpacing") else 1.0
        sz = float(ds.SliceThickness) if hasattr(ds, "SliceThickness") else 1.0
        sitk_image.SetSpacing((sx, sy, sz))
    except Exception as e:
        print(f"[DICOM] spacing 取得失敗（デフォルト 1.0 を使用）: {e}")
        sitk_image.SetSpacing((1.0, 1.0, 1.0))

    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        sitk.WriteImage(sitk_image, output_path)

    return sitk_image
