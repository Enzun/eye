"""
annotation_io.py
NIfTI 読み書き、マスク ↔ ポリゴン変換

SimpleITK は UNC パス (\\server\...) や日本語を含むパスを開けないため、
該当する場合はローカル一時ファイル経由で読み書きする。
"""

import os
import shutil
import tempfile
from pathlib import Path

import cv2
import numpy as np
import SimpleITK as sitk


# ── パス安全判定 ──────────────────────────────────────────

def _needs_temp_copy(path_str: str) -> bool:
    """SimpleITK に渡す前にローカルコピーが必要か判定する"""
    # UNC パス
    if path_str.startswith("\\\\"):
        return True
    # ASCII の範囲外 (日本語フォルダ名など)
    try:
        path_str.encode("ascii")
    except UnicodeEncodeError:
        return True
    return False


# ── NIfTI I/O ────────────────────────────────────────────


def load_nifti(path):
    """NIfTI ファイルを読み込む。

    UNC パスや日本語パスの場合はローカル一時ファイルにコピーしてから読み込む。

    Returns:
        tuple: (array, spacing)
            array  : numpy.ndarray shape=(Z, Y, X) uint8 or int16 etc.
            spacing: tuple (sx, sy, sz) in mm (SimpleITK 順)
    """
    path_str = str(path)
    tmp_dir = None

    try:
        if _needs_temp_copy(path_str):
            # ローカルの temp にコピーして読む (ASCII-only パスが保証される)
            tmp_dir = tempfile.mkdtemp(prefix="annot_io_")
            basename = os.path.basename(path_str)
            # ファイル名もASCII化 (念のため)
            safe_name = "tmp_nifti.nii.gz" if not basename.isascii() else basename
            tmp_path = os.path.join(tmp_dir, safe_name)
            shutil.copy2(path_str, tmp_path)
            img = sitk.ReadImage(tmp_path)
        else:
            img = sitk.ReadImage(path_str)

        array = sitk.GetArrayFromImage(img)  # (Z, Y, X)
        spacing = img.GetSpacing()           # (X, Y, Z)
        return array, spacing
    finally:
        # 一時ファイルを確実にクリーンアップ
        if tmp_dir and os.path.exists(tmp_dir):
            shutil.rmtree(tmp_dir, ignore_errors=True)


def save_nifti(array, spacing, path):
    """numpy 配列を NIfTI ファイルに保存する。

    UNC パスや日本語パスの場合はローカル一時ファイルに書いてからコピーする。

    Args:
        array  : numpy.ndarray shape=(Z, Y, X)
        spacing: tuple (sx, sy, sz) in mm
        path   : 出力先ファイルパス
    """
    path_str = str(path)
    Path(path_str).parent.mkdir(parents=True, exist_ok=True)

    img = sitk.GetImageFromArray(array.astype(np.uint8))
    img.SetSpacing(spacing)

    if _needs_temp_copy(path_str):
        tmp_dir = tempfile.mkdtemp(prefix="annot_io_")
        try:
            tmp_path = os.path.join(tmp_dir, "tmp_nifti.nii.gz")
            sitk.WriteImage(img, tmp_path)
            shutil.copy2(tmp_path, path_str)
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)
    else:
        sitk.WriteImage(img, path_str)


# ── マスク ↔ ポリゴン変換 ─────────────────────────────────


def mask_to_polygons(slice_mask, max_label=11):
    """2D マスク配列をポリゴンリストに変換する。

    Args:
        slice_mask: numpy.ndarray shape=(Y, X) 値 0=背景, 1〜max_label=各ラベル
        max_label : 最大ラベル ID

    Returns:
        list of dict:
            [{"label": int, "label_name": str, "points": [[x,y],...], "is_hole": bool}, ...]
    """
    polygons = []
    for label_id in range(1, max_label + 1):
        mask = (slice_mask == label_id).astype(np.uint8)
        if mask.sum() == 0:
            continue

        contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        if hierarchy is None:
            continue

        # hierarchy[0][i] = [next, previous, first_child, parent]
        for i, contour in enumerate(contours):
            if len(contour) < 3:
                continue
            pts = contour.squeeze().tolist()
            if isinstance(pts[0], int):
                pts = [pts]  # 点が1点だけの場合
            is_hole = (hierarchy[0][i][3] != -1)
            polygons.append({
                "label": label_id,
                "label_name": f"label_{label_id}",
                "points": pts,
                "is_hole": is_hole,
            })
    return polygons


def polygons_to_mask(polygons, shape):
    """ポリゴンリストを 2D マスク配列にラスタライズする。

    Args:
        polygons: mask_to_polygons() の出力形式
        shape   : (height, width)

    Returns:
        numpy.ndarray shape=(Y, X) dtype=uint8
    """
    mask = np.zeros(shape, dtype=np.uint8)
    if not polygons:
        return mask

    max_label = max(p["label"] for p in polygons)
    for label_id in range(1, max_label + 1):
        # 外側輪郭を塗りつぶす
        for poly in polygons:
            if poly["label"] == label_id and not poly.get("is_hole"):
                pts = np.array(poly["points"], dtype=np.int32)
                if len(pts) >= 3:
                    cv2.fillPoly(mask, [pts], color=label_id)
        # 穴を背景で上書き
        for poly in polygons:
            if poly["label"] == label_id and poly.get("is_hole"):
                pts = np.array(poly["points"], dtype=np.int32)
                if len(pts) >= 3:
                    cv2.fillPoly(mask, [pts], color=0)
    return mask
