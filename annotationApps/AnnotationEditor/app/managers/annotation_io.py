"""
annotation_io.py
NIfTI 読み書き、マスク ↔ ポリゴン変換
"""

from pathlib import Path

import cv2
import numpy as np
import SimpleITK as sitk


# ── NIfTI I/O ────────────────────────────────────────────


def load_nifti(path):
    """NIfTI ファイルを読み込む。

    Returns:
        tuple: (array, spacing)
            array  : numpy.ndarray shape=(Z, Y, X) uint8 or int16 etc.
            spacing: tuple (sx, sy, sz) in mm (SimpleITK 順)
    """
    img = sitk.ReadImage(str(path))
    array = sitk.GetArrayFromImage(img)  # (Z, Y, X)
    spacing = img.GetSpacing()           # (X, Y, Z)
    return array, spacing


def save_nifti(array, spacing, path):
    """numpy 配列を NIfTI ファイルに保存する。

    Args:
        array  : numpy.ndarray shape=(Z, Y, X)
        spacing: tuple (sx, sy, sz) in mm
        path   : 出力先ファイルパス
    """
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    img = sitk.GetImageFromArray(array.astype(np.uint8))
    img.SetSpacing(spacing)
    sitk.WriteImage(img, str(path))


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
