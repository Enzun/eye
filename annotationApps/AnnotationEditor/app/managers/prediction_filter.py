"""
prediction_filter.py
予測マスクの後処理フィルタ

フィルタ適用順:
  1. 小面積除去 : area < min_area_px のコンポーネントを削除
  2. 端部除去   : 画像端から edge_margin_pct% 以内のコンポーネントを削除
  3. 個数上限   : 上記フィルタ後に残ったコンポーネントが max_keep_per_label 超なら
                  面積上位 max_keep_per_label 個のみ残す（enable_count_filter=True 時）
"""

import numpy as np
import cv2


# デフォルト設定
DEFAULT_CONFIG = {
    "min_area_px": 20,
    "edge_margin_pct": 12,
    "max_keep_per_label": 2,
    "enable_count_filter": True,
}


def filter_predictions(mask_array: np.ndarray, cfg: dict | None = None):
    """
    予測マスクにフィルタを適用する。

    Args:
        mask_array : (Z, Y, X) uint8 のマスク配列
        cfg        : フィルタ設定辞書（省略時はデフォルト値を使用）

    Returns:
        (filtered_mask, stats)
        stats = {
            "removed_small": int,   # 小面積で削除したコンポーネント数
            "removed_edge":  int,   # 端部で削除したコンポーネント数
            "removed_count": int,   # 個数上限で削除したコンポーネント数
            "slices_affected": int, # 変化があったスライス数
        }
    """
    if cfg is None:
        cfg = {}
    min_area    = int(cfg.get("min_area_px",        DEFAULT_CONFIG["min_area_px"]))
    margin_pct  = float(cfg.get("edge_margin_pct",  DEFAULT_CONFIG["edge_margin_pct"])) / 100.0
    max_keep    = int(cfg.get("max_keep_per_label",  DEFAULT_CONFIG["max_keep_per_label"]))
    use_count   = bool(cfg.get("enable_count_filter", DEFAULT_CONFIG["enable_count_filter"]))

    result = mask_array.copy()
    stats = {"removed_small": 0, "removed_edge": 0, "removed_count": 0, "slices_affected": 0}

    Z, H, W = mask_array.shape
    margin_y = int(H * margin_pct)
    margin_x = int(W * margin_pct)

    for z in range(Z):
        slice_mask  = mask_array[z]
        new_slice   = slice_mask.copy()
        changed     = False

        for label_id in np.unique(slice_mask):
            if label_id == 0:
                continue

            label_bin = (slice_mask == label_id).astype(np.uint8)
            n_comp, labels_map, comp_stats, _ = cv2.connectedComponentsWithStats(label_bin)

            # ── フィルタ 1 & 2: 小面積 / 端部 ─────────────────
            survivors = []
            for cid in range(1, n_comp):
                area = int(comp_stats[cid, cv2.CC_STAT_AREA])
                x0   = int(comp_stats[cid, cv2.CC_STAT_LEFT])
                y0   = int(comp_stats[cid, cv2.CC_STAT_TOP])
                cw   = int(comp_stats[cid, cv2.CC_STAT_WIDTH])
                ch   = int(comp_stats[cid, cv2.CC_STAT_HEIGHT])
                x1, y1 = x0 + cw, y0 + ch

                remove_reason = None
                if area < min_area:
                    remove_reason = "small"
                elif y0 < margin_y or y1 > H - margin_y:
                    remove_reason = "edge"
                elif x0 < margin_x or x1 > W - margin_x:
                    remove_reason = "edge"

                if remove_reason:
                    new_slice[labels_map == cid] = 0
                    stats[f"removed_{remove_reason}"] += 1
                    changed = True
                else:
                    survivors.append((cid, area))

            # ── フィルタ 3: 個数上限 ──────────────────────────
            if use_count and len(survivors) > max_keep:
                # 面積の大きい順に並べ、上位 max_keep 個以外を削除
                survivors_sorted = sorted(survivors, key=lambda x: x[1], reverse=True)
                for cid, _ in survivors_sorted[max_keep:]:
                    new_slice[labels_map == cid] = 0
                    stats["removed_count"] += 1
                    changed = True

        result[z] = new_slice
        if changed:
            stats["slices_affected"] += 1

    return result, stats


def stats_summary(stats: dict) -> str:
    """統計を人間が読みやすい文字列に変換する"""
    total = stats["removed_small"] + stats["removed_edge"] + stats["removed_count"]
    if total == 0:
        return "削除対象なし"
    parts = []
    if stats["removed_small"]:
        parts.append(f"極小 {stats['removed_small']}個")
    if stats["removed_edge"]:
        parts.append(f"端部 {stats['removed_edge']}個")
    if stats["removed_count"]:
        parts.append(f"過剰個数 {stats['removed_count']}個")
    return f"{' / '.join(parts)} を削除  ({stats['slices_affected']} スライス影響)"
