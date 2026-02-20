# セグメンテーション編集ツール仕様書

最終更新: 2026-02-20

---

## 1. 概要

`MuscleGUI` 内で予測マスクの誤りを手動修正できるポリゴンベース編集ツール。
外部ツールへのエクスポート不要で「レビュー → 修正 → 確定」のシームレスなワークフローを実現する。

**編集方式**: AnnotationTool と同じポリゴン編集（頂点ドラッグ・描画・削除）
**データ変換**: NIfTI マスク ⇔ ポリゴンリスト を相互変換

---

## 2. ユーザーワークフロー

### 2.1. 編集開始

| ステータス | 表示ボタン | 動作 |
|---|---|---|
| `pending` | ✓ 確認 / ✎ 修正 | 修正クリック → `needs_correction` に変更 |
| `needs_correction` | ✓ 確認 / ✎ 編集 | 編集クリック → 編集モード開始 |
| `confirmed` | ✓ 確認 / ✎ 修正 | 修正クリック → `needs_correction` に変更（確認ボタンは押しても無効だが表示は維持） |
| `corrected` | 再編集 | 再編集クリック → 編集モード開始 |

### 2.2. 編集中

1. **UI ロック**:
   - 患者リスト・スライスバー・ズーム・タブ・バッチ処理ボタン等すべて無効化
   - 編集パネルのみ操作可能
2. **編集操作**（AnnotationTool と同じ）:
   - **ポリゴン選択**: リストクリック → 頂点が表示される
   - **頂点ドラッグ**: マウスで頂点を移動して形状調整
   - **新規描画**: ラベル選択 → 左クリックで点追加 → 右クリックで完成
   - **削除**: ポリゴン選択 → Delete キーまたは削除ボタン
   - **ラベル別一括削除**: ラベル選択 → 「選択ラベルをクリア」ボタン
3. **自動保存**: スライス切替・描画完了時に一時 NIfTI (`temp_edit/{id}_temp.nii.gz`) を自動保存
4. **表示**: 編集中のラベルは不透明、他ラベルは半透明（alpha 0.3〜0.5）

### 2.3. 編集完了

- **「編集完了」ボタン**:
  - 全スライスのポリゴンを NIfTI にラスタライズ
  - 体積・面積を再計算（3方式: 全スライス・動的範囲・固定範囲）
  - `edited/{id}_corrected.nii.gz` に保存
  - `edited/csv/corrected_measurements.csv` に追記
  - 元 CSV の `review_status` を `corrected` に更新
  - ステータス: `needs_correction` → `corrected` / `corrected` → `corrected`（変化なし）
  - UI ロック解除
- **「キャンセル」ボタン**:
  - 確認ダイアログ表示
  - OK → 一時ファイル削除・編集内容破棄・UI ロック解除

---

## 3. データ変換仕様

### 3.1. NIfTI → ポリゴン（編集開始時）

```python
# スライスごとに実行
for slice_idx in range(num_slices):
    slice_mask = pred_array[slice_idx]  # (Y, X)
    polygons = []

    for label_id in [1, 2, 3, 4, 5, 6]:  # SR, IR, MR, LR, SO, IO
        mask = (slice_mask == label_id).astype(np.uint8)
        contours, hierarchy = cv2.findContours(
            mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
        )

        for contour in contours:
            points = contour.squeeze().tolist()  # [[x, y], ...]
            polygons.append({
                "label": label_id,
                "label_name": label_names[label_id],  # "sr", "ir", ...
                "points": points
            })

    slice_polygons[slice_idx] = polygons
```

**注**:
- `cv2.RETR_TREE` で外側輪郭と穴（内側輪郭）の両方を取得
- 穴は hierarchy で判定し、ラスタライズ時に黒塗り（背景に戻す）

### 3.2. ポリゴン → NIfTI（編集完了時）

```python
# スライスごとにラスタライズ
for slice_idx, polygons in slice_polygons.items():
    canvas = np.zeros((height, width), dtype=np.uint8)

    # ラベル順に描画（後のラベルが上書き）
    for label_id in [1, 2, 3, 4, 5, 6]:
        for poly in polygons:
            if poly["label"] == label_id:
                pts = np.array(poly["points"], dtype=np.int32)
                cv2.fillPoly(canvas, [pts], color=label_id)

    pred_array[slice_idx] = canvas

# SimpleITK で保存
pred_sitk = sitk.GetImageFromArray(pred_array)
pred_sitk.SetSpacing(spacing)
sitk.WriteImage(pred_sitk, edited_path)
```

---

## 4. UI 構成

### 4.1. 編集パネル（左サイドバーまたはオーバーレイ）

```
┌─────────────────────────────┐
│ 【編集モード】              │
├─────────────────────────────┤
│ ラベル選択:                 │
│  ○ SR (上直筋)     [1]      │
│  ○ IR (下直筋)     [2]      │
│  ○ MR (内直筋)     [3]      │
│  ○ LR (外直筋)     [4]      │
│  ○ SO (上斜筋)     [5]      │
│  ○ IO (下斜筋)     [6]      │
│                             │
│ [選択ラベルをクリア]        │
├─────────────────────────────┤
│ ポリゴン一覧:               │
│ ┌─────────────────────────┐ │
│ │ 1. sr (4点)             │ │
│ │ 2. ir (6点)             │ │
│ │ ...                     │ │
│ └─────────────────────────┘ │
│ [選択を削除 (Delete)]       │
├─────────────────────────────┤
│ [💾 編集完了]  [❌ キャンセル] │
└─────────────────────────────┘
```

### 4.2. キーボードショートカット

| キー | 動作 |
|---|---|
| 1〜6 | ラベル切替（SR, IR, MR, LR, SO, IO） |
| 左クリック | 点を追加 |
| 右クリック | ポリゴン完成 |
| Delete | 選択ポリゴン削除 |
| Esc | 描画中キャンセル |
| ← → | スライス切替（編集モード中も有効） |

---

## 5. ファイル構造

```
output/119/
├── predictions/
│   └── {id}_pred.nii.gz              # 元の予測（不変）
├── temp_edit/
│   └── {id}_temp.nii.gz              # 編集中の一時保存（編集完了/キャンセルで削除）
├── edited/
│   ├── {id}_corrected.nii.gz         # 編集完了後の最終版
│   └── csv/
│       └── corrected_measurements.csv  # 編集済み患者の体積データ
└── csv/
    └── muscle_measurements.csv       # 全患者の元データ（review_status 列のみ更新）
```

---

## 6. CSV 管理

### 6.1. 元 CSV (`muscle_measurements.csv`)
- **更新内容**: `review_status` 列のみ
- **体積データ**: 編集前の値を保持（モデル検証用）

### 6.2. 編集後 CSV (`corrected_measurements.csv`)
- **対象**: `review_status == "corrected"` の患者のみ
- **形式**: 元 CSV と同じ列構成（folder_name, timestamp, 体積, 面積, review_status）
- **使い分け**:
  - GUI 表示時: status が `corrected` なら edited CSV から読み込み、それ以外は元 CSV
  - エクスポート時: 両方を結合して最新データを提供

---

## 7. 技術的実装詳細

### 7.1. コード再利用

| モジュール | 流用元 | 用途 |
|---|---|---|
| `AnnotationCanvas` | AnnotationTool | ポリゴン描画・編集 UI |
| `visualize_slice` | MuscleGUI (既存) | 輪郭抽出ロジック (`cv2.findContours`) |
| `ResultManager` | MuscleGUI (既存) | NIfTI 保存・CSV 管理 |

### 7.2. 編集状態管理

```python
class MuscleSegmentationGUI:
    def __init__(self):
        self.is_editing = False
        self.edit_slice_polygons = {}  # {slice_idx: [polygons]}
        self.edit_temp_dir = None
        self.edit_original_pred_array = None
```

### 7.3. 編集完了時の処理フロー

```python
def finish_editing(self):
    # 1. ポリゴン → NIfTI 変換
    pred_array = self.rasterize_all_slices(self.edit_slice_polygons)

    # 2. 体積再計算
    volumes_all, volumes_dyn, volumes_fix, dyn_range = self.recalculate_volumes(pred_array)

    # 3. 保存
    self.result_manager.save_corrected_nifti(folder_name, pred_array, spacing)
    self.result_manager.append_to_corrected_csv(folder_name, volumes_all, volumes_dyn, volumes_fix, dyn_range, max_areas)

    # 4. ステータス更新
    self.result_manager.update_review_status(folder_name, "corrected")

    # 5. 一時ファイル削除
    self.cleanup_temp_edit()

    # 6. UI ロック解除
    self.is_editing = False
    self.unlock_ui()
```

---

## 8. 制約・注意事項

- **L/R の自動判定**: ポリゴンの重心 X 座標で左右を判定（現在の `visualize_slice` と同じロジック）
- **穴あきマスク**: `cv2.RETR_TREE` + hierarchy で内側輪郭も保持し、ラスタライズ時に黒塗り
- **複数輪郭**: 同一ラベルで複数の独立した領域がある場合、それぞれ別ポリゴンとして扱う
- **編集中のスライス切替**: 可能（一時保存が自動実行される）
- **編集中の患者切替**: 不可（完了/キャンセルが必須）

---

## 9. 将来拡張

- [ ] Undo/Redo（編集履歴スタック）
- [ ] ポリゴンの自動スムージング（輪郭をなめらかに）
- [ ] 複数スライス間のポリゴン補間（3D 編集支援）
- [ ] 編集データの nnU-Net 学習データ自動エクスポート
