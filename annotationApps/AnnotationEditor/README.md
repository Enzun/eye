# AnnotationEditor

眼筋 MRI アノテーション補正専用 GUI

最終更新: 2026-04-04

---

## 概要

nnU-Net の推論結果（予測マスク NIfTI）を対象とした **アノテーション補正専用ツール**。  
読み込んだ瞬間から編集可能な状態になり、面積・体積の計測機能は持たない。

### HospitalMuscleBatch との主な違い

| 項目 | HospitalMuscleBatch | AnnotationEditor |
|------|---------------------|------------------|
| 目的 | 計測 + 補正 | 補正に特化 |
| 編集モード | ボタンで切り替え | 常時編集可能 |
| ラベル数 | 6（モデル出力のみ） | 11（全構造物） |
| 面積・体積計算 | あり | なし |
| 推論機能 | あり | なし |

---

## 主な機能

### 1. 常時編集モード

- 読み込み後すぐにポリゴン編集が可能
- ラベル切り替えのみで操作（「編集開始」ボタン不要）

### 2. 11 ラベル対応

モデルの推論は 6 筋肉だが、人がアノテーションする際は以下の全 11 ラベルを自由に使用できる。

| ID | 名前 | 日本語 | キー |
|----|------|--------|------|
| 1  | ir   | 下直筋 | 1 |
| 2  | mr   | 内直筋 | 2 |
| 3  | sr   | 上直筋 | 3 |
| 4  | so   | 上斜筋 | 4 |
| 5  | lr   | 外直筋 | 5 |
| 6  | lev  | 挙筋   | 6 |
| 7  | io   | 下斜筋 | 7 |
| 8  | lac  | 涙腺   | 8 |
| 9  | on   | 視神経 | 9 |
| 10 | ball | 眼球   | q |
| 11 | orbit| 眼窩   | w |

ラベル定義は `labels.json` で変更可能。

### 3. ポリゴン編集

**選択ツール（キー: 0）**
- ポリゴンクリック → 頂点ドラッグで形状調整
- 右クリック → 透明度切り替え（通常 / 薄め / 輪郭のみ）

**描画モード（キー: 1〜9 / q / w）**
- 左クリックで点を追加
- 右クリックまたはダブルクリックでポリゴン完成
- Ctrl+Z で最後の点を削除
- Esc でキャンセル

**共通操作**
- マウスホイール: ズーム（マウス位置中心）
- 中ボタンドラッグ: パン
- Delete: 選択ポリゴンを削除
- Tab: マスク表示 ON/OFF

### 4. Undo / Redo

- Ctrl+Z / Ctrl+Y でスライス単位に 30 ステップまで取り消し・やり直し
- 頂点ドラッグ後も 1 ステップとして記録

### 5. Window / Level 調整

- トップバーの WC（Window Center）・WW（Window Width）スライダーで調整
- 画像読み込み時に統計から自動設定（中央値 / パーセンタイル）

### 6. 隣接スライスからのコピー

- 「← コピー」「コピー →」ボタン、または Ctrl+← / Ctrl+→
- 形状が近いスライスで入力コストを削減

### 7. リセット

- 「リセット」ボタンで現在スライスを予測結果に戻す
- 予測ファイルがない場合は使用不可

### 8. セッション管理

- `session.json` でケース状態を永続保存
- 再起動後に最後に開いていたケースを自動復元
- ステータス（未完了 / 完了）を一覧表示

### 9. 保存

- Ctrl+S または「💾 保存」ボタン
- 保存先: `output/corrected/{case_id}_corrected.nii.gz`
- 編集履歴: `output/corrected/{case_id}_edit_log.json`
- 未保存変更があるスライスはナビバーに `*` 表示

---

## ディレクトリ構成

```
AnnotationEditor/
├── labels.json               # ラベル定義（編集可能）
├── config.json               # 設定ファイル
├── session.json              # セッション状態（自動生成）
├── requirements.txt          # Python パッケージ一覧
├── RunAnnotationEditor.bat   # 起動スクリプト（Windows）
├── app/
│   ├── gui/
│   │   ├── AnnotationEditorGUI.py  # メインアプリ
│   │   ├── editor_canvas.py        # ポリゴン編集キャンバス
│   │   └── dicom_handler.py        # DICOM → NIfTI 変換
│   └── managers/
│       ├── label_config.py         # labels.json 管理
│       ├── session_manager.py      # session.json 管理
│       └── annotation_io.py        # NIfTI 読み書き / マスク変換
├── images/                   # DICOM変換後の元画像 NIfTI（自動生成）
└── output/
    └── corrected/
        ├── {id}_corrected.nii.gz   # 編集済みマスク
        └── {id}_edit_log.json      # 編集ログ
```

---

## 入力ファイル形式

### NIfTI フォルダから一括追加

```
フォルダ/
├── case001_0000.nii.gz    # 元画像
├── case001_pred.nii.gz    # AI予測（省略可）
├── case002_0000.nii.gz
└── case002_pred.nii.gz
```

- `*_0000.nii.gz` が画像として認識される
- `*_pred.nii.gz` が予測マスクとして対応付けられる
- 予測がなくても追加可能（空のマスクから編集開始）

### DICOM フォルダを変換して追加

- IMG 番号順のシングルフレーム DICOM フォルダを選択
- Enhanced DICOM（マルチフレーム）は `dicom_handler.py` の `convert_multiframe_dicom_to_nifti()` で対応

---

## 出力ファイル

### 編集済みマスク（`*_corrected.nii.gz`）

- パス: `output/corrected/{case_id}_corrected.nii.gz`
- 形式: ラベル値 0〜11 の整数マスク、shape=(Z, Y, X)
- nnU-Net の学習データとして直接利用可能

### 編集ログ（`*_edit_log.json`）

```json
{
  "2026-04-04 12:00:00": {
    "case_id": "case001",
    "edited_slices": [3, 5, 8]
  }
}
```

---

## キーボードショートカット一覧

| キー | 動作 |
|------|------|
| 0 | 選択ツール |
| 1〜9 | ラベル切替（ir/mr/sr/so/lr/lev/io/lac/on） |
| q | ラベル ball（眼球） |
| w | ラベル orbit（眼窩） |
| 左クリック | 点追加（描画モード）/ 頂点ドラッグ（選択ツール） |
| 右クリック | ポリゴン完成（描画モード）/ 透明度切替（選択ツール） |
| ダブルクリック | ポリゴン完成 |
| Delete | 選択ポリゴン削除 |
| Esc | 描画キャンセル |
| Tab | マスク表示 ON/OFF |
| ← / → | スライス移動 |
| Ctrl+← / Ctrl+→ | 隣接スライスからコピー |
| Ctrl+Z | Undo（描画中は最後の点を削除） |
| Ctrl+Y | Redo |
| Ctrl+S | 保存 |
| マウスホイール | ズーム |
| 中ボタンドラッグ | パン |

---

## セットアップ

### 必要パッケージ

```bash
pip install -r requirements.txt
# PyQt5, numpy, SimpleITK, pydicom, opencv-python
```

### 起動方法

**Windows（推奨）**

```
RunAnnotationEditor.bat をダブルクリック
```

同リポジトリの `HospitalMuscleBatch/python311/` が存在する場合は自動的にそちらを使用する。

**Python 直接実行**

```bash
cd annotationApps/AnnotationEditor
python app/gui/AnnotationEditorGUI.py
```

---

## ラベルのカスタマイズ

`labels.json` を編集することでラベルを変更・追加できる。

```json
{
  "labels": {
    "1": {"name": "ir", "display": "下直筋", "color": [255,255,0], "key": "1"},
    ...
    "12": {"name": "new", "display": "新規", "color": [100,200,100], "key": "e"}
  }
}
```

- `key`: キーボードショートカット（1 文字）
- `color`: RGB 値（0〜255）
- 変更後にアプリを再起動すると反映される

---

## 注意事項

- 推論機能はなし（予測 NIfTI は外部で生成してから読み込む）
- CSV への計測値出力なし（補正に特化）
- 保存前に別ケースへ移動しようとすると未保存変更の確認ダイアログが表示される
- 編集済みマスクがある場合、次回読み込み時は編集済みが優先（予測より新しい）

---

## 今後の拡張候補

- スライスサムネイル一覧（全スライスのアノテーション状態を一目で確認）
- ポリゴンの自動スムージング
- 複数スライス間のポリゴン補間（3D 編集支援）
- ラベルごとの表示 ON/OFF
