# CVATからnnU-Netへの移行ガイド

## 概要
このガイドでは、CVATでアノテーションしたデータをnnU-Netで学習可能な形式に変換する手順を説明します。

## ワークフロー全体像

```
1. DICOMファイル → TIFFファイル (既存のdcmToTiff.py使用)
2. TIFFファイルをCVATにアップロード
3. CVATでアノテーション作業
4. CVATから「LabelMe JSON 1.1」形式でエクスポート
5. 変換スクリプトで患者ごとにまとめてNIfTI形式に変換
6. nnU-Netで学習
```

## 詳細手順

### ステップ1: DICOMからTIFFへの変換（既存）

既存の `dcmToTiff.py` を使用してDICOMファイルをTIFF形式に変換します。

```bash
python dcmToTiff.py
```

### ステップ2: CVATでのアノテーション

1. CVATに患者ごと（または複数患者まとめて）TIFFファイルをアップロード
2. アノテーション作業を実施
   - ラベル: ir, mr, sr, so, lr, io (l_/r_プレフィックスも可)

### ステップ3: CVATからのエクスポート

**重要: 出力形式は「LabelMe JSON 1.1」を選択してください**

エクスポート手順:
1. CVATでプロジェクトまたはタスクを開く
2. メニューから「Export annotations」を選択
3. 形式: **LabelMe JSON 1.1** を選択
4. エクスポートボタンをクリック
5. ダウンロードしたZIPファイルを解凍

### ステップ4: nnU-Net環境のセットアップ

```bash
# nnU-Netのインストール（まだの場合）
pip install nnunetv2

# 環境変数の設定
export nnUNet_raw="パス/to/nnUNet_raw"
export nnUNet_preprocessed="パス/to/nnUNet_preprocessed"
export nnUNet_results="パス/to/nnUNet_results"
```

Windowsの場合:
```cmd
set nnUNet_raw=パス\to\nnUNet_raw
set nnUNet_preprocessed=パス\to\nnUNet_preprocessed
set nnUNet_results=パス\to\nnUNet_results
```

### ステップ5: 変換スクリプトの設定

`cvat_to_nnunet.py` を編集して、以下の設定を変更します:

```python
# タスクID（001-999の範囲で他のタスクと重複しないように）
TASK_ID = 102
TASK_NAME = "EyeMuscleSegmentation"

# 入力データのパス
CVAT_JSON_DIR = "datasets/cvat_output"  # CVATからエクスポートしたJSONがある場所
TIFF_IMAGE_DIR = "DATA/Tiffs"           # 元のTIFF画像がある場所

# ラベルマッピング
LABEL_MAP = {
    "ir": 1,
    "mr": 2,
    "sr": 3,
    "so": 4,
    "lr": 5,
    "io": 6,
}
```

### ステップ6: 変換の実行

```bash
python cvat_to_nnunet.py
```

出力例:
```
============================================================
CVAT → nnU-Net 変換スクリプト
============================================================

JSONファイルをグループ化中...
患者数: 5
  EX7_SE3: 15スライス
  EX9_SE2: 12スライス
  ...

[1/5] 処理中: EX7_SE3 (15スライス)
  保存完了: EX7_SE3
...

変換完了!
============================================================
変換されたケース数: 5
出力先: /path/to/nnUNet_raw/Dataset102_EyeMuscleSegmentation
```

### ステップ7: nnU-Netでの前処理と学習

```bash
# データセットの検証と前処理計画
nnUNetv2_plan_and_preprocess -d 102 --verify_dataset_integrity

# 2D学習の場合
nnUNetv2_train 102 2d 0

# 3D学習の場合（スライス数が十分にある場合）
nnUNetv2_train 102 3d_fullres 0
```

## ディレクトリ構造

### 変換前
```
Data/
└── cvat_output/
|   ├── EX7SE3IMG01.json
|   ├── EX7SE3IMG02.json
|   ├── ...
|   ├── EX9SE2IMG01.json
|   └── ...
└── Tiffs/
    ├── EX7SE3IMG01.tiff
    ├── EX7SE3IMG02.tiff
    └── ...
```

### 変換後（nnU-Net形式）
```
nnUNet_raw/
└── Dataset102_EyeMuscleSegmentation/
    ├── dataset.json
    ├── imagesTr/
    │   ├── EX7_SE3_0000.nii.gz
    │   ├── EX9_SE2_0000.nii.gz
    │   └── ...
    └── labelsTr/
        ├── EX7_SE3.nii.gz
        ├── EX9_SE2.nii.gz
        └── ...
```

## トラブルシューティング

### Q: 「TIFF画像が見つかりません」というエラーが出る

**A:** `TIFF_IMAGE_DIR` のパスが正しいか確認してください。スクリプトはサブディレクトリも再帰的に検索します。

### Q: スライス数が少ない患者がいる

**A:** nnU-Netは自動的に最適な学習方法を選択します。2Dモードを使用すれば、スライス数が少なくても学習可能です。

### Q: メモリ不足エラーが出る

**A:** 以下を試してください:
- バッチサイズを小さくする（学習時）
- 2Dモードを使用する
- 画像サイズを小さくする（前処理で自動調整されます）

### Q: ラベルが正しく認識されない

**A:** `LABEL_MAP` の設定がCVATで使用したラベル名と一致しているか確認してください。l_/r_プレフィックスは自動的に削除されます。

## YOLOとnnU-Netの比較

| 項目 | YOLO | nnU-Net |
|------|------|---------|
| 用途 | 汎用物体検出 | 医療画像セグメンテーション |
| データ形式 | JPEG/PNG | NIfTI (3D対応) |
| 学習速度 | 速い | やや遅い |
| 精度（医療画像） | 良い | 非常に良い |
| 前処理 | 手動設定 | 自動最適化 |
| 3D対応 | なし | あり |

## 参考資料

- nnU-Net公式ドキュメント: https://github.com/MIC-DKFZ/nnUNet
- CVAT公式ドキュメント: https://opencv.github.io/cvat/
