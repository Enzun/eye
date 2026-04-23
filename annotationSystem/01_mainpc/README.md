# 01_mainpc — 主PC（管理者）専用の作業領域

このフォルダは主PC上にのみ存在します。生データの保管から管理ツールの配置まで、
「主PC内で完結する機能」をすべてここに集約します。
共有フォルダには**絶対に置かない**でください。

## フォルダ構成

```
01_mainpc/
├── rawData_nifti/    ← エクスポートされた生データ（元の患者IDやファイル名を持つ）
│   ├── 147757_20260308_112048_EX1_SE3_0000.nii.gz
│   └── ...
│
├── AdminApp/         ← 管理用GUIアプリ（主PC管理ツール）
│   ├── main_app.py           # GUI起動スクリプト
│   ├── core/                 # ロジック（インポート、推論、配布等）
│   └── utils/                # 補助スクリプト
│
└── mapping/          ← 対応表（絶対に外部へ出さない）
    └── case_mapping.csv      # case_id ↔ 元のファイル名・患者ID等の対応表
```

## 対応表（case_mapping.csv）フォーマット
```csv
case_id,original_filename,patient_id,exam_id,series_id,convert_date
Case001,147757_20260308_112048_EX1_SE3,147757,EX1,SE3,2026-03-08
Case002,...
```
