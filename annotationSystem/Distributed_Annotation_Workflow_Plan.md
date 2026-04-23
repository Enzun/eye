# 分散アノテーション作業に向けたワークフロー方針と設計まとめ

本ドキュメントは、複数人でのアノテーション作業を安全・効率的に進めるために確定したアーキテクチャ・運用方針をまとめたものです。

---

## 1. 全体方針：役割の完全分離（主PC vs 作業者PC）

これまで1つのアプリで行っていた機能群を、「データ準備（管理者）」と「データ編集（作業者）」に完全に役割分担させます。

### 🏥 主PC（管理者）の役割
高度な処理や個人情報に関わる処理はすべて主PC（＋強PC）に集約します。
- **データ収集**: 病院の共有フォルダ等からDICOMデータを読み込む
- **フォーマット変換**: DICOMからNIfTI（`*_0000.nii.gz`）への変換
- **AI予測生成（GPU強PCが実行）**: nnUNetで事前予測（`*_pred.nii.gz`）を実行
- **匿名化処理**: 患者IDを含むファイル名を連番（例: `Case001_0000.nii.gz`）に変換
    - 変換前後の対応表（CSV）を作成し、主PC内から**絶対に出さない**
- **作業振り分け**: `assignments.json` にグループを定義し、作業者ごとの担当ケース範囲を設定
- **進捗可視化**: 完了済みケース数・未完了ケースを一覧できる管理ツールを使用
- **データ回収・復元**: `*_corrected.nii.gz` を回収し、対応表をもとに元のファイル名へ戻す

### 🧑‍💻 作業者PC（知り合い等）の役割
専用の編集アプリのみを使用し、アノテーションの修正作業だけに集中します。
- 共有フォルダから `AnnotationEditor.exe` を起動（またはローカルコピーして起動）
- 起動時にグループ選択ダイアログで自分の担当グループを選択
- 匿名化された画像・予測結果を編集し、`*_corrected.nii.gz` として保存

---

## 2. 開発対象ソフトウェア

| ソフト | 対象PC | 元となるコード |
|---|---|---|
| **AnnotationEditor（Lite版）** | 作業者PC | 現行 AnnotationEditor から推論・DICOM機能を削除 |
| **主PC管理ツール** | 主PC（管理者） | 新規作成（匿名化・振り分け・進捗可視化・回収） |

---

## 3. フォルダ構成（01_mainpc と 02_shared）

主PC用と共有ネットワーク用の2つの親フォルダに集約します。

### 01_mainpc（主PCのローカル領域・外部非公開）
生データの一時保管と、管理用ツール・匿名化対応表を配置します。
```text
C:\01_mainpc\
  ├── rawData_nifti/        ← DICOMから変換された生データ
  ├── codes/                ← 管理用ツール（匿名化、振り分け、回収など）
  └── mapping/              ← 匿名化対応表（case_mapping.csv）
```

### 02_shared（全員がアクセスする共有フォルダ）
作業者PCからアクセスする編集用データ群です。
```text
\\192.168.0.81\02_shared\
  │
  ├── AnnotationEditor.exe                   ← 作業者用アプリ（1つだけ配置）
  ├── config.json                            ← 共通設定（workspace_root など）
  ├── labels.json                            ← ラベル定義（共通）
  ├── assignments.json                       ← グループ定義（主PCが管理・作業者は読み取り専用）
  │
  ├── images/                                ← 匿名化済みNIfTI画像（全員が読み取り専用）
  │   ├── raw/                               # 生データ（*_0000.nii.gz）
  │   │   └── Case001_0000.nii.gz
  │   └── pred/                              # AI予測結果（*_pred.nii.gz）
  │       └── Case001_pred.nii.gz
  │
  ├── corrected/                             ← 完了ファイル保存先（全員が書き込む・競合なし）
  │   ├── Case001_corrected.nii.gz
  │   └── ...
  │
  └── sessions/                              ← グループごとの session ファイル
      ├── grp001_Aさん.json                  ← Aさん担当グループの状態管理
      └── grp002_Bさん.json
```

### 競合リスク分析
| ファイル | 書き込み者 | 競合リスク |
|---|---|---|
| `images/` 内ファイル | 読み取り専用 | なし |
| `corrected/` 内ファイル | 各作業者（ファイル名が一意） | なし |
| `config.json`, `labels.json` | 読み取り専用 | なし |
| `assignments.json` | 主PCのみ | なし |
| `sessions/{group}.json` | 担当作業者のみ | **グループ割り当てが1対1なら実質なし** |

> ⚠️ **唯一のリスク箇所**：2人が同じグループの session.json を同時に開いた場合。
> グループは1人1担当で運用することで回避する。

### AnnotationEditor（Lite版）の起動フロー
1. 作業者が `AnnotationEditor.exe` をダブルクリック
2. `config.json` の `workspace_root`（絶対パス）を読み込み
3. `assignments.json` を参照してグループ一覧を表示するダイアログを表示
4. 作業者が自分のグループを選択
5. 以降、`images/` から担当ケースの画像を表示・`sessions/grp00X_名前.json` を読み書き・`corrected/` に保存

> 💡 `config.json` に `workspace_root` の絶対UNCパスを書けば、共有フォルダから直接起動しても
> ローカルに exe をコピーして起動しても、同じ共有データにアクセスできる。

---

## 4. assignments.json の設計

```json
{
  "version": "1.0",
  "groups": [
    {
      "group_id": "grp001",
      "group_name": "Aさん担当",
      "case_start": 1,
      "case_end": 20,
      "session_file": "sessions/grp001_A.json",
      "note": ""
    },
    {
      "group_id": "grp002",
      "group_name": "Bさん担当",
      "case_start": 21,
      "case_end": 40,
      "session_file": "sessions/grp002_B.json",
      "note": ""
    }
  ]
}
```

| フィールド | 意味 |
|---|---|
| `group_id` | システム内部ID（変更しない） |
| `group_name` | 表示名（自由に設定可能） |
| `case_start` / `case_end` | 担当ケースの連番範囲（例: 1〜20 → Case001〜Case020） |
| `session_file` | 対応する session.json のパス（`workspace_root` からの相対パス） |
| `note` | 備考（任意） |

---

## 5. ファイル命名規則（匿名化）

### 元のファイル名（主PC内のみ。外部に出さない）
```
147757_20260308_112048_EX1_SE3_0000.nii.gz
```

### 匿名化後（共有フォルダへ配置するもの）
```
Case001_0000.nii.gz       ← 画像ファイル
Case001_pred.nii.gz       ← AI事前予測
Case001_corrected.nii.gz  ← 作業者の編集完了ファイル
```

- 形式は `Case{3〜4桁の連番}` を基本とする（症例数に応じて桁数を決定）
- `EX`・`SE` 等のシリーズ情報は**対応表（CSV）に組み込み**、ファイル名からは除去
- 対応表は `case_id, original_filename, exam_id, series_id, patient_id, ...` 等の列を持つ

### 対応表（主PC内のみ・絶対に出さない）
```csv
case_id, original_filename, patient_id, exam_id, series_id, convert_date
Case001, 147757_20260308_112048_EX1_SE3, 147757, EX1, SE3, 2026-03-08
Case002, ...
```

---

## 6. 主PC管理ツールの機能一覧

| 機能 | 説明 |
|---|---|
| **DICOM→NIfTI変換** | 指定フォルダのDICOMを変換・シリーズ名フィルタで対象を絞る |
| **匿名化・配置** | 連番に変換・対応表CSV生成・`images/` へコピー |
| **assignments.json 編集** | グループの新規作成・担当ケース範囲の設定 |
| **進捗可視化** | `corrected/` フォルダと各 session.json を読み、完了率を一覧表示 |
| **データ回収・復元** | `corrected/` から完了ファイルを回収し、対応表で元のIDに戻して最終保管 |

---

## 7. AnnotationEditor（Lite版）の機能変更方針

### 削除する機能（推論・入力周り）
- DICOM インポート機能（`dicom_handler.py`）
- nnUNet 推論実行機能（`run_nnunet.py`）
- 共有フォルダ自動スキャン機能（`auto_loader.py`）
- 予測フィルタリング（`prediction_filter.py`）
- `config.json` の推論関連フィールド（`model_id`, `series_filter`, `prediction_filter`）

### 追加・変更する機能
- **起動時グループ選択ダイアログ**（`assignments.json` を読んでグループ一覧を表示）
- **担当ケースのみ表示**（選択グループの `case_start`〜`case_end` の範囲に絞る）
- **session.json の相対パス化**（`workspace_root` 基点の相対パスで記録）
- **`corrected/` への保存先統一**（作業者フォルダ分けなし）
- ゼロからのアノテーション対応（予測ファイルがなくても起動できる）

---

## 8. セキュリティ（匿名化）についての補足
NIfTI（`.nii.gz`）ファイル自体には、DICOMとは異なりヘッダー情報等による患者個人情報（氏名や生年月日）は格納されません。
そのため、**ファイル名を連番などにリネーム（匿名化）するだけで、患者データをそのまま外部のPC環境に送ってアノテーションさせることが可能**です。
（※各施設の倫理規定や情報の持ち出しルールに準拠できる範囲で運用可能）

---

## 9. 開発環境・病院環境を行き来する際の注意点

### 9-1. JSONファイルでのUNCパスの記述（エスケープ）
`config.json` に `\\192.168.0.81\03_Shared_Workspace` のようなパスを書く場合、バックスラッシュは2倍にする。
- ❌ 誤: `"workspace_root": "\\192.168.0.81\03_Shared_Workspace"`
- ✅ 正: `"workspace_root": "\\\\192.168.0.81\\03_Shared_Workspace"`

### 9-2. Lite版から PyTorch を除去する
Lite版には推論機能がないため、`python311` 実行環境から PyTorch・CUDA 関連パッケージを削除し、exeサイズと起動時間を軽量化する。

### 9-3. ファイル名の変遷管理
```
【主PC】 DICOMインポート
    → 147757_20260308_112048_EX1_SE3_0000.nii.gz  （一時保管）
    → 匿名化・対応表生成
    → Case001_0000.nii.gz  （images/ へコピー）

【GPU強PC】 AI予測
    → Case001_pred.nii.gz  （images/ へコピー）

【作業者PC】 AnnotationEditor（Lite版）で編集
    → Case001_corrected.nii.gz  （corrected/ へ保存）

【主PC】 回収・復元
    → 対応表参照 → 147757_20260308_112048_EX1_SE3_corrected.nii.gz  （最終保管）
```
