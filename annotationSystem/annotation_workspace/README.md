# 02_shared — 全員がアクセスする共有フォルダ

作業者PCからアクセスする編集用データ群です。
作業者PC側で使用する `AnnotationEditor.exe` や、匿名化済みの画像ファイルが置かれます。

## フォルダ構成

```
02_shared/
├── AnnotationEditor.exe     ← 作業者用アプリ（1つだけ配置、作業者はこれを起動）
├── config.json              ← 共通設定（workspace_rootなど）
├── labels.json              ← ラベル定義（共通）
├── assignments.json         ← グループ定義（主PCが管理・作業者は読み取り専用）
│
├── images/                  ← 匿名化済みNIfTI画像（全員が読み取り専用）
│   ├── raw/                 # 生データ（*_0000.nii.gz）
│   │   └── Case001_0000.nii.gz
│   └── pred/                # AI予測結果（*_pred.nii.gz）
│       └── Case001_pred.nii.gz
│
├── corrected/               ← 完了ファイル保存先（全員が書き込む・競合なし）
│   └── Case001_corrected.nii.gz
│
└── sessions/                ← グループごとの session ファイル
    ├── grp001_Aさん.json
    └── grp002_Bさん.json
```

## ⚠️ 共有フォルダのアクセス権限設定

ネットワーク（別PC）経由で作業を行えるようにするため、このワークスペースをネットワーク共有する際は、**作業者からのアクセスに対して以下の権限設定**を行ってください。

| ファイル / フォルダ | 作業者に必要な権限 | 備考 |
|:---|:---|:---|
| `images/` | **読み取り専用** | 生データおよび推論結果を誤って変更・削除させないため。 |
| `*.json` 設定ファイル一式 | **読み取り専用** | `config.json`, `labels.json`, `assignments.json` など、設定の不用意な変更を防ぐため。 |
| `corrected/` | **変更（読み書き更新）** | 各作業者がアノテーション修正後の NIfTI ファイルを保存するために必須。 |
| `sessions/` | **変更（読み書き更新）** | アプリが現在の作業進捗（セッション情報）を保存するために必須。 |

> **💡 Tips: AnnotationEditor の配布について**
> ネットワーク越しに exe を起動すると遅い場合があります。作業者が各自のPCのローカル（デスクトップ等）にコピーして使用する場合は、アプリ内の `config.json` にある `"workspace_root"` の値をこの共有フォルダのUNCパス（例: `"\\\\192.168.0.x\\shared\\annotation_workspace"`）に書き換えてから配布してください。
