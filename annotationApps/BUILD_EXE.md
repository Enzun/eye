# ポータブルアプリ EXE ランチャーのビルド手順

`.exe` ファイルは、`.bat` ファイルにアイコンを付けて起動するためのランチャーです。  
PyInstaller を使って作成します。

---

## 前提

- 開発用 Python 環境に PyInstaller がインストールされていること
  ```
  pip install pyinstaller
  ```
- アイコンファイル（`app_icon.ico`）が用意されていること

---

## EyeMuscleAnnotationPortable

### ランチャースクリプト（`launcher_eye.py`）を作成

```python
import subprocess
import sys
import os

bat = os.path.join(os.path.dirname(sys.executable), 'AnnotationTool.bat')
subprocess.Popen(['cmd', '/c', bat], creationflags=subprocess.CREATE_NO_WINDOW)
```

### ビルドコマンド

```bat
pyinstaller --onefile --noconsole --icon=app_icon.ico --name=AnnotationTool launcher_eye.py
```

### 出力先

`dist\AnnotationTool.exe` を `annotationApps\EyeMuscleAnnotationPortable\` にコピーする。

---

## HospitalMusclePortable

### ランチャースクリプト（`launcher_hospital.py`）を作成

```python
import subprocess
import sys
import os

bat = os.path.join(os.path.dirname(sys.executable), 'MuscleSegmentation.bat')
subprocess.Popen(['cmd', '/c', bat], creationflags=subprocess.CREATE_NO_WINDOW)
```

### ビルドコマンド

```bat
pyinstaller --onefile --noconsole --icon=app_icon.ico --name=MuscleSegmentation launcher_hospital.py
```

### 出力先

`dist\MuscleSegmentation.exe` を `annotationApps\HospitalMusclePortable\` にコピーする。

---

## ポイント

| オプション | 説明 |
|-----------|------|
| `--onefile` | 単一の `.exe` にまとめる |
| `--noconsole` | **コンソール画面を非表示にする**（これがないと黒い画面が出る） |
| `--icon` | アイコンを設定する |
| `--name` | 出力ファイル名 |

> [!NOTE]
> ビルド後に生成される `build/`、`dist/`、`*.spec`、`launcher_*.py` は不要なので削除してよい。

---

## 注意: `.bat` 内のエラー表示について

`--noconsole` にすると `.bat` のコンソールも非表示になるため、  
エラーが起きても画面に表示されない。  
デバッグ時は一時的に `--noconsole` を外してビルドするか、  
`.bat` 内でエラーをログファイルに書き出すようにするとよい。

```bat
python311\python.exe app\gui\annotation_tool.py >> "%~dp0error.log" 2>&1
```
