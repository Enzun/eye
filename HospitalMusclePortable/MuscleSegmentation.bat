@echo off
cd /d "%~dp0"

REM nnU-Net 環境変数を設定（正しいパス）
set nnUNet_results=%~dp0app\nnUNet_results
set nnUNet_raw=%~dp0app\nnUNet_raw
set nnUNet_preprocessed=%~dp0app\nnUNet_preprocessed

REM ポータブルPythonのScriptsをPATHに追加
set PATH=%~dp0python311\Scripts;%PATH%

REM アプリを起動
python311\python.exe app\gui\MuscleGUI.py

REM エラーがあった場合は表示
if errorlevel 1 (
    echo.
    echo エラーが発生しました。上記のメッセージを確認してください。
    pause
)

