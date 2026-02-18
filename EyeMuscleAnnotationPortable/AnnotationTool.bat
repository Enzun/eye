@echo off
cd /d "%~dp0"

REM ポータブルPythonのScriptsをPATHに追加
set PATH=%~dp0python311\Scripts;%PATH%

REM アプリを起動
python311\python.exe app\gui\annotation_tool.py

REM エラーがあった場合は表示
if errorlevel 1 (
    echo.
    echo エラーが発生しました。上記のメッセージを確認してください。
    pause
)
