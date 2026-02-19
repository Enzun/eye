@echo off
cd /d "%~dp0"

REM ==============================
REM VC++ Redistributable の確認・インストール
REM （PyTorchのDLL読み込みに必要）
REM ==============================
reg query "HKLM\SOFTWARE\Microsoft\VisualStudio\14.0\VC\Runtimes\x64" /v Version >nul 2>&1
if errorlevel 1 (
    echo VC++ Redistributable が見つかりません。インストールしています...
    if exist "%~dp0vc_redist.x64.exe" (
        "%~dp0vc_redist.x64.exe" /install /quiet /norestart
        echo VC++ Redistributable のインストールが完了しました。
    ) else (
        echo [エラー] vc_redist.x64.exe が見つかりません。
        echo Microsoft Visual C++ Redistributable をインストールしてください。
        pause
        exit /b 1
    )
)

REM nnU-Net 環境変数を設定（正しいパス）
set nnUNet_results=%~dp0app\nnUNet_results
set nnUNet_raw=%~dp0app\nnUNet_raw
set nnUNet_preprocessed=%~dp0app\nnUNet_preprocessed

REM ポータブルPythonのScriptsをPATHに追加
REM torch\libとPyQt5\binも追加（VC++ランタイムDLLの解決用）
set PATH=%~dp0python311\Scripts;%~dp0python311\Lib\site-packages\torch\lib;%~dp0python311\Lib\site-packages\PyQt5\Qt5\bin;%~dp0python311;%PATH%

REM Qtプラグインのパスを設定（重要！）
set QT_PLUGIN_PATH=%~dp0python311\Lib\site-packages\PyQt5\Qt5\plugins

REM アプリを起動
python311\python.exe app\gui\MuscleGUI.py

REM エラーがあった場合は表示
if errorlevel 1 (
    echo.
    echo エラーが発生しました。上記のメッセージを確認してください。
    pause
)

