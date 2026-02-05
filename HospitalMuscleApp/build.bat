@echo off
REM Hospital Muscle Segmentation App - Build Script
REM PyInstallerを使用してexeを作成します

echo ============================================
echo   Hospital Muscle Segmentation App Builder
echo ============================================
echo.

REM PyInstallerがインストールされているか確認
pip show pyinstaller >nul 2>&1
if errorlevel 1 (
    echo PyInstallerがインストールされていません。インストールします...
    pip install pyinstaller
)

echo.
echo ビルドを開始します...
echo （これには数分かかる場合があります）
echo.

REM specファイルを使用してビルド
pyinstaller --clean MuscleSegmentation.spec

echo.
if exist "dist\MuscleSegmentation" (
    echo ============================================
    echo   ビルド成功！
    echo ============================================
    echo.
    echo 出力先: dist\MuscleSegmentation\
    echo 実行ファイル: dist\MuscleSegmentation\MuscleSegmentation.exe
    echo.
    echo USBにコピーする内容:
    echo   - dist\MuscleSegmentation\ フォルダ全体
    echo.
) else (
    echo ============================================
    echo   ビルド失敗
    echo ============================================
    echo エラーログを確認してください。
)

pause
