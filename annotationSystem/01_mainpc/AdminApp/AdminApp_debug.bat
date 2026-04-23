@echo off
chcp 65001 > nul
pushd "%~dp0"

echo === Python パスの確認 ===
set PYTHON_EXE=%~dp0..\..\python311\python.exe
echo PYTHON_EXE = %PYTHON_EXE%

if not exist "%PYTHON_EXE%" (
    echo [ERROR] python311\python.exe が見つかりません: %PYTHON_EXE%
    pause
    exit /b 1
)

echo.
echo === PyQt5 DLL の確認 ===
dir "%~dp0..\..\python311\Lib\site-packages\PyQt5\Qt5\bin\Qt5Core.dll" 2>nul
if errorlevel 1 echo [WARNING] PyQt5 Qt5Core.dll が見つかりません

echo.
echo === 起動テスト ===
"%PYTHON_EXE%" -c "import sys; print('Python:', sys.version)"
echo.
"%PYTHON_EXE%" -c "from PyQt5.QtWidgets import QApplication; print('PyQt5 OK')"
echo.
"%PYTHON_EXE%" AdminApp.py

echo.
echo 終了コード: %ERRORLEVEL%
pause
