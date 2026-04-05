@echo off
chcp 65001 > nul
cd /d "%~dp0"

rem Resolve portable Python path (normalizes ".." in path)
rem Priority: annotationApps/python311 > HospitalMuscleBatch/python311 (legacy)
for %%i in ("%~dp0..\python311\python.exe") do set "PYTHON_EXE=%%~fi"
if not exist "%PYTHON_EXE%" (
    for %%i in ("%~dp0..\HospitalMuscleBatch\python311\python.exe") do set "PYTHON_EXE=%%~fi"
)

if exist "%PYTHON_EXE%" (
    echo [INFO] Using portable Python: %PYTHON_EXE%
    "%PYTHON_EXE%" app\gui\AnnotationEditorGUI.py
    goto :check_error
)

echo [INFO] Using system Python
python app\gui\AnnotationEditorGUI.py

:check_error
if %ERRORLEVEL% neq 0 (
    echo.
    echo [ERROR] Application exited with error: %ERRORLEVEL%
    pause
)
