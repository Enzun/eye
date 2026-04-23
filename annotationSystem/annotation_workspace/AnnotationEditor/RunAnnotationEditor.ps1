# RunAnnotationEditor.ps1
# PowerShell launcher - handles UNC paths natively (no drive mapping needed)
$annotEditorDir = $PSScriptRoot                         # ...AnnotationEditor\
$workspaceDir   = Split-Path $annotEditorDir -Parent    # ...annotation_workspace\
$sharedRoot     = Split-Path $workspaceDir   -Parent    # ...\(root of share)

$pythonExe = Join-Path $sharedRoot "python311\python.exe"
$scriptPy  = Join-Path $annotEditorDir "app\gui\AnnotationEditorGUI.py"

# Set env vars using drive-letter equivalent paths (PowerShell resolves cleanly)
$env:ANNOTEDITOR_WORKSPACE       = $workspaceDir
$env:QT_PLUGIN_PATH              = Join-Path $sharedRoot "python311\Lib\site-packages\PyQt5\Qt5\plugins"
$env:QT_QPA_PLATFORM_PLUGIN_PATH = Join-Path $sharedRoot "python311\Lib\site-packages\PyQt5\Qt5\plugins\platforms"

Write-Host "[INFO] Workspace : $workspaceDir"
Write-Host "[INFO] Python    : $pythonExe"

if (Test-Path $pythonExe) {
    Write-Host "[INFO] Using portable Python"
    & $pythonExe -B $scriptPy
} else {
    Write-Host "[WARN] python311 not found. Falling back to system Python."
    python -B $scriptPy
}

if ($LASTEXITCODE -ne 0) {
    Write-Host "[ERROR] Application exited with error: $LASTEXITCODE"
    Read-Host "Press Enter to exit"
}
