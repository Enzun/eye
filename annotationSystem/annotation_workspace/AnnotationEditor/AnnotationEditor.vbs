Set WshShell = CreateObject("WScript.Shell")
WshShell.Run """" & Replace(WScript.ScriptFullName, WScript.ScriptName, "") & "RunAnnotationEditor.bat""", 0, False
