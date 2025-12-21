@echo off
REM Creates a shortcut in the current user's Startup folder to run the web UI minimized at login.
set SCRIPT_PATH=%~dp0start_webui_minimized.ps1
necho Creating startup shortcut...
powershell -NoProfile -ExecutionPolicy Bypass -Command "$WshShell = New-Object -ComObject WScript.Shell; $startup = [Environment]::GetFolderPath('Startup'); $sc = $WshShell.CreateShortcut((Join-Path $startup 'TextGenWebUI.lnk')); $sc.TargetPath = 'powershell.exe'; $sc.Arguments = '-NoProfile -ExecutionPolicy Bypass -File "' + '%SCRIPT_PATH%' + '"'; $sc.IconLocation = '%~dp0venv\Scripts\python.exe'; $sc.WindowStyle = 7; $sc.Save()"
echo Shortcut created in your Startup folder.
echo To remove it, delete the file 'TextGenWebUI.lnk' from the Startup folder.
pause