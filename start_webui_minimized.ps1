# Start text-generation-webui in the project's venv, minimized.
# Created by automation. To edit, update this file in the project folder.

$ProjectPath = 'D:\AI_ML_Development\text-generation-webui'
$Python = Join-Path $ProjectPath 'venv\Scripts\python.exe'

# Launch the server in portable mode with API enabled.
# Use Minimized window style so the console doesn't pop up.
Start-Process -FilePath $Python -ArgumentList 'server.py','--portable','--api' -WorkingDirectory $ProjectPath -WindowStyle Minimized

# Optional: redirect output to a log file instead of showing console.
# To enable logging, uncomment the following lines and edit the log path.
# $log = Join-Path $ProjectPath 'autorun.log'
# Start-Process -FilePath $Python -ArgumentList 'server.py','--portable','--api' -WorkingDirectory $ProjectPath -WindowStyle Minimized -RedirectStandardOutput $log -RedirectStandardError $log
