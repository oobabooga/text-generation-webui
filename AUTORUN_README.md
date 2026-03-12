Install and autorun for text-generation-webui (Windows)

This folder contains helper scripts to install the web UI as an autorun service on your Windows user account.

Files:
- `start_webui_minimized.ps1` — PowerShell script that launches the web UI server using the venv Python; starts minimized.
- `install_autostart.bat` — Creates a shortcut in your Windows Startup folder to run the PowerShell script at login.

How to install (one-time):
1. Double-click `install_autostart.bat` and allow it to run. This will create a shortcut named `TextGenWebUI.lnk` in your Startup folder.
2. The next time you log into Windows, the web UI will start automatically minimized using the project's venv.

Start now (manual):
- To start immediately without waiting for login, run the PowerShell script manually (right-click and select "Run with PowerShell") or run:
  powershell -NoProfile -ExecutionPolicy Bypass -File "start_webui_minimized.ps1"

Stop/remove autorun:
- Delete `TextGenWebUI.lnk` from your Startup folder. You can open the Startup folder with:
  explorer.exe shell:startup

Notes & troubleshooting:
- The scripts assume the project is at `D:\AI_ML_Development\text-generation-webui` and the venv exists at `venv` inside the project. If you move the project, edit `start_webui_minimized.ps1`.
- Ensure your PowerShell ExecutionPolicy allows running scripts. `install_autostart.bat` sets `-ExecutionPolicy Bypass` when launching the script.
- Logs: by default the script starts the server without redirecting output. To capture logs, edit `start_webui_minimized.ps1` and use the `-RedirectStandardOutput` / `-RedirectStandardError` options (commented in file).
