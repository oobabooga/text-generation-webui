@echo off

set INSTALL_ENV_DIR=%cd%\installer_files\env
set PATH=%INSTALL_ENV_DIR%;%INSTALL_ENV_DIR%\Library\bin;%INSTALL_ENV_DIR%\Scripts;%INSTALL_ENV_DIR%\Library\usr\bin;%PATH%
call conda activate
cd text-generation-webui
python download-model.py

pause
