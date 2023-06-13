@echo off

cd /D "%~dp0"

set PATH=%PATH%;%SystemRoot%\system32

@rem sed -i 's/\x0D$//' ./wsl.sh converts newlines to unix format in the wsl script
call wsl -e bash -lic "sed -i 's/\x0D$//' ./wsl.sh; source ./wsl.sh"

:end
pause
