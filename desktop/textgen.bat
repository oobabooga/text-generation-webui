@echo off
set "APP=%~dp0__APP__"
for %%a in (%*) do (
    if /i "%%~a"=="--help" goto :help
    if /i "%%~a"=="-h" goto :help
    if /i "%%~a"=="--nowebui" goto :server
    if /i "%%~a"=="--listen" goto :server
)
"%APP%\electron\electron.exe" "%APP%" -- %*
exit /b %errorlevel%
:help
"%APP%\portable_env\python.exe" "%APP%\server.py" --help
exit /b %errorlevel%
:server
"%APP%\portable_env\python.exe" "%APP%\server.py" --portable --api %*
exit /b %errorlevel%
