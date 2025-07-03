@echo off
setlocal enabledelayedexpansion

@rem environment isolation
set PYTHONNOUSERSITE=1
set PYTHONPATH=
set PYTHONHOME=

cd /D "%~dp0"

@rem Portable install case
if exist "portable_env" (
    .\portable_env\python.exe server.py --portable --api --auto-launch %*
    exit /b %errorlevel%
)

set PATH=%PATH%;%SystemRoot%\system32

echo "%CD%"| findstr /C:" " >nul && echo This script relies on Miniforge which can not be silently installed under a path with spaces. && goto end

@rem Check for special characters in installation path
set "SPCHARMESSAGE="WARNING: Special characters were detected in the installation path!" "         This can cause the installation to fail!""
echo "%CD%"| findstr /R /C:"[!#\$%&()\*+,;<=>?@\[\]\^`{|}~]" >nul && (
	call :PrintBigMessage %SPCHARMESSAGE%
)
set SPCHARMESSAGE=

@rem fix failed install when installing to a separate drive
set TMP=%cd%\installer_files
set TEMP=%cd%\installer_files

@rem deactivate existing conda envs as needed to avoid conflicts
(call conda deactivate && call conda deactivate && call conda deactivate) 2>nul

@rem config
set INSTALL_DIR=%cd%\installer_files
set CONDA_ROOT_PREFIX=%cd%\installer_files\conda
set INSTALL_ENV_DIR=%cd%\installer_files\env
set MINIFORGE_DOWNLOAD_URL=https://github.com/conda-forge/miniforge/releases/download/25.3.0-3/Miniforge3-25.3.0-3-Windows-x86_64.exe
set MINIFORGE_CHECKSUM=b48cd98430170983076dfb51769a6d37668176f59bf3b59c4b21ac4c9bc24f39
set conda_exists=F

@rem figure out whether git and conda needs to be installed
call "%CONDA_ROOT_PREFIX%\_conda.exe" --version >nul 2>&1
if "%ERRORLEVEL%" EQU "0" set conda_exists=T

@rem (if necessary) install git and conda into a contained environment
@rem download conda
if "%conda_exists%" == "F" (
	echo Downloading Miniforge from %MINIFORGE_DOWNLOAD_URL% to %INSTALL_DIR%\miniforge_installer.exe

	mkdir "%INSTALL_DIR%"
	call curl -Lk "%MINIFORGE_DOWNLOAD_URL%" > "%INSTALL_DIR%\miniforge_installer.exe" || ( echo. && echo Miniforge failed to download. && goto end )

	@rem Try CertUtil first
	for /f %%a in ('CertUtil -hashfile "%INSTALL_DIR%\miniforge_installer.exe" SHA256 ^| find /i /v " " ^| find /i "%MINIFORGE_CHECKSUM%"') do (
		set "output=%%a"
	)

	@rem If CertUtil fails, try PowerShell
	if not defined output (
		for /f %%a in ('powershell -Command "if((Get-FileHash \"%INSTALL_DIR%\miniforge_installer.exe\" -Algorithm SHA256).Hash -eq ''%MINIFORGE_CHECKSUM%''){echo true}"') do (
			set "output=%%a"
		)
	)

	if not defined output (
		echo The checksum verification for miniforge_installer.exe has failed.
		del "%INSTALL_DIR%\miniforge_installer.exe"
		goto end
	) else (
		echo The checksum verification for miniforge_installer.exe has passed successfully.
	)

	echo Installing Miniforge to %CONDA_ROOT_PREFIX%
	start /wait "" "%INSTALL_DIR%\miniforge_installer.exe" /InstallationType=JustMe /NoShortcuts=1 /AddToPath=0 /RegisterPython=0 /NoRegistry=1 /S /D=%CONDA_ROOT_PREFIX%

	@rem test the conda binary
	echo Miniforge version:
	call "%CONDA_ROOT_PREFIX%\_conda.exe" --version || ( echo. && echo Miniforge not found. && goto end )

	@rem delete the Miniforge installer
	del "%INSTALL_DIR%\miniforge_installer.exe"
)

@rem create the installer env
if not exist "%INSTALL_ENV_DIR%" (
	echo Packages to install: %PACKAGES_TO_INSTALL%
	call "%CONDA_ROOT_PREFIX%\_conda.exe" create --no-shortcuts -y -k --prefix "%INSTALL_ENV_DIR%" python=3.11 || ( echo. && echo Conda environment creation failed. && goto end )
)

@rem check if conda environment was actually created
if not exist "%INSTALL_ENV_DIR%\python.exe" ( echo. && echo Conda environment is empty. && goto end )

set "CUDA_PATH=%INSTALL_ENV_DIR%"
set "CUDA_HOME=%CUDA_PATH%"

@rem activate installer env
call "%CONDA_ROOT_PREFIX%\condabin\conda.bat" activate "%INSTALL_ENV_DIR%" || ( echo. && echo Miniforge hook not found. && goto end )

@rem setup installer env
call python one_click.py %*

@rem below are functions for the script   next line skips these during normal execution
goto end

:PrintBigMessage
echo. && echo.
echo *******************************************************************
for %%M in (%*) do echo * %%~M
echo *******************************************************************
echo. && echo.
exit /b

:end
pause
