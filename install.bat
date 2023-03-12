@echo off

@rem Based on the installer found here: https://github.com/Sygil-Dev/sygil-webui
@rem This script will install git and conda (if not found on the PATH variable)
@rem using micromamba (an 8mb static-linked single-file binary, conda replacement).
@rem This enables a user to install this project without manually installing conda and git.

echo What is your GPU?
echo.
echo A) NVIDIA
echo B) None (I want to run in CPU mode)
echo.
set /p "gpuchoice=Input> "
set gpuchoice=%gpuchoice:~0,1%
setlocal enabledelayedexpansion
set gpuchoice=!gpuchoice:a=A!
set gpuchoice=!gpuchoice:b=B!

if "%gpuchoice%" == "A" (
    set "PACKAGES_TO_INSTALL=torchvision torchaudio pytorch-cuda=11.7 conda git"
    set "CHANNEL=-c nvidia"
) else if "%gpuchoice%" == "B" (
    set "PACKAGES_TO_INSTALL=torchvision torchaudio pytorch conda git"
    set "CHANNEL="
) else (
    echo Invalid choice. Exiting...
    exit
)

set MAMBA_ROOT_PREFIX=%cd%\installer_files\mamba
set INSTALL_ENV_DIR=%cd%\installer_files\env
set MICROMAMBA_DOWNLOAD_URL=https://github.com/cmdr2/stable-diffusion-ui/releases/download/v1.1/micromamba.exe
set REPO_URL=https://github.com/oobabooga/text-generation-webui.git
set umamba_exists=F

@rem figure out whether git and conda needs to be installed
if exist "%INSTALL_ENV_DIR%" set PATH=%INSTALL_ENV_DIR%;%INSTALL_ENV_DIR%\Library\bin;%INSTALL_ENV_DIR%\Scripts;%INSTALL_ENV_DIR%\Library\usr\bin;%PATH%
call "%MAMBA_ROOT_PREFIX%\micromamba.exe" --version >.tmp1 2>.tmp2
if "%ERRORLEVEL%" EQU "0" set umamba_exists=T

@rem (if necessary) install git and conda into a contained environment
if "%PACKAGES_TO_INSTALL%" NEQ "" (
    @rem download micromamba
    if "%umamba_exists%" == "F" (
        echo "Downloading micromamba from %MICROMAMBA_DOWNLOAD_URL% to %MAMBA_ROOT_PREFIX%\micromamba.exe"

        mkdir "%MAMBA_ROOT_PREFIX%"
        call curl -L "%MICROMAMBA_DOWNLOAD_URL%" > "%MAMBA_ROOT_PREFIX%\micromamba.exe"

        @rem test the mamba binary
        echo Micromamba version:
        call "%MAMBA_ROOT_PREFIX%\micromamba.exe" --version
    )

    @rem create the installer env
    if not exist "%INSTALL_ENV_DIR%" (
        call "%MAMBA_ROOT_PREFIX%\micromamba.exe" create -y --prefix "%INSTALL_ENV_DIR%"
    )

    echo "Packages to install: %PACKAGES_TO_INSTALL%"

    call "%MAMBA_ROOT_PREFIX%\micromamba.exe" install -y --prefix "%INSTALL_ENV_DIR%" -c conda-forge -c pytorch %CHANNEL% %PACKAGES_TO_INSTALL%
    call "%MAMBA_ROOT_PREFIX%\micromamba.exe" install -y --prefix "%INSTALL_ENV_DIR%" -c conda-forge -c pytorch %CHANNEL% %PACKAGES_TO_INSTALL%

    if not exist "%INSTALL_ENV_DIR%" (
        echo "There was a problem while installing%PACKAGES_TO_INSTALL% using micromamba. Cannot continue."
        pause
        exit /b
    )
)

set PATH=%INSTALL_ENV_DIR%;%INSTALL_ENV_DIR%\Library\bin;%INSTALL_ENV_DIR%\Scripts;%INSTALL_ENV_DIR%\Library\usr\bin;%PATH%

@rem clone the repository and install the pip requirements
call conda activate
if exist text-generation-webui\ (
  cd text-generation-webui
  git pull
) else (
  git clone https://github.com/oobabooga/text-generation-webui.git
  cd text-generation-webui
)
call python -m pip install -r requirements.txt
call python -m pip install -r extensions\google_translate\requirements.txt
call python -m pip install -r extensions\silero_tts\requirements.txt

cd ..
del .tmp1 .tmp2

pause
