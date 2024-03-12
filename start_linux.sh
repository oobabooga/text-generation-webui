#!/bin/bash

export LANG=zh_CN.UTF-8

cd "$(dirname "${BASH_SOURCE[0]}")"

if [[ "$(pwd)" =~ " " ]]; then echo 此脚本依赖Miniconda，而它无法在包含空格的路径下静默安装。 && exit; fi

# deactivate existing conda envs as needed to avoid conflicts
{ conda deactivate && conda deactivate && conda deactivate; } 2> /dev/null

OS_ARCH=$(uname -m)
case "${OS_ARCH}" in
    x86_64*)    OS_ARCH="x86_64";;
    arm64*)     OS_ARCH="aarch64";;
    aarch64*)     OS_ARCH="aarch64";;
    *)          echo "未知系统架构： $OS_ARCH！此脚本只在x86_64或arm64运行" && exit
esac

# config
INSTALL_DIR="$(pwd)/installer_files"
CONDA_ROOT_PREFIX="$(pwd)/installer_files/conda"
INSTALL_ENV_DIR="$(pwd)/installer_files/env"
MINICONDA_DOWNLOAD_URL="https://mirrors.tuna.tsinghua.edu.cn/anaconda/miniconda/Miniconda3-py310_23.3.1-0-Linux-${OS_ARCH}.sh"
conda_exists="F"

# figure out whether git and conda needs to be installed
if "$CONDA_ROOT_PREFIX/bin/conda" --version &>/dev/null; then conda_exists="T"; fi

# (if necessary) install git and conda into a contained environment
# download miniconda
if [ "$conda_exists" == "F" ]; then
    echo "正在从 $MINICONDA_DOWNLOAD_URL 下载Miniconda至 $INSTALL_DIR/miniconda_installer.sh"

    mkdir -p "$INSTALL_DIR"
    curl -L "$MINICONDA_DOWNLOAD_URL" > "$INSTALL_DIR/miniconda_installer.sh"

    chmod u+x "$INSTALL_DIR/miniconda_installer.sh"
    bash "$INSTALL_DIR/miniconda_installer.sh" -b -p $CONDA_ROOT_PREFIX

    # test the conda binary
    echo "Miniconda版本："
    "$CONDA_ROOT_PREFIX/bin/conda" --version

    # delete the Miniconda installer
    rm "$INSTALL_DIR/miniconda_installer.sh"
fi

# create the installer env
if [ ! -e "$INSTALL_ENV_DIR" ]; then
    "$CONDA_ROOT_PREFIX/bin/conda" create -y -k --prefix "$INSTALL_ENV_DIR" -c https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/ python=3.11
fi

# check if conda environment was actually created
if [ ! -e "$INSTALL_ENV_DIR/bin/python" ]; then
    echo "Conda 环境未创建。"
    exit
fi

# environment isolation
export PYTHONNOUSERSITE=1
unset PYTHONPATH
unset PYTHONHOME
export CUDA_PATH="$INSTALL_ENV_DIR"
export CUDA_HOME="$CUDA_PATH"

# activate installer env
source "$CONDA_ROOT_PREFIX/etc/profile.d/conda.sh" # otherwise conda complains about 'shell not initialized' (needed when running in a script)
conda activate "$INSTALL_ENV_DIR"

# setup installer env
python one_click.py $@
