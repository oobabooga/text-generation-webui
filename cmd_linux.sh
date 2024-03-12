#!/bin/bash

export LANG=zh_CN.UTF-8

cd "$(dirname "${BASH_SOURCE[0]}")"

if [[ "$(pwd)" =~ " " ]]; then echo 此脚本依赖Miniconda，而它无法在包含空格的路径下静默安装。 && exit; fi

# deactivate existing conda envs as needed to avoid conflicts
{ conda deactivate && conda deactivate && conda deactivate; } 2> /dev/null

# config
CONDA_ROOT_PREFIX="$(pwd)/installer_files/conda"
INSTALL_ENV_DIR="$(pwd)/installer_files/env"

# environment isolation
export PYTHONNOUSERSITE=1
unset PYTHONPATH
unset PYTHONHOME
export CUDA_PATH="$INSTALL_ENV_DIR"
export CUDA_HOME="$CUDA_PATH"

# activate env
bash --init-file <(echo "source \"$CONDA_ROOT_PREFIX/etc/profile.d/conda.sh\" && conda activate \"$INSTALL_ENV_DIR\"")
