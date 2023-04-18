#!/bin/bash

cd "$(dirname "${BASH_SOURCE[0]}")"

if [[ "$(pwd)" =~ " " ]]; then echo This script relies on Miniconda which can not be silently installed under a path with spaces. && exit; fi

# config
CONDA_ROOT_PREFIX="$(pwd)/installer_files/conda"
INSTALL_ENV_DIR="$(pwd)/installer_files/env"

# activate installer env
source "$CONDA_ROOT_PREFIX/etc/profile.d/conda.sh" # otherwise conda complains about 'shell not initialized' (needed when running in a script)
conda activate "$INSTALL_ENV_DIR"

# update installer env
python webui.py --update

echo
echo "Done!"
