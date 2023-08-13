#!/usr/bin/env bash

# A simple script that you can call to install any extensions that have been added with a requirements.txt file in their subdirectory.

USERPKGS=/app/venv_user/lib/python3.10/site-packages
VENV=/app/venv/lib/python3.10/site-packages

source /app/venv/bin/activate

# If the directory /app/venv_user exists
if [ -d "/app/venv_user" ]; then

  echo "Copying any missing files from ${USERPKGS}/ to ${VENV}/"
  cp -R -n "${USERPKGS}/"* "${VENV}/"

  for d in "${USERPKGS}"/*/; do
    cd "${VENV}/$d" || exit
    git pull || true
    if [ -f "requirements.txt" ]; then
      pip install -r "${VENV}/$d/requirements.txt" || true
      cd - || exit
    fi
  done
fi
