#!/usr/bin/env bash

# This is basically like an rsync but without needing to install rsync

USERPKGS=/app/venv_user/lib/python3.10/site-packages
VENV=/app/venv/lib/python3.10/site-packages
EXTENSIONS=/app/extensions
export GIT_DISCOVERY_ACROSS_FILESYSTEM=1

git config --global --add safe.directory '*' || true

. app/venv/bin/activate

# If the directory /app/venv_user exists
if [ -d "/app/venv_user" ]; then

  echo "Copying any missing files from the cache ${USERPKGS}/ to ${VENV}/"
  cp -anR "${USERPKGS}/"* "${VENV}/"

  for d in "${EXTENSIONS}"/*/; do
    cd "$d" || exit
    if [ -d ".git" ]; then
      echo "Updating extension $d"
      git pull || true
    fi
    if [ -f "requirements.txt" ]; then
      pip install -r "requirements.txt" || true
      cd - || exit
    fi
  done
fi

# Copy any missing files from /app/venv to /app/user_venv, without overwriting any existing files
echo "Caching files in ${VENV}/ by copying to ${USERPKGS}/"
cp -anR "${VENV}/"* "${USERPKGS}/"
