#!/bin/bash
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
APP="$DIR/__APP__"
PY="$APP/portable_env/bin/python3"
for arg; do
    case "$arg" in
        --help|-h)
            exec "$PY" "$APP/server.py" --help
            ;;
        --nowebui|--listen)
            exec "$PY" "$APP/server.py" --portable --api "$@"
            ;;
    esac
done
# --no-sandbox / --no-zygote needed on Linux: chrome-sandbox can't be SUID
# in an unzipped tarball, and the zygote's mount namespace hides /dev/shm
# and /tmp. Must be on the actual command line — appendSwitch in main.js
# runs too late on Ubuntu 24.04+ with restricted unprivileged userns.
if [[ "$(uname)" == "Linux" ]]; then
    exec "$APP/__ELECTRON__" --no-sandbox --no-zygote "$APP" -- "$@"
else
    exec "$APP/__ELECTRON__" "$APP" -- "$@"
fi
