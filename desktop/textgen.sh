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
exec "$APP/__ELECTRON__" "$APP" -- "$@"
