# From https://github.com/AUTOMATIC1111/stable-diffusion-webui

import glob
import json
import os
from modules import shared
from modules.logging_colors import logger


def list_localizations(dirname):
    shared.localizations.clear()

    for file in os.listdir(dirname):
        fn, ext = os.path.splitext(file)
        if ext.lower() != ".json":
            continue

        shared.localizations[fn] = [os.path.join(dirname, file)]

    for file in glob.glob(os.path.join(dirname, "extensions", "*", "localizations", "*.json")):
        fn, ext = os.path.splitext(file.filename)
        if fn not in shared.localizations:
            shared.localizations[fn] = []
        shared.localizations[fn].append(file.path)


def localization_js(current_localization_name: str) -> str:
    fns = shared.localizations.get(current_localization_name, None)
    data = {}
    if fns is not None:
        for fn in fns:
            try:
                with open(fn, "r", encoding="utf8") as file:
                    data.update(json.load(file))
                    print(f"Loaded localization from {fn}")
            except Exception:
                logger.error(f"Error loading localization from {fn}")
    return f"window.localization = {json.dumps(data)};"