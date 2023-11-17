from pathlib import Path

from modules.logging_colors import logger

if Path('../webui.py').exists():
    logger.warning('\nIt looks like you are running an outdated version of '
                   'the one-click-installers.\n'
                   'Please migrate your installation following the instructions here:\n'
                   'https://github.com/oobabooga/text-generation-webui/wiki/Migrating-an-old-one%E2%80%90click-install')
