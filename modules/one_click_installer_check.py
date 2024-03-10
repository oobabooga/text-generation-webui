from pathlib import Path

from modules.logging_colors import logger

if Path('../webui.py').exists():
    logger.warning('\n看起来您正在运行的是过时版本的一键安装程序。\n'
               '请按照以下说明迁移您的安装：\n'
               'https://github.com/Touch-Night/text-generation-webui/wiki/Migrating-an-old-one%E2%80%90click-install')
