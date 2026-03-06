import sys
from pathlib import Path


def resolve_user_data_dir():
    """
    Resolve the user_data directory path. Order of precedence:
    1. --user-data-dir CLI flag (pre-parsed from sys.argv before argparse)
    2. Auto-detect: if ./user_data doesn't exist but ../user_data does, use ../user_data
    3. Default: 'user_data'
    """
    script_dir = Path(__file__).resolve().parent.parent

    # Check sys.argv for --user-data-dir before argparse runs
    for i, arg in enumerate(sys.argv):
        if arg == '--user-data-dir' and i + 1 < len(sys.argv):
            return Path(sys.argv[i + 1])
        elif arg.startswith('--user-data-dir='):
            return Path(arg.split('=', 1)[1])

    # Auto-detect: check if user_data exists locally vs one folder up
    local_path = script_dir / 'user_data'
    parent_path = script_dir.parent / 'user_data'

    if not local_path.exists() and parent_path.exists():
        return parent_path

    return Path('user_data')
