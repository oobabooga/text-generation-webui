import sys
from pathlib import Path


def resolve_user_data_dir():
    """
    Resolve the user_data directory path. Order of precedence:
    1. --user-data-dir CLI flag (pre-parsed from sys.argv before argparse)
    2. --portable + ../app exists: ../../user_data (electron build, where
       the project lives one level deeper than in classic portable mode)
    3. --portable: ../user_data
    4. Default: 'user_data'
    """
    script_dir = Path(__file__).resolve().parent.parent

    # Check sys.argv for --user-data-dir before argparse runs
    for i, arg in enumerate(sys.argv):
        if arg == '--user-data-dir' and i + 1 < len(sys.argv):
            return Path(sys.argv[i + 1])
        elif arg.startswith('--user-data-dir='):
            return Path(arg.split('=', 1)[1])

    is_portable = '--portable' in sys.argv
    if is_portable:
        # Electron build: ../app exists alongside the project; prefer ../../user_data.
        if (script_dir.parent / 'app').exists():
            electron_user_data = script_dir.parent.parent / 'user_data'
            if electron_user_data.exists():
                return electron_user_data

        # Classic portable build: ../user_data.
        parent_path = script_dir.parent / 'user_data'
        if parent_path.exists():
            return parent_path

    return Path('user_data')
