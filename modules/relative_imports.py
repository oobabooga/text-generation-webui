import sys
from pathlib import Path


class RelativeImport:
    def __init__(self, path):
        self.import_path = Path(path)

    def __enter__(self):
        sys.path.insert(0, str(self.import_path))

    def __exit__(self, exc_type, exc_value, traceback):
        sys.path.remove(str(self.import_path))
