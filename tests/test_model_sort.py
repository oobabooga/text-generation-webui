import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

with patch.object(sys, 'argv', ['test_model_sort']):
    from modules import shared, utils


class ModelSortTests(unittest.TestCase):
    def test_newest_first_and_default_order(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(shared.args, 'model_dir', tmp):
            for name, modified in [('model2.gguf', 200), ('model10.gguf', 300), ('model1.gguf', 100)]:
                path = Path(tmp) / name
                path.touch()
                os.utime(path, (modified, modified))
            self.assertEqual(utils.get_available_models(), ['model1.gguf', 'model2.gguf', 'model10.gguf'])
            self.assertEqual(utils.get_available_models('modified'), ['model10.gguf', 'model2.gguf', 'model1.gguf'])

    def test_directories_and_split_models(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(shared.args, 'model_dir', tmp):
            directory = Path(tmp) / 'transformers-model'
            directory.mkdir()
            (directory / 'model.safetensors').touch()
            for name in ['split-00001-of-00002.gguf', 'split-00002-of-00002.gguf']:
                path = Path(tmp) / name
                path.touch()
                os.utime(path, (100, 100))
            os.utime(directory, (200, 200))
            self.assertEqual(utils.get_available_models('modified'), ['transformers-model', 'split-00001-of-00002.gguf'])

    def test_equal_timestamps_preserve_default_order(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(shared.args, 'model_dir', tmp):
            for name in ['model10.gguf', 'model2.gguf']:
                path = Path(tmp) / name
                path.touch()
                os.utime(path, (100, 100))
            self.assertEqual(utils.get_available_models('modified'), utils.get_available_models())

    def test_missing_file_during_sort(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(shared.args, 'model_dir', tmp), patch.object(utils, 'get_available_ggufs', return_value=['gone.gguf']):
            self.assertEqual(utils.get_available_models('modified'), ['gone.gguf'])


if __name__ == '__main__':
    unittest.main()
