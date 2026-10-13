import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import yaml
from PIL import Image

with patch.object(sys, 'argv', [sys.argv[0]]):
    from modules import chat, shared


class CharacterImportTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        (self.root / 'characters').mkdir()
        self.user_data = patch.object(shared, 'user_data_dir', self.root)
        self.user_data.start()
        self.addCleanup(self.user_data.stop)

    def upload(self, name):
        return chat.upload_tavern_character(None, {
            'name': name, 'description': 'A character', 'first_mes': 'Hello',
            'mes_example': '', 'scenario': '',
        })

    def test_windows_filename_characters(self):
        for index, char in enumerate('<>:"/\\|?*\x00\x1f'):
            with self.subTest(char=repr(char)):
                name = f'Character{index}{char}Name'
                result = self.upload(name)
                self.assertEqual(result['value'], f'Character{index}_Name')
                data = yaml.safe_load((self.root / 'characters' / f"{result['value']}.yaml").read_text())
                self.assertEqual(data['name'], name)

    def test_reserved_and_empty_names(self):
        for name, expected in [('CON', '_CON'), ('nul.txt', '_nul.txt'),
                               ('COM1', '_COM1'), ('lpt9', '_lpt9'),
                               ('Name. ', 'Name'), ('...', 'character')]:
            with self.subTest(name=name):
                self.assertEqual(self.upload(name)['value'], expected)

    def test_unicode_and_collision_with_image(self):
        image_path = self.root / 'avatar.png'
        Image.new('RGB', (2, 2)).save(image_path)
        data = {'name': '角色🌟:Name', 'greeting': 'Hello', 'context': 'A character'}
        first = chat.upload_character(json.dumps(data), str(image_path))
        second = chat.upload_character(json.dumps(data), str(image_path))
        self.assertEqual(first['value'], '角色🌟_Name')
        self.assertEqual(second['value'], '角色🌟_Name_001')
        for result in (first, second):
            path = self.root / 'characters' / result['value']
            self.assertTrue(path.with_name(path.name + '.png').is_file())
            saved = yaml.safe_load(path.with_name(path.name + '.yaml').read_text())
            self.assertEqual(saved['name'], data['name'])

    def test_ordinary_name_unchanged(self):
        self.assertEqual(self.upload('Alice')['value'], 'Alice')


if __name__ == '__main__':
    unittest.main()
