import importlib
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

with patch.object(sys, "argv", [sys.argv[0]]):
    from modules import utils


def load_ui_file_saving():
    gradio_stub = types.ModuleType("gradio")
    gradio_stub.Error = type("Error", (Exception,), {})
    gradio_stub.update = Mock(side_effect=lambda **kwargs: kwargs)

    module_stubs = {
        "gradio": gradio_stub,
        "modules.chat": types.ModuleType("modules.chat"),
        "modules.presets": types.ModuleType("modules.presets"),
        "modules.ui": types.ModuleType("modules.ui"),
    }
    with patch.dict(sys.modules, module_stubs):
        module = importlib.import_module("modules.ui_file_saving")

    return module


ui_file_saving = load_ui_file_saving()


class TestValidateFilename(unittest.TestCase):
    def test_accepts_valid_filename(self):
        self.assertEqual(utils.validate_filename("My Preset"), "My Preset")

    def test_rejects_empty_filename(self):
        with self.assertRaisesRegex(ValueError, "not valid on all operating systems"):
            utils.validate_filename("")

    def test_rejects_colon(self):
        with self.assertRaisesRegex(ValueError, "not valid"):
            utils.validate_filename("Custom: min_p")

    def test_rejects_trailing_dot_or_space(self):
        for filename in ("My Preset.", "My Preset "):
            with self.subTest(filename=filename):
                with self.assertRaisesRegex(ValueError, "not valid"):
                    utils.validate_filename(filename)

    def test_rejects_path_components(self):
        for filename in ("../My Preset", "folder/My Preset", "folder\\My Preset"):
            with self.subTest(filename=filename):
                with self.assertRaisesRegex(ValueError, "not valid"):
                    utils.validate_filename(filename)

    def test_rejects_windows_device_names(self):
        filenames = (
            "CON", "prn", "Aux", "NUL.txt", "com1", "COM9.yaml", "lpt1", "LPT9.backup",
            "COM¹", "COM²", "COM³", "LPT¹", "LPT²", "LPT³", "COM¹.txt",
        )
        for filename in filenames:
            with self.subTest(filename=filename):
                with self.assertRaisesRegex(ValueError, "not valid"):
                    utils.validate_filename(filename)


class TestHandleSavePresetConfirmClick(unittest.TestCase):
    def setUp(self):
        ui_file_saving.gr.update.reset_mock()

    def test_writes_valid_preset(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            preset_dir = Path(temp_dir) / "presets"
            preset_dir.mkdir()
            contents = "temperature: 0.7\n"

            with (
                patch.object(ui_file_saving.shared, "user_data_dir", Path(temp_dir)),
                patch.object(ui_file_saving.utils, "get_available_presets", return_value=["My Preset"]),
            ):
                result = ui_file_saving.handle_save_preset_confirm_click("My Preset", contents)

            self.assertEqual((preset_dir / "My Preset.yaml").read_text(encoding="utf-8"), contents)
            self.assertEqual(result[0], {"choices": ["My Preset"], "value": "My Preset"})
            self.assertEqual(result[1], {"visible": False})

    def test_keeps_dialog_open_after_invalid_filename(self):
        with (
            patch.object(ui_file_saving.utils, "save_file") as save_file,
            self.assertRaisesRegex(ui_file_saving.gr.Error, "not valid"),
        ):
            ui_file_saving.handle_save_preset_confirm_click("folder/My Preset", "temperature: 0.7\n")

        save_file.assert_not_called()
        ui_file_saving.gr.update.assert_not_called()

    def test_keeps_dialog_open_after_write_failure(self):
        with (
            patch.object(ui_file_saving.utils, "save_file", side_effect=OSError("disk full")),
            self.assertRaisesRegex(ui_file_saving.gr.Error, "Check the server logs"),
        ):
            ui_file_saving.handle_save_preset_confirm_click("My Preset", "temperature: 0.7\n")

        ui_file_saving.gr.update.assert_not_called()


if __name__ == "__main__":
    unittest.main()
