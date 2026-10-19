import sys
import unittest
from unittest.mock import Mock, patch

from PIL import Image

from modules.image_utils import convert_pil_to_base64, decode_base64_image


class ImageEncodingTests(unittest.TestCase):
    def encode(self, image, **kwargs):
        return decode_base64_image(convert_pil_to_base64(image, **kwargs))

    def test_default_preserves_png_pixels_and_size(self):
        image = Image.new('RGBA', (80, 40), (10, 20, 30, 40))
        result = self.encode(image)
        self.assertEqual(result.format, 'PNG')
        self.assertEqual(result.size, image.size)
        self.assertEqual(result.tobytes(), image.tobytes())

    def test_resize_preserves_aspect_ratio_and_source(self):
        image = Image.new('RGB', (200, 100))
        self.assertEqual(self.encode(image, max_dimension=50).size, (50, 25))
        self.assertEqual(image.size, (200, 100))
        self.assertEqual(self.encode(image, max_dimension=400).size, (200, 100))

    def test_jpeg_composites_transparency_on_white(self):
        image = Image.new('RGBA', (40, 80), (0, 0, 0, 0))
        result = self.encode(image, image_format='jpeg', max_dimension=40)
        self.assertEqual(result.format, 'JPEG')
        self.assertEqual(result.size, (20, 40))
        self.assertEqual(result.mode, 'RGB')
        self.assertEqual(result.getpixel((10, 10)), (255, 255, 255))

    def test_palette_transparency_and_grayscale_jpeg(self):
        image = Image.new('P', (10, 10))
        image.info['transparency'] = 0
        self.assertEqual(self.encode(image, image_format='jpeg').getpixel((0, 0)), (255, 255, 255))
        self.assertEqual(self.encode(Image.new('L', (10, 10)), image_format='jpeg').mode, 'RGB')

    def test_invalid_options(self):
        image = Image.new('RGB', (10, 10))
        for kwargs in ({'max_dimension': -1}, {'image_format': 'gif'}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                self.encode(image, **kwargs)


class ImageRequestTests(unittest.TestCase):
    def test_generation_encodes_images_from_each_input_source(self):
        with patch.object(sys, 'argv', ['test_image_encoding']):
            from modules import shared
            from modules.llama_cpp_server import LlamaServer

        image = Image.new('RGB', (200, 100), 'red')
        encoded = convert_pil_to_base64(image)
        sources = [
            {'raw_images': [image]},
            {'image_attachments': [{'type': 'image', 'image_data': encoded}]},
            {'history': {'messages': [{'role': 'user', 'content': [
                {'type': 'image_url', 'image_url': {'url': 'data:image/png;base64,' + encoded}}
            ]}]}},
        ]
        server = LlamaServer.__new__(LlamaServer)
        server.process = None
        server.port = 8080
        server.media_marker = '<__media__>'
        server.prepare_payload = Mock(return_value={})
        server.encode = Mock(return_value=[1])
        server.session = Mock()
        server.session.post.return_value.iter_lines.return_value = []
        state = {'add_bos_token': False, 'auto_max_new_tokens': False, 'max_new_tokens': 10}
        with patch.object(shared, 'is_multimodal', True), patch.object(shared.args, 'image_format', 'jpeg'), patch.object(shared.args, 'image_max_dimension', 50):
            for source in sources:
                with self.subTest(source=list(source)):
                    list(server.generate_with_streaming('<__media__>', state | source))
                    payload = server.session.post.call_args.kwargs['json']
                    images = payload['prompt']['multimodal_data']
                    self.assertEqual(len(images), 1)
                    result = decode_base64_image(images[0])
                    self.assertEqual(result.format, 'JPEG')
                    self.assertEqual(result.size, (50, 25))


if __name__ == '__main__':
    unittest.main()
