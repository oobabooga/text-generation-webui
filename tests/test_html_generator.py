import sys
import unittest
from unittest.mock import patch

with patch.object(sys, 'argv', [sys.argv[0]]):
    from modules.html_generator import process_markdown_content


class HtmlGeneratorLatexBackslashTest(unittest.TestCase):
    def test_preserves_double_backslashes_in_latex_formats(self):
        cases = (
            '$$\nx \\\\ y\n$$',
            '\\[x \\\\ y\\]',
            '\\(x \\\\ y\\)',
            '$x \\\\ y$',
        )

        for source in cases:
            with self.subTest(source=source):
                self.assertIn('x \\\\ y', process_markdown_content(source))

    def test_keeps_latex_command_backslashes_single(self):
        source = '$$\nA = \\begin{bmatrix}\na & b \\\\ \nc & d\n\\end{bmatrix}\n$$'
        result = process_markdown_content(source)

        self.assertIn('\\begin{bmatrix}', result)
        self.assertNotIn('\\\\begin{bmatrix}', result)
        self.assertIn('a &amp; b \\\\', result)
        self.assertIn('\\end{bmatrix}', result)
        self.assertNotIn('\\\\end{bmatrix}', result)

    def test_preserves_double_backslashes_before_inline_math_delimiter(self):
        result = process_markdown_content('$x \\\\$')

        self.assertIn('$x \\\\$', result)

    def test_avoids_placeholder_collisions(self):
        source = (
            '$$\nLATEXDOUBLEBACKSLASHPLACEHOLDER0END '
            'LATEXDOUBLEBACKSLASHPLACEHOLDER1END x \\\\ y\n$$'
        )
        result = process_markdown_content(source)

        self.assertIn('LATEXDOUBLEBACKSLASHPLACEHOLDER0END', result)
        self.assertIn('LATEXDOUBLEBACKSLASHPLACEHOLDER1END', result)
        self.assertIn('x \\\\ y', result)

    def test_avoids_placeholder_collisions_after_html_unescape(self):
        source = (
            '<code>LATEXDOUBLEBACKSLASHPLACEHOLDER&#48;END</code>\n\n'
            '$$x \\\\ y$$'
        )
        result = process_markdown_content(source)

        self.assertIn('<code>LATEXDOUBLEBACKSLASHPLACEHOLDER0END</code>', result)
        self.assertIn('$$x \\\\ y$$', result)

    def test_keeps_plain_text_backslash_behavior(self):
        source = 'ordinary C:\\temp path and double \\\\ separators'
        expected = '<p>ordinary C:\\temp path and double \\ separators</p>'

        self.assertEqual(expected, process_markdown_content(source))

    def test_keeps_escaped_dollar_text_backslash_behavior(self):
        source = 'Cost is \\$5, path C:\\\\tmp, then \\$10'
        expected = '<p>Cost is \\$5, path C:\\tmp, then \\$10</p>'

        self.assertEqual(expected, process_markdown_content(source))

    def test_keeps_code_backslash_behavior(self):
        expected_fence = '<pre><code class="language-latex">$$\nx \\\\ y\n$$</code></pre>'
        cases = (
            ('```latex\n$$\nx \\\\ y\n$$\n```', expected_fence),
            ('~~~latex\n$$\nx \\\\ y\n$$\n~~~', expected_fence),
            ('`$x \\\\ y$`', '<p><code>$x \\\\ y$</code></p>'),
        )

        for source, expected in cases:
            with self.subTest(source=source):
                self.assertEqual(expected, process_markdown_content(source))


if __name__ == '__main__':
    unittest.main()
