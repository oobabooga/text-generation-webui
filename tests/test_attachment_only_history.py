import copy
import sys
import unittest
from unittest.mock import patch

with patch.object(sys, 'argv', [sys.argv[0]]):
    from modules import chat, shared


class AttachmentOnlyHistoryTest(unittest.TestCase):
    def prompt(self, attachments, user_text='', include_past=True):
        state = copy.deepcopy(shared.settings)
        state.update({
            'mode': 'instruct',
            'instruction_template_str': '{% for message in messages %}{{ message.role }}:{{ message.content }}\n{% endfor %}',
            'chat_template_str': '',
            'include_past_attachments': include_past,
            'history': {
                'internal': [[user_text, 'First answer']],
                'metadata': {'user_0': {'attachments': attachments}},
            },
        })
        return chat.generate_chat_prompt('Next question', state)

    def test_image_only_turn_keeps_marker_on_next_request(self):
        for include_past in (True, False):
            with self.subTest(include_past=include_past):
                prompt = self.prompt([{'type': 'image'}], include_past=include_past)
                self.assertEqual(prompt, 'user:<__media__>\n\n\nassistant:First answer\nuser:Next question\n')

    def test_text_attachment_only_turn_keeps_contents(self):
        prompt = self.prompt([{'type': 'text/plain', 'name': 'notes.txt', 'content': 'Attached notes'}])
        self.assertIn('user:\n\nATTACHMENTS:', prompt)
        self.assertIn('Attached notes', prompt)

    def test_disabled_past_text_attachments_do_not_add_empty_turn(self):
        prompt = self.prompt([{'type': 'text/plain', 'content': 'Hidden'}], include_past=False)
        self.assertEqual(prompt, 'assistant:First answer\nuser:Next question\n')

    def test_empty_turn_and_greeting_remain_omitted(self):
        for text in ('', '<|BEGIN-VISIBLE-CHAT|>'):
            with self.subTest(text=text):
                self.assertEqual(self.prompt([], text), 'assistant:First answer\nuser:Next question\n')

    def test_text_and_image_turn_retains_both(self):
        self.assertIn('user:<__media__>\n\nDescribe it\n', self.prompt([{'type': 'image'}], 'Describe it'))


if __name__ == '__main__':
    unittest.main()
