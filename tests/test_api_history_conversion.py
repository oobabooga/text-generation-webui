import copy
import sys
import unittest
from unittest.mock import patch

with patch.object(sys, 'argv', [sys.argv[0]]):
    from modules import chat, shared
    from modules.api.completions import convert_history


def user_message(image_count=0):
    return {'role': 'user', 'content': [
        {'type': 'text', 'text': 'Describe this.'},
        *[{'type': 'image_url', 'image_url': {'url': 'data:image/png;base64,TEST'}}
          for _ in range(image_count)],
    ]}


class HistoryConversionTests(unittest.TestCase):
    def test_trailing_instruction_consumes_pending_user(self):
        for role in ('system', 'developer'):
            for image_count in (0, 1, 2):
                with self.subTest(role=role, image_count=image_count):
                    messages = [user_message(image_count), {'role': role, 'content': 'Be accurate.'}]
                    original = copy.deepcopy(messages)
                    pending, system, history = convert_history(messages)
                    self.assertEqual(pending, '')
                    self.assertEqual(system, '')
                    self.assertEqual(len(history['internal']), 2)
                    self.assertEqual(history['internal'][0][0].count('<__media__>'), image_count)
                    self.assertEqual(history['internal'][1], ['Be accurate.', '', '', {'role': 'system'}])
                    self.assertEqual(history['visible'], history['internal'])
                    self.assertEqual(history['messages'], original)
                    self.assertEqual(messages, original)

    def test_rendered_prompt_has_one_user_turn_and_one_marker_per_image(self):
        for image_count in (0, 1, 2):
            with self.subTest(image_count=image_count):
                pending, system, history = convert_history([
                    user_message(image_count), {'role': 'system', 'content': 'Be accurate.'},
                ])
                state = copy.deepcopy(shared.settings)
                state.update({
                    'mode': 'instruct',
                    'instruction_template_str': '{% for message in messages %}{{ message.role }}:{{ message.content }}\n{% endfor %}',
                    'chat_template_str': '',
                    'custom_system_message': system,
                    'history': history,
                })
                prompt = chat.generate_chat_prompt(pending, state)
                self.assertEqual(prompt.count('user:'), 1)
                self.assertEqual(prompt.count('<__media__>'), image_count)
                self.assertLess(prompt.index('Describe this.'), prompt.index('Be accurate.'))

    def test_multiple_trailing_instructions_preserve_order(self):
        pending, _, history = convert_history([
            user_message(1), {'role': 'system', 'content': 'First instruction.'},
            {'role': 'developer', 'content': 'Second instruction.'},
        ])
        self.assertEqual(pending, '')
        self.assertEqual([row[0] for row in history['internal'][1:]],
                         ['First instruction.', 'Second instruction.'])

    def test_final_user_after_instruction_is_still_pending(self):
        pending, _, history = convert_history([
            user_message(1), {'role': 'system', 'content': 'Be accurate.'},
            {'role': 'user', 'content': 'A follow-up question.'},
        ])
        self.assertEqual(pending, 'A follow-up question.')
        self.assertEqual(len(history['internal']), 2)

    def test_ordinary_image_request_remains_pending(self):
        pending, system, history = convert_history([
            {'role': 'system', 'content': 'Initial instruction.'}, user_message(1),
        ])
        self.assertEqual(pending.count('<__media__>'), 1)
        self.assertEqual(system, 'Initial instruction.')
        self.assertEqual(history['internal'], [])

    def test_leading_instructions_still_combine(self):
        pending, system, history = convert_history([
            {'role': 'system', 'content': 'First.'},
            {'role': 'developer', 'content': 'Second.'}, user_message(),
        ])
        self.assertEqual(pending, 'Describe this.')
        self.assertEqual(system, 'First.\nSecond.')
        self.assertEqual(history['internal'], [])

    def test_assistant_after_instruction_does_not_duplicate_user(self):
        pending, _, history = convert_history([
            user_message(1), {'role': 'system', 'content': 'Be accurate.'},
            {'role': 'assistant', 'content': 'An answer.'},
        ])
        self.assertEqual(pending, '')
        self.assertEqual(len(history['internal']), 3)
        self.assertEqual(history['internal'][2][1], 'An answer.')

    def test_tool_calls_and_results_are_preserved(self):
        tool_calls = [{'id': 'call_1', 'type': 'function',
                       'function': {'name': 'test', 'arguments': '{}'}}]
        pending, _, history = convert_history([
            user_message(),
            {'role': 'assistant', 'content': None, 'tool_calls': tool_calls},
            {'role': 'tool', 'content': 'Result', 'tool_call_id': 'call_1'},
            {'role': 'system', 'content': 'Continue checking.'},
        ])
        self.assertEqual(pending, '')
        self.assertEqual(history['internal'][0][3]['tool_calls'], tool_calls)
        self.assertEqual(history['internal'][1][2], 'Result')
        self.assertEqual(history['internal'][1][3]['tool_call_id'], 'call_1')


if __name__ == '__main__':
    unittest.main()
