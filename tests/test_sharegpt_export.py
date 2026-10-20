import copy
import json
import sys
import unittest
from unittest.mock import patch

with patch.object(sys, 'argv', ['test_sharegpt_export']):
    from modules import chat


class ShareGPTExportTests(unittest.TestCase):
    def state(self, rows, **kwargs):
        return dict(history={'internal': rows, 'visible': [], 'metadata': {}}, mode='chat',
                    context='{{char}} helps {{user}}.', custom_system_message='Be helpful.',
                    name1='Alice', name2='Assistant', **kwargs)

    def export(self, state):
        return json.loads(chat.export_history_sharegpt(state))[0]['conversations']

    def test_chat_roles_and_character_context(self):
        self.assertEqual(self.export(self.state([['Hi', 'Hello']])), [
            {'from': 'system', 'value': 'Assistant helps Alice.'},
            {'from': 'human', 'value': 'Hi'}, {'from': 'gpt', 'value': 'Hello'}])

    def test_instruct_uses_custom_system_message(self):
        state = self.state([['Question', 'Answer']])
        state['mode'] = 'instruct'
        self.assertEqual(self.export(state)[0], {'from': 'system', 'value': 'Be helpful.'})

    def test_chat_instruct_uses_character_context(self):
        state = self.state([])
        state['mode'] = 'chat-instruct'
        self.assertEqual(self.export(state), [{'from': 'system', 'value': 'Assistant helps Alice.'}])

    def test_greeting_empty_turns_and_unanswered_message(self):
        state = self.state([['<|BEGIN-VISIBLE-CHAT|>', 'Welcome'], ['', ''], ['Question', '']])
        state['context'] = ''
        self.assertEqual(self.export(state), [{'from': 'gpt', 'value': 'Welcome'},
                                              {'from': 'human', 'value': 'Question'}])

    def test_raw_text_and_history_are_preserved(self):
        state = self.state([['你好\n<script>&', '  reply\n']])
        original = copy.deepcopy(state)
        self.assertEqual(self.export(state)[1]['value'], '你好\n<script>&')
        self.assertEqual(self.export(state)[2]['value'], '  reply\n')
        self.assertEqual(state, original)

    def test_tool_call_and_response(self):
        calls = [{'id': 'call_1', 'type': 'function', 'function': {'name': 'search', 'arguments': '{}'}}]
        state = self.state([['Find it', '', 'Found it', {'tool_calls': calls, 'tool_call_id': 'call_1'}], ['', 'Done']])
        turns = self.export(state)
        self.assertEqual(turns[2], {'from': 'gpt', 'value': '', 'tool_calls': calls})
        self.assertEqual(turns[3], {'from': 'tool', 'value': 'Found it', 'tool_call_id': 'call_1'})
        self.assertEqual(turns[4], {'from': 'gpt', 'value': 'Done'})

    def test_empty_history(self):
        state = self.state([])
        state['context'] = '  '
        self.assertEqual(self.export(state), [])


if __name__ == '__main__':
    unittest.main()
