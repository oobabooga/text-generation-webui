import unittest

from modules.api.errors import InvalidRequestError
from modules.api.message_validation import SUPPORTED_CHAT_ROLES, validate_chat_messages


class ChatMessageValidationTests(unittest.TestCase):
    def assert_invalid(self, messages, expected_message):
        with self.assertRaises(InvalidRequestError) as raised:
            validate_chat_messages(messages)

        self.assertEqual(raised.exception.code, 400)
        self.assertEqual(raised.exception.param, 'messages')
        self.assertEqual(raised.exception.message, expected_message)

    def test_rejects_non_string_and_unknown_roles(self):
        self.assert_invalid(
            [{'role': 0, 'content': 0}],
            'role: 0 is not supported.'
        )
        self.assert_invalid(
            [{'role': 'unknown', 'content': 'hello'}],
            'role: unknown is not supported.'
        )

    def test_accepts_supported_roles(self):
        messages = [
            {'role': role, 'content': 'hello'}
            for role in SUPPORTED_CHAT_ROLES
        ]

        validate_chat_messages(messages)

    def test_preserves_assistant_tool_call_without_text(self):
        messages = [{
            'role': 'assistant',
            'content': None,
            'tool_calls': [{'id': 'call-1'}]
        }]

        validate_chat_messages(messages)

        self.assertEqual(messages[0]['content'], '')

    def test_rejects_non_string_text_content(self):
        self.assert_invalid(
            [{'role': 'user', 'content': [{'type': 'text', 'text': 0}]}],
            'messages: text content must be a string'
        )

    def test_rejects_malformed_image_url_content(self):
        self.assert_invalid(
            [{'role': 'user', 'content': [{'type': 'image_url', 'image_url': 0}]}],
            'messages: missing image_url in content item'
        )
        self.assert_invalid(
            [{'role': 'user', 'content': [{'type': 'image_url', 'image_url': {'url': 0}}]}],
            'messages: image_url must contain a string URL'
        )

    def test_accepts_valid_multimodal_content(self):
        messages = [{
            'role': 'user',
            'content': [
                {'type': 'text', 'text': 'Describe this image.'},
                {'type': 'image_url', 'image_url': {'url': 'data:image/png;base64,AA==', 'detail': 'auto'}}
            ]
        }]

        validate_chat_messages(messages)

    def test_keeps_existing_missing_field_errors(self):
        self.assert_invalid(
            [{'content': 'hello'}],
            'messages: missing role'
        )
        self.assert_invalid(
            [{'role': 'user'}],
            'messages: missing content'
        )


if __name__ == '__main__':
    unittest.main()
