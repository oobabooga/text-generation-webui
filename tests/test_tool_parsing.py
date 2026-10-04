import unittest

from modules.tool_parsing import detect_tool_call_format, parse_tool_call


class BareToolCallTests(unittest.TestCase):
    def test_longer_name_with_shared_prefix(self):
        calls = parse_tool_call('search_web{"query":"weather"}', ['search', 'search_web'])

        self.assertEqual(calls, [{
            'type': 'function',
            'function': {'name': 'search_web', 'arguments': {'query': 'weather'}},
        }])

    def test_longer_name_with_template_selected_parsers(self):
        parsers, _, _ = detect_tool_call_format('[TOOL_CALLS]')
        calls = parse_tool_call(
            'search_web {"query":"weather"}', ['search', 'search_web'], parsers=parsers
        )

        self.assertEqual(calls[0]['function']['name'], 'search_web')

    def test_multiple_calls_with_shared_prefix(self):
        calls = parse_tool_call(
            'search{"query":"local"} search_web{"query":"remote"}', ['search', 'search_web']
        )

        self.assertEqual([call['function']['name'] for call in calls], ['search', 'search_web'])
        self.assertEqual([call['function']['arguments'] for call in calls], [
            {'query': 'local'}, {'query': 'remote'},
        ])

    def test_content_prefix_and_nested_arguments(self):
        prefix = '<think>Find the latest forecast.</think>Checking now. '
        calls, content = parse_tool_call(
            prefix + 'search_web{"options":{"query":"a } brace", "days":[1,2]}}',
            ['search', 'search_web'], return_prefix=True
        )

        self.assertEqual(content, prefix)
        self.assertEqual(calls[0]['function'], {
            'name': 'search_web',
            'arguments': {'options': {'query': 'a } brace', 'days': [1, 2]}},
        })

    def test_shorter_name_still_matches(self):
        calls = parse_tool_call('search{"query":"local"}', ['search', 'search_web'])

        self.assertEqual(calls[0]['function']['name'], 'search')

    def test_longer_name_registered_first(self):
        calls = parse_tool_call('search_web{"query":"remote"}', ['search_web', 'search'])

        self.assertEqual(calls[0]['function']['name'], 'search_web')

    def test_unknown_name_does_not_match_known_prefix(self):
        self.assertEqual(parse_tool_call('search_web{"query":"remote"}', ['search']), [])

    def test_incomplete_arguments_are_not_a_call(self):
        self.assertEqual(parse_tool_call('search_web{"query":', ['search', 'search_web']), [])

    def test_no_registered_tools_are_not_a_call(self):
        for answer in ['{"query":"weather"}', 'search_web{"query":"weather"}']:
            with self.subTest(answer=answer):
                self.assertEqual(parse_tool_call(answer, []), [])
                self.assertEqual(parse_tool_call(answer, [], return_prefix=True), ([], ''))


class OtherToolCallFormatsTests(unittest.TestCase):
    def test_shared_prefix_names_in_other_formats(self):
        answers = [
            '<|start|>assistant to=functions.search_web<|channel|>commentary json<|message|>{"query":"weather"}',
            '[TOOL_CALLS]search_web[ARGS]{"query":"weather"}',
            '<tool_call><function=search_web><parameter=query>weather</parameter></function></tool_call>',
            '<|tool_calls_section_begin|><|tool_call_begin|>functions.search_web:0<|tool_call_argument_begin|>{"query":"weather"}<|tool_call_end|><|tool_calls_section_end|>',
            '<minimax:tool_call><invoke name="search_web"><parameter name="query">weather</parameter></invoke></minimax:tool_call>',
            '<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>search_web<｜tool▁sep｜>{"query":"weather"}<｜tool▁call▁end｜><｜tool▁calls▁end｜>',
            '<tool_call>search_web<arg_key>query</arg_key><arg_value>weather</arg_value></tool_call>',
            '<|tool_call>call:search_web{query:<|"|>weather<|"|>}<tool_call|>',
            '[search_web(query="weather")]',
            '<tool_call>{"name":"search_web","arguments":{"query":"weather"}}</tool_call>',
            '{"name":"search_web","arguments":{"query":"weather"}}',
        ]
        for answer in answers:
            with self.subTest(answer=answer):
                calls = parse_tool_call(answer, ['search', 'search_web'])

                self.assertEqual(calls, [{
                    'type': 'function',
                    'function': {'name': 'search_web', 'arguments': {'query': 'weather'}},
                }])


if __name__ == '__main__':
    unittest.main()
