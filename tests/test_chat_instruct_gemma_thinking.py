"""Prompt-level regression tests without loading model/UI dependencies.

Run with: python -m unittest discover -s tests -p test_chat_instruct_gemma_thinking.py
Requires Jinja2, already a runtime dependency of chat.py.
"""
import ast
import copy
import unittest
from functools import partial
from pathlib import Path
from types import SimpleNamespace

from jinja2.ext import loopcontrols
from jinja2.sandbox import ImmutableSandboxedEnvironment


ROOT = Path(__file__).resolve().parents[1]
# Compile the real prompt builder and its lightweight helpers, not a copy of
# their implementation. This avoids importing Gradio and model backends.
NAMES = {'generate_chat_prompt', 'get_compiled_template', 'replace_character_names'}
source = ast.parse((ROOT / 'modules/chat.py').read_text())
functions = ast.Module(body=[node for node in source.body if isinstance(node, ast.FunctionDef) and node.name in NAMES], type_ignores=[])
namespace = {
    'copy': copy,
    'partial': partial,
    'shared': SimpleNamespace(bos_token='', eos_token='', tokenizer=None),
    'jinja_env': ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True, extensions=[loopcontrols]),
    '_template_cache': {},
    'apply_extensions': lambda name, value, state: value,
    '_expand_tool_sequence': lambda sequence: [],
    'THINKING_FORMATS': [('<|channel>thought', '<channel|>', None)],
}
exec(compile(functions, str(ROOT / 'modules/chat.py'), 'exec'), namespace)
generate_chat_prompt = namespace['generate_chat_prompt']

CHAT_TEMPLATE = "{% for message in messages %}{{ message['role'] }}: {{ message['content'] }}\n{% endfor %}{% if add_generation_prompt %}assistant: {% endif %}"
# Like the canonical Gemma template, assistant-only stripping cannot reach
# dialogue embedded in an outer user message.
INSTRUCT_TEMPLATE = "{% for message in messages %}<start_of_turn>{{ message['role'] }}\n{{ message['content'] }}<end_of_turn>\n{% endfor %}{% if add_generation_prompt %}<start_of_turn>model\n<|channel>thought\n{% endif %}"


def make_state(reply, **overrides):
    state = dict(mode='chat-instruct', chat_template_str=CHAT_TEMPLATE,
                 instruction_template_str=INSTRUCT_TEMPLATE, name1='User', name2='Bot',
                 user_bio='', context='', custom_system_message='', enable_thinking=True,
                 reasoning_effort='medium', preserve_thinking=False,
                 history={'internal': [['Question', reply]]},
                 **{'chat-instruct_command': 'Continue this conversation:\n<|prompt|>'})
    state.update(overrides)
    return state


class GemmaThinkingHistoryTests(unittest.TestCase):
    def test_discards_prior_gemma_thinking(self):
        for opener in ('<|channel>thought\n', 'thought\n'):
            with self.subTest(opener=opener):
                state = make_state(opener + 'PRIVATE_THOUGHT<channel|>Visible answer')
                original = copy.deepcopy(state)
                prompt = generate_chat_prompt('Next question', state)
                self.assertNotIn('PRIVATE_THOUGHT', prompt)
                self.assertIn('Visible answer', prompt)
                self.assertIn('Next question', prompt)
                self.assertEqual(state, original)

    def test_preserve_thinking_keeps_raw_history(self):
        reply = '<|channel>thought\nPRIVATE_THOUGHT<channel|>Visible answer'
        self.assertIn(reply, generate_chat_prompt('Next', make_state(reply, preserve_thinking=True)))

    def test_continue_preserves_current_reply_but_not_older_thinking(self):
        for tail in ('<|channel>thought\nCURRENT_THOUGHT',
                     '<|channel>thought\nCURRENT_THOUGHT<channel|>Partial answer',
                     'thought\nCURRENT_THOUGHT<channel|>Partial answer'):
            with self.subTest(tail=tail):
                state = make_state(tail)
                state['history']['internal'].insert(0, ['Earlier', '<|channel>thought\nOLD_THOUGHT<channel|>Old answer'])
                prompt = generate_chat_prompt('', state, _continue=True)
                self.assertNotIn('OLD_THOUGHT', prompt)
                self.assertIn('Old answer', prompt)
                self.assertTrue(prompt.endswith(tail[tail.index('CURRENT_THOUGHT'):]))

    def test_discards_unfinished_tagged_thinking_in_old_turn(self):
        prompt = generate_chat_prompt('Next', make_state('<|channel>thought\nPRIVATE_THOUGHT'))
        self.assertNotIn('PRIVATE_THOUGHT', prompt)

    def test_ordinary_text_is_unchanged(self):
        for reply in ('though', 'thought', 'thought\nAn ordinary answer',
                      'I thought\nthis was the answer', 'Quoted <|channel>thought\nexample<channel|> text'):
            with self.subTest(reply=reply):
                self.assertIn(reply, generate_chat_prompt('Next', make_state(reply)))

    def test_other_modes_keep_existing_gemma_handling(self):
        reply = '<|channel>thought\nPRIVATE_THOUGHT<channel|>Visible answer'
        for mode in ('chat', 'instruct'):
            with self.subTest(mode=mode):
                self.assertIn(reply, generate_chat_prompt('Next', make_state(reply, mode=mode)))

    def test_user_text_is_not_stripped(self):
        user_text = '<|channel>thought\nUser-provided example<channel|>User answer'
        state = make_state('Ordinary answer')
        state['history']['internal'][0][0] = user_text
        self.assertIn(user_text, generate_chat_prompt('Next', state))

    def test_gpt_oss_and_seed_history_behavior_is_unchanged(self):
        for reply in ('<|channel|>analysis<|message|>PRIVATE_THOUGHT<|end|><|channel|>final<|message|>Visible answer',
                      '<seed:think>PRIVATE_THOUGHT</seed:think>Visible answer'):
            with self.subTest(reply=reply):
                prompt = generate_chat_prompt('Next', make_state(reply))
                self.assertNotIn('PRIVATE_THOUGHT', prompt)
                self.assertIn('Visible answer', prompt)

    def test_qwen_history_behavior_is_unchanged(self):
        for reply in ('<think>PRIVATE_THOUGHT</think>Visible answer', 'PRIVATE_THOUGHT</think>Visible answer'):
            with self.subTest(reply=reply):
                prompt = generate_chat_prompt('Next', make_state(reply))
                self.assertNotIn('PRIVATE_THOUGHT', prompt)
                self.assertIn('Visible answer', prompt)


if __name__ == '__main__':
    unittest.main()
