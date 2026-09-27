import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from modules.reasoning import extract_reasoning


class ExtractReasoningTest(unittest.TestCase):
    def test_word_though_is_the_reply(self):
        self.assertEqual(extract_reasoning("though"), (None, "though"))
        self.assertEqual(extract_reasoning("thought"), (None, "thought"))
        self.assertEqual(extract_reasoning("though", html_escaped=True), (None, "though"))

    def test_sentence_with_thought_is_the_reply(self):
        text = "I thought\nthis was the answer"
        self.assertEqual(extract_reasoning(text), (None, text))

    def test_gemma_thought_prefix_still_splits(self):
        self.assertEqual(
            extract_reasoning("thought\nsecret<channel|>answer"),
            ("secret", "answer"),
        )
        self.assertEqual(
            extract_reasoning("\nthought\nsecret<channel|>answer"),
            ("secret", "answer"),
        )

    def test_partial_think_tag_stays_hidden(self):
        self.assertEqual(extract_reasoning("<thi"), ("", ""))

    def test_think_block_still_splits(self):
        self.assertEqual(
            extract_reasoning("<think>secret</think> answer"),
            ("secret", "answer"),
        )


if __name__ == "__main__":
    unittest.main()
