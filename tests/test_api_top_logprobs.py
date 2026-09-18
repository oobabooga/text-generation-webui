import sys
import unittest

from pydantic import ValidationError

sys.argv = [sys.argv[0]]

from modules.api.typing import ChatCompletionRequest, validate_top_logprobs


class TestTopLogprobsValidation(unittest.TestCase):
    def test_accepts_boundary_values(self):
        for value in (0, 20):
            request = ChatCompletionRequest(
                messages=[{'role': 'user', 'content': 'hello'}], top_logprobs=value
            )
            self.assertEqual(request.top_logprobs, value)

    def test_rejects_values_outside_supported_range(self):
        for value in (-1, 21):
            with self.assertRaises(ValidationError):
                ChatCompletionRequest(
                    messages=[{'role': 'user', 'content': 'hello'}], top_logprobs=value
                )

    def test_internal_validation_rejects_values_outside_supported_range(self):
        for value in (-1, 21):
            with self.assertRaises(ValueError):
                validate_top_logprobs(value)


if __name__ == '__main__':
    unittest.main()
