import json
import unittest

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.testclient import TestClient
from pydantic import BaseModel, ValidationError, model_validator

from modules.api.validation import validation_error_response


class CompletionRequestFixture(BaseModel):
    prompt: str | list[str] | None = None
    messages: list[dict] | None = None

    @model_validator(mode='after')
    def validate_prompt_or_messages(self):
        if self.prompt is None and self.messages is None:
            raise ValueError("Either 'prompt' or 'messages' must be provided")
        return self


app = FastAPI()


@app.exception_handler(RequestValidationError)
async def validation_error_handler(request: Request, exc: RequestValidationError):
    return validation_error_response(request.url.path, exc.errors())


@app.post('/v1/completions')
async def completions(request: CompletionRequestFixture):
    return request


class ValidationErrorResponseTests(unittest.TestCase):
    def test_model_validator_context_is_json_serializable(self):
        response = TestClient(app, raise_server_exceptions=False).post(
            '/v1/completions',
            json={}
        )
        body = response.json()

        self.assertEqual(response.status_code, 422)
        self.assertEqual(
            body['detail'][0]['ctx']['error'],
            "Either 'prompt' or 'messages' must be provided"
        )

    def test_invalid_unicode_in_rejected_input_is_replaced(self):
        response = TestClient(app, raise_server_exceptions=False).post(
            '/v1/completions',
            content=b'{"messages":[[{"role":"R_WR6\\ud800"}]]}',
            headers={'content-type': 'application/json'}
        )
        body = response.json()

        self.assertEqual(response.status_code, 422)
        self.assertEqual(body['detail'][0]['input'][0]['role'], 'R_WR6\ufffd')

    def test_standard_validation_error_shape_is_preserved(self):
        with self.assertRaises(ValidationError) as raised:
            CompletionRequestFixture.model_validate({'prompt': 123})

        original_error = raised.exception.errors()[0]
        response = validation_error_response('/v1/completions', raised.exception.errors())
        returned_error = json.loads(response.body)['detail'][0]

        self.assertEqual(response.status_code, 422)
        self.assertEqual(returned_error['type'], original_error['type'])
        self.assertEqual(returned_error['loc'], list(original_error['loc']))
        self.assertEqual(returned_error['msg'], original_error['msg'])
        self.assertEqual(returned_error['input'], original_error['input'])

    def test_anthropic_validation_errors_keep_the_existing_format(self):
        errors = [{
            'type': 'string_type',
            'loc': ('body', 'model'),
            'msg': 'Input should be a valid string',
            'input': 123
        }]

        response = validation_error_response('/v1/messages', errors)

        self.assertEqual(response.status_code, 400)
        self.assertEqual(json.loads(response.body), {
            'type': 'error',
            'error': {
                'type': 'invalid_request_error',
                'message': 'body.model: Input should be a valid string'
            }
        })


if __name__ == '__main__':
    unittest.main()
