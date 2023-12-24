import openai
from fastapi import HTTPException

class OpenAIError(HTTPException):
    def __init__(self, message, param, code):
        body = {
            'message' : message,
            'code': code,
            'param' : param
        }
        super().__init__(status_code=code, detail=body)
        self.message = message
        self.code = code
        self.param = param

    def __repr__(self):
        return "%s(message=%r, code=%d, param=%s)" % (
            self.__class__.__name__,
            self.message,
            self.code,
            self.param,
        )


class InvalidRequestError(OpenAIError):
    def __init__(self, message, param, code=400):
        super().__init__(message, param, code)


class ServiceUnavailableError(OpenAIError):
    def __init__(self, message="Service unavailable, please try again later.", param='', code=503):
        super().__init__(message, param, code)
