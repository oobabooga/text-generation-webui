class OpenAIError(Exception):
    def __init__(self, message=None, code=500, internal_message=''):
        self.message = message
        self.code = code
        self.internal_message = internal_message

    def __repr__(self):
        return "%s(message=%r, code=%d)" % (
            self.__class__.__name__,
            self.message,
            self.code,
        )


class InvalidRequestError(OpenAIError):
    def __init__(self, message, param, code=400, error_type='InvalidRequestError', internal_message=''):
        super(OpenAIError, self).__init__(message, code, error_type, internal_message)
        self.param = param

    def __repr__(self):
        return "%s(message=%r, code=%d, param=%s)" % (
            self.__class__.__name__,
            self.message,
            self.code,
            self.param,
        )


class ServiceUnavailableError(OpenAIError):
    def __init__(self, message=None, code=500, error_type='ServiceUnavailableError', internal_message=''):
        super(OpenAIError, self).__init__(message, code, error_type, internal_message)
