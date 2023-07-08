class OpenAIError(Exception):
    def __init__(self, message = None, code = 500, error_type ='APIError', internal_message = ''):
        self.message = message
        self.code = code
        self.error_type = error_type
        self.internal_message = internal_message

class InvalidRequestError(OpenAIError):
    def __init__(self, message, param, code = 400, error_type ='InvalidRequestError', internal_message = ''):
        super(OpenAIError, self).__init__(message, code, error_type, internal_message)
        self.param = param
        
class ServiceUnavailableError(OpenAIError):
    def __init__(self, message = None, code = 500, error_type ='ServiceUnavailableError', internal_message = ''):
        super(OpenAIError, self).__init__(message, code, error_type, internal_message)
