from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse


def _replace_invalid_unicode(value):
    if isinstance(value, str):
        return ''.join(
            '\ufffd' if 0xD800 <= ord(character) <= 0xDFFF else character
            for character in value
        )

    if isinstance(value, list):
        return [_replace_invalid_unicode(item) for item in value]

    if isinstance(value, dict):
        return {
            _replace_invalid_unicode(key): _replace_invalid_unicode(item)
            for key, item in value.items()
        }

    return value


def validation_error_response(path, errors):
    errors = jsonable_encoder(errors, custom_encoder={Exception: str})
    errors = _replace_invalid_unicode(errors)

    if path.startswith('/v1/messages'):
        messages = '; '.join(
            f"{'.'.join(str(location) for location in error['loc'])}: {error['msg']}"
            for error in errors
        )
        return JSONResponse(
            status_code=400,
            content={"type": "error", "error": {"type": "invalid_request_error", "message": messages}}
        )

    return JSONResponse(status_code=422, content={"detail": errors})
