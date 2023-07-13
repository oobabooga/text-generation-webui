from extensions.openai.utils import float_list_to_base64
from modules.text_generation import encode, decode


def token_count(prompt):
    tokens = encode(prompt)[0]

    return {
        'results': [{
            'tokens': len(tokens)
        }]
    }


def token_encode(input, encoding_format=''):
    # if isinstance(input, list):
    tokens = encode(input)[0]

    return {
        'results': [{
            'encoding_format': encoding_format,
            'tokens': float_list_to_base64(tokens) if encoding_format == "base64" else tokens,
            'length': len(tokens),
        }]
    }


def token_decode(tokens, encoding_format):
    # if isinstance(input, list):
    #    if encoding_format == "base64":
    #         tokens = base64_to_float_list(tokens)
    output = decode(tokens)[0]

    return {
        'results': [{
            'text': output
        }]
    }
