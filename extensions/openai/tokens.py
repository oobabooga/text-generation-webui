from modules.text_generation import decode, encode


def token_count(prompt):
    tokens = encode(prompt)[0]

    return {
        'results': [{
            'tokens': len(tokens)
        }]
    }


def token_encode(input, encoding_format):
    # if isinstance(input, list):
    tokens = encode(input)[0]

    return {
        'results': [{
            'tokens': tokens,
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
