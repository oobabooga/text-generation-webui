from modules.text_generation import decode, encode


def token_count(prompt):
    tokens = encode(prompt)[0]
    return {
        'length': len(tokens)
    }


def token_encode(input):
    tokens = encode(input)[0]
    if tokens.__class__.__name__ in ['Tensor', 'ndarray']:
        tokens = tokens.tolist()

    return {
        'tokens': tokens,
        'length': len(tokens),
    }


def token_decode(tokens):
    output = decode(tokens)
    return {
        'text': output
    }
