from deep_translator import GoogleTranslator

params = {
    "language string": "ja",
}

def input_modifier(string):
    """
    This function is applied to your text inputs before
    they are fed into the model.
    """ 

    return GoogleTranslator(source=params['language string'], target='en').translate(string)

def output_modifier(string):
    """
    This function is applied to the model outputs.
    """

    return GoogleTranslator(source='en', target=params['language string']).translate(string)

def bot_prefix_modifier(string):
    """
    This function is only applied in chat mode. It modifies
    the prefix text for the Bot and can be used to bias its
    behavior.
    """

    return string
