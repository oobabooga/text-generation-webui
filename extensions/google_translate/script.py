from googletrans import Translator

translator = Translator()

params = {
    "language string": "ja",
}

def input_modifier(string):
    """
    This function is applied to your text inputs before
    they are fed into the model.
    """ 

    return translator.translate(string, src=params['language string'], dest='en').text

def output_modifier(string):
    """
    This function is applied to the model outputs.
    """

    return translator.translate(string, src="en", dest=params['language string']).text

def bot_prefix_modifier(string):
    """
    This function is only applied in chat mode. It modifies
    the prefix text for the Bot and can be used to bias its
    behavior.
    """

    return string
