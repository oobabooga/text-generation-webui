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

    return translator.translate(string, dest='en').text

def output_modifier(string):
    """
    This function is applied to the model outputs.
    """

    return translator.translate(string, dest=params['language string']).text
