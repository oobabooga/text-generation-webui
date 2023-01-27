def input_modifier(string):
    """
    This function is applied to your text inputs before
    they are fed into the model.
    """ 

    return string.replace(' ', '#')

def output_modifier(string):
    """
    This function is applied to the model outputs.
    """

    return string.replace(' ', '_')
