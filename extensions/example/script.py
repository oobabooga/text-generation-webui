params = {
    "input suffix": " *I say as I make a funny face*",
}

def input_modifier(string):
    """
    This function is applied to your text inputs before
    they are fed into the model.
    """ 

    return string + params["input suffix"]

def output_modifier(string):
    """
    This function is applied to the model outputs.
    """

    return string
