params = {
    "input suffix": " *I say as I make a funny face*",
    "bot prefix": " *I speak in a cute way*",
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

def bot_prefix_modifier(string):
    """
    This function is only applied in chat mode. It modifies
    the prefix text for the Bot and can be used to bias its
    behavior.
    """

    return string + params["bot prefix"]
