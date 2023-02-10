params = {
    "soft prompt": " *I speak in an annoyingly cute way*",
}

def input_modifier(string):
    """
    This function is applied to your text inputs before
    they are fed into the model.
    """ 

    return string

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

    return string + params["soft prompt"]
