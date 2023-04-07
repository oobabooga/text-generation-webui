import gradio as gr

params = {
    "activate": True,
    "bias string": " *I am so happy*",
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

    if params['activate']:
        return f'{string} {params["bias string"].strip()} '
    else:
        return string


def ui():
    # Gradio elements
    activate = gr.Checkbox(value=params['activate'], label='Activate character bias')
    string = gr.Textbox(value=params["bias string"], label='Character bias')

    # Event functions to update the parameters in the backend
    string.change(lambda x: params.update({"bias string": x}), string, None)
    activate.change(lambda x: params.update({"activate": x}), activate, None)
