import os

import gradio as gr

# get the current directory of the script
current_dir = os.path.dirname(os.path.abspath(__file__))

# check if the bias_options.txt file exists, if not, create it
bias_file = os.path.join(current_dir, "bias_options.txt")
if not os.path.isfile(bias_file):
    with open(bias_file, "w") as f:
        f.write("*I am so happy*\n*I am so sad*\n*I am so excited*\n*I am so bored*\n*I am so angry*")

# read bias options from the text file
with open(bias_file, "r") as f:
    bias_options = [line.strip() for line in f.readlines()]

params = {
    "activate": True,
    "bias string": " *I am so happy*",
    "use custom string": False,
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
        if params['use custom string']:
            return f'{string} {params["custom string"].strip()} '
        else:
            return f'{string} {params["bias string"].strip()} '
    else:
        return string


def ui():
    # Gradio elements
    activate = gr.Checkbox(value=params['activate'], label='Activate character bias')
    dropdown_string = gr.Dropdown(choices=bias_options, value=params["bias string"], label='Character bias', info='To edit the options in this dropdown edit the "bias_options.txt" file')
    use_custom_string = gr.Checkbox(value=False, label='Use custom bias textbox instead of dropdown')
    custom_string = gr.Textbox(value="", placeholder="Enter custom bias string", label="Custom Character Bias", info='To use this textbox activate the checkbox above')

    # Event functions to update the parameters in the backend
    def update_bias_string(x):
        if x:
            params.update({"bias string": x})
        else:
            params.update({"bias string": dropdown_string.get()})
        return x

    def update_custom_string(x):
        params.update({"custom string": x})

    dropdown_string.change(update_bias_string, dropdown_string, None)
    custom_string.change(update_custom_string, custom_string, None)
    activate.change(lambda x: params.update({"activate": x}), activate, None)
    use_custom_string.change(lambda x: params.update({"use custom string": x}), use_custom_string, None)

    # Group elements together depending on the selected option
    def bias_string_group():
        if use_custom_string.value:
            return gr.Group([use_custom_string, custom_string])
        else:
            return dropdown_string
