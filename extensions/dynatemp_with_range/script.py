import gradio as gr

params = {
    "activate": True,
    "minimum_temperature": 0.1,
    "maximum_temperature": 2,
}

def convert_to_dynatemp():
    temperature = 0.5 * (params["minimum_temperature"] + params["maximum_temperature"])
    dynatemp = params["maximum_temperature"] - temperature
    return temperature, dynatemp


def state_modifier(state):
    """
    Modifies the state variable, which is a dictionary containing the input
    values in the UI like sliders and checkboxes.
    """

    if params["activate"]:
        temperature, dynatemp = convert_to_dynatemp()

        state["temperature"] = temperature
        state["dynatemp"] = dynatemp 

    return state


def generate_info():
    temperature, dynatemp = convert_to_dynatemp()
    return f"The combination above is equivalent to: T={temperature:.2f}, dynatemp={dynatemp:.2f}"


def ui():
    activate = gr.Checkbox(value=params['activate'], label='Activate Dynamic Temperature Range', info='When checked, the default temperature/dynatemp parameters are ignored and the parameters below are used instead.')
    with gr.Row():
        minimum_temperature = gr.Slider(0, 5, step=0.01, label="Minimum temperature", value=params["minimum_temperature"], interactive=True)
        maximum_temperature = gr.Slider(0, 5, step=0.01, label="Maximum temperature", value=params["maximum_temperature"], interactive=True)

    info = gr.HTML(generate_info())

    activate.change(lambda x: params.update({"activate": x}), activate, None)
    minimum_temperature.change(
        lambda x: params.update({"minimum_temperature": x}), minimum_temperature, None).then(
        generate_info, None, info, show_progress=False)

    maximum_temperature.change(
        lambda x: params.update({"maximum_temperature": x}), maximum_temperature, None).then(
        generate_info, None, info, show_progress=False)
