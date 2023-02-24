import extensions
import modules.shared as shared
import gradio as gr

state = {}
available_extensions = []

def load_extensions():
    global state
    for i, name in enumerate(shared.args.extensions):
        if name in available_extensions:
            print(f'Loading the extension "{name}"... ', end='')
            import_string = f"extensions.{name}.script"
            exec(f"import {import_string}")
            state[name] = [True, i]
            print(f'Ok.')

def iterator():
    for name in sorted(state, key=lambda x : state[x][1]):
        if state[name][0] == True:
            yield eval(f"extensions.{name}.script"), name

def apply_extensions(text, typ):
    for extension, _ in iterator():
        if typ == "input" and hasattr(extension, "input_modifier"):
            text = extension.input_modifier(text)
        elif typ == "output" and hasattr(extension, "output_modifier"):
            text = extension.output_modifier(text)
        elif typ == "bot_prefix" and hasattr(extension, "bot_prefix_modifier"):
            text = extension.bot_prefix_modifier(text)
    return text

def update_extensions_parameters(*args):
    i = 0
    for extension, _ in iterator():
        for param in extension.params:
            if len(args) >= i+1:
                extension.params[param] = eval(f"args[{i}]")
                i += 1

def create_extensions_block():
    extensions_ui_elements = []
    default_values = []
    if not (shared.args.chat or shared.args.cai_chat):
        gr.Markdown('## Extensions parameters')
    for extension, name in iterator():
        for param in extension.params:
            _id = f"{name}-{param}"
            default_value = shared.settings[_id] if _id in shared.settings else extension.params[param]
            default_values.append(default_value)
            if type(extension.params[param]) == str:
                extensions_ui_elements.append(gr.Textbox(value=default_value, label=f"{name}-{param}"))
            elif type(extension.params[param]) in [int, float]:
                extensions_ui_elements.append(gr.Number(value=default_value, label=f"{name}-{param}"))
            elif type(extension.params[param]) == bool:
                extensions_ui_elements.append(gr.Checkbox(value=default_value, label=f"{name}-{param}"))

    update_extensions_parameters(*default_values)
    btn_extensions = gr.Button("Apply")
    btn_extensions.click(update_extensions_parameters, [*extensions_ui_elements], [])
