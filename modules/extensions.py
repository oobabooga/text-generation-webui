import extensions
import modules.shared as shared

state = {}
available_extensions = []

def load_extensions():
    global state
    for i, name in enumerate(shared.args.extensions):
        if name in available_extensions:
            print(f'Loading the extension "{name}"... ', end='')
            exec(f"import extensions.{name}.script")
            state[name] = [True, i]
            print('Ok.')

# This iterator returns the extensions in the order specified in the command-line
def iterator():
    for name in sorted(state, key=lambda x : state[x][1]):
        if state[name][0] == True:
            yield eval(f"extensions.{name}.script"), name

# Extension functions that map string -> string
def apply_extensions(text, typ, additional_params={}):
    for extension, _ in iterator():
        if typ == "input" and hasattr(extension, "input_modifier"):
            text = extension.input_modifier(text, additional_params)
        elif typ == "output" and hasattr(extension, "output_modifier"):
            text = extension.output_modifier(text, additional_params)
        elif typ == "bot_prefix" and hasattr(extension, "bot_prefix_modifier"):
            text = extension.bot_prefix_modifier(text, additional_params)
        elif typ == "load_history" and hasattr(extension, "load_history_modifier"):
            text = extension.load_history_modifier(text, additional_params)
        elif typ == "load_visible_history" and hasattr(extension, "load_visible_history_modifier"):
            text = extension.load_visible_history_modifier(text, additional_params)
    return text

def create_extensions_block():
    # Updating the default values
    for extension, name in iterator():
        if hasattr(extension, 'params'):
            for param in extension.params:
                _id = f"{name}-{param}"
                if _id in shared.settings:
                    extension.params[param] = shared.settings[_id]

    # Creating the extension ui elements
    for extension, name in iterator():
        if hasattr(extension, "ui"):
            extension.ui()
