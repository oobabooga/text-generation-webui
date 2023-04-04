from pathlib import Path

import gradio as gr

from modules import shared
from modules import ui as _ui

params = {
    'template': '%input%'
}

def get_available_templates():
    return ['None'] + sorted(set((k.stem for k in Path('extensions/prompt_template/templates').glob('*.txt'))), key=str.lower)

def load_template(fname):
    if fname in ['None', '']:
        return '%input%'
    else:
        with open(Path(f'extensions/prompt_template/templates/{fname}.txt'), 'r', encoding='utf-8') as f:
            text = f.read()
            if text[-1] == '\n':
                text = text[:-1]
            return text

def input_modifier(string):
    """
    This function is applied to your text inputs before
    they are fed into the model.
    """ 

    return params['template'].replace('%input%', string)

def output_modifier(string):
    return f'\n{string}'

def setup():
    shared.args.verbose = True

def ui():
    # Gradio elements

    with gr.Row():
        with gr.Column():
            template = gr.Textbox(value=params['template'], info="%input% will be replaced with your user input.", label='Template')
        with gr.Column():
            with gr.Row():
                template_menu = gr.Dropdown(choices=get_available_templates(), value='None', label='Available templates')
                _ui.create_refresh_button(shared.gradio['model_menu'], lambda : None, lambda : {'choices': get_available_templates()}, 'refresh-button')

    template_menu.change(load_template, template_menu, template)
    template.change(lambda x: params.update({"template": x}), template, None)
