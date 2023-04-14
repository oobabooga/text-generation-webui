from pathlib import Path
import json
import gradio as gr

# Initialize JS and CSS
#   arrive.js - https://raw.githubusercontent.com/uzairfarooq/arrive/cfabddbd2633a866742e98c88ba5e4b75cb5257b/minified/arrive.min.js
#     [SHA256 - 5971DE670AEF1D6F90A63E6ED8D095CA22F95C455FFC0CEB60BE62E30E1A4473]
#
#   highlight.js - https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.7.0/highlight.min.js
#     [SHA256 - 9F19CEBC1D4441AE1F0FFC696A42628B9F865FE5D99DEDA1F1E8CD5BEC878888]
assets_dir = Path(__file__).resolve().parent / 'assets'
with open(assets_dir / 'arrive.min.js', 'r') as f:
    js_modules = '\n' + f.read() + '\n'
with open(assets_dir / 'highlight.min.js', 'r') as f:
    js_modules += f.read() + '\n'
with open(assets_dir / 'main.js', 'r') as f:
    js_modules += f.read() + '\n'
with open(assets_dir / 'github.css', 'r') as f:
    css_theme_light = f.read()
with open(assets_dir / 'github-dark.css', 'r') as f:
    css_theme_dark = f.read()

# Define extension config with global params - https://github.com/oobabooga/text-generation-webui/wiki/Extensions#params-dictionary
params = {
    'activate': True,
    'inline_highlight': False,
    'performance_mode': False,
    'performance_mode_timeout_time': 800,
}

# JS script to initialize the params for JS modules
def js_params_loader():
    return f'''
      document.getElementById('code-syntax-highlight').setAttribute('params', JSON.stringify({json.dumps(params)}));
    '''

# JS script to update the specified param in the params for JS modules
def js_params_updater(paramName):
    return '(x) => { ' + f'''
      const paramName = '{paramName}';
    ''' + '''
      const element = document.getElementById('code-syntax-highlight');
      const params = JSON.parse(element.getAttribute('params'));
      params[paramName] = x;
      element.setAttribute('params', JSON.stringify(params));
    ''' + ' }'

# Build UI and inject CSS and JS
def ui():
    # Load CSS
    gr.HTML(value='<style id="hljs-theme-light" media="not all">' + css_theme_light + '</style>', visible=False)
    gr.HTML(value='<style id="hljs-theme-dark" media="not all">' + css_theme_dark + '</style>', visible=False)
    # Create a DOM element to be used as proxy between Gradio and the injected JS modules
    gr.HTML(value='<code-syntax-highlight id="code-syntax-highlight" style="display: none;"> </code-syntax-highlight>', visible=False)
    # Load JS, instead of using shared.gradio['interface'], we create a new interface to avoid conflicts with the current and future scripts (like ui.main_js and ui.chat_js)
    with gr.Blocks(analytics_enabled=False) as interface:
        interface.load(None, None, None, _js=f'() => {{{js_params_loader()+js_modules}}}')
    # Display extension settings on the Gradio UI
    with gr.Accordion('Settings', elem_id="code-syntax-highlight_accordion", open=True):
        # Setting: activate
        activate = gr.Checkbox(value=params['activate'], label='Enable syntax highlighting of code snippets')
        activate.change(lambda x: params.update({'activate': x}), activate, _js=js_params_updater('activate'))
        # Setting: inline_highlight
        inline_highlight = gr.Checkbox(value=params['inline_highlight'], label='Highlight inline code snippets')
        inline_highlight.change(lambda x: params.update({'inline_highlight': x}), inline_highlight, _js=js_params_updater('inline_highlight'))
        # Setting: performance_mode
        performance_mode = gr.Checkbox(value=params['performance_mode'], label='Reduce CPU usage at the cost of delayed syntax highlighting')
        performance_mode.change(lambda x: params.update({'performance_mode': x}), performance_mode, _js=js_params_updater('performance_mode'))
        # Settings text and accordion style
        gr.HTML(value='''
          <p style="margin-bottom: 0;">Changes are applied during the next text generation or UI reload</p>
          <style>
            #code-syntax-highlight_accordion > div {
              gap: var(--spacing-lg, 8px) !important;
            }
          </style>
        ''')
