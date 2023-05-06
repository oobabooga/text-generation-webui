import textwrap
from .provider import PseudocontextProvider
import gradio as gr

provider = PseudocontextProvider()

data = ''

def input_modifier(string):
    """
    This function is applied to your text inputs before
    they are fed into the model.
    """

    return provider.with_pseudocontext(string)

def ui():
    with gr.Accordion("Extension Configuration", open=False):
        with gr.Tab("Changelog"):
            gr.Markdown(textwrap.dedent("""
            ## Changelog
            v0.0.1-alpha
            - Added 'best source' searching to find the best source among all added sources
            - Condensed the formatted prompts to a single type
            - Modified the project structure to make it easy to export as a package
            """).strip())
        with gr.Tab('Info'):
            gr.Markdown(textwrap.dedent("""
            # SuperBIG
            This extension has two parts:
            - Pseudocontext: Adds a virtual context on top of your real context, so you can use as many characters/tokens as you want in your input!
            - (TBD) Focus: Adds operations that optimizes your context for your model (Not implemented yet!)
            
            ## Quick FAQ
            - I want to add documents such as TXTs, PDFs, or HTMLs to my context:
                You can write filepaths and urls directly in the context: 
                1. Head to `Settings > Injection`
                2. Check `Allow sources to be inferred from raw text`
                3. Make sure to review the default selections. You can uncheck any patterns you don't want SuperBIG to automatically infer
                4. Type a path/URL anywhere in your prompt and hit generate!
                
                Alternatively, you can add sources and manage them manually: 
                1. Head to the `Sources` tab and add whatever sources you'd like to use. Give your source a name that will be used to inject it into the prompt
                2. Look at the list of sources, taking note of the source name. Copy the source name into your prompt wherever you'd like it to be injected
                3. Hit Generate!
                
            - How can I see what SuperBIG is doing under-the-hood?
                You can enable logging in the `Settings` tab:
                1. Head to `Settings > Logging` and configure the settings based on your preferred log verbosity
                2. You can also view Logs in the Log Viewer (Settings > Logging > Log Viewer)
            
            - How much memory is this consuming?
                You can view memory consumption and tune performance in `Settings`:
                1. Head to `Settings > Performance`
                2. The `Database` section contains info on resources consumed by the in-memory database
                3. `Tokens` gives you information on each generation, including both the virtual and real number of tokens
            """).strip())
        with gr.Tab('Sources'):
            pass
        with gr.Tab('Settings'):
            pass
