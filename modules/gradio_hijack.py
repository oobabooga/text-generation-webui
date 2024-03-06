import gradio as gr


def Box(*args, **kwargs):
    return gr.Blocks(*args, **kwargs)


if not hasattr(gr, 'Box'):
    gr.Box = Box
