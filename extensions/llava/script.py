import logging
import gradio as gr

def ui():
    text = gr.Markdown("### This extension is deprecated, use \"multimodal\" extension instead")
    logging.error("LLaVA extension is deprecated, use \"multimodal\" extension instead")
