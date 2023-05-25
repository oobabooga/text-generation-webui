import gradio as gr

from modules.logging_colors import logger


def ui():
    gr.Markdown("### This extension is deprecated, use \"multimodal\" extension instead")
    logger.error("LLaVA extension is deprecated, use \"multimodal\" extension instead")
