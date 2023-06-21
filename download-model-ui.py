import json
import os
import re
import sys
from pathlib import Path
import importlib
import gradio as gr

model_name = 'None'
color_er = gr.themes.colors.blue

theme = gr.themes.Default(
    primary_hue=gr.themes.colors.orange,
    neutral_hue=gr.themes.colors.gray,
    secondary_hue=gr.themes.colors.pink,
    font=['Helvetica', 'ui-sans-serif', 'system-ui', 'sans-serif'],
    font_mono=['IBM Plex Mono', 'ui-monospace', 'Consolas', 'monospace'],
).set(

    # Colors
    input_background_fill_dark="*neutral_800",
    button_cancel_border_color=color_er.c200,
    button_cancel_border_color_dark=color_er.c600,
    button_cancel_text_color=color_er.c600,
    button_cancel_text_color_dark="white",
)

def download_model_wrapper(repo_id):
    try:
        downloader_module = importlib.import_module("download-model")
        downloader = downloader_module.ModelDownloader()
        repo_id_parts = repo_id.split(":")
        model = repo_id_parts[0] if len(repo_id_parts) > 0 else repo_id
        branch = repo_id_parts[1] if len(repo_id_parts) > 1 else "main"
        check = False

        yield ("Cleaning up the model/branch names")
        model, branch = downloader.sanitize_model_and_branch_names(model, branch)

        yield ("Getting the download links from Hugging Face")
        links, sha256, is_lora = downloader.get_download_links_from_huggingface(model, branch, text_only=False)

        yield ("Getting the output folder")
        output_folder = downloader.get_output_folder(model, branch, is_lora)

        if check:
            yield ("Checking previously downloaded files")
            downloader.check_model_files(model, branch, links, sha256, output_folder)
        else:
            yield (f"Downloading files to {output_folder}")
            downloader.download_model_files(model, branch, links, sha256, output_folder, threads=1)
            yield ("Done!")
    except:
        yield ("error")



with gr.Blocks(title="Downloader", theme=theme) as demo:
    gr.HTML("<h1>Download Model</h1>")
    custom_model_menu = gr.Textbox(label="Download custom model or LoRA",
                                                    info="Enter the Hugging Face username/model path, for instance: facebook/galactica-125m. To specify a branch, add it at the end after a \":\" character like this: facebook/galactica-125m:main")
    download_model_button = gr.Button("Download", variant='primary')
    model_status = gr.Markdown('No model is loaded' if model_name == 'None' else 'Ready')

    download_model_button.click(download_model_wrapper, custom_model_menu, model_status, show_progress=False)

demo.queue(concurrency_count=1)
demo.launch(server_port=7861)