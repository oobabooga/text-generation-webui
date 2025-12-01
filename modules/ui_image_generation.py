import os
import time
import traceback
from datetime import datetime
from pathlib import Path

import gradio as gr
import numpy as np
import torch

from modules import shared, ui, utils
from modules.image_models import load_image_model, unload_image_model
from modules.logging_colors import logger
from modules.utils import gradio

ASPECT_RATIOS = {
    "1:1 Square": (1, 1),
    "16:9 Cinema": (16, 9),
    "9:16 Mobile": (9, 16),
    "4:3 Photo": (4, 3),
    "Custom": None,
}

STEP = 32


def round_to_step(value, step=STEP):
    return round(value / step) * step


def clamp(value, min_val, max_val):
    return max(min_val, min(max_val, value))


def apply_aspect_ratio(aspect_ratio, current_width, current_height):
    if aspect_ratio == "Custom" or aspect_ratio not in ASPECT_RATIOS:
        return current_width, current_height

    w_ratio, h_ratio = ASPECT_RATIOS[aspect_ratio]

    if w_ratio == h_ratio:
        base = min(current_width, current_height)
        new_width = base
        new_height = base
    elif w_ratio < h_ratio:
        new_width = current_width
        new_height = round_to_step(current_width * h_ratio / w_ratio)
    else:
        new_height = current_height
        new_width = round_to_step(current_height * w_ratio / h_ratio)

    new_width = clamp(new_width, 256, 2048)
    new_height = clamp(new_height, 256, 2048)

    return int(new_width), int(new_height)


def update_height_from_width(width, aspect_ratio):
    if aspect_ratio == "Custom" or aspect_ratio not in ASPECT_RATIOS:
        return gr.update()

    w_ratio, h_ratio = ASPECT_RATIOS[aspect_ratio]
    new_height = round_to_step(width * h_ratio / w_ratio)
    new_height = clamp(new_height, 256, 2048)

    return int(new_height)


def update_width_from_height(height, aspect_ratio):
    if aspect_ratio == "Custom" or aspect_ratio not in ASPECT_RATIOS:
        return gr.update()

    w_ratio, h_ratio = ASPECT_RATIOS[aspect_ratio]
    new_width = round_to_step(height * w_ratio / h_ratio)
    new_width = clamp(new_width, 256, 2048)

    return int(new_width)


def swap_dimensions_and_update_ratio(width, height, aspect_ratio):
    new_width, new_height = height, width

    new_ratio = "Custom"
    for name, ratios in ASPECT_RATIOS.items():
        if ratios is None:
            continue
        w_r, h_r = ratios
        expected_height = new_width * h_r / w_r
        if abs(expected_height - new_height) < STEP:
            new_ratio = name
            break

    return new_width, new_height, new_ratio


def create_ui():
    if shared.settings['image_model_menu'] != 'None':
        shared.image_model_name = shared.settings['image_model_menu']

    with gr.Tab("Image AI", elem_id="image-ai-tab"):
        with gr.Tabs():
            # TAB 1: GENERATE
            with gr.TabItem("Generate"):
                with gr.Row():
                    with gr.Column(scale=4, min_width=350):
                        shared.gradio['image_prompt'] = gr.Textbox(
                            label="Prompt",
                            placeholder="Describe your imagination...",
                            lines=3,
                            autofocus=True,
                            value=shared.settings['image_prompt']
                        )
                        shared.gradio['image_neg_prompt'] = gr.Textbox(
                            label="Negative Prompt",
                            placeholder="Low quality...",
                            lines=3,
                            value=shared.settings['image_neg_prompt']
                        )

                        shared.gradio['image_generate_btn'] = gr.Button("✨ GENERATE", variant="primary", size="lg", elem_id="gen-btn")
                        gr.HTML("<hr style='border-top: 1px solid #444; margin: 20px 0;'>")

                        gr.Markdown("### Dimensions")
                        with gr.Row():
                            with gr.Column():
                                shared.gradio['image_width'] = gr.Slider(256, 2048, value=shared.settings['image_width'], step=32, label="Width")
                            with gr.Column():
                                shared.gradio['image_height'] = gr.Slider(256, 2048, value=shared.settings['image_height'], step=32, label="Height")
                            shared.gradio['image_swap_btn'] = gr.Button("⇄ Swap", elem_classes='refresh-button', scale=0, min_width=80, elem_id="swap-height-width")

                        with gr.Row():
                            shared.gradio['image_aspect_ratio'] = gr.Radio(
                                choices=["1:1 Square", "16:9 Cinema", "9:16 Mobile", "4:3 Photo", "Custom"],
                                value=shared.settings['image_aspect_ratio'],
                                label="Aspect Ratio",
                                interactive=True
                            )

                        gr.Markdown("### Config")
                        with gr.Row():
                            with gr.Column():
                                shared.gradio['image_steps'] = gr.Slider(1, 15, value=shared.settings['image_steps'], step=1, label="Steps")
                                shared.gradio['image_seed'] = gr.Number(label="Seed", value=shared.settings['image_seed'], precision=0, info="-1 = Random")
                            with gr.Column():
                                shared.gradio['image_batch_size'] = gr.Slider(1, 32, value=shared.settings['image_batch_size'], step=1, label="Batch Size (VRAM Heavy)", info="Generates N images at once.")
                                shared.gradio['image_batch_count'] = gr.Slider(1, 128, value=shared.settings['image_batch_count'], step=1, label="Sequential Count (Loop)", info="Repeats the generation N times.")

                    with gr.Column(scale=6, min_width=500):
                        with gr.Column(elem_classes=["viewport-container"]):
                            shared.gradio['image_output_gallery'] = gr.Gallery(label="Output", show_label=False, columns=2, rows=2, height="80vh", object_fit="contain", preview=True, elem_id="image-output-gallery")

            # TAB 2: GALLERY
            with gr.TabItem("Gallery"):
                with gr.Row():
                    shared.gradio['image_refresh_history'] = gr.Button("🔄 Refresh Gallery", elem_classes="refresh-button")
                shared.gradio['image_history_gallery'] = gr.Gallery(value=lambda : get_history_images(), label="History", show_label=False, columns=6, object_fit="cover", height="auto", allow_preview=True, elem_id="image-history-gallery")

            # TAB 3: MODEL
            with gr.TabItem("Model"):
                with gr.Row():
                    with gr.Column():
                        with gr.Row():
                            shared.gradio['image_model_menu'] = gr.Dropdown(
                                choices=utils.get_available_image_models(),
                                value=shared.settings['image_model_menu'],
                                label='Model',
                                elem_classes='slim-dropdown'
                            )
                            shared.gradio['image_refresh_models'] = gr.Button("🔄", elem_classes='refresh-button', scale=0, min_width=40)
                            shared.gradio['image_load_model'] = gr.Button("Load", variant='primary', elem_classes='refresh-button')
                            shared.gradio['image_unload_model'] = gr.Button("Unload", elem_classes='refresh-button')

                        gr.Markdown("## Settings")
                        with gr.Row():
                            with gr.Column():
                                shared.gradio['image_dtype'] = gr.Dropdown(
                                    choices=['bfloat16', 'float16'],
                                    value=shared.settings['image_dtype'],
                                    label='Data Type',
                                    info='bfloat16 recommended for modern GPUs'
                                )
                                shared.gradio['image_attn_backend'] = gr.Dropdown(
                                    choices=['sdpa', 'flash_attention_2', 'flash_attention_3'],
                                    value=shared.settings['image_attn_backend'],
                                    label='Attention Backend',
                                    info='SDPA is default. Flash Attention requires compatible GPU.'
                                )
                            with gr.Column():
                                shared.gradio['image_compile'] = gr.Checkbox(
                                    value=shared.settings['image_compile'],
                                    label='Compile Model',
                                    info='Faster inference after first run. First run will be slow.'
                                )
                                shared.gradio['image_cpu_offload'] = gr.Checkbox(
                                    value=shared.settings['image_cpu_offload'],
                                    label='CPU Offload',
                                    info='Enable for low VRAM GPUs. Slower but uses less memory.'
                                )

                    with gr.Column():
                        shared.gradio['image_download_path'] = gr.Textbox(
                            label="Download model",
                            placeholder="Tongyi-MAI/Z-Image-Turbo",
                            info="Enter HuggingFace path. Use : for branch, e.g. user/model:main"
                        )
                        shared.gradio['image_download_btn'] = gr.Button("Download", variant='primary')
                        shared.gradio['image_model_status'] = gr.Markdown(
                            value=f"Model: **{shared.settings['image_model_menu']}** (not loaded)" if shared.settings['image_model_menu'] != 'None' else "No model selected"
                        )


def create_event_handlers():
    # Dimension controls
    shared.gradio['image_aspect_ratio'].change(
        apply_aspect_ratio,
        gradio('image_aspect_ratio', 'image_width', 'image_height'),
        gradio('image_width', 'image_height'),
        show_progress=False
    )

    shared.gradio['image_width'].release(
        update_height_from_width,
        gradio('image_width', 'image_aspect_ratio'),
        gradio('image_height'),
        show_progress=False
    )

    shared.gradio['image_height'].release(
        update_width_from_height,
        gradio('image_height', 'image_aspect_ratio'),
        gradio('image_width'),
        show_progress=False
    )

    shared.gradio['image_swap_btn'].click(
        swap_dimensions_and_update_ratio,
        gradio('image_width', 'image_height', 'image_aspect_ratio'),
        gradio('image_width', 'image_height', 'image_aspect_ratio'),
        show_progress=False
    )

    # Generation
    shared.gradio['image_generate_btn'].click(
        ui.gather_interface_values, gradio(shared.input_elements), gradio('interface_state')).then(
        generate, gradio('interface_state'), gradio('image_output_gallery'))

    shared.gradio['image_prompt'].submit(
        ui.gather_interface_values, gradio(shared.input_elements), gradio('interface_state')).then(
        generate, gradio('interface_state'), gradio('image_output_gallery'))

    shared.gradio['image_neg_prompt'].submit(
        ui.gather_interface_values, gradio(shared.input_elements), gradio('interface_state')).then(
        generate, gradio('interface_state'), gradio('image_output_gallery'))

    # Model management
    shared.gradio['image_refresh_models'].click(
        lambda: gr.update(choices=utils.get_available_image_models()),
        None,
        gradio('image_model_menu'),
        show_progress=False
    )

    shared.gradio['image_load_model'].click(
        load_image_model_wrapper,
        gradio('image_model_menu', 'image_dtype', 'image_attn_backend', 'image_cpu_offload', 'image_compile'),
        gradio('image_model_status'),
        show_progress=True
    )

    shared.gradio['image_unload_model'].click(
        unload_image_model_wrapper,
        None,
        gradio('image_model_status'),
        show_progress=False
    )

    shared.gradio['image_download_btn'].click(
        download_image_model_wrapper,
        gradio('image_download_path'),
        gradio('image_model_status', 'image_model_menu'),
        show_progress=True
    )

    # History
    shared.gradio['image_refresh_history'].click(
        get_history_images,
        None,
        gradio('image_history_gallery'),
        show_progress=False
    )


def generate(state):
    model_name = state['image_model_menu']

    if not model_name or model_name == 'None':
        logger.error("No image model selected. Go to the Model tab and select a model.")
        return []

    if shared.image_model is None:
        result = load_image_model(
            model_name,
            dtype=state['image_dtype'],
            attn_backend=state['image_attn_backend'],
            cpu_offload=state['image_cpu_offload'],
            compile_model=state['image_compile']
        )
        if result is None:
            logger.error(f"Failed to load model `{model_name}`.")
            return []

        shared.image_model_name = model_name

    seed = state['image_seed']
    if seed == -1:
        seed = np.random.randint(0, 2**32 - 1)

    generator = torch.Generator("cuda").manual_seed(int(seed))
    all_images = []

    t0 = time.time()
    for i in range(int(state['image_batch_count'])):
        generator.manual_seed(int(seed + i))
        batch_results = shared.image_model(
            prompt=state['image_prompt'],
            negative_prompt=state['image_neg_prompt'],
            height=int(state['image_height']),
            width=int(state['image_width']),
            num_inference_steps=int(state['image_steps']),
            guidance_scale=0.0,
            num_images_per_prompt=int(state['image_batch_size']),
            generator=generator,
        ).images
        all_images.extend(batch_results)

    t1 = time.time()
    save_generated_images(all_images, state['image_prompt'], seed)

    logger.info(f'Images generated in {(t1-t0):.2f} seconds (seed {seed})')
    return all_images


def load_image_model_wrapper(model_name, dtype, attn_backend, cpu_offload, compile_model):
    if not model_name or model_name == 'None':
        yield "No model selected"
        return

    try:
        yield f"Loading `{model_name}`..."
        unload_image_model()

        result = load_image_model(
            model_name,
            dtype=dtype,
            attn_backend=attn_backend,
            cpu_offload=cpu_offload,
            compile_model=compile_model
        )

        if result is not None:
            shared.image_model_name = model_name
            yield f"✓ Loaded **{model_name}**"
        else:
            yield f"✗ Failed to load `{model_name}`"
    except Exception:
        yield f"Error:\n```\n{traceback.format_exc()}\n```"


def unload_image_model_wrapper():
    unload_image_model()
    if shared.image_model_name != 'None':
        return f"Model: **{shared.image_model_name}** (not loaded)"
    return "No model loaded"


def download_image_model_wrapper(model_path):
    from huggingface_hub import snapshot_download

    if not model_path:
        yield "No model specified", gr.update()
        return

    try:
        if ':' in model_path:
            model_id, branch = model_path.rsplit(':', 1)
        else:
            model_id, branch = model_path, 'main'

        folder_name = model_id.replace('/', '_')
        output_folder = Path(shared.args.image_model_dir) / folder_name

        yield f"Downloading `{model_id}` (branch: {branch})...", gr.update()

        snapshot_download(
            repo_id=model_id,
            revision=branch,
            local_dir=output_folder,
            local_dir_use_symlinks=False,
        )

        new_choices = utils.get_available_image_models()
        yield f"✓ Downloaded to `{output_folder}`", gr.update(choices=new_choices, value=folder_name)
    except Exception:
        yield f"Error:\n```\n{traceback.format_exc()}\n```", gr.update()


def save_generated_images(images, prompt, seed):
    date_str = datetime.now().strftime("%Y-%m-%d")
    folder_path = os.path.join("user_data", "image_outputs", date_str)
    os.makedirs(folder_path, exist_ok=True)

    for idx, img in enumerate(images):
        timestamp = datetime.now().strftime("%H-%M-%S")
        filename = f"{timestamp}_{seed}_{idx}.png"
        img.save(os.path.join(folder_path, filename))


def get_history_images():
    output_dir = os.path.join("user_data", "image_outputs")
    if not os.path.exists(output_dir):
        return []

    image_files = []
    for root, _, files in os.walk(output_dir):
        for file in files:
            if file.endswith((".png", ".jpg", ".jpeg")):
                full_path = os.path.join(root, file)
                image_files.append((full_path, os.path.getmtime(full_path)))

    image_files.sort(key=lambda x: x[1], reverse=True)
    return [x[0] for x in image_files]
