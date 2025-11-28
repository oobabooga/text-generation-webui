# modules/ui_image_generation.py
import os
import traceback
from datetime import datetime
from pathlib import Path

import gradio as gr
import numpy as np
import torch

from modules import shared, utils
from modules.image_models import load_image_model, unload_image_model
from modules.image_model_settings import get_effective_settings, save_image_model_settings


# Aspect ratio definitions: name -> (width_ratio, height_ratio)
ASPECT_RATIOS = {
    "1:1 Square": (1, 1),
    "16:9 Cinema": (16, 9),
    "9:16 Mobile": (9, 16),
    "4:3 Photo": (4, 3),
    "Custom": None,
}

STEP = 32  # Slider step for rounding


def round_to_step(value, step=STEP):
    """Round a value to the nearest step."""
    return round(value / step) * step


def clamp(value, min_val, max_val):
    """Clamp value between min and max."""
    return max(min_val, min(max_val, value))


def apply_aspect_ratio(aspect_ratio, current_width, current_height):
    """
    Apply an aspect ratio preset.

    Logic to prevent dimension creep:
    - For tall ratios (like 9:16): keep width fixed, calculate height
    - For wide ratios (like 16:9): keep height fixed, calculate width
    - For square (1:1): use the smaller of the current dimensions

    Returns (new_width, new_height).
    """
    if aspect_ratio == "Custom" or aspect_ratio not in ASPECT_RATIOS:
        return current_width, current_height

    w_ratio, h_ratio = ASPECT_RATIOS[aspect_ratio]

    if w_ratio == h_ratio:
        # Square ratio - use the smaller current dimension to prevent creep
        base = min(current_width, current_height)
        new_width = base
        new_height = base
    elif w_ratio < h_ratio:
        # Tall ratio (like 9:16) - width is the smaller side, keep it fixed
        new_width = current_width
        new_height = round_to_step(current_width * h_ratio / w_ratio)
    else:
        # Wide ratio (like 16:9) - height is the smaller side, keep it fixed
        new_height = current_height
        new_width = round_to_step(current_height * w_ratio / h_ratio)

    # Clamp to slider bounds
    new_width = clamp(new_width, 256, 2048)
    new_height = clamp(new_height, 256, 2048)

    return int(new_width), int(new_height)


def update_height_from_width(width, aspect_ratio):
    """Update height when width changes (if not Custom)."""
    if aspect_ratio == "Custom" or aspect_ratio not in ASPECT_RATIOS:
        return gr.update()

    w_ratio, h_ratio = ASPECT_RATIOS[aspect_ratio]
    new_height = round_to_step(width * h_ratio / w_ratio)
    new_height = clamp(new_height, 256, 2048)

    return int(new_height)


def update_width_from_height(height, aspect_ratio):
    """Update width when height changes (if not Custom)."""
    if aspect_ratio == "Custom" or aspect_ratio not in ASPECT_RATIOS:
        return gr.update()

    w_ratio, h_ratio = ASPECT_RATIOS[aspect_ratio]
    new_width = round_to_step(height * w_ratio / h_ratio)
    new_width = clamp(new_width, 256, 2048)

    return int(new_width)


def swap_dimensions_and_update_ratio(width, height, aspect_ratio):
    """Swap dimensions and update aspect ratio to match (or set to Custom)."""
    new_width, new_height = height, width

    # Try to find a matching aspect ratio for the swapped dimensions
    new_ratio = "Custom"
    for name, ratios in ASPECT_RATIOS.items():
        if ratios is None:
            continue
        w_r, h_r = ratios
        # Check if the swapped dimensions match this ratio (within tolerance)
        expected_height = new_width * h_r / w_r
        if abs(expected_height - new_height) < STEP:
            new_ratio = name
            break

    return new_width, new_height, new_ratio


def create_ui():
    # Get effective settings (CLI > yaml > defaults)
    settings = get_effective_settings()

    # Update shared state (but don't load the model yet)
    if settings['model_name'] != 'None':
        shared.image_model_name = settings['model_name']

    with gr.Tab("Image AI", elem_id="image-ai-tab"):
        with gr.Tabs():
            # TAB 1: GENERATION STUDIO
            with gr.TabItem("Generate"):
                with gr.Row():

                    # === LEFT COLUMN: CONTROLS ===
                    with gr.Column(scale=4, min_width=350):

                        # 1. PROMPT
                        prompt = gr.Textbox(label="Prompt", placeholder="Describe your imagination...", lines=3, autofocus=True)
                        neg_prompt = gr.Textbox(label="Negative Prompt", placeholder="Low quality...", lines=3)

                        # 2. GENERATE BUTTON
                        generate_btn = gr.Button("✨ GENERATE", variant="primary", size="lg", elem_id="gen-btn")
                        gr.HTML("<hr style='border-top: 1px solid #444; margin: 20px 0;'>")

                        # 3. DIMENSIONS
                        gr.Markdown("### 📐 Dimensions")
                        with gr.Row():
                            with gr.Column():
                                width_slider = gr.Slider(256, 2048, value=1024, step=32, label="Width")

                            with gr.Column():
                                height_slider = gr.Slider(256, 2048, value=1024, step=32, label="Height")

                            swap_btn = gr.Button("⇄ Swap", elem_classes='refresh-button', scale=0, min_width=80, elem_id="swap-height-width")

                        with gr.Row():
                            preset_radio = gr.Radio(
                                choices=["1:1 Square", "16:9 Cinema", "9:16 Mobile", "4:3 Photo", "Custom"],
                                value="1:1 Square",
                                label="Aspect Ratio",
                                interactive=True
                            )

                        # 4. SETTINGS & BATCHING
                        gr.Markdown("### ⚙️ Config")
                        with gr.Row():
                            with gr.Column():
                                steps_slider = gr.Slider(1, 15, value=9, step=1, label="Steps")
                                seed_input = gr.Number(label="Seed", value=-1, precision=0, info="-1 = Random")

                            with gr.Column():
                                batch_size_parallel = gr.Slider(1, 32, value=1, step=1, label="Batch Size (VRAM Heavy)", info="Generates N images at once.")
                                batch_count_seq = gr.Slider(1, 128, value=1, step=1, label="Sequential Count (Loop)", info="Repeats the generation N times.")

                    # === RIGHT COLUMN: VIEWPORT ===
                    with gr.Column(scale=6, min_width=500):
                        with gr.Column(elem_classes=["viewport-container"]):
                            output_gallery = gr.Gallery(
                                label="Output", show_label=False, columns=2, rows=2, height="80vh", object_fit="contain", preview=True
                            )
                        with gr.Row():
                            used_seed = gr.Markdown(label="Info", interactive=False)

            # TAB 2: HISTORY VIEWER
            with gr.TabItem("Gallery"):
                with gr.Row():
                    refresh_btn = gr.Button("🔄 Refresh Gallery", elem_classes="refresh-button")

                history_gallery = gr.Gallery(
                    label="History", show_label=False, columns=6, object_fit="cover", height="auto", allow_preview=True
                )

            # TAB 3: MODEL SETTINGS
            with gr.TabItem("Model"):
                with gr.Row():
                    with gr.Column():
                        with gr.Row():
                            image_model_menu = gr.Dropdown(
                                choices=utils.get_available_image_models(),
                                value=settings['model_name'],
                                label='Model',
                                elem_classes='slim-dropdown'
                            )
                            image_refresh_models = gr.Button("🔄", elem_classes='refresh-button', scale=0, min_width=40)
                            image_load_model = gr.Button("Load", variant='primary', elem_classes='refresh-button')
                            image_unload_model = gr.Button("Unload", elem_classes='refresh-button')

                        gr.Markdown("## Settings")

                        with gr.Row():
                            with gr.Column():
                                image_dtype = gr.Dropdown(
                                    choices=['bfloat16', 'float16'],
                                    value=settings['dtype'],
                                    label='Data Type',
                                    info='bfloat16 recommended for modern GPUs'
                                )

                                image_attn_backend = gr.Dropdown(
                                    choices=['sdpa', 'flash_attention_2', 'flash_attention_3'],
                                    value=settings['attn_backend'],
                                    label='Attention Backend',
                                    info='SDPA is default. Flash Attention requires compatible GPU.'
                                )

                            with gr.Column():
                                image_compile = gr.Checkbox(
                                    value=settings['compile_model'],
                                    label='Compile Model',
                                    info='Faster inference after first run. First run will be slow.'
                                )

                                image_cpu_offload = gr.Checkbox(
                                    value=settings['cpu_offload'],
                                    label='CPU Offload',
                                    info='Enable for low VRAM GPUs. Slower but uses less memory.'
                                )

                    with gr.Column():
                        image_download_path = gr.Textbox(
                            label="Download model",
                            placeholder="Tongyi-MAI/Z-Image-Turbo",
                            info="Enter the HuggingFace model path like Tongyi-MAI/Z-Image-Turbo. Use : for branch, e.g. Tongyi-MAI/Z-Image-Turbo:main"
                        )
                        image_download_btn = gr.Button("Download", variant='primary')
                        image_model_status = gr.Markdown(
                            value=f"Model: **{settings['model_name']}** (not loaded)" if settings['model_name'] != 'None' else "No model selected"
                        )

        # === WIRING ===

        # Aspect ratio preset changes -> update dimensions
        preset_radio.change(
            fn=apply_aspect_ratio,
            inputs=[preset_radio, width_slider, height_slider],
            outputs=[width_slider, height_slider],
            show_progress=False
        )

        # Width slider changes -> update height (if not Custom)
        width_slider.release(
            fn=update_height_from_width,
            inputs=[width_slider, preset_radio],
            outputs=[height_slider],
            show_progress=False
        )

        # Height slider changes -> update width (if not Custom)
        height_slider.release(
            fn=update_width_from_height,
            inputs=[height_slider, preset_radio],
            outputs=[width_slider],
            show_progress=False
        )

        # Swap button -> swap dimensions and update aspect ratio
        swap_btn.click(
            fn=swap_dimensions_and_update_ratio,
            inputs=[width_slider, height_slider, preset_radio],
            outputs=[width_slider, height_slider, preset_radio],
            show_progress=False
        )

        # Generation
        inputs = [prompt, neg_prompt, width_slider, height_slider, steps_slider, seed_input, batch_size_parallel, batch_count_seq]
        outputs = [output_gallery, used_seed]

        generate_btn.click(
            fn=lambda *args: generate(*args, image_model_menu, image_dtype, image_attn_backend, image_cpu_offload, image_compile),
            inputs=inputs,
            outputs=outputs
        )
        prompt.submit(
            fn=lambda *args: generate(*args, image_model_menu, image_dtype, image_attn_backend, image_cpu_offload, image_compile),
            inputs=inputs,
            outputs=outputs
        )
        neg_prompt.submit(
            fn=lambda *args: generate(*args, image_model_menu, image_dtype, image_attn_backend, image_cpu_offload, image_compile),
            inputs=inputs,
            outputs=outputs
        )

        # Model tab events
        image_refresh_models.click(
            fn=lambda: gr.update(choices=utils.get_available_image_models()),
            inputs=None,
            outputs=[image_model_menu],
            show_progress=False
        )

        image_load_model.click(
            fn=load_image_model_wrapper,
            inputs=[image_model_menu, image_dtype, image_attn_backend, image_cpu_offload, image_compile],
            outputs=[image_model_status],
            show_progress=True
        )

        image_unload_model.click(
            fn=unload_image_model_wrapper,
            inputs=None,
            outputs=[image_model_status],
            show_progress=False
        )

        image_download_btn.click(
            fn=download_image_model_wrapper,
            inputs=[image_download_path],
            outputs=[image_model_status, image_model_menu],
            show_progress=True
        )

        # History
        refresh_btn.click(fn=get_history_images, inputs=None, outputs=history_gallery, show_progress=False)


def generate(prompt, neg_prompt, width, height, steps, seed, batch_size_parallel, batch_count_seq,
             model_menu, dtype_dropdown, attn_dropdown, cpu_offload_checkbox, compile_checkbox):
    """Generate images with the current model settings."""

    model_name = shared.image_model_name

    if model_name == 'None':
        return [], "No image model selected. Go to the Model tab and select a model."

    # Auto-load model if not loaded
    if shared.image_model is None:
        # Get effective settings (CLI > yaml > defaults)
        settings = get_effective_settings()

        result = load_image_model(
            model_name,
            dtype=settings['dtype'],
            attn_backend=settings['attn_backend'],
            cpu_offload=settings['cpu_offload'],
            compile_model=settings['compile_model']
        )

        if result is None:
            return [], f"Failed to load model `{model_name}`."

    if seed == -1:
        seed = np.random.randint(0, 2**32 - 1)

    generator = torch.Generator("cuda").manual_seed(int(seed))
    all_images = []

    # Sequential loop (easier on VRAM)
    for i in range(int(batch_count_seq)):
        current_seed = seed + i
        generator.manual_seed(int(current_seed))

        # Parallel generation
        batch_results = shared.image_model(
            prompt=prompt,
            negative_prompt=neg_prompt,
            height=int(height),
            width=int(width),
            num_inference_steps=int(steps),
            guidance_scale=0.0,
            num_images_per_prompt=int(batch_size_parallel),
            generator=generator,
        ).images

        all_images.extend(batch_results)

    # Save to disk
    save_generated_images(all_images, prompt, seed)

    return all_images, f"Seed: {seed}"


def load_image_model_wrapper(model_name, dtype, attn_backend, cpu_offload, compile_model):
    """Load model and save settings."""
    if model_name == 'None' or not model_name:
        yield "No model selected"
        return

    try:
        yield f"Loading `{model_name}`..."

        # Unload existing model first
        unload_image_model()

        # Load the new model
        result = load_image_model(
            model_name,
            dtype=dtype,
            attn_backend=attn_backend,
            cpu_offload=cpu_offload,
            compile_model=compile_model
        )

        if result is not None:
            # Save settings to yaml
            save_image_model_settings(model_name, dtype, attn_backend, cpu_offload, compile_model)
            yield f"✓ Loaded **{model_name}**"
        else:
            yield f"✗ Failed to load `{model_name}`"

    except Exception:
        exc = traceback.format_exc()
        yield f"Error:\n```\n{exc}\n```"


def unload_image_model_wrapper():
    """Unload model wrapper."""
    unload_image_model()

    if shared.image_model_name != 'None':
        return f"Model: **{shared.image_model_name}** (not loaded)"
    else:
        return "No model loaded"


def download_image_model_wrapper(model_path):
    """Download a model from Hugging Face."""
    from huggingface_hub import snapshot_download

    if not model_path:
        yield "No model specified", gr.update()
        return

    try:
        # Parse model name and branch
        if ':' in model_path:
            model_id, branch = model_path.rsplit(':', 1)
        else:
            model_id, branch = model_path, 'main'

        # Output folder name
        folder_name = model_id.split('/')[-1]
        output_folder = Path(shared.args.image_model_dir) / folder_name

        yield f"Downloading `{model_id}` (branch: {branch})...", gr.update()

        snapshot_download(
            repo_id=model_id,
            revision=branch,
            local_dir=output_folder,
            local_dir_use_symlinks=False,
        )

        # Refresh the model list
        new_choices = utils.get_available_image_models()

        yield f"✓ Downloaded to `{output_folder}`", gr.update(choices=new_choices, value=folder_name)

    except Exception:
        exc = traceback.format_exc()
        yield f"Error:\n```\n{exc}\n```", gr.update()


def save_generated_images(images, prompt, seed):
    """Save generated images to disk."""
    date_str = datetime.now().strftime("%Y-%m-%d")
    folder_path = os.path.join("user_data", "image_outputs", date_str)
    os.makedirs(folder_path, exist_ok=True)

    saved_paths = []

    for idx, img in enumerate(images):
        timestamp = datetime.now().strftime("%H-%M-%S")
        filename = f"{timestamp}_{seed}_{idx}.png"
        full_path = os.path.join(folder_path, filename)

        img.save(full_path)
        saved_paths.append(full_path)

    return saved_paths


def get_history_images():
    """Scan the outputs folder and return all images, newest first."""
    output_dir = os.path.join("user_data", "image_outputs")
    if not os.path.exists(output_dir):
        return []

    image_files = []
    for root, dirs, files in os.walk(output_dir):
        for file in files:
            if file.endswith((".png", ".jpg", ".jpeg")):
                full_path = os.path.join(root, file)
                mtime = os.path.getmtime(full_path)
                image_files.append((full_path, mtime))

    image_files.sort(key=lambda x: x[1], reverse=True)
    return [x[0] for x in image_files]
