import json
import os
import random
import time
import traceback
from datetime import datetime
from pathlib import Path

import gradio as gr
from PIL.PngImagePlugin import PngInfo

from modules import shared, ui, utils
from modules.image_models import (
    get_pipeline_type,
    load_image_model,
    unload_image_model
)
from modules.image_utils import open_image_safely
from modules.logging_colors import logger
from modules.text_generation import stop_everything_event
from modules.utils import check_model_loaded, gradio

ASPECT_RATIOS = {
    "1:1 Square": (1, 1),
    "16:9 Cinema": (16, 9),
    "9:16 Mobile": (9, 16),
    "4:3 Photo": (4, 3),
    "Custom": None,
}

STEP = 16
IMAGES_PER_PAGE = 32

# Settings keys to save in PNG metadata (Generate tab only)
METADATA_SETTINGS_KEYS = [
    'image_prompt',
    'image_neg_prompt',
    'image_width',
    'image_height',
    'image_aspect_ratio',
    'image_steps',
    'image_seed',
    'image_cfg_scale',
]

# Cache for all image paths
_image_cache = []
_cache_timestamp = 0


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


def build_generation_metadata(state, actual_seed):
    """Build metadata dict from generation settings."""
    metadata = {}
    for key in METADATA_SETTINGS_KEYS:
        if key in state:
            metadata[key] = state[key]

    # Store the actual seed used (not -1)
    metadata['image_seed'] = actual_seed
    metadata['generated_at'] = datetime.now().isoformat()
    metadata['model'] = shared.image_model_name

    return metadata


def save_generated_images(images, state, actual_seed):
    """Save images with generation metadata embedded in PNG. Returns list of saved file paths."""
    if shared.args.multi_user:
        return []

    date_str = datetime.now().strftime("%Y-%m-%d")
    folder_path = os.path.join("user_data", "image_outputs", date_str)
    os.makedirs(folder_path, exist_ok=True)

    metadata = build_generation_metadata(state, actual_seed)
    metadata_json = json.dumps(metadata, ensure_ascii=False)

    saved_paths = []
    for idx, img in enumerate(images):
        timestamp = datetime.now().strftime("%H-%M-%S")
        filename = f"TGW_{timestamp}_{actual_seed:010d}_{idx:03d}.png"
        filepath = os.path.join(folder_path, filename)

        # Create PNG metadata
        png_info = PngInfo()
        png_info.add_text("image_gen_settings", metadata_json)

        # Save with metadata
        img.save(filepath, pnginfo=png_info)
        saved_paths.append(filepath)

    return saved_paths


def read_image_metadata(image_path):
    """Read generation metadata from PNG file."""
    try:
        img = open_image_safely(image_path)
        if img is None:
            return None
        try:
            if hasattr(img, 'text') and 'image_gen_settings' in img.text:
                return json.loads(img.text['image_gen_settings'])
        finally:
            img.close()
    except Exception as e:
        logger.debug(f"Could not read metadata from {image_path}: {e}")
    return None


def format_metadata_for_display(metadata):
    """Format metadata as readable text."""
    if not metadata:
        return "No generation settings found in this image."

    lines = []

    # Display in a nice order
    display_order = [
        ('image_prompt', 'Prompt'),
        ('image_neg_prompt', 'Negative Prompt'),
        ('image_width', 'Width'),
        ('image_height', 'Height'),
        ('image_aspect_ratio', 'Aspect Ratio'),
        ('image_steps', 'Steps'),
        ('image_cfg_scale', 'CFG Scale'),
        ('image_seed', 'Seed'),
        ('model', 'Model'),
        ('generated_at', 'Generated At'),
    ]

    for key, label in display_order:
        if key in metadata:
            value = metadata[key]
            if key in ['image_prompt', 'image_neg_prompt'] and value:
                # Truncate long prompts for display
                if len(str(value)) > 200:
                    value = str(value)[:200] + "..."
            lines.append(f"**{label}:** {value}")

    return "\n\n".join(lines)


def get_all_history_images(force_refresh=False):
    """Get all history images sorted by modification time (newest first). Uses caching."""
    global _image_cache, _cache_timestamp

    output_dir = os.path.join("user_data", "image_outputs")
    if not os.path.exists(output_dir):
        return []

    # Check if we need to refresh cache
    current_time = time.time()
    if not force_refresh and _image_cache and (current_time - _cache_timestamp) < 2:
        return _image_cache

    image_files = []
    for root, _, files in os.walk(output_dir):
        for file in files:
            if file.endswith((".png", ".jpg", ".jpeg")):
                full_path = os.path.join(root, file)
                image_files.append((full_path, os.path.getmtime(full_path)))

    image_files.sort(key=lambda x: x[1], reverse=True)
    _image_cache = [x[0] for x in image_files]
    _cache_timestamp = current_time

    return _image_cache


def get_paginated_images(page=0, force_refresh=False):
    """Get images for a specific page."""
    all_images = get_all_history_images(force_refresh)
    total_images = len(all_images)
    total_pages = max(1, (total_images + IMAGES_PER_PAGE - 1) // IMAGES_PER_PAGE)

    # Clamp page to valid range
    page = max(0, min(page, total_pages - 1))

    start_idx = page * IMAGES_PER_PAGE
    end_idx = min(start_idx + IMAGES_PER_PAGE, total_images)

    page_images = all_images[start_idx:end_idx]

    return page_images, page, total_pages, total_images


def get_initial_page_info():
    """Get page info string for initial load."""
    _, page, total_pages, total_images = get_paginated_images(0)
    return f"Page {page + 1} of {total_pages} ({total_images} total images)"


def refresh_gallery(current_page=0):
    """Refresh gallery with current page."""
    images, page, total_pages, total_images = get_paginated_images(current_page, force_refresh=True)
    page_info = f"Page {page + 1} of {total_pages} ({total_images} total images)"
    return images, page, page_info


def go_to_page(page_num, current_page):
    """Go to a specific page (1-indexed input)."""
    try:
        page = int(page_num) - 1  # Convert to 0-indexed
    except (ValueError, TypeError):
        page = current_page

    images, page, total_pages, total_images = get_paginated_images(page)
    page_info = f"Page {page + 1} of {total_pages} ({total_images} total images)"
    return images, page, page_info


def next_page(current_page):
    """Go to next page."""
    images, page, total_pages, total_images = get_paginated_images(current_page + 1)
    page_info = f"Page {page + 1} of {total_pages} ({total_images} total images)"
    return images, page, page_info


def prev_page(current_page):
    """Go to previous page."""
    images, page, total_pages, total_images = get_paginated_images(current_page - 1)
    page_info = f"Page {page + 1} of {total_pages} ({total_images} total images)"
    return images, page, page_info


def on_gallery_select(evt: gr.SelectData, current_page):
    """Handle image selection from gallery."""
    if evt.index is None:
        return "", "Select an image to view its settings"

    if not _image_cache:
        get_all_history_images()

    all_images = _image_cache
    total_images = len(all_images)

    # Calculate the actual index in the full list
    start_idx = current_page * IMAGES_PER_PAGE
    actual_idx = start_idx + evt.index

    if actual_idx >= total_images:
        return "", "Image not found"

    image_path = all_images[actual_idx]
    metadata = read_image_metadata(image_path)
    metadata_display = format_metadata_for_display(metadata)

    return image_path, metadata_display


def send_to_generate(selected_image_path):
    """Load settings from selected image and return updates for all Generate tab inputs."""
    if not selected_image_path or not os.path.exists(selected_image_path):
        return [gr.update()] * 8 + ["No image selected"]

    metadata = read_image_metadata(selected_image_path)
    if not metadata:
        return [gr.update()] * 8 + ["No settings found in this image"]

    # Return updates for each input element in order
    updates = [
        gr.update(value=metadata.get('image_prompt', '')),
        gr.update(value=metadata.get('image_neg_prompt', '')),
        gr.update(value=metadata.get('image_width', 1024)),
        gr.update(value=metadata.get('image_height', 1024)),
        gr.update(value=metadata.get('image_aspect_ratio', '1:1 Square')),
        gr.update(value=metadata.get('image_steps', 9)),
        gr.update(value=metadata.get('image_seed', -1)),
        gr.update(value=metadata.get('image_cfg_scale', 0.0)),
    ]

    status = f"✓ Settings loaded from image (seed: {metadata.get('image_seed', 'unknown')})"
    return updates + [status]


def read_dropped_image_metadata(image_path):
    """Read metadata from a dropped/uploaded image."""
    if not image_path:
        return "Drop an image to view its generation settings."

    metadata = read_image_metadata(image_path)
    return format_metadata_for_display(metadata)


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
                        shared.gradio['image_llm_variations'] = gr.Checkbox(
                            value=shared.settings['image_llm_variations'],
                            label='LLM Prompt Variations',
                            elem_id="llm-prompt-variations",
                        )
                        shared.gradio['image_llm_variations_prompt'] = gr.Textbox(
                            value=shared.settings['image_llm_variations_prompt'],
                            label='Variation Prompt',
                            lines=3,
                            placeholder='Instructions for generating prompt variations...',
                            visible=shared.settings['image_llm_variations'],
                            info='Use the loaded LLM to generate creative prompt variations for each sequential batch.'
                        )

                        shared.gradio['image_generate_btn'] = gr.Button("Generate", variant="primary", size="lg")
                        shared.gradio['image_stop_btn'] = gr.Button("Stop", size="lg", visible=False)
                        shared.gradio['image_progress'] = gr.HTML(
                            value=progress_bar_html(),
                            elem_id="image-progress"
                        )

                        gr.Markdown("### Dimensions")
                        with gr.Row():
                            with gr.Column():
                                shared.gradio['image_width'] = gr.Slider(256, 2048, value=shared.settings['image_width'], step=STEP, label="Width")
                            with gr.Column():
                                shared.gradio['image_height'] = gr.Slider(256, 2048, value=shared.settings['image_height'], step=STEP, label="Height")
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
                                shared.gradio['image_steps'] = gr.Slider(1, 100, value=shared.settings['image_steps'], step=1, label="Steps")
                                shared.gradio['image_cfg_scale'] = gr.Slider(
                                    0.0, 10.0,
                                    value=shared.settings['image_cfg_scale'],
                                    step=0.1,
                                    label="CFG Scale",
                                    info="Z-Image Turbo: 0.0 | Qwen: 4.0"
                                )
                                shared.gradio['image_seed'] = gr.Number(label="Seed", value=shared.settings['image_seed'], precision=0, info="-1 = Random")

                            with gr.Column():
                                shared.gradio['image_batch_size'] = gr.Slider(1, 32, value=shared.settings['image_batch_size'], step=1, label="Batch Size (VRAM Heavy)", info="Generates N images at once.")
                                shared.gradio['image_batch_count'] = gr.Slider(1, 128, value=shared.settings['image_batch_count'], step=1, label="Sequential Count (Loop)", info="Repeats the generation N times.")

                    with gr.Column(scale=6, min_width=500):
                        with gr.Column(elem_classes=["viewport-container"]):
                            shared.gradio['image_output_gallery'] = gr.Gallery(label="Output", show_label=False, columns=2, rows=2, height="80vh", object_fit="contain", preview=True, elem_id="image-output-gallery")

            # TAB 2: GALLERY (with pagination)
            with gr.TabItem("Gallery"):
                with gr.Row():
                    with gr.Column(scale=3):
                        # Pagination controls
                        with gr.Row():
                            shared.gradio['image_refresh_history'] = gr.Button("🔄 Refresh", elem_classes="refresh-button")
                            shared.gradio['image_prev_page'] = gr.Button("◀ Prev Page", elem_classes="refresh-button")
                            shared.gradio['image_page_info'] = gr.Markdown(value=get_initial_page_info, elem_id="image-page-info")
                            shared.gradio['image_next_page'] = gr.Button("Next Page ▶", elem_classes="refresh-button")
                            shared.gradio['image_page_input'] = gr.Number(value=1, label="Page", precision=0, minimum=1, scale=0, min_width=80)
                            shared.gradio['image_go_to_page'] = gr.Button("Go", elem_classes="refresh-button", scale=0, min_width=50)

                        # State for current page and selected image path
                        shared.gradio['image_current_page'] = gr.State(value=0)
                        shared.gradio['image_selected_path'] = gr.State(value="")

                        # Paginated gallery using gr.Gallery
                        shared.gradio['image_history_gallery'] = gr.Gallery(
                            value=lambda: get_paginated_images(0)[0],
                            label="Image History",
                            show_label=False,
                            columns=6,
                            object_fit="cover",
                            height="auto",
                            allow_preview=True,
                            elem_id="image-history-gallery"
                        )

                    with gr.Column(scale=1):
                        gr.Markdown("### Generation Settings")
                        shared.gradio['image_settings_display'] = gr.Markdown("Select an image to view its settings")
                        shared.gradio['image_send_to_generate'] = gr.Button("Send to Generate", variant="primary")
                        shared.gradio['image_gallery_status'] = gr.Markdown("")

                        gr.Markdown("### Import Image")
                        shared.gradio['image_drop_upload'] = gr.Image(
                            label="Drop image here to view settings",
                            type="filepath",
                            height=150
                        )

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
                                shared.gradio['image_quant'] = gr.Dropdown(
                                    label='Quantization',
                                    choices=['none', 'bnb-8bit', 'bnb-4bit', 'torchao-int8wo', 'torchao-fp4', 'torchao-float8wo'],
                                    value=shared.settings['image_quant'],
                                    info='BnB: bitsandbytes quantization. torchao: int8wo, fp4, float8wo.'
                                )

                                shared.gradio['image_dtype'] = gr.Dropdown(
                                    choices=['bfloat16', 'float16'],
                                    value=shared.settings['image_dtype'],
                                    label='Data Type',
                                    info='bfloat16 recommended for modern GPUs'
                                )
                                shared.gradio['image_attn_backend'] = gr.Dropdown(
                                    choices=['sdpa', 'flash_attention_2'],
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
                        shared.gradio['image_model_status'] = gr.Markdown(value="")


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
        lambda: [gr.update(visible=True), gr.update(visible=False)], None, gradio('image_stop_btn', 'image_generate_btn')).then(
        generate, gradio('interface_state'), gradio('image_output_gallery', 'image_progress'), show_progress=False).then(
        lambda: [gr.update(visible=False), gr.update(visible=True)], None, gradio('image_stop_btn', 'image_generate_btn'))

    shared.gradio['image_prompt'].submit(
        ui.gather_interface_values, gradio(shared.input_elements), gradio('interface_state')).then(
        lambda: [gr.update(visible=True), gr.update(visible=False)], None, gradio('image_stop_btn', 'image_generate_btn')).then(
        generate, gradio('interface_state'), gradio('image_output_gallery', 'image_progress'), show_progress=False).then(
        lambda: [gr.update(visible=False), gr.update(visible=True)], None, gradio('image_stop_btn', 'image_generate_btn'))

    shared.gradio['image_neg_prompt'].submit(
        ui.gather_interface_values, gradio(shared.input_elements), gradio('interface_state')).then(
        lambda: [gr.update(visible=True), gr.update(visible=False)], None, gradio('image_stop_btn', 'image_generate_btn')).then(
        generate, gradio('interface_state'), gradio('image_output_gallery', 'image_progress'), show_progress=False).then(
        lambda: [gr.update(visible=False), gr.update(visible=True)], None, gradio('image_stop_btn', 'image_generate_btn'))

    # Stop button
    shared.gradio['image_stop_btn'].click(
        stop_everything_event, None, None, show_progress=False
    )

    # Model management
    shared.gradio['image_refresh_models'].click(
        lambda: gr.update(choices=utils.get_available_image_models()),
        None,
        gradio('image_model_menu'),
        show_progress=False
    )

    shared.gradio['image_load_model'].click(
        load_image_model_wrapper,
        gradio('image_model_menu', 'image_dtype', 'image_attn_backend', 'image_cpu_offload', 'image_compile', 'image_quant'),
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

    # Gallery pagination handlers
    shared.gradio['image_refresh_history'].click(
        refresh_gallery,
        gradio('image_current_page'),
        gradio('image_history_gallery', 'image_current_page', 'image_page_info'),
        show_progress=False
    )

    shared.gradio['image_next_page'].click(
        next_page,
        gradio('image_current_page'),
        gradio('image_history_gallery', 'image_current_page', 'image_page_info'),
        show_progress=False
    )

    shared.gradio['image_prev_page'].click(
        prev_page,
        gradio('image_current_page'),
        gradio('image_history_gallery', 'image_current_page', 'image_page_info'),
        show_progress=False
    )

    shared.gradio['image_go_to_page'].click(
        go_to_page,
        gradio('image_page_input', 'image_current_page'),
        gradio('image_history_gallery', 'image_current_page', 'image_page_info'),
        show_progress=False
    )

    # Image selection from gallery
    shared.gradio['image_history_gallery'].select(
        on_gallery_select,
        gradio('image_current_page'),
        gradio('image_selected_path', 'image_settings_display'),
        show_progress=False
    )

    # Send to Generate
    shared.gradio['image_send_to_generate'].click(
        send_to_generate,
        gradio('image_selected_path'),
        gradio(
            'image_prompt',
            'image_neg_prompt',
            'image_width',
            'image_height',
            'image_aspect_ratio',
            'image_steps',
            'image_seed',
            'image_cfg_scale',
            'image_gallery_status'
        ),
        js=f'() => {{{ui.switch_tabs_js}; switch_to_image_ai_generate()}}',
        show_progress=False
    )

    shared.gradio['image_drop_upload'].change(
        read_dropped_image_metadata,
        gradio('image_drop_upload'),
        gradio('image_settings_display'),
        show_progress=False
    )

    # LLM Variations visibility toggle
    shared.gradio['image_llm_variations'].change(
        lambda x: gr.update(visible=x),
        gradio('image_llm_variations'),
        gradio('image_llm_variations_prompt'),
        show_progress=False
    )


def generate_prompt_variation(state):
    """Generate a creative variation of the image prompt using the LLM."""
    from modules.chat import generate_chat_prompt
    from modules.text_generation import generate_reply

    prompt = state['image_prompt']

    # Check if LLM is loaded
    model_loaded, _ = check_model_loaded()
    if not model_loaded:
        logger.warning("No LLM loaded for prompt variation. Using original prompt.")
        return prompt

    # Get the custom variation prompt or use default
    variation_instruction = state.get('image_llm_variations_prompt', '')
    if not variation_instruction:
        variation_instruction = 'Write a variation of the image generation prompt above. Consider the intent of the user with that prompt and write something that will likely please them, with added details. Output only the new prompt. Do not add any explanations, prefixes, or additional text.'

    augmented_message = f"{prompt}\n\n=====\n\n{variation_instruction}"

    # Use minimal state for generation
    var_state = state.copy()
    var_state['history'] = {'internal': [], 'visible': [], 'metadata': {}}
    var_state['auto_max_new_tokens'] = True
    var_state['enable_thinking'] = False
    var_state['reasoning_effort'] = 'low'
    var_state['start_with'] = ""

    formatted_prompt = generate_chat_prompt(augmented_message, var_state)

    variation = ""
    for reply in generate_reply(formatted_prompt, var_state, stopping_strings=[], is_chat=True):
        variation = reply

    # Strip thinking blocks if present
    if "</think>" in variation:
        variation = variation.rsplit("</think>", 1)[1]
    elif "<|start|>assistant<|channel|>final<|message|>" in variation:
        variation = variation.rsplit("<|start|>assistant<|channel|>final<|message|>", 1)[1]
    elif "</seed:think>" in variation:
        variation = variation.rsplit("</seed:think>", 1)[1]

    variation = variation.strip()
    if len(variation) >= 2 and variation.startswith('"') and variation.endswith('"'):
        variation = variation[1:-1]

    if variation:
        logger.info("Prompt variation:")
        print(variation)
        return variation

    return prompt


def progress_bar_html(progress=0, text=""):
    """Generate HTML for progress bar. Empty div when progress <= 0."""
    if progress <= 0:
        return '<div class="image-ai-separator"></div>'

    return f'''<div class="image-ai-progress-wrapper">
        <div class="image-ai-progress-track">
            <div class="image-ai-progress-fill" style="width: {progress * 100:.1f}%;"></div>
        </div>
        <div class="image-ai-progress-text">{text}</div>
    </div>'''


def generate(state, save_images=True):
    """
    Generate images using the loaded model.
    Automatically adjusts parameters based on pipeline type.
    """
    import queue
    import threading

    import torch

    from modules.torch_utils import clear_torch_cache, get_device

    try:
        model_name = state['image_model_menu']

        if not model_name or model_name == 'None':
            logger.error("No image model selected. Go to the Model tab and select a model.")
            yield [], progress_bar_html()
            return

        if shared.image_model is None:
            result = load_image_model(
                model_name,
                dtype=state['image_dtype'],
                attn_backend=state['image_attn_backend'],
                cpu_offload=state['image_cpu_offload'],
                compile_model=state['image_compile'],
                quant_method=state['image_quant']
            )
            if result is None:
                logger.error(f"Failed to load model `{model_name}`.")
                yield [], progress_bar_html()
                return

            shared.image_model_name = model_name

        seed = state['image_seed']
        if seed == -1:
            seed = random.randint(0, 2**32 - 1)

        device = get_device()
        if device is None:
            device = "cpu"
        generator = torch.Generator(device)

        all_images = []

        # Get pipeline type for parameter adjustment
        pipeline_type = getattr(shared, 'image_pipeline_type', None)
        if pipeline_type is None:
            pipeline_type = get_pipeline_type(shared.image_model)

        prompt = state['image_prompt']

        shared.stop_everything = False

        batch_count = int(state['image_batch_count'])
        steps_per_batch = int(state['image_steps'])
        total_steps = steps_per_batch * batch_count

        # Queue for progress updates from callback
        progress_queue = queue.Queue()

        def interrupt_callback(pipe, step_index, timestep, callback_kwargs):
            if shared.stop_everything:
                pipe._interrupt = True
            progress_queue.put(step_index + 1)
            return callback_kwargs

        gen_kwargs = {
            "prompt": prompt,
            "negative_prompt": state['image_neg_prompt'],
            "height": int(state['image_height']),
            "width": int(state['image_width']),
            "num_inference_steps": steps_per_batch,
            "num_images_per_prompt": int(state['image_batch_size']),
            "generator": generator,
            "callback_on_step_end": interrupt_callback,
        }

        cfg_val = state.get('image_cfg_scale', 0.0)
        if pipeline_type == 'qwenimage':
            gen_kwargs["true_cfg_scale"] = cfg_val
        else:
            gen_kwargs["guidance_scale"] = cfg_val

        t0 = time.time()

        for batch_idx in range(batch_count):
            if shared.stop_everything:
                break

            generator.manual_seed(int(seed + batch_idx))

            # Generate prompt variation if enabled
            if state['image_llm_variations']:
                gen_kwargs["prompt"] = generate_prompt_variation(state)

            # Run generation in thread so we can yield progress
            result_holder = []
            error_holder = []

            def run_batch():
                try:
                    # Apply magic suffix only at generation time for qwenimage
                    clean_prompt = gen_kwargs["prompt"]
                    if pipeline_type == 'qwenimage':
                        magic_suffix = ", Ultra HD, 4K, cinematic composition"
                        if magic_suffix.strip(", ") not in clean_prompt:
                            gen_kwargs["prompt"] = clean_prompt + magic_suffix

                    result_holder.extend(shared.image_model(**gen_kwargs).images)
                    gen_kwargs["prompt"] = clean_prompt  # restore
                except Exception as e:
                    error_holder.append(e)

            thread = threading.Thread(target=run_batch)
            thread.start()

            # Yield progress updates while generation runs
            while thread.is_alive():
                try:
                    step = progress_queue.get(timeout=0.1)
                    absolute_step = batch_idx * steps_per_batch + step
                    pct = absolute_step / total_steps
                    text = f"Batch {batch_idx + 1}/{batch_count} — Step {step}/{steps_per_batch}"
                    yield all_images, progress_bar_html(pct, text)
                except queue.Empty:
                    pass

            thread.join()

            if error_holder:
                raise error_holder[0]

            # Save this batch's images with the actual prompt and seed used
            if save_images:
                batch_seed = seed + batch_idx
                original_prompt = state['image_prompt']
                state['image_prompt'] = gen_kwargs["prompt"]
                saved_paths = save_generated_images(result_holder, state, batch_seed)
                state['image_prompt'] = original_prompt
                # Use file paths so gallery serves actual PNGs with metadata
                all_images.extend(saved_paths)
            else:
                # Fallback to PIL objects if not saving
                all_images.extend(result_holder)

            yield all_images, progress_bar_html((batch_idx + 1) / batch_count, f"Batch {batch_idx + 1}/{batch_count} complete")

        t1 = time.time()

        total_images = batch_count * int(state['image_batch_size'])
        logger.info(f'Generated {total_images} {"image" if total_images == 1 else "images"} in {(t1 - t0):.2f} seconds ({total_steps / (t1 - t0):.2f} steps/s, seed {seed})')

        yield all_images, progress_bar_html()
        clear_torch_cache()

    except Exception as e:
        logger.error(f"Image generation failed: {e}")
        traceback.print_exc()
        yield [], progress_bar_html()
        clear_torch_cache()


def load_image_model_wrapper(model_name, dtype, attn_backend, cpu_offload, compile_model, quant_method):
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
            compile_model=compile_model,
            quant_method=quant_method
        )

        if result is not None:
            shared.image_model_name = model_name
            yield f"✓ Loaded **{model_name}** (quantization: {quant_method})"
        else:
            yield f"✗ Failed to load `{model_name}`"
    except Exception:
        yield f"Error:\n```\n{traceback.format_exc()}\n```"


def unload_image_model_wrapper():
    previous_name = shared.image_model_name
    unload_image_model()
    if previous_name != 'None':
        return f"Model: **{previous_name}** (unloaded)"
    return "No model loaded"


def download_image_model_wrapper(model_path):
    from huggingface_hub import snapshot_download

    if not model_path:
        yield "No model specified", gr.update()
        return

    try:
        model_path = model_path.strip()
        if model_path.startswith('https://huggingface.co/'):
            model_path = model_path[len('https://huggingface.co/'):]
        elif model_path.startswith('huggingface.co/'):
            model_path = model_path[len('huggingface.co/'):]

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
