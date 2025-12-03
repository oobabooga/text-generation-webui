"""
OpenAI-compatible image generation using local diffusion models.
"""

import base64
import io
import json
import os
import time
from datetime import datetime

import numpy as np
from extensions.openai.errors import ServiceUnavailableError
from modules import shared
from modules.logging_colors import logger
from PIL.PngImagePlugin import PngInfo


def generations(prompt: str, size: str, response_format: str, n: int,
                negative_prompt: str = "", steps: int = 9, seed: int = -1,
                cfg_scale: float = 0.0, batch_count: int = 1):
    """
    Generate images using the loaded diffusion model.

    Args:
        prompt: Text description of the desired image
        size: Image dimensions as "WIDTHxHEIGHT"
        response_format: 'url' or 'b64_json'
        n: Number of images per batch
        negative_prompt: What to avoid in the image
        steps: Number of inference steps
        seed: Random seed (-1 for random)
        cfg_scale: Classifier-free guidance scale
        batch_count: Number of sequential batches

    Returns:
        dict with 'created' timestamp and 'data' list of images
    """
    import torch
    from modules.image_models import get_pipeline_type
    from modules.torch_utils import clear_torch_cache, get_device

    if shared.image_model is None:
        raise ServiceUnavailableError("No image model loaded. Load a model via the UI first.")

    clear_torch_cache()

    # Parse dimensions
    try:
        width, height = [int(x) for x in size.split('x')]
    except (ValueError, IndexError):
        width, height = 1024, 1024

    # Handle seed
    if seed == -1:
        seed = np.random.randint(0, 2**32 - 1)

    device = get_device() or "cpu"
    generator = torch.Generator(device).manual_seed(int(seed))

    # Get pipeline type for CFG parameter name
    pipeline_type = getattr(shared, 'image_pipeline_type', None) or get_pipeline_type(shared.image_model)

    # Build generation kwargs
    gen_kwargs = {
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "height": height,
        "width": width,
        "num_inference_steps": steps,
        "num_images_per_prompt": n,
        "generator": generator,
    }

    # Pipeline-specific CFG parameter
    if pipeline_type == 'qwenimage':
        gen_kwargs["true_cfg_scale"] = cfg_scale
    else:
        gen_kwargs["guidance_scale"] = cfg_scale

    # Generate
    all_images = []
    t0 = time.time()

    shared.stop_everything = False

    def interrupt_callback(pipe, step_index, timestep, callback_kwargs):
        if shared.stop_everything:
            pipe._interrupt = True
        return callback_kwargs

    gen_kwargs["callback_on_step_end"] = interrupt_callback

    for i in range(batch_count):
        if shared.stop_everything:
            break
        generator.manual_seed(int(seed + i))
        batch_results = shared.image_model(**gen_kwargs).images
        all_images.extend(batch_results)

    t1 = time.time()
    total_images = len(all_images)
    total_steps = steps * batch_count
    logger.info(f'Generated {total_images} {"image" if total_images == 1 else "images"} in {(t1 - t0):.2f} seconds ({total_steps / (t1 - t0):.2f} steps/s, seed {seed})')

    # Save images
    _save_images(all_images, prompt, negative_prompt, width, height, steps, seed, cfg_scale)

    # Build response
    resp = {
        'created': int(time.time()),
        'data': []
    }

    for img in all_images:
        b64 = _image_to_base64(img)
        if response_format == 'b64_json':
            resp['data'].append({'b64_json': b64})
        else:
            resp['data'].append({'url': f'data:image/png;base64,{b64}'})

    return resp


def _image_to_base64(image) -> str:
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode('utf-8')


def _save_images(images, prompt, negative_prompt, width, height, steps, seed, cfg_scale):
    """Save images with metadata."""
    date_str = datetime.now().strftime("%Y-%m-%d")
    folder = os.path.join("user_data", "image_outputs", date_str)
    os.makedirs(folder, exist_ok=True)

    metadata = {
        'image_prompt': prompt,
        'image_neg_prompt': negative_prompt,
        'image_width': width,
        'image_height': height,
        'image_steps': steps,
        'image_seed': seed,
        'image_cfg_scale': cfg_scale,
        'model': getattr(shared, 'image_model_name', 'unknown'),
    }

    for idx, img in enumerate(images):
        ts = datetime.now().strftime("%H-%M-%S")
        filepath = os.path.join(folder, f"{ts}_{seed:010d}_{idx:03d}.png")

        png_info = PngInfo()
        png_info.add_text("image_gen_settings", json.dumps(metadata))
        img.save(filepath, pnginfo=png_info)
