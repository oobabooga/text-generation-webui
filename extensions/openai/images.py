"""
OpenAI-compatible image generation using local diffusion models.
"""

import base64
import io
import time

from extensions.openai.errors import ServiceUnavailableError
from modules import shared


def generations(request):
    """
    Generate images using the loaded diffusion model.
    Returns dict with 'created' timestamp and 'data' list of images.
    """
    from modules.ui_image_generation import generate

    if shared.image_model is None:
        raise ServiceUnavailableError("No image model loaded. Load a model via the UI first.")

    width, height = request.get_width_height()

    # Build state dict: GenerationOptions fields + image-specific keys
    state = request.model_dump()
    state.update({
        'image_model_menu': shared.image_model_name,
        'image_prompt': request.prompt,
        'image_neg_prompt': request.negative_prompt,
        'image_width': width,
        'image_height': height,
        'image_steps': request.steps,
        'image_seed': request.image_seed,
        'image_batch_size': request.batch_size,
        'image_batch_count': request.batch_count,
        'image_cfg_scale': request.cfg_scale,
        'image_llm_variations': False,
    })

    # Exhaust generator, keep final result
    images = []
    for images, _ in generate(state, save_images=False):
        pass

    if not images:
        raise ServiceUnavailableError("Image generation failed or produced no images.")

    # Build response
    resp = {'created': int(time.time()), 'data': []}
    for img in images:
        b64 = _image_to_base64(img)

        image_obj = {'revised_prompt': request.prompt}

        if request.response_format == 'b64_json':
            image_obj['b64_json'] = b64
        else:
            image_obj['url'] = f'data:image/png;base64,{b64}'

        resp['data'].append(image_obj)

    return resp


def _image_to_base64(image) -> str:
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode('utf-8')
