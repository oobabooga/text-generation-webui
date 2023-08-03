import os
import time
import requests
from extensions.openai.errors import *


def generations(prompt: str, size: str, response_format: str, n: int):
    # Stable Diffusion callout wrapper for txt2img
    # Low effort implementation for compatibility. With only "prompt" being passed and assuming DALL-E
    # the results will be limited and likely poor. SD has hundreds of models and dozens of settings.
    # If you want high quality tailored results you should just use the Stable Diffusion API directly.
    # it's too general an API to try and shape the result with specific tags like negative prompts
    # or "masterpiece", etc. SD configuration is beyond the scope of this API.
    # At this point I will not add the edits and variations endpoints (ie. img2img) because they
    # require changing the form data handling to accept multipart form data, also to properly support
    # url return types will require file management and a web serving files... Perhaps later!
    base_model_size = 512 if not 'SD_BASE_MODEL_SIZE' in os.environ else int(os.environ.get('SD_BASE_MODEL_SIZE', 512))
    sd_defaults = {
        'sampler_name': 'DPM++ 2M Karras',  # vast improvement
        'steps': 30,
    }

    width, height = [int(x) for x in size.split('x')]  # ignore the restrictions on size

    # to hack on better generation, edit default payload.
    payload = {
        'prompt': prompt,  # ignore prompt limit of 1000 characters
        'width': width,
        'height': height,
        'batch_size': n,
    }
    payload.update(sd_defaults)

    scale = min(width, height) / base_model_size
    if scale >= 1.2:
        # for better performance with the default size (1024), and larger res.
        scaler = {
            'width': width // scale,
            'height': height // scale,
            'hr_scale': scale,
            'enable_hr': True,
            'hr_upscaler': 'Latent',
            'denoising_strength': 0.68,
        }
        payload.update(scaler)

    resp = {
        'created': int(time.time()),
        'data': []
    }

    # TODO: support SD_WEBUI_AUTH username:password pair.
    sd_url = f"{os.environ['SD_WEBUI_URL']}/sdapi/v1/txt2img"

    response = requests.post(url=sd_url, json=payload)
    r = response.json()
    if response.status_code != 200 or 'images' not in r:
        print(r)
        raise ServiceUnavailableError(r.get('error', 'Unknown error calling Stable Diffusion'), code=response.status_code, internal_message=r.get('errors',None))
    # r['parameters']...
    for b64_json in r['images']:
        if response_format == 'b64_json':
            resp['data'].extend([{'b64_json': b64_json}])
        else:
            resp['data'].extend([{'url': f'data:image/png;base64,{b64_json}'}])  # yeah it's lazy. requests.get() will not work with this

    return resp
