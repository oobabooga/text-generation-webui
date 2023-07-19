import os
import time
import requests
from extensions.openai.errors import *


def generations(prompt: str, size: str, response_format: str, n: int):
    # Stable Diffusion callout wrapper for txt2img
    # Low effort implementation for compatibility. With only "prompt" being passed and assuming DALL-E
    # the results will be limited and likely poor. SD has hundreds of models and dozens of settings.
    # If you want high quality tailored results you should just use the Stable Diffusion API directly.
    # it's too general an API to try and shape the result with specific tags like "masterpiece", etc,
    # Will probably work best with the stock SD models.
    # SD configuration is beyond the scope of this API.
    # At this point I will not add the edits and variations endpoints (ie. img2img) because they
    # require changing the form data handling to accept multipart form data, also to properly support
    # url return types will require file management and a web serving files... Perhaps later!

    width, height = [int(x) for x in size.split('x')]  # ignore the restrictions on size

    # to hack on better generation, edit default payload.
    payload = {
        'prompt': prompt,  # ignore prompt limit of 1000 characters
        'width': width,
        'height': height,
        'batch_size': n,
        'restore_faces': True,  # slightly less horrible
    }

    resp = {
        'created': int(time.time()),
        'data': []
    }

    # TODO: support SD_WEBUI_AUTH username:password pair.
    sd_url = f"{os.environ['SD_WEBUI_URL']}/sdapi/v1/txt2img"

    response = requests.post(url=sd_url, json=payload)
    r = response.json()
    if response.status_code != 200 or 'images' not in r:
        raise ServiceUnavailableError(r.get('detail', [{'msg': 'Unknown error calling Stable Diffusion'}])[0]['msg'], code=response.status_code)
    # r['parameters']...
    for b64_json in r['images']:
        if response_format == 'b64_json':
            resp['data'].extend([{'b64_json': b64_json}])
        else:
            resp['data'].extend([{'url': f'data:image/png;base64,{b64_json}'}])  # yeah it's lazy. requests.get() will not work with this

    return resp
