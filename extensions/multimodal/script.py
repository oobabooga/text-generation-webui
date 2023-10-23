import base64
import re
import time
from functools import partial
from io import BytesIO

import gradio as gr
import torch

from extensions.multimodal.multimodal_embedder import MultimodalEmbedder
from modules import shared
from modules.logging_colors import logger

params = {
    "add_all_images_to_prompt": False,
    # device to run vision encoder on
    "vision_device": None,
    # bits to load vision encoder in, either 16 or 32
    "vision_bits": 32,
    # device to run multimodal projector on
    "projector_device": None,
    # multimodal projector bits, either 32 or 16
    "projector_bits": 32
}


# If 'state' is True, will hijack the next chat generation
input_hijack = {
    'state': False,
    'value': ["", ""]
}


# initialized in ui, so that params are loaded from settings
multimodal_embedder: MultimodalEmbedder = None


def chat_input_modifier(text, visible_text, state):
    global input_hijack
    if input_hijack['state']:
        input_hijack['state'] = False
        return input_hijack['value'](text, visible_text)
    else:
        return text, visible_text


def add_chat_picture(picture, text, visible_text):
    # resize the image, so that shortest edge is at least 224 (size for CLIP), and at most 300 (to keep history manageable)
    # Adjusted to 336 for the values here, due to the increased resolution in llava-v1.5
    max_hw, min_hw = max(picture.size), min(picture.size)
    aspect_ratio = max_hw / min_hw
    shortest_edge = int(max(336 / aspect_ratio, 336))
    longest_edge = int(shortest_edge * aspect_ratio)
    w = shortest_edge if picture.width < picture.height else longest_edge
    h = shortest_edge if picture.width >= picture.height else longest_edge
    picture = picture.resize((w, h))

    buffer = BytesIO()
    picture.save(buffer, format="PNG")
    img_str = base64.b64encode(buffer.getvalue()).decode('utf-8')
    image = f'<img src="data:image/jpeg;base64,{img_str}">'

    if '<image>' in text:
        text = text.replace('<image>', image)
    else:
        text = image + '\n' + text

    if visible_text == '' or visible_text is None:
        visible_text = text
    elif '<image>' in visible_text:
        visible_text = visible_text.replace('<image>', image)
    else:
        visible_text = visible_text + '\n' + image

    return text, visible_text


def custom_tokenized_length(prompt):
    return multimodal_embedder.len_in_tokens(prompt)


def tokenizer_modifier(state, prompt, input_ids, input_embeds):
    global params
    start_ts = time.time()
    image_match = re.search(r'<img src="data:image/jpeg;base64,[A-Za-z0-9+/=]+">', prompt)

    if image_match is None:
        return prompt, input_ids, input_embeds

    prompt, input_ids, input_embeds, total_embedded = multimodal_embedder.forward(prompt, state, params)
    logger.info(f'Embedded {total_embedded} image(s) in {time.time()-start_ts:.2f}s')
    return (prompt,
            input_ids.unsqueeze(0).to(shared.model.device, dtype=torch.int64),
            input_embeds.unsqueeze(0).to(shared.model.device, dtype=shared.model.dtype))


def ui():
    global multimodal_embedder
    multimodal_embedder = MultimodalEmbedder(params)
    with gr.Column():
        picture_select = gr.Image(label='Send a picture', type='pil')
        # The models don't seem to deal well with multiple images
        single_image_checkbox = gr.Checkbox(False, label='Embed all images, not only the last one')
    # Prepare the input hijack
    picture_select.upload(
        lambda picture: input_hijack.update({"state": True, "value": partial(add_chat_picture, picture)}),
        [picture_select],
        None
    )
    picture_select.clear(lambda: input_hijack.update({"state": False, "value": ["", ""]}), None, None)
    single_image_checkbox.change(lambda x: params.update({"add_all_images_to_prompt": x}), single_image_checkbox, None)
    shared.gradio['Generate'].click(lambda: None, None, picture_select)
    shared.gradio['textbox'].submit(lambda: None, None, picture_select)
