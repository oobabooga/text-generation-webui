import base64
from functools import partial
from io import BytesIO
import re
import time
import gradio as gr
import torch

from modules import shared
from modules.extensions import apply_extensions
from modules.text_generation import (encode, get_max_prompt_length)
from PIL import Image

from transformers import CLIPImageProcessor, CLIPVisionModel
from huggingface_hub import hf_hub_download

projector_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
image_processor = CLIPImageProcessor.from_pretrained('openai/clip-vit-large-patch14', torch_dtype=torch.float16)
vision_tower = CLIPVisionModel.from_pretrained('openai/clip-vit-large-patch14', torch_dtype=torch.float16).to(projector_device)

projector_path = hf_hub_download('liuhaotian/LLaVA-13b-pretrain-projector-v0', 'LLaVA-13b-pretrain-projector-v0-CC3M-595K-original_caption.bin')
mm_projector = torch.nn.Linear(1024, 5120)
projector_data = torch.load(projector_path)
mm_projector.weight = torch.nn.Parameter(projector_data['model.mm_projector.weight'].to(dtype=torch.float16), False)
mm_projector.bias = torch.nn.Parameter(projector_data['model.mm_projector.bias'].to(dtype=torch.float16), False)
mm_projector = mm_projector.to(projector_device)


# If 'state' is True, will hijack the next chat generation with
# custom input text given by 'value' in the format [text, visible_text]
input_hijack = {
    'state': False,
    'value': ["", ""]
}

params = {
    "add_all_images_to_prompt": False
}

DEFAULT_IMAGE_PATCH_TOKEN = "<im_patch>"
DEFAULT_IM_START_TOKEN = "<im_start>"
DEFAULT_IM_END_TOKEN = "<im_end>"
IM_PATCH_ID = 32000
IM_START_ID = 32001
IM_END_ID = 32002


def generate_chat_picture(picture, text, visible_text):
    # resize the image, so that shortest edge is at least 224 (size for CLIP), and at most 300 (to keep history manageable)
    max_hw, min_hw = max(picture.size), min(picture.size)
    aspect_ratio = max_hw / min_hw
    shortest_edge = int(max(300 / aspect_ratio, 224))
    longest_edge = int(shortest_edge * aspect_ratio)
    w = shortest_edge if picture.width < picture.height else longest_edge
    h = shortest_edge if picture.width >= picture.height else longest_edge
    picture = picture.resize((w,h))

    buffer = BytesIO()
    picture.save(buffer, format="JPEG")
    img_str = base64.b64encode(buffer.getvalue()).decode('utf-8')
    visible = f'<img src="data:image/jpeg;base64,{img_str}">'
    internal = f'<image:{img_str}>'

    if visible_text == '' or visible_text is None:
        visible_text = text

    if '<image>' in text:
        text.replace('<image>', internal)
    else:
        text = text + '\n' + internal

    if '<image>' in visible_text:
        visible_text.replace('<image>', visible)
    else:
        visible_text = visible_text + '\n' + visible

    return text, visible_text

def len_in_tokens(text):
    images = re.findall(r"<image:[A-Za-z0-9+/=]+>", text)
    image_tokens = 0
    for image in images:
        image_tokens += 258
    return len(encode(re.sub(r"<image:[A-Za-z0-9+/=]+>", '', text))[0]) + image_tokens

def custom_generate_chat_prompt(user_input, state, **kwargs):
    impersonate = kwargs['impersonate'] if 'impersonate' in kwargs else False
    _continue = kwargs['_continue'] if '_continue' in kwargs else False
    also_return_rows = kwargs['also_return_rows'] if 'also_return_rows' in kwargs else False
    rows = [f"{state['context'].strip()}\n"]
    min_rows = 3

    # Finding the maximum prompt size
    chat_prompt_size = state['chat_prompt_size']
    if shared.soft_prompt:
        chat_prompt_size -= shared.soft_prompt_tensor.shape[1]
    max_length = min(get_max_prompt_length(state), chat_prompt_size)

    prefix1 = f"{state['name1']}: "
    prefix2 = f"{state['name2']}: "

    i = len(shared.history['internal']) - 1
    while i >= 0 and len_in_tokens(''.join(rows)) < max_length:
        if _continue and i == len(shared.history['internal']) - 1:
            rows.insert(1, f"{prefix2}{shared.history['internal'][i][1]}")
        else:
            rows.insert(1, f"{prefix2}{shared.history['internal'][i][1].strip()}{state['end_of_turn']}\n")

        string = shared.history['internal'][i][0]
        if string != '':
            rows.insert(1, f"{prefix1}{string.strip()}{state['end_of_turn']}\n")

        i -= 1

    if impersonate:
        min_rows = 2
        rows.append(f"{prefix1}")
    elif not _continue:
        # Adding the user message
        if len(user_input) > 0:
            rows.append(f"{prefix1}{user_input}{state['end_of_turn']}\n")

        # Adding the Character prefix
        rows.append(apply_extensions("bot_prefix", f"{prefix2}"))

    while len(rows) > min_rows and len_in_tokens(''.join(rows)) >= max_length:
        rows.pop(1)
    prompt = ''.join(rows)

    if also_return_rows:
        return prompt, rows
    else:
        return prompt

def tokenizer_modifier(state, prompt, input_ids, input_embeds):
    global params
    start_ts = time.time()
    image_matches = re.finditer(r"<image:([A-Za-z0-9+/=]+)>", prompt)
    images = [Image.open(BytesIO(base64.b64decode(match.group(1)))) for match in image_matches]

    if len(images) == 0:
        return prompt, input_ids, input_embeds

    for _ in images:
        # replace the image token with the image patch token in the prompt (each occurrence)
        replace_token = DEFAULT_IMAGE_PATCH_TOKEN * 256
        replace_token = DEFAULT_IM_START_TOKEN + replace_token + DEFAULT_IM_END_TOKEN
        prompt = re.sub(r"<image:([A-Za-z0-9+/=]+)>", replace_token, prompt, 1)
    input_ids = encode(prompt, add_bos_token=state['add_bos_token'], truncation_length=get_max_prompt_length(state))
    input_embeds = shared.model.model.embed_tokens(input_ids)

    images = image_processor(images, return_tensors='pt')['pixel_values']
    images = images.to(vision_tower.device, dtype=torch.float16)

    with torch.no_grad():
        image_forward_outs = vision_tower(images, output_hidden_states=True)
        select_hidden_state_layer = -2
        select_hidden_state = image_forward_outs.hidden_states[select_hidden_state_layer]
        image_features = select_hidden_state[:, 1:].to(projector_device)
        image_features = mm_projector(image_features)

    new_input_embeds = []
    cur_image_idx = 0
    total_embedded = 0
    for cur_input_ids, cur_input_embeds in zip(input_ids, input_embeds):
        image_start_tokens = torch.where(cur_input_ids == IM_START_ID)[0]
        if not torch.any(cur_input_ids == IM_PATCH_ID) or len(image_start_tokens) == 0:
            # multimodal LLM, but the current sample is not multimodal/truncated
            new_input_embeds.append(cur_input_embeds)
            continue

        if not params['add_all_images_to_prompt']:
            image_start_tokens = [image_start_tokens[-1]]
            cur_image_idx = -1

        for image_start_token_pos in image_start_tokens:
            cur_image_features = image_features[cur_image_idx]
            num_patches = cur_image_features.shape[0]
            cur_input_embeds = torch.cat((cur_input_embeds[:image_start_token_pos+1], cur_image_features, cur_input_embeds[image_start_token_pos + num_patches + 1:]), dim=0)
            cur_image_idx += 1
            total_embedded += 1
        new_input_embeds.append(cur_input_embeds)
    input_embeds = torch.stack(new_input_embeds, dim=0)
    print(f'Embedded {total_embedded} image(s) in {time.time()-start_ts:.2f}s')
    return prompt, input_ids.to(shared.model.device), input_embeds.to(shared.model.device)


def ui():
    with gr.Column():
        picture_select = gr.Image(label='Send a picture', type='pil')
        # I found that it doesn't deal super well with multiple images, and demo ui had a bug where it included only the last image anyway
        single_image_checkbox = gr.Checkbox(False, label='Embed all images, not only the last one')
    # Prepare the input hijack
    picture_select.upload(
        lambda picture: input_hijack.update({"state": True, "value": partial(generate_chat_picture, picture)}),
        [picture_select],
        None
    )
    single_image_checkbox.change(lambda x: params.update({"add_all_images_to_prompt": x}), single_image_checkbox, None)
    shared.gradio['Generate'].click(lambda: None, None, picture_select)
