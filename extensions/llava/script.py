import base64
import re
import time
from dataclasses import dataclass
from functools import partial
from io import BytesIO

import gradio as gr
import torch
from huggingface_hub import hf_hub_download
from PIL import Image
from transformers import CLIPImageProcessor, CLIPVisionModel

from modules import shared
from modules.extensions import apply_extensions
from modules.text_generation import encode, get_max_prompt_length

params = {
    "add_all_images_to_prompt": False,
    # device to run CLIP on
    "clip_device": None,
    # bits to load clip in either 32 or 16 (it doesn't support 8-bit)
    "clip_bits": 32,
    # clip repository
    "clip_repo": "openai/clip-vit-large-patch14",
    # device to run projector on
    "projector_device": None,
    # projector bits, either 32 or 16
    "projector_bits": 32,
    # projector repository
    "projector_repo": "liuhaotian/LLaVA-13b-delta-v0",
    # file with the projector weights
    "projector_file": "mm_projector.bin"
}


# If 'state' is True, will hijack the next chat generation
input_hijack = {
    'state': False,
    'value': ["", ""]
}


# initialized in ui, so that params are loaded from settings
llava_embedder = None


@dataclass
class Token:
    token: str
    id: int


class LLaVAEmbedder:
    IM_PATCH = Token("<im_patch>", 32000)
    IM_START = Token("<im_start>", 32001)
    IM_END = Token("<im_end>", 32002)

    def __init__(self):
        self.clip_device = self._get_device("clip_device")
        self.clip_dtype = self._get_dtype("clip_bits")
        self.projector_device = self._get_device("projector_device")
        self.projector_dtype = self._get_dtype("projector_bits")
        self.image_processor, self.vision_tower, self.mm_projector = self._load_models()

    def _get_device(self, setting_name):
        if params[setting_name] is None:
            return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        return torch.device(params[setting_name])

    def _get_dtype(self, setting_name):
        return torch.float32 if int(params[setting_name]) == 32 else torch.float16

    def _load_models(self):
        start_ts = time.time()

        print(f"LLaVA - Loading CLIP from {params['clip_repo']} as {self.clip_dtype} on {self.clip_device}...")
        image_processor = CLIPImageProcessor.from_pretrained(params["clip_repo"], torch_dtype=self.clip_dtype)
        vision_tower = CLIPVisionModel.from_pretrained(params["clip_repo"], torch_dtype=self.clip_dtype).to(self.clip_device)

        print(f"LLaVA - Loading projector from {params['projector_repo']} as {self.projector_dtype} on {self.projector_device}...")
        projector_path = hf_hub_download(params["projector_repo"], params["projector_file"])
        mm_projector = torch.nn.Linear(1024, 5120)
        projector_data = torch.load(projector_path)
        mm_projector.weight = torch.nn.Parameter(projector_data['model.mm_projector.weight'].to(dtype=self.projector_dtype), False)
        mm_projector.bias = torch.nn.Parameter(projector_data['model.mm_projector.bias'].to(dtype=self.projector_dtype), False)
        mm_projector = mm_projector.to(self.projector_device)

        print(f"LLaVA supporting models loaded, took {time.time() - start_ts:.2f} seconds")
        return image_processor, vision_tower, mm_projector

    def _update_prompt(self, prompt, images):
        for _ in images:
            # replace the image token with the image patch token in the prompt (each occurrence)
            replace_token = LLaVAEmbedder.IM_PATCH.token * 256
            replace_token = LLaVAEmbedder.IM_START.token + replace_token + LLaVAEmbedder.IM_END.token
            prompt = re.sub(r'<img src="data:image/jpeg;base64,([A-Za-z0-9+/=]+)">', replace_token, prompt, 1)
        return prompt

    def _extract_image_features(self, images):
        images = self.image_processor(images, return_tensors='pt')['pixel_values']
        images = images.to(self.clip_device, dtype=self.clip_dtype)

        with torch.no_grad():
            image_forward_outs = self.vision_tower(images, output_hidden_states=True)
            select_hidden_state_layer = -2
            select_hidden_state = image_forward_outs.hidden_states[select_hidden_state_layer]
            image_features = select_hidden_state[:, 1:].to(self.projector_device, dtype=self.projector_dtype)
            image_features = self.mm_projector(image_features)
        return image_features

    def forward(self, prompt, images, state):
        prompt = self._update_prompt(prompt, images)
        input_ids = encode(prompt, add_bos_token=state['add_bos_token'], truncation_length=get_max_prompt_length(state))[0]
        input_embeds = shared.model.model.embed_tokens(input_ids).to(self.projector_device)

        if input_ids[0] == LLaVAEmbedder.IM_PATCH.id:
            # prompt got truncated in the middle of an image, remove the image data
            im_end = torch.where(input_ids == LLaVAEmbedder.IM_END.id)[0][0]
            input_ids = input_ids[im_end+1:]
            input_embeds = input_embeds[im_end+1:]
            leftover_images = torch.where(input_ids == LLaVAEmbedder.IM_START.id)[0].shape[0]
            print(f"LLaVA - WARNING: removed {len(images) - leftover_images} image(s) from prompt. The generation might be broken, try decreasing max_new_tokens")
            images = images[-leftover_images:]
            if len(images) == 0:
                return prompt, input_ids, input_embeds, 0

        total_embedded = 0
        image_features = self._extract_image_features(images).to(self.projector_device)
        image_start_tokens = torch.where(input_ids == LLaVAEmbedder.IM_START.id)[0]

        if not torch.any(input_ids == LLaVAEmbedder.IM_PATCH.id) or len(image_start_tokens) == 0:
            # multimodal LLM, but the current prompt is not multimodal/truncated
            return prompt, input_ids, input_embeds, total_embedded

        cur_image_idx = 0
        if not params['add_all_images_to_prompt']:
            image_start_tokens = [image_start_tokens[-1]]
            cur_image_idx = -1

        for image_start_token_pos in image_start_tokens:
            cur_image_features = image_features[cur_image_idx]
            num_patches = cur_image_features.shape[0]
            input_embeds = torch.cat((input_embeds[:image_start_token_pos+1], cur_image_features, input_embeds[image_start_token_pos + num_patches + 1:]), dim=0)
            cur_image_idx += 1
            total_embedded += 1

        return prompt, input_ids, input_embeds, total_embedded

    @staticmethod
    def len_in_tokens(text):
        images = re.findall(r'<img src="data:image/jpeg;base64,[A-Za-z0-9+/=]+">', text)
        image_tokens = 0
        for _ in images:
            image_tokens += 258
        return len(encode(re.sub(r'<img src="data:image/jpeg;base64,[A-Za-z0-9+/=]+">', '', text))[0]) + image_tokens


def add_chat_picture(picture, text, visible_text):
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
    image = f'<img src="data:image/jpeg;base64,{img_str}">'


    if '<image>' in text:
        text = text.replace('<image>', image)
    else:
        text = text + '\n' + image

    if visible_text == '' or visible_text is None:
        visible_text = text
    elif '<image>' in visible_text:
        visible_text = visible_text.replace('<image>', image)
    else:
        visible_text = visible_text + '\n' + image

    return text, visible_text


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
    while i >= 0 and LLaVAEmbedder.len_in_tokens(''.join(rows)) < max_length:
        if _continue and i == len(shared.history['internal']) - 1:
            rows.insert(1, f"{prefix2}{shared.history['internal'][i][1]}")
        else:
            rows.insert(1, f"{prefix2}{shared.history['internal'][i][1].strip()}\n")

        string = shared.history['internal'][i][0]
        if string != '':
            rows.insert(1, f"{prefix1}{string.strip()}\n")

        i -= 1

    if impersonate:
        min_rows = 2
        rows.append(f"{prefix1}")
    elif not _continue:
        # Adding the user message
        if len(user_input) > 0:
            rows.append(f"{prefix1}{user_input}\n")

        # Adding the Character prefix
        rows.append(apply_extensions("bot_prefix", f"{prefix2}"))

    while len(rows) > min_rows and LLaVAEmbedder.len_in_tokens(''.join(rows)) >= max_length:
        rows.pop(1)
    prompt = ''.join(rows)

    if also_return_rows:
        return prompt, rows
    else:
        return prompt


def tokenizer_modifier(state, prompt, input_ids, input_embeds):
    global params
    start_ts = time.time()
    image_matches = re.finditer(r'<img src="data:image/jpeg;base64,([A-Za-z0-9+/=]+)">', prompt)
    images = [Image.open(BytesIO(base64.b64decode(match.group(1)))) for match in image_matches]

    if len(images) == 0:
        return prompt, input_ids, input_embeds

    prompt, input_ids, input_embeds, total_embedded = llava_embedder.forward(prompt, images, state)
    print(f'LLaVA - Embedded {total_embedded} image(s) in {time.time()-start_ts:.2f}s')
    return (prompt,
        input_ids.unsqueeze(0).to(shared.model.device, dtype=torch.int64),
        input_embeds.unsqueeze(0).to(shared.model.device, dtype=shared.model.dtype))


def ui():
    global llava_embedder
    llava_embedder = LLaVAEmbedder()
    with gr.Column():
        picture_select = gr.Image(label='Send a picture', type='pil')
        # I found that it doesn't deal super well with multiple images, and demo ui had a bug where it included only the last image anyway
        single_image_checkbox = gr.Checkbox(False, label='Embed all images, not only the last one')
    # Prepare the input hijack
    picture_select.upload(
        lambda picture: input_hijack.update({"state": True, "value": partial(add_chat_picture, picture)}),
        [picture_select],
        None
    )
    picture_select.clear(lambda: input_hijack.update({"state": False, "value": ["",""]}), None, None)
    single_image_checkbox.change(lambda x: params.update({"add_all_images_to_prompt": x}), single_image_checkbox, None)
    shared.gradio['Generate'].click(lambda: None, None, picture_select)
    shared.gradio['textbox'].submit(lambda: None, None, picture_select)
