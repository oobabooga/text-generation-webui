import base64
import re
import time
from pathlib import Path
from io import BytesIO

import torch
import importlib.util
from transformers import is_torch_npu_available, is_torch_xpu_available

from modules import shared
from modules.logging_colors import logger

def get_available_devices():
    devices = [None]
    # Check for CUDA and ROCm devices
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            devices.append(f"cuda:{i}")
    # Check for Apple M devices
    if torch.backends.mps.is_available():
        devices.append("mps")
    # Check for NPU
    if is_torch_npu_available():
        for i in range(torch.npu.device_count()):
            devices.append(f"npu:{i}")
    # Check for XPU
    if is_torch_xpu_available():
        for i in range(torch.xpu.device_count()):
            devices.append(f"xpu:{i}")
    # Add CPU
    devices.append("cpu")
    return devices

def get_available_pipelines(loader):
    if loader in ['llama.cpp', 'llamacpp_HF', 'AutoGPTQ', 'GPTQ-for-LLaMa']:
        pipelines_dir = Path(__file__).parent / 'pipelines'
        available_pipelines = [None]

        for subdir in pipelines_dir.iterdir():
            if subdir.is_dir():
                pipelines_file = subdir / 'pipelines.py'
                if pipelines_file.is_file():
                    spec = importlib.util.spec_from_file_location("pipelines", pipelines_file)
                    pipelines_module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(pipelines_module)
                    if hasattr(pipelines_module, 'available_pipelines'):
                        available_pipelines.extend(pipelines_module.available_pipelines)

        return available_pipelines
    else:
        return [None]

def add_chat_picture(picture, text, visible_text):
    # resize the image, so that shortest edge is at least 224 (size for CLIP), and at most 300 (to keep history manageable)
    # Adjusted to 336 for the values here, due to the increased resolution in llava-v1.5
    # Adjustable by user, between 224 and 500
    max_hw, min_hw = max(picture.size), min(picture.size)
    aspect_ratio = max_hw / min_hw
    shortest_edge = int(max(shared.args.shortest_edge_size / aspect_ratio, shared.args.shortest_edge_size))
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

def tokenizer_modifier(state, prompt, input_ids, input_embeds):
    params = {
        'add_all_images_to_prompt': shared.args.add_all_images_to_prompt,
        'vision_device': shared.args.vision_device,
        'vision_bits': shared.args.vision_bits,
        'projector_device': shared.args.projector_device,
        'projector_bits': shared.args.projector_bits,
    }
    start_ts = time.time()
    image_match = re.search(r'<img src="data:image/jpeg;base64,[A-Za-z0-9+/=]+">', prompt)

    if image_match is None:
        return prompt, input_ids, input_embeds

    prompt, input_ids, input_embeds, total_embedded = shared.multimodal_embedder.forward(prompt, state, params)
    logger.info(f'Embedded {total_embedded} image(s) in {time.time()-start_ts:.2f}s')
    return (prompt,
            input_ids.unsqueeze(0).to(shared.model.device, dtype=torch.int64),
            input_embeds.unsqueeze(0).to(shared.model.device, dtype=shared.model.dtype))